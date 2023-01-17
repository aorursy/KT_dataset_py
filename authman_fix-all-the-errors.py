# Note: This code was intended to be ran by your home machine

# and not the Kaggle kernel



from multiprocessing import Pool

from collections import Counter

from time import time



import numba, json, os, cv2, glob



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import skimage.feature
class_names = ['adult_females', 'adult_males', 'juveniles', 'pups', 'subadult_males']



traincsv = pd.read_csv('../Train/train.csv', index_col=0)



bad_train_ids = [str(i) + '.jpg' for i in [

    # 7 no longer here. Manually rotate it 180* and your should be good

    3, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 

    268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 

    507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 

    779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 

    913, 927, 946]]



file_names = os.listdir("../TrainDotted/")

file_names = sorted(file_names, key=lambda 

                    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item)) 



#processed = glob.glob('../dots/*.txt')

#if len(processed)>0: processed = [f[8:-4] for f in processed]



# remove trash

file_names = list(set(file_names) - set(bad_train_ids))# - set(processed))
@numba.jit

def jit_blob(img):

    # blob_log is the slowest executing of the three:

    #  skimage.feature.blob_dog

    #  skimage.feature.blob_doh

    #  skimage.feature.blob_log

    

    # I've experimented with OpenCV Hough circles and that was even slower.

    # Then again, that was before I added morphological erosion... hmm...

    # Feel free to replace this with a cv2 circle detection method. If

    # successfully implemented, it'll save us from weird annotations on

    # our images like 'DEAD' seen on one of the images, ROFL.

    return skimage.feature.blob_log(img, min_sigma=2.5, max_sigma=5, num_sigma=1, threshold=0.01, overlap=0.25)



@numba.jit

def meanpatch(img, x,y, mask=None):

    # Here's my first contribution:

    # Average a patch of pixels to get better accuracy

    if mask is not None:

        i = img[x-1:x+1, y-1:y+1].reshape(-1,3)

        m = mask[x-1:x+1, y-1:y+1].reshape(-1,1)

        m[m>0] = 1

        return (m*i).mean(axis=0)

    return img[x-1:x+1, y-1:y+1].reshape(-1,3).mean(axis=0)
# This method takes in a single filename



def coords(filename):

    # Read the Train and Train Dotted images

    # Looks like they're stored as BGR, which is weird to look at, so let's make it RGB

    image_1 = cv2.cvtColor(cv2.imread("../TrainDotted/" + filename), cv2.COLOR_BGR2RGB)

    image_2 = cv2.cvtColor(cv2.imread("../Train/" + filename), cv2.COLOR_BGR2RGB)



    # For safe keepings...

    width = image_1.shape[0]

    height = image_1.shape[1]

    if width != image_2.shape[0] or height != image_2.shape[1]:

        with open('../dots/anomalies', 'a') as f:

            f.write('width x height mismatch', filename)

        return

    

    # Let's mask out the blackened regions that exist on *Either* image

    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)

    mask_1[mask_1 < 17] = 0

    mask_1[mask_1 > 0] = 255



    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

    mask_2[mask_2 < 17] = 0

    mask_2[mask_2 > 0] = 255



    # Absolute difference between Train and Train Dotted

    img_diff = cv2.absdiff(image_1, image_2)

    img_diff = cv2.bitwise_or(img_diff, img_diff, mask=mask_1 & mask_2)



    # Convert to grayscale to be accepted by skimage.feature.blob_log

    img_diff = cv2.cvtColor(img_diff, cv2.COLOR_RGB2GRAY)

    

    # Second contribution:

    # JPEG compression causes some artification, so get rid of very small

    # discrepancies between source images.

    img_diff[img_diff < 5] = 0



    # Third contribution - we have a lot of faint and bright dots

    # left over to classify. Faint dots are at risk of being erased if we

    # increase the #7 value above, or if use morphological erosion since

    # that will make it harder to detect them. So instead, let's split the

    # image into two parts, bright and dark regions... run erosion on the

    # light portions only, then merge the results back together?

    kernel = np.ones((3,3),np.uint8)

    img_diff = cv2.morphologyEx(img_diff, cv2.MORPH_OPEN, kernel)



    # Bilateral Filtering

    # I've tried this by itself (e.g. without the MORPH_OPENING) and it

    # was a disaster. Maybe try it w/ the opening? Anyhow, looking at the

    # reported errors, I'm not seeing much where the script is wrong so...

    # maybe it's not needed...?

    #filter_dist = 7

    #img_diff = cv2.bilateralFilter(img_diff.astype(np.float32), d=filter_dist, sigmaColor=filter_dist*2, sigmaSpace=int(filter_dist/2))

    

    #cv2.imwrite("diff.png", img_diff) # For inspection ;-)

    

    # Detect bad data. If train and dotted images are very different then somethings wrong.

    avg_diff = img_diff.sum() / (img_diff.shape[0] * img_diff.shape[1])

    if avg_diff > 10:

        #print(avg_diff, 'diff. Skipping bad data:', filename)

        #skipped.append(filename)

        with open('../dots/anomalies', 'a') as f: f.write(avg_diff, filename)

        return # We're dipping since this will be multi-threaded



    # detect blobs

    blobs = jit_blob(img_diff)



    # Init buckets

    adult_males = []

    subadult_males = []

    pups = []

    juveniles = []

    adult_females = []

    

    # These centroids come from https://www.kaggle.com/threeplusone

    dist = 32**2

    cls_colors = np.array((

        (243,8,5),          # red

        (244,8,242),        # magenta

        (87,46,10),         # brown 

        (25,56,176),        # blue

        (38,174,21),        # green

    ), dtype=np.int16)



    for y, x, _ in blobs:

        # The documentation for skimage says we get back y,x,sigma...

        # but it seems like we're actually getting back x,y,sigma.

        # Anyhow, don't let this bother you, it works fine... trust me.

        

        # Get the coordinates for each blob, and give some padding for 

        # our mean function:

        y = int(max(1,y))

        x = int(max(1,x))

        y = min(y,width-2)

        x = min(x,height-2)

            

        # Get the avg color of the pixelpatch from Train Dotted in the center of the blob

        red = np.sum((cls_colors[0] - mp)**2) < dist

        magenta = np.sum((cls_colors[1] - mp)**2) < dist

        brown = np.sum((cls_colors[2] - mp)**2) < dist

        blue = np.sum((cls_colors[3] - mp)**2) < dist

        green = np.sum((cls_colors[4] - mp)**2) < dist



        # Decision tree to pick the class of the blob by looking at the color in Train Dotted

        if red: # r>204 and b<29 and g<26: # RED

            adult_males.append((x,y))

        elif magenta: # r>220 and b>204 and g<25: # MAGENTA

            subadult_males.append((x,y))

        elif green: # 6<r<64 and b<52 and 156<g<199: # GREEN

            pups.append((x,y))

        elif blue: #r<78 and  124<b<221 and 31<g<85: # BLUE

            juveniles.append((x,y)) 

        elif brown: #59<r<115 and b<49 and 19<g<80:  # BROWN

            adult_females.append((x,y))

            #cv2.circle(image_circles, (int(x),int(y)), 20, (0,0,255), 2)

        else:

            #errors += 1

            #cv2.circle(image_circles, (int(x),int(y)), 20, (255,255,255), 3)

            pass

        

    obj = json.dumps({

        'adult_males':   adult_males,

        'subadult_males':subadult_males,

        'adult_females': adult_females,

        'juveniles':     juveniles,

        'pups':          pups,

        'counts':        [len(adult_males), len(subadult_males), len(adult_females), len(juveniles), len(pups)]

    }, separators=(',',':'))



    #print(len(adult_males), len(subadult_males), len(adult_females), len(juveniles), len(pups))

    with open('../dots/' + filename[:-4] + '.txt', 'w') as f:

        f.write(obj)

    

    #cv2.imwrite("error.png", cv2.cvtColor(image_circles, cv2.COLOR_RGB2BGR))
# I have a quad-core desktop, so dedicate 6 cores

# That way I can browse the net w/o things getting choppy..

# This process takes ~10 min on my Intel® Core™ i7-4770 CPU @ 3.40GHz × 8, 64bit:

thread_pool = Pool(6)

thread_pool.map(coords, file_names)

thread_pool.close()

thread_pool.join()
def merge():

    # The threads don't share memory, so this process will load up all the 

    # separately written .csv outputs and create a unified dataset called train_cs.csv

    output = 'jpg,adult_males,subadult_males,adult_females,juveniles,pups\n'

    for filename in file_names:

        sfname = filename[:-4]

        with open('../dots/' + sfname + '.txt', 'r') as f:

            data = json.loads(f.read())

            output += "%s,%i,%i,%i,%i,%i\n" %(sfname,data['counts'][0],data['counts'][1],data['counts'][2],data['counts'][3],data['counts'][4])



    with open('../dots/train_cv.csv', 'w') as f:

        f.write(output)





def diff():

    # This guy will produce the final output which tells us how

    # well or bad we did compared to the provide dtrain.csv file.

    cv = pd.read_csv('../dots/train_cv.csv', index_col=0)



    results = []

    for index, row in cv.iterrows():

        fact = traincsv.loc[index].values

        test = np.array([row['adult_males'], row['subadult_males'], row['adult_females'], row['juveniles'], row['pups']])

        diff = fact-test

        if np.abs(diff).sum() > 0:

            results.append([index, fact-test, np.abs(fact-test).sum()])



    results = pd.DataFrame(results, columns=['index', 'diff', 'absdiff'])

    results.index = results['index']

    results.sort_index(inplace=True)

    results.sort_values(by='absdiff', inplace=True)



    print(results)

    totalerr=0

    with open('../dots/train_diff.csv', 'w') as f:

        for index, row in results.iterrows():

            f.write(str(row['absdiff']) + '\t' + str(row['index']) + '\t' + str(row['diff']) + '\n')

            totalerr+=int(row['absdiff'])



    print('totalerr',totalerr)

    

# Alright, let's run them:

merge()

diff()