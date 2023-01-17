import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt
train = pd.read_csv( "../input/train.csv")

train.head()
df = pd.DataFrame(train)

print(df.columns)

df.head()
#df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

#df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

#df.head()

#df['hasmask'] = ~ df['EncodedPixels'].isna()

#df.head()

#mask_count_df = df1.groupby('ImageId').agg(np.sum).reset_index()

#mask_count_df.sort_values('hasmask', ascending=False, inplace=True)

#print(mask_count_df.shape)

#mask_count_df.head()

#sub_df = pd.read_csv('../input/sample_submission.csv')

#sub_df['ImageId'] = sub_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

#test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

#test_imgs.head()

#non_missing_train_id = mask_count_df[mask_count_df['hasmask'] > 0]

#non_missing_train_id.head()
df = df[ df['EncodedPixels'].notnull() ]

print( df.shape )

df.head()
import cv2

def rle2mask(rle, imgshape):

    width = imgshape[0]

    height= imgshape[1]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mask.reshape(height,width), k=1 ) )



fig=plt.figure(figsize=(20,100))

columns = 2

rows = 50

for i in range(1, 100+1):

    fig.add_subplot(rows, columns, i)

    

    fn = df['ImageId'].iloc[i].split('_')[0]

    img = cv2.imread( '../input/train_images/'+fn )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = rle2mask( df['EncodedPixels'].iloc[i], img.shape  )

    img[mask==1,0] = 255

    

    plt.imshow(img)

plt.show()
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import cv2
train_path = "../input/train_images"
def loadImages(path):

    # Put files into lists and return them as one list of size 4

    image_files = sorted([os.path.join(path, file)

         for file in os.listdir(path) if      file.endswith('.jpg')])

 

    return image_files
data = loadImages(train_path)
# Display one image

def display_one(a, title1 = "Original"):

    plt.imshow(a), plt.title(title1)

    plt.xticks([]), plt.yticks([])

    plt.show()

    

    

# Display two images

def display(a, b, title1 = "Original", title2 = "Edited"):

    plt.subplot(121), plt.imshow(a), plt.title(title1)

    plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(b), plt.title(title2)

    plt.xticks([]), plt.yticks([])

    plt.show()

    

    

# Preprocessing

def processing(data):

    # loading image

    # Getting 3 images to work with 

    img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data[:3]]

    print('Original size',img[0].shape)

    # --------------------------------

    # setting dim of the resize

    height = 220

    width = 220

    dim = (width, height)

    res_img = []

    for i in range(len(img)):

        res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)

        res_img.append(res)



    # Checking the size

    print("RESIZED", res_img[1].shape)

    

    # Visualizing one of the images in the array

    original = res_img[1]

    display_one(original)

    

    no_noise = []

    for i in range(len(res_img)):

        blur = cv2.GaussianBlur(res_img[i], (5, 5), 0)

        no_noise.append(blur)





    image = no_noise[1]

    display(original, image, 'Original', 'Blured')

    

    # Segmentation

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)



# Displaying segmented images

    display(original, thresh, 'Original', 'Segmented')

    

    # Further noise removal

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)



# sure background area

    sure_bg = cv2.dilate(opening, kernel, iterations=3)



# Finding sure foreground area

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)



# Finding unknown region

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)



#Displaying segmented back ground

    display(original, sure_bg, 'Original', 'Segmented Background')

    

    # Marker labelling

    ret, markers = cv2.connectedComponents(sure_fg)



# Add one to all labels so that sure background is not 0, but 1

    markers = markers + 1



# Now, marking the region of unknown with zero

    markers[unknown == 255] = 0



    markers = cv2.watershed(image, markers)

    image[markers == -1] = [255, 0, 0]



# Displaying markers on the image

    display(image, markers, 'Original', 'Marked')
processing(data)