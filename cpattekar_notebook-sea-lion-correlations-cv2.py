import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

print(check_output(["ls", "../input"]).decode("utf8"))

from glob import glob

import seaborn as sns

from scipy import stats



df = pd.read_csv('../input/Train/train.csv')

print("{} training samples total".format(df.shape[0]))

df.head()
df[['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']].sum(axis=0).plot.barh()
def corrfunc(x, y, **kws):

    r, _ = stats.pearsonr(x, y)

    ax = plt.gca()

    ax.annotate("r = {:.3f}".format(r),

                xy=(.1, .9), xycoords=ax.transAxes)



g = sns.PairGrid(df[['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']], palette=["red"])

#g.map_upper(plt.scatter, s=10)

g.map_lower(plt.scatter, s=10)

g.map_diag(sns.distplot, kde=False)

#g.map_lower(sns.kdeplot, cmap="Blues_d")

g.map_lower(corrfunc)

#sns.pairplot(df)
all_types = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']

all_normalized_types = ['normalized_'+t for t in all_types]

row_counts = df[['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']].sum(axis=1)



for t in all_types:

    df['normalized_'+t] = df[t].divide(row_counts)



df.head()
sns.clustermap(

    df[all_normalized_types].fillna(0.0),

    col_cluster=False,

    cmap=plt.get_cmap('viridis'),

    figsize=(12,10)

)
from scipy.spatial.distance import pdist, squareform



sq_dists = squareform(pdist(df[all_normalized_types].values))

sq_dists[np.isnan(sq_dists)] = 0.0

sns.clustermap(

    sq_dists,

    cmap=plt.get_cmap('viridis'),

    figsize=(12,10)

)
training_images = glob('../input/Train/*.jpg')

training_dotted = glob('../input/TrainDotted/*.jpg')

len(training_images), len(training_dotted)
fig = plt.figure(figsize=(16,10))

for i in range(4):

    ax = fig.add_subplot(2,2,i+1)

    plt.imshow(plt.imread(training_images[i]))
from skimage.io import imread, imshow

from skimage.util import crop

import cv2



cropped_dotted = cv2.cvtColor(cv2.imread('../input/TrainDotted/8.jpg'), cv2.COLOR_BGR2RGB)[500:1500,2000:2800,:]

cropped_raw = cv2.cvtColor(cv2.imread('../input/Train/8.jpg'), cv2.COLOR_BGR2RGB)[500:1500,2000:2800,:]



fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(1,2,1)

plt.imshow(cropped_dotted)

ax = fig.add_subplot(1,2,2)

plt.imshow(cropped_raw)
diff = cv2.subtract(cropped_dotted, cropped_raw)

diff = diff/diff.max()

plt.figure(figsize=(12,8))

plt.imshow((diff > 0.20).astype(float))

plt.grid(False)
diff = cv2.absdiff(cropped_dotted, cropped_raw)

gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

ret,th1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)



cnts = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

print("Sea Lions Found: {}".format(len(cnts)))



x, y = [], []



lion_patches = []



for loc in cnts:

    ((xx, yy), _) = cv2.minEnclosingCircle(loc)



    # store patches of some sea lions

    if xx > 10 and xx < gray.shape[1] - 10:

        lion_patches.append(cropped_raw[yy-10:yy+10, xx-10:xx+10])



    x.append(xx)

    y.append(yy)



x = np.array(x)

y = np.array(y)
from scipy.stats.kde import gaussian_kde



k = gaussian_kde(np.vstack([x, y]), bw_method=0.5)

xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]

zi = k(np.vstack([xi.flatten(), yi.flatten()]))
fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)



# alpha=0.5 will make the plots semitransparent

ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)

ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)



ax1.set_xlim(x.min(), x.max())

ax1.set_ylim(y.min(), y.max())

ax2.set_xlim(x.min(), x.max())

ax2.set_ylim(y.min(), y.max())



ax1.imshow(cropped_raw, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')

ax2.imshow(cropped_raw, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
n_images_total = 16

n_images_per_row = 4



fig = plt.figure(figsize=(16,12))

for i in range(n_images_total):

    ax = fig.add_subplot(4,n_images_per_row,i+1)

    plt.grid(False)

    imshow(lion_patches[i])
# read images again for template matching code

cropped_dotted = cv2.cvtColor(cv2.imread('../input/TrainDotted/8.jpg'), cv2.COLOR_BGR2RGB)[1000:2000,2000:2800,:]

cropped_raw = cv2.cvtColor(cv2.imread('../input/Train/8.jpg'), cv2.COLOR_BGR2RGB)[1000:2000,2000:2800,:]

plt.imshow(cropped_raw)
plt.clf()

sealions = [

    cropped_raw[35:90, 505:520],

    cropped_raw[40:60, 510:515],

    cropped_raw[930:945, 610:665],

    cropped_raw[935:940, 630:645],

    cropped_raw[658:678, 395:448],

    cropped_raw[668:673, 415:420]

]

fig = plt.figure(figsize=(12,8))

for i in range(len(sealions)):

    ax = fig.add_subplot(1,len(sealions),i+1)

    imshow(sealions[i])
# All the 6 methods for comparison in a list

# not using all, to let kernel finish quickly

methods = [

    'cv2.TM_CCOEFF',

    'cv2.TM_CCOEFF_NORMED',

    'cv2.TM_CCORR',

    #'cv2.TM_CCORR_NORMED',

    #'cv2.TM_SQDIFF',

    #'cv2.TM_SQDIFF_NORMED'

]



def templateMatchFor(image, sealion):

    w, h = sealion.shape[1], sealion.shape[0]

    for meth in methods:

        method = eval(meth)



        # Apply template Matching

        res = cv2.matchTemplate(image,sealion,method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)



        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:

            top_left = min_loc

        else:

            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)



        cv2.rectangle(image,top_left, bottom_right, 255, 2)



        plt.figure(figsize=(12,8))

        plt.subplot(121)

        plt.imshow(res,cmap = 'gray')

        plt.title('Matching Result')

        plt.xticks([]), plt.yticks([])

        plt.subplot(122)

        plt.imshow(image,cmap = 'gray')

        plt.title('Detected Point')

        plt.xticks([]), plt.yticks([])

        plt.suptitle(meth)

        plt.show()



[templateMatchFor(cropped_raw, sealion) for sealion in sealions]
