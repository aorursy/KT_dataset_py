import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import missingno as msno

from tqdm import tqdm

import cv2

from keras.preprocessing import image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



labels = pd.read_csv('../input/labels.csv')



print (labels.shape)
labels.isnull().values.any()
msno.matrix(labels, width_ratios=(5,1),\

            figsize=(16,8), color=(0.2,0.2,0.2), fontsize=18, sparkline=True, labels=True)
labels.breed.nunique()
yy = pd.value_counts(labels['breed'])



fig, ax = plt.subplots()

fig.set_size_inches(15, 9)

sns.set_style("whitegrid")



ax = sns.barplot(x = yy.index, y = yy, data = labels)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 8)

ax.set(xlabel='Dog Breed', ylabel='Count')

ax.set_title('Distribution of Dog breeds')
yy.head()
yy.tail()
i = 0

img_height = []

img_width = []

for f, breed in tqdm(labels.values):

    img = cv2.imread('../input/train/{}.jpg'.format(f))

    img_height.append(img.shape[0])

    img_width.append(img.shape[1])

    i = f

i    
print(np.mean(img_height))

print(np.mean(img_width))
'''

import pandas as pd

import numpy as np

import os

import shutil





path = "/home/ubuntu/Kaggle/Dog_Classification/Data/"



label = pd.read_csv(path + "labels.csv")

train_path = '/home/ubuntu/Kaggle/Dog_Classification/Data/train/'

new_train_path = '/home/ubuntu/Kaggle/Dog_Classification/Data/new_train/'

#--- snippet to split train images into 120 folders ---



c = 0

for i in range(len(label)):

    l = label.id[i]

    for filename in os.listdir(train_path):	    

        f = filename[:-4]

        if (l == f):

            print c

            c+=1

            if not os.path.exists(new_train_path + label.breed[i]):

                os.makedirs(new_train_path + label.breed[i])

                shutil.copy2(train_path + filename, new_train_path + label.breed[i])				

            else:

                shutil.copy2(train_path + filename, new_train_path + label.breed[i])

'''                




fig, ax = plt.subplots()

img = image.load_img('../input/train/fff43b07992508bc822f33d8ffd902ae.jpg')

img = image.img_to_array(img)

ax.imshow(img / 255.) 

ax.axis('off')

plt.show()



fimg = np.fliplr(img)

fimg = image.img_to_array(fimg)

ax.imshow(fimg / 255.) 

ax.axis('off')

plt.show()

np.flipud



fig, ax = plt.subplots()

im = plt.imread('../input/train/fff43b07992508bc822f33d8ffd902ae.jpg')

plt.subplot(2, 1, 1)

ax.axis('off')

plt.imshow(im)

plt.subplot(2, 1, 2)

plt.imshow(np.fliplr(im))

ax.axis('off')

plt.show()


fig, ax = plt.subplots()

img = np.fliplr(img)

img = image.img_to_array(img)

ax.imshow(img / 255.) 

ax.axis('off')

plt.imshow(img)



img = image.img_to_array(fimg)

ax.imshow(img / 255.) 

ax.axis('off')

plt.show()

#plt.show(rimg)