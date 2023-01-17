import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import glob

import PIL.Image as Image



def get_folder_names(mode, dataid= str):

    return f'../input/landmark-recognition-2020/{mode}/{dataid[0]}/{dataid[1]}/{dataid[2]}/{dataid}.jpg'
train_csv = pd.read_csv('../input/landmark-recognition-2020/train.csv')

print('Number of landmark images in train set:', len(train_csv))

train_csv.head()

submission_csv= pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')

submission_csv.head()
print('Any missing values?\n', train_csv.isnull().values.any()) 
print('Any Duplicates?', train_csv.duplicated().values.any())
landmark_count = train_csv.landmark_id.value_counts()
fig = plt.figure(figsize = (20, 7))

sns.distplot(landmark_count, hist = False);

plt.title('Class Distribution', size = 20);

plt.xlabel('number of images', size = 15);
limits = [None, (0,200), (0,100)]

fig = plt.figure(figsize = (20, 7))

for i, lim in enumerate(limits):

    plt.subplot(len(limits),1,i+1)

    sns.boxplot(landmark_count)

    plt.xlim(lim)

    plt.title(lim)

plt.tight_layout()



print((landmark_count>200).sum())
landmark_count_id = list(landmark_count.index)
def get_images(data, landmarkid, num):

    sub = data.id[data.landmark_id == landmarkid] 

    fig = plt.figure(figsize= (10, 10))

    fig.suptitle(f"landmark ID  {landmarkid}")

    for i in range(num):

        if num > 3:

            plt.subplot(num ** (1/2), num ** (1/2), i+1)

        else:

            plt.subplot(1, num, i+1)

        img = Image.open(get_folder_names('train' ,list(sub)[i]))

        plt.imshow(img)

        plt.axis('off')

    



get_images(train_csv, landmark_count_id[0], 9)
get_images(train_csv, landmark_count_id[1], 9)
get_images(train_csv, landmark_count_id[8], 9)
get_images(train_csv, landmark_count_id[-1], 2)
get_images(train_csv, landmark_count_id[-2], 2)
get_images(train_csv, landmark_count_id[-3], 2)
a = glob.glob('../input/landmark-recognition-2020/test/*/*/*/*.jpg')

num = 10

fig = plt.figure(figsize = (20, 10))

for i in range(num):

    plt.subplot(1, num, i+1)

    plt.axis('off')

    randint = np.random.randint(0, len(a))

    img = Image.open(a[randint])

    plt.imshow(img)

    

    