import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from tqdm import tqdm_notebook as tqdm



import glob

import cv2

import os



from colorama import Fore, Back, Style



# Setting color palette.

plt.rcdefaults()

plt.style.use('dark_background')



import warnings

warnings.filterwarnings("ignore")
# Assigning paths to variables

INPUT_PATH = os.path.join('..', 'input')

DATASET_PATH = os.path.join(INPUT_PATH, 'landmark-recognition-2020')

TRAIN_IMAGE_PATH = os.path.join(DATASET_PATH, 'train')

TEST_IMAGE_PATH = os.path.join(DATASET_PATH, 'test')

TRAIN_CSV_PATH = os.path.join(DATASET_PATH, 'train.csv')

SUBMISSION_CSV_PATH = os.path.join(DATASET_PATH, 'sample_submission.csv')
train = pd.read_csv(TRAIN_CSV_PATH)

print("training dataset has {} rows and {} columns".format(train.shape[0],train.shape[1]))



submission = pd.read_csv(SUBMISSION_CSV_PATH)

print("submission dataset has {} rows and {} columns \n".format(submission.shape[0],submission.shape[1]))
# understand folder structure

print(Fore.YELLOW + "If you want to access image a40d00dc4fcc3a10, you should traverse as shown below:\n",Style.RESET_ALL)



print(Fore.GREEN + f"Image name: {train['id'].iloc[9]}\n",Style.RESET_ALL)



print(Fore.BLUE + f"First folder to look inside: {train['id'][9][0]}")

print(Fore.BLUE + f"Second folder to look inside: {train['id'][9][1]}")

print(Fore.BLUE + f"Second folder to look inside: {train['id'][9][2]}",Style.RESET_ALL)
print(Fore.BLUE + f"{'---'*20} \n Mapping for Training Data \n {'---'*20}")

data_label_dict = {'image': [], 'target': []}

for i in tqdm(range(train.shape[0])):

    data_label_dict['image'].append(

        TRAIN_IMAGE_PATH + '/' +

        train['id'][i][0] + '/' + 

        train['id'][i][1]+ '/' +

        train['id'][i][2]+ '/' +

        train['id'][i] + ".jpg")

    data_label_dict['target'].append(

        train['landmark_id'][i])



#Convert to dataframe

train_pathlabel = pd.DataFrame(data_label_dict)

print(train_pathlabel.head())

    

print(Fore.BLUE + f"{'---'*20} \n Mapping for Test Data \n {'---'*20}",Style.RESET_ALL)

data_label_dict = {'image': []}

for i in tqdm(range(submission.shape[0])):

    data_label_dict['image'].append(

        TEST_IMAGE_PATH + '/' +

        submission['id'][i][0] + '/' + 

        submission['id'][i][1]+ '/' +

        submission['id'][i][2]+ '/' +

        submission['id'][i] + ".jpg")



test_pathlabel = pd.DataFrame(data_label_dict)

print(test_pathlabel.head())
# list of unique landmark ids

train.landmark_id.unique()
# count of unique landmark_ids

print("There are", train.landmark_id.nunique(), "landmarks in the training dataset")
# each class count-wise

train.landmark_id.value_counts()
files = train_pathlabel.image[:10]

print(Fore.BLUE + "Shape of files from training dataset",Style.RESET_ALL)

for i in range(10):

    im = cv2.imread(files[i])

    print(im.shape)





print("------------------------------------")    

print("------------------------------------")    

print("------------------------------------")    



files = test_pathlabel.image[:10]

print(Fore.BLUE + "Shape of files from test dataset",Style.RESET_ALL)

for i in range(10):

    im = cv2.imread(files[i])

    print(im.shape)
plt.figure(figsize = (12, 8))



sns.kdeplot(train['landmark_id'], color="yellow",shade=True)

plt.xlabel("LandMark IDs")

plt.ylabel("Probability Density")

plt.title('Class Distribution - Density plot')



plt.show()
fig = plt.figure(figsize = (12,8))



count = train.landmark_id.value_counts().sort_values(ascending=False)[:10]



sns.countplot(x=train.landmark_id,

             order = train.landmark_id.value_counts().sort_values(ascending=False).iloc[:10].index)



plt.xlabel("LandMark Id")

plt.ylabel("Frequency")

plt.title("Top 10 Classes in the Dataset")



plt.show()
top6 = train.landmark_id.value_counts().sort_values(ascending=False)[:6].index



images = []



for i in range(6):

    img=cv2.imread(train_pathlabel[train_pathlabel.target == top6[i]]['image'].values[1])   

    images.append(img)



f, ax = plt.subplots(3,2, figsize=(20,15))

for i, img in enumerate(images):        

        ax[i//2, i%2].imshow(img)

        ax[i//2, i%2].axis('off')
fig = plt.figure(figsize = (12,8))



count = train.landmark_id.value_counts().sort_values(ascending=False)[:50]



sns.countplot(x=train.landmark_id,

             order = train.landmark_id.value_counts().sort_values(ascending=False).iloc[:50].index)



plt.xticks(rotation = 90)



plt.xlabel("LandMark Id")

plt.ylabel("Frequency")

plt.title("Top 50 Classes in the Dataset")



plt.show()
top50 = train.landmark_id.value_counts().sort_values(ascending=False).index[:50]



images = []



for i in range(50):

    img=cv2.imread(train_pathlabel[train_pathlabel.target == top50[i]]['image'].values[1])   

    images.append(img)



f, ax = plt.subplots(10,5, figsize=(20,15))

for i, img in enumerate(images):        

        ax[i//5, i%5].imshow(img)

        ax[i//5, i%5].axis('off')
fig = plt.figure(figsize = (10,6))



count = train.landmark_id.value_counts()[-10:]



sns.countplot(x=train.landmark_id,

             order = train_pathlabel.target.value_counts().iloc[-10:].index)



plt.xlabel("LandMark Id")

plt.ylabel("Frequency")

plt.title("Bottom 10 Classes in the Dataset")



plt.show()
files = train_pathlabel.image[:4]



fig = plt.figure(figsize = (20,9))



for i in range(4):

    img=cv2.imread(files[i])   

    plt.subplot(2,2,i+1)

    plt.hist(img.ravel(), bins = 256,color = 'gold')

    

plt.suptitle("Histogram for Grayscale Images",fontsize = 25)    

plt.show()
fig = plt.figure(figsize = (20,9))



for i in range(4):

    img=cv2.imread(files[i])   

    plt.subplot(2,2,i+1)

    plt.hist(img.ravel(), bins = 8, color = "coral")



plt.suptitle("Cumulative Histogram for Grayscale Images - Bin Size = 8",fontsize = 25)    

plt.show()
fig = plt.figure(figsize = (20,9))



for i in range(4):

    img=cv2.imread(files[i])   

    plt.subplot(2,2,i+1)

    plt.hist(img.ravel(), bins = 256,color = 'magenta',cumulative = True)



plt.suptitle("Cumulative Histogram for Grayscale Images",fontsize = 25)    

plt.show()
fig = plt.figure(figsize = (20,9))



for i in range(4):

    img=cv2.imread(files[i])   

    plt.subplot(2,2,i+1)

    plt.hist(img.ravel(), bins = 256, color = 'orange', )

    plt.hist(img[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)

    plt.hist(img[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)

    plt.hist(img[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)

    plt.xlabel('Intensity Value')

    plt.ylabel('Count')

    plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])



plt.suptitle("Color Histograms",fontsize = 25)    

plt.show()