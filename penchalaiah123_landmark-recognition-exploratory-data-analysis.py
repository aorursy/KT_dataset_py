import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from scipy import stats

import glob

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/landmark-recognition-2020/train.csv")

train.head()
# Check for Duplicates

train.duplicated().sum()
print("Training data size",train.shape)

submission = pd.read_csv("../input/landmark-recognition-2020/sample_submission.csv")

submission.head()
train['landmark_id'].value_counts().hist()
# missing data in training data 

total = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)

missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train_data.head()
# Occurance of landmark_id in decreasing order(Top categories)

temp = pd.DataFrame(train.landmark_id.value_counts().head(8))

temp.reset_index(inplace=True)

temp.columns = ['landmark_id','count']

temp
# Plot the most frequent landmark_ids

plt.figure(figsize = (9, 8))

plt.title('Most frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=temp,

            label="Count")

plt.show()
# Occurance of landmark_id in increasing order

temp = pd.DataFrame(train.landmark_id.value_counts().tail(8))

temp.reset_index(inplace=True)

temp.columns = ['landmark_id','count']

temp
# Plot the least frequent landmark_ids

plt.figure(figsize = (9, 8))

plt.title('Least frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=temp,

            label="Count")

plt.show()
train.nunique()
#Landmark ID distribution

plt.figure(figsize = (10, 8))

plt.title('Landmark ID Distribuition')

sns.distplot(train['landmark_id'])



plt.show()
print("Number of classes under 20 occurences",(train['landmark_id'].value_counts() <= 20).sum(),'out of total number of categories',len(train['landmark_id'].unique()))
# Landmark Id Density Plot

plt.figure(figsize = (8, 8))

plt.title('Landmark id density plot')

sns.kdeplot(train['landmark_id'], color="tomato", shade=True)

plt.show()
#Landmark id distribuition and density plot

plt.figure(figsize = (8, 8))

plt.title('Landmark id distribuition and density plot')

sns.distplot(train['landmark_id'],color='green', kde=True,bins=100)

plt.show()
sns.set()

plt.title('Training set: number of images per class(line plot)')

sns.set_color_codes("pastel")

landmarks_fold = pd.DataFrame(train['landmark_id'].value_counts())

landmarks_fold.reset_index(inplace=True)

landmarks_fold.columns = ['landmark_id','count']

ax = landmarks_fold['count'].plot(logy=True, grid=True)

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images")
#Training set: number of images per class(statter plot)

sns.set()

landmarks_fold_sorted = pd.DataFrame(train['landmark_id'].value_counts())

landmarks_fold_sorted.reset_index(inplace=True)

landmarks_fold_sorted.columns = ['landmark_id','count']

landmarks_fold_sorted = landmarks_fold_sorted.sort_values('landmark_id')

ax = landmarks_fold_sorted.plot.scatter(\

     x='landmark_id',y='count',

     title='Training set: number of images per class(statter plot)')

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images")
# Visualize outliers, min/max or quantiles of the landmarks count

sns.set()

ax = landmarks_fold_sorted.boxplot(column='count')

ax.set_yscale('log')
# Probability Plot

sns.set()

res = stats.probplot(train['landmark_id'], plot=plt)
train_list = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')
plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(4, 3, figsize=(24, 22))



curr_row = 0

for i in range(12):

    example = cv2.imread(train_list[i])

    example = example[:,:,::-1]

    

    col = i%4

    axarr[col, curr_row].imshow(example)

    if col == 3:

        curr_row += 1