import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from scipy import stats

import glob

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/landmark-recognition-2020/train.csv")

df.head()
df.duplicated().sum()
sub = pd.read_csv("../input/landmark-recognition-2020/sample_submission.csv")

sub.head()
sub.shape
df['landmark_id'].value_counts().hist()
total = df.isnull().sum().sort_values(ascending = False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending = False)

missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train_data.head()
temp = pd.DataFrame(df.landmark_id.value_counts().head(8))

temp.reset_index(inplace=True)

temp.columns = ['landmark_id','count']

temp
plt.figure(figsize = (9, 8))

plt.title('Most frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=temp,

            label="Count")

plt.show()
temp = pd.DataFrame(df.landmark_id.value_counts().tail(8))

temp.reset_index(inplace=True)

temp.columns = ['landmark_id','count']

temp
plt.figure(figsize = (9, 8))

plt.title('Least frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=temp,

            label="Count")

plt.show()
df.nunique()
plt.figure(figsize = (8, 8))

plt.title('Landmark ID Distribuition')

sns.distplot(df['landmark_id'])



plt.show()
print("Number of classes under 20 occurences",(df['landmark_id'].value_counts() <= 20).sum(),'out of total number of categories',len(df['landmark_id'].unique()))
# Landmark Id Density Plot

plt.figure(figsize = (8, 8))

plt.title('Landmark id density plot')

sns.kdeplot(df['landmark_id'], color="green", shade=True)

plt.show()
#Landmark id distribuition and density plot

plt.figure(figsize = (8, 8))

plt.title('Landmark id distribuition and density plot')

sns.distplot(df['landmark_id'],color='blue', kde=True,bins=100)

plt.show()
sns.set()

plt.title('Training set: number of images per class(line plot)')

sns.set_color_codes("pastel")

landmarks_fold = pd.DataFrame(df['landmark_id'].value_counts())

landmarks_fold.reset_index(inplace=True)

landmarks_fold.columns = ['landmark_id','count']

ax = landmarks_fold['count'].plot(logy=True, grid=True)

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images")
#Training set: number of images per class(statter plot)

sns.set()

landmarks_fold_sorted = pd.DataFrame(df['landmark_id'].value_counts())

landmarks_fold_sorted.reset_index(inplace=True)

landmarks_fold_sorted.columns = ['landmark_id','count']

landmarks_fold_sorted = landmarks_fold_sorted.sort_values('landmark_id')

ax = landmarks_fold_sorted.plot.scatter(\

     x='landmark_id',y='count',

     title='Training set: number of images per class(statter plot)')

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images")
sns.set()

ax = landmarks_fold_sorted.boxplot(column='count')

ax.set_yscale('log')
sns.set()

res = stats.probplot(df['landmark_id'], plot=plt)
train_list = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')
plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(4, 3, figsize=(25, 25))



curr_row = 0

for i in range(12):

    example = cv2.imread(train_list[i])

    example = example[:,:,::-1]

    

    col = i%4

    axarr[col, curr_row].imshow(example)

    if col == 3:

        curr_row += 1