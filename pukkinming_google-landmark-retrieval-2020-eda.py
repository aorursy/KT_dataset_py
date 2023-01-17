import os

import glob

import math

import numpy as np 

import pandas as pd 

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt

import cv2
!ls ../input/landmark-retrieval-2020
!ls ../input/landmark-retrieval-2020/train
!ls ../input/landmark-retrieval-2020/train/0
!ls ../input/landmark-retrieval-2020/train/0/0/
!ls ../input/landmark-retrieval-2020/train/0/0/0 | wc -l
!ls ../input/landmark-retrieval-2020/train/0/0/1 | wc -l
!ls ../input/landmark-retrieval-2020/train/9/9/f | wc -l
!ls ../input/landmark-retrieval-2020/test/0
!ls ../input/landmark-retrieval-2020/test/1/0
train_list = glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')

test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')

index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')

print("There are {} images in train folder, {} images in test folder, and {} images in index folder.".format(

    len(train_list), len(test_list), len(index_list)))
train_file_path = '../input/landmark-retrieval-2020/train.csv'

df_train = pd.read_csv(train_file_path)



print("Training data size:", df_train.shape)

print("Training data columns: {}\n\n".format(df_train.columns))

print(df_train.info())
df_train.head(5)
df_train.tail(5)
missing = df_train.isnull().sum()

percent = missing/df_train.count()

missing_train_data = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])

missing_train_data.head()
print("Minimum of landmark_id: {}, maximum of landmark_id: {}".format(df_train['landmark_id'].min(), df_train['landmark_id'].max()))

print("Number of unique landmark_id: {}".format(len(df_train['landmark_id'].unique())))

print(df_train['landmark_id'].unique())
sns.set()

plt.title('Training set: number of images per class(line plot)')

sns.set_color_codes("pastel")

landmarks_fold = pd.DataFrame(df_train['landmark_id'].value_counts())

landmarks_fold.reset_index(inplace=True)

landmarks_fold.columns = ['landmark_id','count']

ax = landmarks_fold['count'].plot(logy=True, grid=True)

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images")
df_count = pd.DataFrame(df_train.landmark_id.value_counts().sort_values(ascending=False))

df_count.reset_index(inplace=True)

df_count.columns = ['landmark_id', 'count']

df_count
sns.set()

plt.figure(figsize=(9, 4))

plt.title('Most frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(

    x="landmark_id",

    y="count",

    data=df_count.head(10),

    label="Count",

    order=df_count.head(10).landmark_id)

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.show()
def print_img(class_id, df_class, figsize):

    file_path = "../input/landmark-retrieval-2020/train/"

    df = df_train[df_train['landmark_id'] == class_id].reset_index()

    

    print("Class {} - {}".format(class_id, df_class[class_id].split(':')[-1]))

    print("Number of images: {}".format(len(df)))

    

    plt.rcParams["axes.grid"] = False

    no_row = math.ceil(min(len(df), 12)/3) 

    f, axarr = plt.subplots(no_row, 3, figsize=figsize)



    curr_row = 0

    len_img = min(12, len(df))

    for i in range(len_img):

        img_name = df['id'][i] + ".jpg"

        img_path = os.path.join(

            file_path, img_name[0], img_name[1], img_name[2], img_name)

        example = cv2.imread(img_path)

        # uncomment the following if u wanna rotate the image

        # example = cv2.rotate(example, cv2.ROTATE_180)

        example = example[:,:,::-1]



        col = i % 3

        axarr[curr_row, col].imshow(example)

        axarr[curr_row, col].set_title("{}. {} ({})".format(

            class_id, df_class[class_id].split(':')[-1], df['id'][i]))

        if col == 2:

            curr_row += 1
# From: https://www.kaggle.com/sudeepshouche/identify-landmark-name-from-landmark-id

url = 'https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv'

df_class = pd.read_csv(url, index_col = 'landmark_id', encoding='latin', engine='python')['category'].to_dict()
class_id = 138982

print_img(class_id, df_class, (24, 18))
class_id = 126637

print_img(class_id, df_class, (24, 18))
threshold = [2, 3, 5, 10, 20, 50, 100, 200, 1000]

total = len(df_train['landmark_id'].unique())

for num in threshold:

    cnt = (df_train['landmark_id'].value_counts() < num).sum()

    print("Number of classes with {} images or less: {}/{} ({:.2f}%)".format(

        num, cnt, total, cnt/total*100))
plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(4, 3, figsize=(24, 22))



curr_row = 0

for i in range(12):

    example = cv2.imread(train_list[i])

    # uncomment the following if u wanna rotate the image

    # example = cv2.rotate(example, cv2.ROTATE_180)

    example = example[:,:,::-1]

    

    col = i % 4

    axarr[col, curr_row].imshow(example)

    axarr[col, curr_row].set_title(train_list[i])

    if col == 3:

        curr_row += 1
plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(4, 3, figsize=(24, 22))



curr_row = 0

for i in range(12):

    example = cv2.imread(test_list[i])

    # uncomment the following if u wanna rotate the image

    # example = cv2.rotate(example, cv2.ROTATE_180)

    example = example[:,:,::-1]

    

    col = i % 4

    axarr[col, curr_row].imshow(example)

    axarr[col, curr_row].set_title(test_list[i])

    if col == 3:

        curr_row += 1
plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(4, 3, figsize=(24, 22))



curr_row = 0

for i in range(12):

    example = cv2.imread(index_list[i])

    # uncomment the following if u wanna rotate the image

    # example = cv2.rotate(example, cv2.ROTATE_180)

    example = example[:,:,::-1]

    

    col = i % 4

    axarr[col, curr_row].imshow(example)

    axarr[col, curr_row].set_title(index_list[i])

    if col == 3:

        curr_row += 1
class_id = 1

print_img(class_id, df_class, (24, 12))
class_id = 7

print_img(class_id, df_class, (24, 12))
class_id = 9

print_img(class_id, df_class, (24, 16))
class_id = 11

print_img(class_id, df_class, (24, 16))
class_id = 12

print_img(class_id, df_class, (24, 16))
class_id = 22

print_img(class_id, df_class, (24, 16))