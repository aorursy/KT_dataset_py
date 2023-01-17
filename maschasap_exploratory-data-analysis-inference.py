import numpy as np 

import pandas as pd 



import os

import glob

import cv2



import seaborn as sns

import matplotlib.pyplot as plt

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



from scipy import stats



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline

%config InlineBackend.figure_format = 'png'
df_train = pd.read_csv('../input/landmark-retrieval-2020/train.csv')

print(f'Train df consumes {df_train.memory_usage().sum() / 1024**2:.4f} MB of RAM and has a shape {df_train.shape}')

df_train.sample(5)
value_counts = df_train.landmark_id.value_counts().reset_index().rename(columns={'landmark_id': 'count', 'index': 'landmark_id'})

value_counts_sorted = value_counts.sort_values('count')

value_counts
plt.figure(figsize=(12, 6))

plt.title('landmark_id distribution')

sns.distplot(df_train['landmark_id']);
plt.figure(figsize=(12,6))

p1=sns.distplot(value_counts, color="b").set_title('Number of images per class')
plt.figure(figsize=(12, 6))

sns.set()

plt.title('Training set: number of images per class (line plot logarithmically scaled)')

ax = value_counts['count'].plot(logy=True, grid=True)

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images");
sns.set()

ax = value_counts.boxplot(column='landmark_id')

ax.set_yscale('log')
sns.set()

res = stats.probplot(df_train['landmark_id'], plot=plt)
plt.figure(figsize=(12, 6))

sns.set()

landmarks_fold_sorted = value_counts_sorted

ax = landmarks_fold_sorted.plot.scatter(

     x='landmark_id',y='count',

     title='Training set: number of images per class(statter plot)')

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images");
threshold = [2, 3, 5, 10, 20, 50, 100]

for num in threshold:    

    print("Number of classes under {}: {}/{} "

          .format(num, (df_train['landmark_id'].value_counts() < num).sum(), 

                  len(df_train['landmark_id'].unique()))

          )
sns.set()

plt.figure(figsize=(14, 9))

plt.title('Most frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=value_counts_sorted.tail(25),

            label="Count")

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.show()
sns.set()

plt.figure(figsize=(14, 9))

plt.title('Least frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=value_counts_sorted.head(25),

            label="Count")

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.show()
train_list = glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')

test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')

index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')
print( 'Query', len(test_list), ' test images in ', len(index_list), 'index images')
def plot_random_images(data_list, seed=2020_07_02, n_rows=3, n_cols=3):

    

    np.random.seed(seed)

    random_ids = np.random.choice(range(len(data_list)), n_rows * n_cols, False)



    plt.rcParams["axes.grid"] = False

    f, axarr = plt.subplots(n_rows, n_cols, figsize=(24, 22))



    curr_row = 0

    for i, random_id in enumerate(random_ids):

        example = cv2.imread(data_list[random_id])

        example = example[:,:,::-1]



        col = i % n_cols

        axarr[col, curr_row].imshow(example)

        if col == n_cols - 1:

            curr_row += 1
plot_random_images(test_list)
plot_random_images(index_list)
all_ids = df_train.landmark_id.unique()



np.random.seed(2020)

n_random, len_row = 5, 3

random_ids = np.append(np.random.choice(all_ids, n_random, False), [138982])
plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(n_random+1, len_row, figsize=(len_row*7, 6*(n_random+1)))



curr_row = 0

for random_id in random_ids:

    images = df_train.query(f'landmark_id == {random_id}').sample(len_row)['id']

    for i, img in enumerate(images):

        arg_img = int(np.argwhere(list(map(lambda x: img in x, train_list))).ravel())

        example = cv2.imread(train_list[arg_img])

        example = example[:,:,::-1]



        col = i % len_row

        axarr[curr_row, col].imshow(example)

        if col == len_row - 1:

            curr_row += 1
np.random.seed(0)

n_random, len_row = 3, 3

random_ids = np.random.choice(all_ids, n_random, False)



plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(n_random, len_row, figsize=(len_row*7, 6*n_random))



curr_row = 0

for random_id in random_ids:

    images = df_train.query(f'landmark_id == {random_id}').sample(len_row)['id']

    for i, img in enumerate(images):

        arg_img = int(np.argwhere(list(map(lambda x: img in x, train_list))).ravel())

        example = cv2.imread(train_list[arg_img])

        example = example[:,:,::-1]



        col = i % len_row

        axarr[curr_row, col].imshow(example)

        if col == len_row - 1:

            curr_row += 1