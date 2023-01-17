import os

import numpy as np

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from matplotlib import rcParams

import seaborn as sns

import cv2

import tqdm

from IPython.display import Image

%matplotlib inline
data = pd.read_csv('../input/identify-the-dance-form/train.csv')
data.describe()
sns.set()

data.target.value_counts().plot(kind='bar', y='Count', colormap='Paired').set_title('Dance Forms')
data.target.value_counts().plot(figsize=(20,10))
y = data.target
Image(filename='../input/identify-the-dance-form/train/1.jpg') 
Image(filename='../input/identify-the-dance-form/train/10.jpg') 
Image(filename='../input/identify-the-dance-form/train/12.jpg') 
X = []

y = []

j=0

path='/kaggle/input/identify-the-dance-form/train'

for img in tqdm.tqdm(data['Image']):

    img_path = os.path.join(path,img)

    img = cv2.imread(img_path)

    X.append(img)

    y.append(data['target'][j])

    j = j+1
data['Actual_Image'] = X
type(data.iloc[1,2]),data.iloc[1,2].shape
type(data.iloc[1,2]),data.iloc[18,2].shape
data_group = data.groupby('target', as_index=False)
rows, columns = (4,2)

fig=plt.figure(figsize=(20, 20))

i = 1

for group_name, group_df in data_group:

    fig.add_subplot(rows, columns, i)

    group_df = group_df.reset_index()

    plt.imshow(group_df['Actual_Image'][np.random.randint(low=0, high=len(group_df))])

    plt.gca().set_title(group_name)

    i = i+1

plt.show()
rows, columns = (4,2)

fig=plt.figure(figsize=(20, 20))

i = 1

for group_name, group_df in data_group:

    fig.add_subplot(rows, columns, i)

    group_df = group_df.reset_index()

    plt.imshow(group_df['Actual_Image'][np.random.randint(low=0, high=len(group_df))])

    plt.gca().set_title(group_name)

    i = i+1

plt.show()
img_height, img_width = (224,224)
data['Actual_Image'] = data['Actual_Image'].apply(lambda x : cv2.resize(x,(img_height, img_width)))
X_train, X_test, y_train, y_test = train_test_split(data['Actual_Image'].apply(lambda x: x.flatten()), data['target'], test_size=0.3, random_state=42 )
X_train.shape, y_train.shape, X_test.shape, y_test.shape
arr = []

for each in X_train:

    arr.append(each)

arr = np.array(arr)

arr.shape
X_train = arr
arr = []

for each in X_test:

    arr.append(each)

arr = np.array(arr)
arr.shape
X_test = arr
X_train.shape, y_train.shape, X_test.shape, y_test.shape
knn_model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

knn_model.fit(X_train, y_train)
knn_model.score(X_test, y_test)
accuracy_score_list = []

for k in tqdm.tqdm(range(3, 50)):

    knn_model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    knn_model.fit(X_train, y_train)

    accuracy_score_list.append(knn_model.score(X_test, y_test))

sns.lineplot(range(3,50), accuracy_score_list)

plt.title("K Vs Accuracy")

plt.xlabel("K")

plt.ylabel("Accuracy")

plt.show()