import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

from scipy.stats import bernoulli

import seaborn as sns

print(check_output(["ls", "../input"]).decode("utf8"))

#print(check_output(["ls", "../input/train-jpg"]).decode("utf8"))
sample = pd.read_csv('../input/sample_submission_v2.csv')

print(sample.shape)

sample.head()
df = pd.read_csv('../input/train_v2.csv')

df.head()
df.shape
all_tags = [item for sublist in list(df['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]

print('total of {} non-unique tags in all training images'.format(len(all_tags)))

print('average number of labels per image {}'.format(1.0*len(all_tags)/df.shape[0]))
tags_counted_and_sorted = pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().sort_values(0, ascending=False)

tags_counted_and_sorted.head()
tags_counted_and_sorted.plot.barh(x='tag', y=0, figsize=(12,8))
from glob import glob

image_paths = sorted(glob('../input/train-jpg/*.jpg'))[0:1000]

image_names = list(map(lambda row: row.split("/")[-1][:-4], image_paths))

image_names[0:10]
plt.figure(figsize=(12,8))

for i in range(6):

    plt.subplot(2,3,i+1)

    plt.imshow(plt.imread(image_paths[i]))

    plt.title(str(df[df.image_name == image_names[i]].tags.values))
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

from sklearn.metrics import fbeta_score, precision_score, make_scorer, average_precision_score

import cv2

import warnings

import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfTransformer





n_samples = 200

rescaled_dim = 20



df['split_tags'] = df['tags'].map(lambda row: row.split(" "))

lb = MultiLabelBinarizer()

y = lb.fit_transform(df['split_tags'])

y = y[:n_samples]

X = np.squeeze(np.array([cv2.resize(plt.imread('../input/train-jpg/{}.jpg'.format(name)), (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR).reshape(1, -1) for name in df.head(n_samples)['image_name'].values]))

X = MinMaxScaler().fit_transform(X)



print(X.shape, y.shape, lb.classes_)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



from keras.models import Sequential

from keras.layers import Dense

import numpy

# fix random seed for reproducibility

numpy.random.seed(7)

model = Sequential()

model.add(Dense(12, input_dim=1600, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(17, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50, batch_size=5000)
scores = model.evaluate(X_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
X_sub = np.squeeze(np.array([cv2.resize(plt.imread('../input/test-jpg-v2/{}.jpg'.format(name)), (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR).reshape(1, -1) for name in sample['image_name'].values]))

X_sub = MinMaxScaler().fit_transform(X_sub)

X_sub.shape

y_sub = model.predict(X_sub)

y_sub.shape

print(y_sub)
test_tags = []

for index in range(y_sub.shape[0]):

    test_tags.append(' '.join(list(lb.classes_[np.where(y_sub[index, :] == 1)[0]])))



sample.head()
test_tags[0:20]
sample['tags'] = test_tags

sample.head()
image_paths = sorted(glob('../input/test-jpg-v2/*.jpg'))[0:1000]

image_names = list(map(lambda row: row.split("/")[-1][:-4], image_paths))

image_names[0:10]
plt.figure(figsize=(12,8))

for i in range(12):

    plt.subplot(3,4,i+1)

    plt.imshow(plt.imread(image_paths[i]))

    plt.title(str(sample[sample.image_name == image_names[i]].tags.values))