# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Some libraries for visualization purposes

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from pandas.plotting import scatter_matrix

from collections import Counter



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
# Loading the data

df = pd.read_csv('/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')

df.head()
# Info and Describe of the data

print(df.shape)

print(df.describe())

print(df.info())
sns.heatmap(df.isnull());
df.dropna(inplace = True)

df.info()
df.head()
plt.figure(figsize = (15, 7))

plt.subplot(1, 3, 1)

sns.countplot(data=df, x='species')

plt.subplot(1, 3, 2)

sns.countplot(data=df, x='sex')

plt.subplot(1, 3, 3)

sns.countplot(data=df, x='island')

plt.tight_layout()

plt.show()
print(df.shape)

df = df.loc[df.sex != '.', :]

print(df.shape)
plt.figure(figsize = (8, 6))

plt.subplot(2, 2, 1)

sns.distplot(df['culmen_length_mm'])

plt.subplot(2, 2, 2)

sns.distplot(df['culmen_depth_mm'])

plt.subplot(2, 2, 3)

sns.distplot(df['flipper_length_mm'])

plt.subplot(2, 2, 4)

sns.distplot(df['body_mass_g'])

plt.tight_layout()

plt.show()
plt.figure(figsize = (8, 6))

plt.subplot(2, 2, 1)

sns.boxplot(df['culmen_length_mm'])

plt.subplot(2, 2, 2)

sns.boxplot(df['culmen_depth_mm'])

plt.subplot(2, 2, 3)

sns.boxplot(df['flipper_length_mm'])

plt.subplot(2, 2, 4)

sns.boxplot(df['body_mass_g'])

plt.tight_layout()

plt.show()
sns.heatmap(df.corr(), annot=True)

plt.show()
sns.pairplot(data=df, hue='species');
df.head()
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

train = df[df['is_train'] == True]

test = df[df['is_train'] == False]



train_x = train[train.columns[:len(train.columns) - 1]]

train_x = train_x.drop('species', axis=1) # Dropping the label

train_y = train['species']





test_x = test[test.columns[:len(test.columns) - 1]]

test_x = test_x.drop('species', axis=1) # Dropping the label

test_y = test['species']
train_x = pd.get_dummies(train_x)

test_x = pd.get_dummies(test_x);



# One hot encoding the label data to fit in the model
print(train_x.shape)

train_x.head()
print(train_y.shape)
# Euclidean Distance

def euclidean_distance(point1, point2):

    distance = 0

    for i in range(point1.shape[0]):

        distance += np.square(point1[i] - point2[i])

    return np.sqrt(distance)



# Manhattan Distance

def manhattan_distance(point1, point2):

    distance = 0

    for i in range(point1.shape[0]):

        distance += abs(point1[i] - point2[i])

    return distance
def knn(train_x, train_y, dis_func, sample, k):

    """

    Parameters:

    train_x: training samples

    train_y: corresponding labels

    dis_func: calculates distance

    sample: one test sample

    k: number of nearest neighbors

    

    Returns:

    cl: class of the sample

    """

    

    distances = {}

    for i in range(len(train_x)):

        d = dis_func(sample, train_x.iloc[i])

        distances[i] = d

    sorted_dist = sorted(distances.items(), key = lambda x : (x[1], x[0]))

    

    # take k nearest neighbors

    neighbors = []

    for i in range(k):

        neighbors.append(sorted_dist[i][0])

    

    #convert indices into classes

    classes = [train_y.iloc[c] for c in neighbors]

    

    #count each classes in top k

    counts = Counter(classes)

    

    #take vote of max number of samples of a class

    list_values = list(counts.values())

    list_keys = list(counts.keys())

    cl = list_keys[list_values.index(max(list_values))]

    

    return cl
model = knn(train_x, train_y, euclidean_distance, test_x.iloc[3], k=5)

print(model)

print(test_y.iloc[3])
# Calculate the accuracy

def get_accuracy(test_x, test_y, train_x, train_y, k):

    correct = 0

    for i in range(len(test_x)):

        sample = test_x.iloc[i]

        true_label = test_y.iloc[i]

        predicted_label_euclidean = knn(train_x, train_y, euclidean_distance, sample, k)

        if predicted_label_euclidean == true_label:

            correct += 1

    

    accuracy_euclidean = (correct / len(test_x)) * 100

    

    correct = 0

    for i in range(len(test_x)):

        sample = test_x.iloc[i]

        true_label = test_y.iloc[i]

        predicted_label_euclidean = knn(train_x, train_y, manhattan_distance, sample, k)

        if predicted_label_euclidean == true_label:

            correct += 1

    

    accuracy_manhattan = (correct / len(test_x)) * 100

    

    print("Model accuracy with Euclidean Distance is %.2f" %(accuracy_euclidean))

    print("Model accuracy with Manhattan Distance is %.2f" %(accuracy_manhattan))  
get_accuracy(test_x, test_y, train_x, train_y, k=5)
from sklearn import neighbors

from sklearn.metrics import accuracy_score
classifier = neighbors.KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)

classifier.fit(train_x, train_y)
y_pred = classifier.predict(test_x)

accuracy = accuracy_score(test_y, y_pred)

print('Accuracy with sklearn KNN with the same hyperparameters: {:.4f}'.format(accuracy))