# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# Database processing

data = pd.read_csv("/kaggle/input/cs98xspotifyclassification/CS98XClassificationTrain.csv",index_col="Id")

data.dropna(inplace=True)

data.drop(columns=['year','artist','title'],axis=0,inplace=True)

data.rename(index=str, columns={"top genre":'genre'},inplace = True)

data.head()



# Get the count of each value

value_counts = data['genre'].value_counts()



# Select the values where the count is less than 3 (or 5 if you like)

to_remove = value_counts[value_counts <= 2].index



# Keep rows where the city column is not in to_remove

data = data[~data.genre.isin(to_remove)]
data['genre'].value_counts()
# selection of wanted clounms

df = pd.DataFrame(data,columns = ("bpm","nrgy","dnce","dB","live","val","dur","acous","spch", "pop"))

df['genre'] = data.genre

df.head()

df.dtypes

df.columns
df['genre'].value_counts()
# creat features cloumns and target cloumn

X = df.drop(['genre'],axis='columns')

y = df.genre

from sklearn import preprocessing

X = preprocessing.normalize(X)
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

X_scaled = std_scaler.fit_transform(X)
# change data to array

X_scaled = np.asarray(X_scaled)

print(X)

y = np.asarray(y)

print(y)

# Divide the data as Train/Test dataset

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,random_state=4)

X_train.shape
y_train.shape
# Modeling (SVM with Sciklit-learn)

from sklearn import svm



classifier = svm.SVC(kernel='linear', gamma='auto', C=10)

classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_train)

y_predict[:150]
from sklearn.metrics import classification_report

print(classification_report(y_train,y_predict))


accuracy = classifier.score(X_train, y_train)

print(accuracy)
from sklearn import neighbors
X1 = df.drop(['genre'],axis='columns')

y1 = df.genre
std_scaler = StandardScaler()

X_scaled_1 = std_scaler.fit_transform(X1)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_scaled_1, y1, test_size=0.2)



clf = neighbors.KNeighborsClassifier()

clf.fit(X_train_1, y_train_1)
y1_predict = clf.predict(X_train_1)



y1_predict[:114]

accuracy = clf.score(X_train, y_train)

print(accuracy)


test = pd.read_csv("../input/cs98xspotifyclassification/CS98XClassificationTest.csv",index_col="Id")

test.drop(columns=['year','artist','title'],axis=0,inplace=True)

test.dropna(inplace=True)

test = preprocessing.normalize(test)

std_scaler = StandardScaler()

X_test_scaled = std_scaler.fit_transform(test)

z_predict = classifier.predict(X_test_scaled)

z = z_predict

print(z)
from sklearn import metrics

#SVM

print(metrics.classification_report(y_train, y_predict))
#KNN

print(metrics.classification_report(y_train_1, y1_predict))
import seaborn as sns
X = df.drop(['genre'],axis='columns')

y = df.genre
for index, item in enumerate(y):

    if (item == 'adult standards'):

        y[index] = 1

    elif(item == 'british invasion'):

        y[index] = 1

    elif(item == 'disco'):

        y[index] = 1

    elif(item == 'deep adult standards'):

        y[index] = 1

    elif(item == 'album rock'):

        y[index] = 2

    elif(item == 'glam rock'):

        y[index] = 2

    elif(item == 'dance rock'):

        y[index] = 2

    elif(item == 'art rock'):

        y[index] = 2

    elif(item == 'soft rock'):

        y[index] = 2

    elif(item == 'dance pop'):

        y[index] = 3

    elif(item == 'brill building pop'):

        y[index] = 3

    elif(item == 'europop'):

        y[index] = 3

    elif(item == 'boy band'):

        y[index] = 3

    elif(item == 'bubblegum dance'):

        y[index] = 3

    elif(item == 'barbadian pop'):

        y[index] = 3

    elif(item == 'eurodance'):

        y[index] = 3

    elif(item == 'pop'):

        y[index] = 3

    elif(item == 'classic uk pop'):

        y[index] = 3

    elif(item == 'disco house'):

        y[index] = 3

    elif(item == 'new wave pop'):

        y[index] = 3

    elif(item == 'atl hip hop'):

        y[index] = 4

    elif(item == 'british soul'):

        y[index] = 4

    elif(item == 'classic soul'):

        y[index] = 4

         

print(y)
Z = X.join(y)
g = sns.pairplot(Z,hue='genre')