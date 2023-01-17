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
iris = pd.read_csv(os.path.join(dirname, filename))
iris.head()
# Various Species Classes are:



iris['species'].value_counts()
iris.describe()
iris.shape
iris_df = iris[iris['species'] != 'Iris-virginica']
iris_df['species'].value_counts()
# Converting categorical value to numeric value



from sklearn import preprocessing
flower_class = preprocessing.LabelEncoder()

flower_class.fit(['Iris-setosa', 'Iris-versicolor'])

iris_df['species'] = flower_class.transform(iris_df['species'])
iris_df.head()
iris_df['species'].unique()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



x = iris_df[['sepal_length','sepal_width','petal_length','petal_width']]

y = iris_df['species']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=3)
x_train.shape , x_test.shape
y_train.shape , y_test.shape
LR = LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)
yhat = LR.predict(x_test)

yhat
y_test
yhat_prob = LR.predict_proba(x_test)

yhat_prob
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,yhat,labels=[0,1])
from sklearn.metrics import accuracy_score



score =accuracy_score(y_test,yhat)

score
iris.head()
# Convert categorical values to numeric values
iris['species'].value_counts()
iris['species'].unique()
flowers = preprocessing.LabelEncoder()

flowers.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

iris['species'] = flowers.transform(iris['species'])
iris['species'].unique()
iris.dtypes
# Convert the label types to int



iris[['sepal_length','sepal_width','petal_length','petal_width']] = iris[['sepal_length','sepal_width','petal_length','petal_width']].astype('int')
iris.dtypes
x = iris[['sepal_length','sepal_width','petal_length','petal_width']]

y = iris['species']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=3)
x_train.shape , x_test.shape
y_train.shape,y_test.shape
from sklearn import svm
clf = svm.SVC(kernel='rbf')

clf.fit(x_train,y_train)

yhat = clf.predict(x_test)
yhat
confusion_matrix(y_test,yhat)
from sklearn.metrics import f1_score

f1_score(y_test, yhat, average='weighted') 
from sklearn.metrics import jaccard_similarity_score

jaccard_similarity_score(y_test, yhat)