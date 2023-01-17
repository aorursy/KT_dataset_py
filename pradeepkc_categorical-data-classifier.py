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
import pandas as pd

import  numpy as np

import seaborn as sb

import matplotlib.pyplot as plt

%matplotlib inline
train=pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

test=pd.read_csv("../input/cat-in-the-dat-ii/test.csv")
train.head()
test.head()
#size of the train and test data

print(train.shape)

print(test.shape)
train.isnull().sum()
#train.bin_1.mode()

train.isnull().sum().sum()
#NaN value count in test data

test.isnull().sum().sum()
#drop all the NaN values from columns in train as well as test data

train.dropna(inplace=True)

test.dropna(inplace=True)
train.head()
#value counts with graph

sb.countplot(train["bin_3"])
# when you observe the feature's in train data you can see that from nom_5 to nom_9 it just have some random values and it is not specifing any detail about the data we
sb.countplot(train["bin_4"])
sb.countplot(train["target"])
sb.countplot(train["month"])
sb.countplot(train["day"])
print(sb.countplot(train["nom_3"]))

print(sb.countplot(train["nom_2"]))



print(sb.countplot(train["nom_1"]))

print(sb.countplot(train["nom_0"]))
train.drop(labels="nom_7",axis=1,inplace=True)

train.drop(labels="nom_6",axis=1,inplace=True)
train.drop(labels="nom_5",axis=1,inplace=True)
train.head()
test.drop(labels=["nom_5","nom_6","nom_7","nom_8","nom_9"],axis=1,inplace=True)
test.info()
list1=["bin_3","bin_4","nom_0","nom_1","nom_2","nom_3","nom_4","ord_1","ord_2","ord_3","ord_4","ord_5"]

from sklearn.preprocessing import LabelEncoder

enc=LabelEncoder()

train["bin_3"]=enc.fit_transform(train.bin_3)
class multicolumnencoder:

    def __init__(self,columns=None):

        self.columns=columns

    def fit(self,X,y=None):

        return self

    def transform(self,X):

        """Transform value assigned to columns if values are not assigned transform all values"""

        output=X.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col]=LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname]=LabelEncoder().fit_transform(col)

        return output

    

    def fit_transform(self,X,y=None):

        return self.fit(X,y).transform(X)
train=multicolumnencoder(columns=["bin_4","nom_0","nom_1","nom_2","nom_3","nom_4","ord_1","ord_2","ord_3","ord_4","ord_5"]).fit_transform(train)
test=multicolumnencoder(columns=["bin_3","bin_4","nom_0","nom_1","nom_2","nom_3","nom_4","ord_1","ord_2","ord_3","ord_4","ord_5"]).fit_transform(test)
#train data is converted into train and test data



from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()

train.head()
test.head()
train.drop(labels="id",axis=1,inplace=True)

test.drop(labels="id",axis=1,inplace=True)
#train test split of train data

from sklearn.model_selection import train_test_split

X=train.iloc[:,:-1]

y=train.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=9)
model.fit(X_train,y_train)

y_predict=model.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
accuracy_score(y_test,y_predict)
y_predict=model.predict(test)