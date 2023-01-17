# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression   #逻辑回归

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',7)

data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

print(data_train.describe())

print(data_train.info())



print(data_test.describe())

print(data_test.info())
plt.figure(1)

plt.subplot(2,2,1)

sns.barplot(x='Pclass',y='Survived',hue='Sex',data=data_train) #hue为以这个为分界



plt.subplot(2,2,2)

sns.barplot(x='Embarked',y='Survived',hue='Sex',data=data_train)



plt.subplot(2,2,3)

sns.barplot(x='SibSp',y='Survived',hue='Sex',data=data_train)



plt.subplot(2,2,4)

sns.barplot(x='Parch',y='Survived',hue='Sex',data=data_train)



plt.show()

def data_analy(df):

    cols=df.columns

    for col in cols:

        if col=='Age':

            df[col]=df[col].fillna(df[col].mean())

            bins=(-1,0,6,12,18,25,35,60,100)  #元组

            group_names=['Unknown','Baby','Child','Teenager','Young','Adult','Wrinky','Old']

            categories=pd.cut(df[col],bins,labels=group_names)

            df.Age=categories



        elif col=='Cabin':

            df[col]=df[col].fillna('N')

            df[col]=df[col].apply(lambda x:x[0])



        elif col=='Fare':

            df[col] = df[col].fillna(df[col].mean())

            bins = (-1,0, 8, 15, 32, 600)

            group_names = ['zeros','1_quartile', '2_quartile', '3_quartile', '4_quartile']

            catagories_F = pd.cut(df[col], bins, labels=group_names)

            df[col] = catagories_F



        else:

            pass



    return df



data_train_deal=data_analy(data_train)

data_train_deal=data_train_deal.drop(['Name','Ticket','Embarked'],axis=1)



data_test_deal=data_analy(data_test)

data_test_deal=data_test_deal.drop(['Name','Ticket','Embarked'],axis=1)
plt.figure(2)



plt.subplot(3,1,1)

sns.barplot(x='Age',y='Survived',hue='Sex',data=data_train_deal)



plt.subplot(3,1,2)

sns.barplot(x='Cabin',y='Survived',hue='Sex',data=data_train_deal)



plt.subplot(3,1,3)

sns.barplot(x='Fare',y='Survived',hue='Sex',data=data_train_deal)

plt.show()
from sklearn.preprocessing import LabelEncoder 



def features_encode(df_train,df_test):

    cols = ['Sex', 'Age', 'Fare', 'Cabin']

    df_combined = pd.concat([df_train[cols], df_test[cols]])

    for col in cols:

        labelencoder = LabelEncoder()

        le = labelencoder.fit(df_combined[col])

        df_train[col] = le.transform(df_train[col])

        df_test[col] = le.transform(df_test[col])

    return df_train,df_test



data_train,test_data=features_encode(data_train_deal,data_test_deal)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression  

from sklearn.metrics import accuracy_score



X=data_train.iloc[:,2:]

y=data_train['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)



model=LogisticRegression()

model.fit(X_train,y_train)



Y_pred=model.predict(X_test)

print(accuracy_score(y_test,Y_pred))



# K-Fold

from sklearn.model_selection import KFold



def run_kfold(clf):

    kf = KFold(n_splits=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf.split(X):

        fold = fold + 1

        X_train, X_test = X.values[train_index], X.values[test_index]

        y_train, y_test = y.values[train_index], y.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy:", mean_outcome)



clf=LogisticRegression()

run_kfold(clf)
predictions=clf.predict(test_data.drop('PassengerId',axis=1))

pre_output=pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predictions})

pre_output.to_csv('predit.csv')

pre_output.head()