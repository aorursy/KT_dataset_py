# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# load data train data
path = '/kaggle/input/titanic/'
df_train = pd.read_csv(path+'train.csv')
df_test = pd.read_csv(path+'test.csv')
df_train.head()
print(df_train.shape, df_test.shape)
df_train.isnull().sum()

# Replace the missing values by the median of the columns 
df_train['Age'].fillna((df_train['Age'].median()), inplace =True)
df_train['Embarked'].fillna(('NaN'), inplace =True)

df_test['Age'].fillna((df_test['Age'].median()), inplace =True)
df_test['Embarked'].fillna(('NaN'), inplace =True)

df_train.head()
df_train.info()
df_train.head()
# # Wrangling dataset

# # The columns PassengerId and Cabin can be droped because not realy relevant for the data exploaration
df_train2 = df_train.drop(columns =['PassengerId', 'Cabin', 'Name', 'Ticket'], axis =1)
# # Same with the test datas
df_test2 = df_test.drop(columns =['PassengerId','Cabin', 'Name', 'Ticket'], axis =1)
print(df_train2.head())

df_train2
df_train2.info()
# Check if  having the survival relation wit other variables

plt.figure(figsize=(10,10))

plt.subplot(4,1,1)
sns.countplot(x="SibSp", data=df_train2 ,hue="Survived")

plt.subplot(4,1,2)
sns.countplot(x='Pclass' ,data=df_train2 ,hue='Survived')

plt.subplot(4,1,3)
sns.countplot(x='Parch' ,data=df_train2 ,hue='Survived')


plt.subplot(4,1,4)
sns.distplot(df_train2[df_train2["Survived"] == 0]['Age'], label='0')
sns.distplot(df_train2[df_train2["Survived"] == 1]['Age'], label ='1')
plt.legend(title ='Survived')
# # Assign  a dummie number manually 
df_train3 = df_train2.copy()
df_test3 = df_test2.copy()
Replacement = {'Sex': {'female':0,'male':1}, 
               'Embarked':{ 'NaN':0,'S':1, 'C':2, 'Q':3}}
df_test3.replace(Replacement, inplace =True) # to run this only once.
df_test3.head()

df_train3.replace(Replacement, inplace =True) # to run this only once.
df_train3.head()
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text

class TextHandler(HandlerBase):
    def create_artists(self, legend, tup ,xdescent, ydescent,
                        width, height, fontsize,trans):
        tx = Text(width/2.,height/2,tup[0], fontsize=fontsize,
                  ha="center", va="center", color=tup[1], fontweight="bold")
        return [tx]

# sns.barplot(train["Embarked"])
ax =sns.countplot(x="Embarked", data= df_train)
handltext = df_train.Embarked.unique().tolist() #
labels = ["Southampton","Cherbourg","Queenstown", "Not Specified"]

t = ax.get_xticklabels()
labeldic = dict(zip(handltext, labels))
labels = [labeldic[h.get_text()]  for h in t]
handles = [(h.get_text(),c.get_fc()) for h,c in zip(t,ax.patches)]

ax.legend(handles, labels, handler_map={tuple : TextHandler()}) 

plt.show()

sns.countplot(x="Survived", data= df_train, hue="Embarked")
plt.show()
sns.barplot(df_train["Sex"],df_train['Age'])
plt.show()
sns.countplot(x="Sex", data= df_train ,hue="Embarked")
plt.show()
# Train and test data 

# test['Fare'].fillna(('NaN'), inplace =True)
test['Fare'].fillna((test['Fare'].median()), inplace =True)
test.isna().sum()


test = df_test3
train = df_train3

x_train = train.drop(columns =["Survived"],axis=1)
y_train = train.Survived

x_test = test


# Decision Tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import  BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

def model(algorithm, xtrain, ytrain, xtest):
    m = algorithm
    algorithm.fit(xtrain,ytrain)
    ypred = algorithm.predict(xtest)
    score = algorithm.score(xtrain, ytrain)
#     print('score of the model with training data:', score)

    return score,m

score_DT, mDT = model(DecisionTreeClassifier(), x_train, y_train, x_test)
score_RF, mRF = model(RandomForestClassifier(), x_train, y_train, x_test)
score_GB, mGB = model(GradientBoostingClassifier(), x_train, y_train, x_test)
score_Bagg, mBagg = model(BaggingClassifier(), x_train, y_train, x_test)
score_Ada, mAda = model(AdaBoostClassifier(), x_train, y_train, x_test)

score_list = [score_DT, score_RF, score_Ada, score_Bagg, score_GB]
names = ['Decision Tree Classifier',
         'Random Forest Classifier', 'Adaboost Classifier',
          'Bagging Classifier','Gradient Boosting Classifier']

evaluation = pd.DataFrame({'Model': names,'Score': score_list})
evaluation.sort_values(by="Score",ascending=False)
#Random Forest clssifier seems to perform best. Decission Tree also, but with huge number of Trees, it probably overfits
# Randome Forest:
RF = RandomForestClassifier()
RF.fit(x_train,y_train)
ypred = RF.predict(x_test)
print(ypred.tolist().count(0), ypred.tolist().count(1))
# So we seems to have 259 deaths and 159 survived
# The model can be improved with cross-validation, Kfold, and tuning the hyperparameters to have better accuracy. 
d = {'Survived': [159], 'Death': [259]}
df = pd.DataFrame(d)
df.to_csv('first_sub.csv', index=False)
