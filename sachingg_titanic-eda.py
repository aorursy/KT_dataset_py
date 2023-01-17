# THIS ONE IS DONE using Randomboost 
# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

import statsmodels.api as sm

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn.metrics import make_scorer, accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from xgboost import XGBClassifier

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

test_df = pd.read_csv(r'../input/test.csv')

train_df = pd.read_csv(r'../input/train.csv')

train_df.head()
#Let's figure out how many passengers were actulay Survived

train_df[train_df.Survived == 1].shape[0]/train_df.shape[0]

#This data shows that only 38% of the people survived, which also matches with the data available on WikiPedia
train_df.info()
train_df.describe()
train_df.Age.hist()
test_df.Age.hist()
train_df.isna().sum()
train_df.Age.fillna(train_df.Age.mean(),inplace=True)

train_df.Age.hist()


def mean_normalize(df,col):

    df[col] = (df[col] - df[col].mean())/df[col].std()



def do_label_encoding(df,col):

    le =LabelEncoder()

    le.fit(df[col])

    df[col] = le.transform(df[col])



def do_mean_encoding(train,test,col1,target):

    _mean = train.groupby(col1)[target].mean()

    train[col1+"_mean"] = train[col1].map(_mean)

    test[col1+"_mean"] = test[col1].map(_mean)



def do_frequency_encoding(train,col1):

    _freq = train.groupby(col1).size()

    _freq = _freq/train.shape[0]

    train[col1+"_freq"] = train[col1].map(_freq)



def plot_corr(df):

    corr = df.corr()



    # Generate a mask for the upper triangle

    mask = np.triu(np.ones_like(corr, dtype=np.bool))



    # Set up the matplotlib figure

    f, ax = plt.subplots(figsize=(11, 9))



    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(220, 10, as_cmap=True)



    # Draw the heatmap with the mask and correct aspect ratio

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

                square=True, linewidths=.5, cbar_kws={"shrink": .5})



mean_normalize(train_df,'Age')

train_df.Age.hist()
sns.boxplot(train_df.Age)
#As per the Wikipedia ,en.wikivoyage.org › wiki › RMS_Titanic. 

train_df['Embarked'].fillna('S',inplace=True)
train_df.describe()
train_df.Fare.hist()
mean_normalize(train_df,'Fare')

train_df.Fare.hist()
sns.boxplot(train_df.Fare)
test_df.isna().sum()
test_df.Fare.fillna(test_df.Fare.mean(),inplace=True)

test_df.Age.fillna(test_df.Age.mean(),inplace=True)
mean_normalize(test_df,'Fare')

mean_normalize(test_df,'Age')

test_df.Age.hist()
test_df.Fare.hist()
sns.barplot(x='Pclass',y='Survived',data = train_df)
sns.barplot(x='Sex',y='Survived',data = train_df)
sns.barplot(x='SibSp',y='Survived',data = train_df)
def getDummy(df):

    #Lets create a feature which combines every combination of Sex,Pclass

    df['PclassSex'] = df["Sex"].str.cat(df["Pclass"].astype(str), sep="")

    output1 = pd.get_dummies(df.PclassSex)

    df = df.join(output1)

    df.head()

    return df
train_df = getDummy(train_df)

train_df.head()
#Let's plot a correlation Matrix, to understand the features better

plot_corr(train_df)
sns.barplot(x='male3',y='Survived',data = train_df)
sns.barplot(x='female3',y='Survived',data = train_df)
do_frequency_encoding(train_df,'Embarked')

drop = ['Sex','Pclass','Embarked','Name','Cabin','PassengerId','Ticket','PclassSex']

train_df.drop(drop,axis=1,inplace=True)

train_df.head()
test_df = getDummy(test_df)

do_frequency_encoding(test_df,'Embarked')

test_df.drop(drop,axis=1,inplace=True)

test_df.head()
X=train_df

Y = X.Survived

X.drop('Survived',axis=1,inplace=True)



clf = RandomForestClassifier(n_estimators=100,max_depth=10)

model = clf.fit(X,Y)



import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model).fit(X,Y)

eli5.show_weights(perm, feature_names = X.columns.tolist())