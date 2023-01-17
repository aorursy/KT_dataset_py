# THIS ONE IS DONE using Ensemble of Randomboost and XGBoost (score: 0.764 )
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



seed = 5



test_df = pd.read_csv(r'../input/test.csv')

train_df = pd.read_csv(r'../input/train.csv')
train_df.info()
train_df.isna().sum()
train_df.describe()
def mean_normalize(df,col):

    df[col] = (df[col] - df[col].mean())/df[col].std()



def do_label_encoding(df,col):

    le =LabelEncoder()

    le.fit(df[col])

    df[col] = le.transform(df[col])





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

train_df.isna().sum()
test_df.isna().sum()
#As per the Wikipedia ,en.wikivoyage.org › wiki › RMS_Titanic. 

train_df['Embarked'].fillna('S',inplace=True)

train_df.Age.fillna(train_df.Age.mean(),inplace=True)
test_df.Fare.fillna(test_df.Fare.mean(),inplace=True)

test_df.Age.fillna(test_df.Age.mean(),inplace=True)
train_df.isna().sum()
#Let's plot a correlation Matrix, to understand the features better

plot_corr(train_df)
do_label_encoding(train_df,'Sex')

do_label_encoding(train_df,'Pclass')

do_label_encoding(train_df,'Embarked')
do_label_encoding(test_df,'Sex')

do_label_encoding(test_df,'Embarked')

do_label_encoding(train_df,'Pclass')
mean_normalize(train_df,'Age')

mean_normalize(train_df,'Fare')

train_df['Fare'].hist()
sns.boxplot(train_df['Fare'])
#Since both test and train data have similar distributions and outliers, lets Normalize them and remove outliers
lower,upper = np.percentile(train_df['Fare'],[1,80])

y = np.clip(train_df['Fare'],lower,upper)

train_df['Fare'] = y
sns.boxplot(train_df['Fare'])
train_df['Fare'].hist()
test_df.isna().sum()
#Since there is one Nan, let'f fix it first

mean_normalize(test_df,'Age')

test_df.Age.hist()
lower,upper = np.percentile(test_df['Age'],[1,99])

print(upper,lower)

y = np.clip(test_df['Age'],lower,upper)

sns.boxplot(y)

test_df.Age = y

test_df.Age.hist()
upper,lower = np.percentile(train_df.Age,[0,93])

print(lower,upper)

y = np.clip(train_df.Age,lower,upper)

sns.boxplot(y)

#pd.Series(y).hist()
train_df.Age = y

train_df['Family Members'] = train_df.SibSp + train_df.Parch

test_df['Family Members'] = test_df.SibSp + test_df.Parch
#Let's see how Survived is related to Embarked

sns.barplot(x='Embarked',y='Survived',data=train_df)
#Let's see how Survived is related to Embarked

sns.barplot(x='Family Members',y='Survived',data=train_df)
#Let's see how Survived is related to Embarked

sns.lineplot(x='Age',y='Survived',data=train_df)
sns.barplot(x='Sex',y='Survived',hue='Sex',data=train_df)
#Let's plot a correlation Matrix Again

plot_corr(train_df)
X = train_df

Y = train_df.Survived

X.drop('Survived',inplace=True,axis=1)
drop = ['Name','Ticket','Cabin','PassengerId']

X.drop(drop,axis=1,inplace=True)

X.info()
len(X.columns)

X.head()
train_df.isna().sum()
# Since Age is mean Normalized, with mean point as 0, we need to replace Nans with 0

train_df.Age.fillna(0,inplace=True)

train_df.isna().sum()
train_accuracy = []

test_accuracy = []

xtrain_accuracy = []

xtest_accuracy = []



rclf = RandomForestClassifier(n_estimators=100,max_features='auto',max_depth=18,random_state=seed)



xclf = XGBClassifier(eta = 0.1,max_depth=20,n_estimators=100,seed=seed,max_features='auto',gamma=0.1)



sk = StratifiedKFold(n_splits=5,shuffle=True)

for train_index,test_index in sk.split(X,Y):

    X_train,Y_train = X.loc[train_index],Y.loc[train_index]

    X_test,Y_test = X.loc[test_index],Y.loc[test_index]



    rclf.fit(X_train,Y_train)

    #Get the Prediction on Train Set

    Y_pred = rclf.predict(X_train)

    score = accuracy_score(Y_pred,Y_train)

    print("Random Forest Train Score", score)

    train_accuracy.append(score)



    #Get the Prediction on Test Set

    Y_pred = rclf.predict(X_test)

    score = accuracy_score(Y_pred,Y_test)

    print("Random Forest Test Score", score)



    test_accuracy.append(score)



    # Add XGBoost Classifier

    xclf.fit(X_train,Y_train)

    #Get the Prediction on Train Set

    Y_pred = xclf.predict(X_train)

    score = accuracy_score(Y_pred,Y_train)

    print("XGBoost Train Score", score)

    xtrain_accuracy.append(score)

    #Get the Prediction on Train Set

    Y_pred = xclf.predict(X_test)

    score = accuracy_score(Y_pred,Y_test)

    print("XGBoost Test Score", score)

    xtest_accuracy.append(score)

print("Training Accuracy RamdomForest")

plt.plot(train_accuracy)

plt.plot(test_accuracy)

print("Training Accuracy XGBoost")

plt.plot(xtrain_accuracy)

plt.plot(xtest_accuracy)
test_df.head()
passengerId = test_df.PassengerId

test_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

len(test_df.columns)
test_df.isna().sum()
preds1 = rclf.predict(test_df)

preds2 = xclf.predict(test_df)

stacked_pred = np.column_stack((preds1,preds2))

def sigmoid(z):

    return (1/(1+np.exp(-z)))
W = np.ones((2,1))

W[0][0] = 0.5

W[1][0] = 0.5


final_preds = sigmoid(np.matmul(stacked_pred,W))
final_preds1 = np.where(final_preds <= 0.5, 0, 1)
final_preds1 = final_preds1.astype(int)

final_df = pd.DataFrame()

final_df['PassengerId'] = passengerId

final_df['Survived'] = final_preds1

final_df.head()
final_df.info()
final_df.to_csv("gender_submission.csv",index=False)
print(os.listdir("../working"))

os.chdir(r'../working')
from IPython.display import FileLink

#FileLink('gender_submission.csv')
