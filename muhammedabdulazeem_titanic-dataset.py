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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.shape
test.shape
train['Survived'].value_counts()
train.isnull().sum()
test.isnull().sum()
combined_df = train.append(test)
combined_df.head()
combined_df.shape
combined_df.isnull().mean().sort_values(ascending=False)
def impute_nan(df,variable,median):

    df[variable+"_median"]=df[variable].fillna(median)

    df[variable+"_random"]=df[variable]

    ##It will have the random sample to fill the na

    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)

    ##pandas need to have same index in order to merge the dataset

    random_sample.index=df[df[variable].isnull()].index

    df.loc[df[variable].isnull(),variable+'_random']=random_sample
median = combined_df['Age'].median()

impute_nan(combined_df,'Age',median)
import matplotlib.pyplot as plt

import seaborn as sns



fig = plt.figure()

ax = fig.add_subplot(111)

combined_df['Age'].plot(kind='kde', ax=ax)

combined_df.Age_median.plot(kind='kde', ax=ax, color='red')

lines, labels = ax.get_legend_handles_labels()

ax.legend(lines, labels, loc='best')
fig = plt.figure()

ax = fig.add_subplot(111)

combined_df['Age'].plot(kind='kde', ax=ax)

combined_df.Age_random.plot(kind='kde', ax=ax, color='red')

lines, labels = ax.get_legend_handles_labels()

ax.legend(lines, labels, loc='best')
from statsmodels.graphics.gofplots import qqplot



fig = qqplot(combined_df['Age_random'],line='s')

plt.show()
combined_df['Age_log'] = np.log1p(combined_df['Age_random'])
fig = qqplot(combined_df['Age_log'],line='s')

plt.show()
combined_df.drop(['Age_log'],axis=1,inplace=True)
combined_df.head()
combined_df['Cabin'].value_counts()
combined_df['Cabin'].unique()
combined_df['Cabin'].fillna('Missing',inplace=True)
combined_df.isnull().sum()
combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0],inplace=True)
combined_df.drop(['Age','Age_median','PassengerId'],inplace=True,axis=1)
combined_df.drop('Ticket',inplace=True,axis=1)
combined_df.head()
plt.figure(figsize=(15,10))

sns.distplot(combined_df['Fare'])
fig = qqplot(combined_df['Fare'],line='s')

plt.show()
fare_less_than_50 = combined_df[combined_df['Fare'] < 50] 
sns.distplot(fare_less_than_50['Fare'])
combined_df[combined_df['Fare'] < 0.0]
people_how_travelled_for_free = combined_df[combined_df['Fare'] == 0.0]
people_how_travelled_for_free
plt.figure(figsize=(15,10))

sns.distplot(people_how_travelled_for_free['Age_random'])
survived_on_the_basis_of_pclass = pd.crosstab(combined_df['Pclass'].dropna(),combined_df['Survived'].dropna())
survived_on_the_basis_of_pclass
plt.figure(figsize=(15,10))

sns.countplot(x=combined_df['Pclass'],hue=combined_df['Survived'])
survived_on_the_basis_of_gender = pd.crosstab(combined_df['Sex'].dropna(),combined_df['Survived'].dropna())
survived_on_the_basis_of_gender
plt.figure(figsize=(15,10))

sns.countplot(x=combined_df['Sex'],hue=combined_df['Survived'])
combined_df.head()
combined_df['SibSp'].value_counts()
combined_df['Parch'].value_counts()
plt.figure(figsize=(15,10))

sns.countplot(x=combined_df['Embarked'],hue=combined_df['Survived'])
combined_df['Fare'].fillna(0.0,inplace=True)
combined_df.drop(['Name'],axis=1,inplace=True)
combined_df.head()
combined_df['Embarked'].value_counts()
combined_df['Sex'] = combined_df['Sex'].map({'male':0,'female':1})

combined_df['Embarked'] = combined_df['Embarked'].map({'S':0,'C':1,'Q':2})
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
combined_df['Cabin'] = le.fit_transform(combined_df['Cabin'])
train.shape
new_train = combined_df.iloc[:891,:]
new_train.isnull().sum()
new_test = combined_df.iloc[891:,:]
new_test.isnull().sum()
new_test.shape
new_train
new_test
new_test.drop('Survived',axis=1,inplace=True)
features_train = new_train.drop('Survived',axis=1)

label_train = new_train['Survived']
from sklearn.preprocessing import StandardScaler



ss = StandardScaler()
X_train = ss.fit_transform(features_train)

X_test = ss.transform(new_test)
X_train
X_test
X_train.mean()
X_test.mean()
sns.distplot(X_train)
sns.distplot(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from skopt import space

from skopt import gp_minimize
from sklearn import ensemble

from sklearn import model_selection

from sklearn import metrics

from functools import partial
def optimize(params, params_names, x, y):

    params_dict = dict(zip(params_names, params))



    model = ensemble.RandomForestClassifier(**params_dict)



    kf = model_selection.StratifiedKFold(n_splits=5)

    accuracies = []



    for idx in kf.split(x, y):

        train_idx, test_idx = idx[0], idx[1]



        xtrain = x[train_idx]

        xtest = x[test_idx]

        ytrain = y[train_idx]

        ytest = y[test_idx]



        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)

        fold_acc = metrics.accuracy_score(ytest, preds)



        accuracies.append(fold_acc)



    return -1.0 * np.mean(accuracies)
params_space = [

        space.Integer(3, 15, name='max_depth'),

        space.Integer(100, 500, name='n_estimators'),

        space.Categorical(['gini', 'entropy'], name='criterion'),

        space.Real(0.01, 1, prior='uniform', name='max_features')

    ]



params_names = [

        'max_depth',

        'n_estimators',

        'criterion',

        'max_features'

    ]



optimization_function = partial(

        optimize,

        params_names=params_names,

        x=X_train,

        y=label_train

    )



result = gp_minimize(

        optimization_function,

        dimensions=params_space,

        n_calls=15,

        n_random_starts=10,

        verbose=10

    )



print(dict(zip(params_names,result.x)))
from sklearn.model_selection import StratifiedKFold
classifier = ensemble.RandomForestClassifier(n_jobs=-1)
accuracies = []



skf = StratifiedKFold(n_splits=5,random_state=None)

skf.get_n_splits(X_train,label_train)



for train_index,test_index in skf.split(X_train,label_train):

    X1_train,X1_test = X_train[train_index],X_train[test_index]

    y1_train,y1_test = label_train[train_index],label_train[test_index]

    

    classifier.fit(X1_train,y1_train)

    predication = classifier.predict(X1_test)

    score = metrics.accuracy_score(y1_test,predication)

    accuracies.append(score)

    
accuracies
np.mean(accuracies)
classifier = LogisticRegression()
accuracies = []



skf = StratifiedKFold(n_splits=5,random_state=None)

skf.get_n_splits(X_train,label_train)



for train_index,test_index in skf.split(X_train,label_train):

    X1_train,X1_test = X_train[train_index],X_train[test_index]

    y1_train,y1_test = label_train[train_index],label_train[test_index]

    

    classifier.fit(X1_train,y1_train)

    predication = classifier.predict(X1_test)

    score = metrics.accuracy_score(y1_test,predication)

    accuracies.append(score)
accuracies
np.mean(accuracies)
random_forest = RandomForestClassifier(n_estimators=50)

random_forest.fit(X_train, label_train)

Y_pred = random_forest.predict(new_test)

random_forest.score(X_train, label_train)

acc_random_forest = round(random_forest.score(X_train, label_train) * 100, 2)

print('training accuracy of the model is',acc_random_forest)
Y_pred
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

submission
test
submission_ = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission_.to_csv('submission_.csv', index=False)
#submission['pred_Survival'] = Y_pred

#submission.to_csv('submission.csv', index=False)
testing_submission_ = pd.read_csv('submission_.csv')
testing_submission_
metrics.accuracy_score(Y_pred,submission['Survived'])
submission['Survived'] = Y_pred

submission.to_csv('submission_new.csv', index=False)