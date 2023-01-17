# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns #plotting package
import re

color = sns.color_palette()
sns.set_style('darkgrid')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew 
from scipy.special import boxcox1p #for Box Cox transformation
from sklearn.preprocessing import LabelEncoder

#Regressors
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

#Pipeline related
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel

#Base classes to be inherited
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

#Cross Validation related
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

#Model Tuning related
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
pd.options.display.max_rows = 80

from subprocess import check_output
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

sns.set()

train.drop("PassengerId", axis = 1, inplace = True)
train.drop("Name", axis = 1, inplace = True)
train.drop("Ticket", axis = 1, inplace = True)
#train.drop("Cabin", axis = 1, inplace = True)

test_id = test['PassengerId']
test.drop("PassengerId", axis = 1, inplace = True)
test.drop("Name", axis = 1, inplace = True)
test.drop("Ticket", axis = 1, inplace = True)
#test.drop("Cabin", axis = 1, inplace = True)
# Any results you write to the current directory are saved as output.
def plot_dist_norm(dist, title):
    sns.distplot(dist, fit=norm);
    (mu, sigma) = norm.fit(dist);
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title(title)
    fig = plt.figure()
    res = stats.probplot(dist, plot=plt)
    plt.show()
train.dtypes
train.head()
decks = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

train['Cabin'] = train['Cabin'].fillna("U0")

train['Deck'] = train['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group()).map(decks)

train['Deck'] = train['Deck'].fillna(0)
train['Deck'] = train['Deck'].astype(int)
train.drop("Cabin", axis = 1, inplace = True)
#train['Pclass'].value_counts().plot.bar()
x_axis_a = train['Survived'].sort_values().unique()
y_axis_a = train['Pclass'].sort_values().unique()

y_axis_b = train['Survived'].sort_values().unique()
x_axis_b = train['Pclass'].sort_values().unique()

y_axis_0_0 = train[(train['Pclass']==y_axis_a[0])&(train['Survived']==x_axis_a[0])]['Survived'].count()
y_axis_1_0 = train[(train['Pclass']==y_axis_a[1])&(train['Survived']==x_axis_a[0])]['Survived'].count()
y_axis_2_0 = train[(train['Pclass']==y_axis_a[2])&(train['Survived']==x_axis_a[0])]['Survived'].count()
y_axis_0_1 = train[(train['Pclass']==y_axis_a[0])&(train['Survived']==x_axis_a[1])]['Survived'].count()
y_axis_1_1 = train[(train['Pclass']==y_axis_a[1])&(train['Survived']==x_axis_a[1])]['Survived'].count()
y_axis_2_1 = train[(train['Pclass']==y_axis_a[2])&(train['Survived']==x_axis_a[1])]['Survived'].count()

total_0a = y_axis_0_0+y_axis_1_0+y_axis_2_0
total_1a = y_axis_0_1+y_axis_1_1+y_axis_2_1

total_0b = y_axis_0_0+y_axis_0_1
total_1b = y_axis_1_0+y_axis_1_1
total_2b = y_axis_2_0+y_axis_2_1

y_axis_0a = (y_axis_0_0/total_0a,y_axis_0_1/total_1a)
y_axis_1a = (y_axis_1_0/total_0a,y_axis_1_1/total_1a)
y_axis_2a = (y_axis_2_0/total_0a,y_axis_2_1/total_1a)
y_axis_01a = ((y_axis_0_0+y_axis_1_0)/total_0a,(y_axis_0_1+y_axis_1_1)/total_1a)

y_axis_0b = (y_axis_0_0/total_0b,y_axis_1_0/total_1b,y_axis_2_0/total_2b)
y_axis_1b = (y_axis_0_1/total_0b,y_axis_1_1/total_1b,y_axis_2_1/total_2b)

plot_1a = plt.bar(x_axis_a,y_axis_0a)
plot_2a = plt.bar(x_axis_a,y_axis_1a,bottom=y_axis_0a)
plot_3a = plt.bar(x_axis_a,y_axis_2a,bottom=y_axis_01a)
plt.title('Pclasses')
plt.xlabel('Survived')
plt.xticks(x_axis_a,x_axis_a)
plt.legend((plot_1a[0],plot_2a[0],plot_3a[0]),y_axis_a)
plt.show()

plot_1b = plt.bar(x_axis_b,y_axis_0b)
plot_2b = plt.bar(x_axis_b,y_axis_1b,bottom=y_axis_0b)
plt.title('Survived')
plt.xlabel('Pclass')
plt.xticks(x_axis_b,x_axis_b)
plt.legend((plot_1b[0],plot_2b[0]),y_axis_b)
plt.show()
x_axis_a = train['Survived'].sort_values().unique()
y_axis_a = train['Sex'].sort_values().unique()

y_axis_b = train['Survived'].sort_values().unique()
x_axis_b = train['Sex'].sort_values().unique()

y_axis_0_0 = train[(train['Sex']==y_axis_a[0])&(train['Survived']==x_axis_a[0])]['Survived'].count()
y_axis_1_0 = train[(train['Sex']==y_axis_a[1])&(train['Survived']==x_axis_a[0])]['Survived'].count()

y_axis_0_1 = train[(train['Sex']==y_axis_a[0])&(train['Survived']==x_axis_a[1])]['Survived'].count()
y_axis_1_1 = train[(train['Sex']==y_axis_a[1])&(train['Survived']==x_axis_a[1])]['Survived'].count()

total_0a = y_axis_0_0+y_axis_1_0
total_1a = y_axis_0_1+y_axis_1_1

total_0b = y_axis_0_0+y_axis_0_1
total_1b = y_axis_1_0+y_axis_1_1


y_axis_0a = (y_axis_0_0/total_0a,y_axis_0_1/total_1a)
y_axis_1a = (y_axis_1_0/total_0a,y_axis_1_1/total_1a)

y_axis_0b = (y_axis_0_0/total_0b,y_axis_1_0/total_1b)
y_axis_1b = (y_axis_0_1/total_0b,y_axis_1_1/total_1b)

plot_1a = plt.bar(x_axis_a,y_axis_0a)
plot_2a = plt.bar(x_axis_a,y_axis_1a,bottom=y_axis_0a)

plt.title('Sex')
plt.xlabel('Survived')
plt.xticks(x_axis_a,x_axis_a)
plt.legend((plot_1a[0],plot_2a[0]),y_axis_a)
plt.show()

plot_1b = plt.bar(x_axis_b,y_axis_0b)
plot_2b = plt.bar(x_axis_b,y_axis_1b,bottom=y_axis_0b)
plt.title('Survived')
plt.xlabel('Sex')
plt.xticks(x_axis_b,x_axis_b)
plt.legend((plot_1b[0],plot_2b[0]),y_axis_b)
plt.show()
train['Age']=train['Age'].fillna(train['Age'].mean())
plot_dist_norm(train['Age'],'Age')
train['SibSp'].value_counts().sort_index().plot.bar()
train['Parch'].value_counts().sort_index().plot.bar()
train['Fare'].replace(0,train[train['Fare']>0]['Fare'].mode()[0],inplace=True)
plot_dist_norm(np.log(train['Fare']),'Fare')
train['Embarked']=train['Embarked'].fillna(train['Embarked'].mode()[0])
x_axis_a = train['Survived'].sort_values().unique()
y_axis_a = train['Embarked'].sort_values().unique()

y_axis_b = train['Survived'].sort_values().unique()
x_axis_b = train['Embarked'].sort_values().unique()

y_axis_0_0 = train[(train['Embarked']==y_axis_a[0])&(train['Survived']==x_axis_a[0])]['Survived'].count()
y_axis_1_0 = train[(train['Embarked']==y_axis_a[1])&(train['Survived']==x_axis_a[0])]['Survived'].count()
y_axis_2_0 = train[(train['Embarked']==y_axis_a[2])&(train['Survived']==x_axis_a[0])]['Survived'].count()
y_axis_0_1 = train[(train['Embarked']==y_axis_a[0])&(train['Survived']==x_axis_a[1])]['Survived'].count()
y_axis_1_1 = train[(train['Embarked']==y_axis_a[1])&(train['Survived']==x_axis_a[1])]['Survived'].count()
y_axis_2_1 = train[(train['Embarked']==y_axis_a[2])&(train['Survived']==x_axis_a[1])]['Survived'].count()

total_0a = y_axis_0_0+y_axis_1_0+y_axis_2_0
total_1a = y_axis_0_1+y_axis_1_1+y_axis_2_1

total_0b = y_axis_0_0+y_axis_0_1
total_1b = y_axis_1_0+y_axis_1_1
total_2b = y_axis_2_0+y_axis_2_1

y_axis_0a = (y_axis_0_0/total_0a,y_axis_0_1/total_1a)
y_axis_1a = (y_axis_1_0/total_0a,y_axis_1_1/total_1a)
y_axis_2a = (y_axis_2_0/total_0a,y_axis_2_1/total_1a)
y_axis_01a = ((y_axis_0_0+y_axis_1_0)/total_0a,(y_axis_0_1+y_axis_1_1)/total_1a)

y_axis_0b = (y_axis_0_0/total_0b,y_axis_1_0/total_1b,y_axis_2_0/total_2b)
y_axis_1b = (y_axis_0_1/total_0b,y_axis_1_1/total_1b,y_axis_2_1/total_2b)

plot_1a = plt.bar(x_axis_a,y_axis_0a)
plot_2a = plt.bar(x_axis_a,y_axis_1a,bottom=y_axis_0a)
plot_3a = plt.bar(x_axis_a,y_axis_2a,bottom=y_axis_01a)
plt.title('Embarked')
plt.xlabel('Survived')
plt.xticks(x_axis_a,x_axis_a)
plt.legend((plot_1a[0],plot_2a[0],plot_3a[0]),y_axis_a)
plt.show()

plot_1b = plt.bar(x_axis_b,y_axis_0b)
plot_2b = plt.bar(x_axis_b,y_axis_1b,bottom=y_axis_0b)
plt.title('Survived')
plt.xlabel('Embarked')
plt.xticks(x_axis_b,x_axis_b)
plt.legend((plot_1b[0],plot_2b[0]),y_axis_b)
plt.show()
train['Family']=train['SibSp']+train['Parch']

#plt.bar(train['Family'].sort_values().unique(), train.groupby(['Family'])['Survived'].sum()/train.groupby(['Family'])['Survived'].count())
#plt.show()

train.drop("SibSp", axis = 1, inplace = True)
train.drop("Parch", axis = 1, inplace = True)
new_train = train

new_train['Fare'] = np.log(new_train['Fare'])
#new_train.Embarked = new_train.Embarked.astype('category', ordered=True, categories=['C','Q','S']).cat.codes
new_train=pd.get_dummies(new_train)
new_train.drop("Sex_male", axis = 1, inplace = True)
corrmat = new_train.corr()
features = corrmat.nlargest(12,'Survived')['Survived'].index
sns.set(font_scale=1.2)
plt.subplots(figsize=(12,9))
relevant_features = features
sns.heatmap(new_train[relevant_features].corr(), cbar=True, annot=True, fmt='.2f', annot_kws={'size':10}, yticklabels=relevant_features.values, xticklabels=relevant_features.values, vmax=1, square=True, cmap='Blues')
test['Fare'].replace(0,test[test['Fare']>0]['Fare'].mode()[0],inplace=True)
test['Embarked']=test['Embarked'].fillna(test['Embarked'].mode()[0])
test['Age']=test['Age'].fillna(test['Age'].mean())
test['Family']=test['SibSp']+test['Parch']
test.drop("SibSp", axis = 1, inplace = True)
test.drop("Parch", axis = 1, inplace = True)
test['Fare'].fillna(test['Fare'].mode()[0],inplace=True)

decks = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

test['Cabin'] = test['Cabin'].fillna("U0")

test['Deck'] = test['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group()).map(decks)

test['Deck'] = test['Deck'].fillna(0)
test['Deck'] = test['Deck'].astype(int)
test.drop("Cabin", axis = 1, inplace = True)

new_test = test

new_test['Fare'] = np.log(new_test['Fare'])
#new_test.Embarked = new_test.Embarked.astype('category', ordered=True, categories=['C','Q','S']).cat.codes

new_test=pd.get_dummies(new_test)
new_test.drop("Sex_male", axis = 1, inplace = True)
nfold=5
y_train = new_train.Survived.values
new_train.drop('Survived', axis=1, inplace=True)

def cv_score(model):
    kf = KFold(nfold, shuffle=True, random_state=42).get_n_splits(new_train.values)
    return cross_val_score(model, new_train.values, y_train, scoring="balanced_accuracy", cv = kf)

flag=True
if flag:
    steps = [('scaler',RobustScaler()),('select',SelectFromModel(SVC(C=1,kernel='linear') )),
            ('randomforest',RandomForestClassifier(n_estimators=100,criterion='gini',bootstrap=True  ) )]
    randomforest_p = Pipeline(steps)
    n_estimatorss=[100,200,400,800]
    criteria=['gini','entropy']
    bootstraps=[True,False]
    gscv = GridSearchCV(randomforest_p, cv=nfold, param_grid={'randomforest__n_estimators': n_estimatorss, 'randomforest__criterion': criteria, 'randomforest__bootstrap': bootstraps}, n_jobs=-1, verbose=1,scoring='balanced_accuracy')
    gscv.fit(new_train.values, y_train)
    randomforest_ = gscv.best_estimator_.named_steps.randomforest
    randomforest = gscv.best_estimator_
    score = cv_score(randomforest_)
    print("\nRandomForest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    print('Best RandomForest: ',randomforest_)

randomforest.fit(new_train.values,y_train)
randomforest_pred = randomforest.predict(new_test.values)
prediction = randomforest_pred
sub = pd.DataFrame()
sub['PassengerId'] = test_id
sub['Survived'] = prediction
sub.to_csv('submission.csv',index=False)