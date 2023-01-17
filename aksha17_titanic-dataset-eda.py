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
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# load data

data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_test  = pd.read_csv("/kaggle/input/titanic/test.csv")

#preview data

print (data_train.info())

print("-"*20)

print('Train columns with null values:\n', data_train.isnull().sum())

print("-"*20)

print (data_test.info())

print('Train columns with null values:\n', data_test.isnull().sum())

print("-"*20)

data_train.sample(10)
f,ax=plt.subplots(1,2,figsize=(18,8))

data_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=data_train,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
#sex

f,ax=plt.subplots(1,2,figsize=(18,8))

data_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=data_train,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

data_train['Pclass'].value_counts().plot.bar(color=['#FF8C00','#FFD700','#FF4500'],ax=ax[0])

ax[0].set_title('Number Of Passengers By Pclass')

ax[0].set_ylabel('Count')

sns.countplot('Pclass',hue='Survived',data=data_train,ax=ax[1])

ax[1].set_title('Pclass:Survived vs Dead')

plt.show()
plt.figure(figsize=(18,8))

plt.subplot(1, 2, 1)

sns.barplot(x = "Pclass", 

            y = "Survived", 

            data=data_train, 

            linewidth=5,

            capsize = .1)

plt.title("Paclass - Survived")

plt.xlabel("Socio-Economic class");

plt.ylabel("% of Passenger Survived");

labels = ['Upper', 'Middle', 'Lower']

#val = sorted(train.Pclass.unique())

val = [0,1,2] ## this is just a temporary trick to get the label right. 

plt.xticks(val, labels);

# Kernel Density Plot

plt.subplot(1, 2, 2)

ax=sns.kdeplot(data_train.Pclass[data_train.Survived == 0] , 

               color='gray',

               shade=True,

               label='not survived')

ax=sns.kdeplot(data_train.loc[(data_train['Survived'] == 1),'Pclass'] ,

               color='g',

               shade=True, 

               label='survived', )

plt.title('Passenger Class Distribution - Survived vs Non-Survived')

plt.ylabel("Frequency of Passenger Survived")

plt.xlabel("Passenger Class")

plt.xticks(sorted(data_train.Pclass.unique()), labels);
sns.factorplot('Pclass','Survived',hue='Sex',data=data_train)

plt.show()
print('Oldest Passenger was of:',data_train['Age'].max(),'Years')

print('Youngest Passenger was of:',data_train['Age'].min(),'Years')

print('Average Age on the ship:',data_train['Age'].mean(),'Years')

plt.figure(figsize=(18,8))

plt.hist(data_train['Age'], 

        bins = np.arange(data_train['Age'].min(),data_train['Age'].max(),5),

        normed = True, 

        color = 'steelblue',

        edgecolor = 'k')

plt.title('Distribution of Age')

plt.xlabel('Age')

plt.ylabel('Frequency')

ax=sns.kdeplot(data_train['Age'] , 

               color='red',

               shade=False,

               label='not survived')

plt.tick_params(top='off', right='off')

plt.legend([ax],['KDE'],loc='best')

plt.show()
data_cleaner = [data_train, data_test]

data_train['Title'], data_test['Title']=0,0

for dataset in data_cleaner:

    dataset['Title']=dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

print(data_train['Title'].value_counts())

print("-"*20)

print(data_test['Title'].value_counts())
for dataset in data_cleaner:

    dataset['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],

                                ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Mr','Mrs'],inplace=True)

print(data_train.groupby('Title')['Age'].mean())

print("-"*20)

print(data_test.groupby('Title')['Age'].mean())
# Assigning the NaN Values with the Ceil values of the mean ages

data_train.loc[(data_train.Age.isnull())&(data_train.Title=='Mr'),'Age']=33

data_train.loc[(data_train.Age.isnull())&(data_train.Title=='Mrs'),'Age']=36

data_train.loc[(data_train.Age.isnull())&(data_train.Title=='Master'),'Age']=5

data_train.loc[(data_train.Age.isnull())&(data_train.Title=='Miss'),'Age']=22



data_test.loc[(data_test.Age.isnull())&(data_test.Title=='Mr'),'Age']=32

data_test.loc[(data_test.Age.isnull())&(data_test.Title=='Mrs'),'Age']=40

data_test.loc[(data_test.Age.isnull())&(data_test.Title=='Master'),'Age']=7

data_test.loc[(data_test.Age.isnull())&(data_test.Title=='Miss'),'Age']=22



for dataset in data_cleaner:

    print(dataset.Age.isnull().any())
#now we see the distribution again

print('Oldest Passenger was of:',data_train['Age'].max(),'Years')

print('Youngest Passenger was of:',data_train['Age'].min(),'Years')

print('Average Age on the ship:',data_train['Age'].mean(),'Years')

plt.figure(figsize=(18,8))

plt.hist(data_train['Age'], 

        bins = np.arange(data_train['Age'].min(),data_train['Age'].max(),5),

        normed = True, 

        color = 'steelblue',

        edgecolor = 'k')

plt.title('Distribution of Age')

plt.xlabel('Age')

plt.ylabel('Frequency')

ax=sns.kdeplot(data_train['Age'] , 

               color='red',

               shade=False,

               label='not survived')

plt.tick_params(top='off', right='off')

plt.legend([ax],['KDE'],loc='best')

plt.show()

f,ax=plt.subplots(1,2,figsize=(18,8))

data_train[data_train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='g')

ax[0].set_title('Survived= 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)

data_train[data_train['Survived']==1].Age.plot.hist(ax=ax[1],color='gray',bins=20,edgecolor='black')

ax[1].set_title('Survived= 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)

plt.show()



fig = plt.figure(figsize=(15,8),)

ax=sns.kdeplot(data_train.loc[(data_train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')

ax=sns.kdeplot(data_train.loc[(data_train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')

plt.title('Age Distribution - Surviver V.S. Non Survivors', fontsize = 25, pad = 40)

plt.xlabel("Age", fontsize = 15, labelpad = 20)

plt.ylabel('Frequency', fontsize = 15, labelpad= 20);
# check the missing value

data_train[data_train.Embarked.isnull()]
sns.set_style('darkgrid')

fig, ax = plt.subplots(figsize=(16,12),ncols=2)

ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=data_train, ax = ax[0]);

ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=data_test, ax = ax[1]);

ax1.set_title("Training Set", fontsize = 18)

ax2.set_title('Test Set',  fontsize = 18)
data_train.Embarked.fillna("C", inplace=True)

data_train.Embarked.isnull().any()
f,ax=plt.subplots(1,3,figsize=(20,6))

sns.countplot('Embarked',data=data_train,ax=ax[0])

ax[0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Survived',data=data_train,ax=ax[1])

ax[1].set_title('Embarked vs Survived')

sns.barplot(x = 'Embarked', y = 'Survived', data=data_train,ax=ax[2])

ax[2].set_title('Embarked Survival Rate')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()
f,ax=plt.subplots(1,2,figsize=(20,8))

sns.countplot('Embarked',hue='Sex',data=data_train,ax=ax[0])

ax[0].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Pclass',data=data_train,ax=ax[1])

ax[1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()
sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data_train)

plt.show()
pd.crosstab([data_train.SibSp],data_train.Survived).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(2,2,figsize=(20,12))

sns.barplot('SibSp','Survived',data=data_train,ax=ax[0,0])

ax[0,0].set_title('SibSp vs Survived')

sns.factorplot('SibSp','Survived',data=data_train,ax=ax[0,1])

ax[0,1].set_title('SibSp vs Survived')

sns.barplot('Parch','Survived',data=data_train,ax=ax[1,0])

ax[1,0].set_title('Parch vs Survived')

sns.factorplot('Parch','Survived',data=data_train,ax=ax[1,1])

ax[1,1].set_title('Parch vs Survived')

plt.close(2)

plt.show()
pd.crosstab(data_train.SibSp,data_train.Pclass).style.background_gradient(cmap='summer_r')
# Fare

# first check the missing value

data_test[data_test.Fare.isnull()]
missing_value = data_test[(data_test.Pclass == 3) & (data_test.Embarked == "S") & (data_test.Sex == "male")].Fare.mean()

# replace the test.fare null values with test.fare mean

data_test.Fare.fillna(missing_value, inplace=True)

data_test.Embarked.isnull().any()
# Kernel Density Plot

fig = plt.figure(figsize=(15,8),)

ax=sns.kdeplot(data_train.loc[(data_train['Survived'] == 0),'Fare'] , color='gray',shade=True,label='not survived')

ax=sns.kdeplot(data_train.loc[(data_train['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')

plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25, pad = 40)

plt.ylabel("Frequency of Passenger Survived", fontsize = 15, labelpad = 20)

plt.xlabel("Fare", fontsize = 15, labelpad = 20);
sns.scatterplot(data_train["Fare"], data_train["Age"])
data_train[data_train.Fare > 300]
# drop the outliers

data_train = data_train[data_train.Fare < 300]
# check the missing value

print("Train Cabin missing: " + str(data_train.Cabin.isnull().sum()/len(data_train.Cabin)))

print("Test Cabin missing: " + str(data_test.Cabin.isnull().sum()/len(data_test.Cabin)))
survivers = data_train.Survived # save the label for a while

data_train.drop(["Survived"],axis=1, inplace=True)

all_data = pd.concat([data_train,data_test], ignore_index=False)

## Assign all the null values to N

all_data.Cabin.fillna("N", inplace=True)

all_data.sort_values("Cabin").head(10)
all_data.Cabin = [i[0] for i in all_data.Cabin]

all_data["Cabin"].value_counts(normalize=True)
with_N = all_data[all_data.Cabin == "N"]



without_N = all_data[all_data.Cabin != "N"]



all_data.groupby("Cabin")['Fare'].mean().sort_values()
def cabin_estimator(i):

    a = 0

    if i<16:

        a = "G"

    elif i>=16 and i<27:

        a = "F"

    elif i>=27 and i<38:

        a = "T"

    elif i>=38 and i<47:

        a = "A"

    elif i>= 47 and i<53:

        a = "E"

    elif i>= 53 and i<54:

        a = "D"

    elif i>=54 and i<116:

        a = 'C'

    else:

        a = "B"

    return a

##applying cabin estimator function. 

with_N['Cabin'] = with_N.Fare.apply(lambda x: cabin_estimator(x))



## getting back train. 

all_data = pd.concat([with_N, without_N], axis=0)



## PassengerId helps us separate train and test. 

all_data.sort_values(by = 'PassengerId', inplace=True)



## Separating train and test from all_data. 

data_train = all_data[:891]



data_test = all_data[891:]



# adding saved target variable with train. 

data_train['Survived'] = survivers
f,ax=plt.subplots(1,2,figsize=(20,8))

sns.countplot('Cabin',data=data_train, ax=ax[0])

sns.barplot('Cabin','Survived',data=data_train, ax=ax[1])
# Placing 0 for female and 1 for male in the "Sex" column. 

data_train['Sex'] = data_train.Sex.apply(lambda x: 0 if x == "female" else 1)

data_test['Sex'] = data_test.Sex.apply(lambda x: 0 if x == "female" else 1)



sns.heatmap(data_train.corr(),annot=True,cmap='coolwarm',linewidths=0.2) 

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
#family size

data_train['family_size'] = data_train.SibSp + data_train.Parch+1

data_test['family_size'] = data_test.SibSp + data_test.Parch+1
#is_alone

data_train['is_alone'] = [1 if i<2 else 0 for i in data_train.family_size]

data_test['is_alone'] = [1 if i<2 else 0 for i in data_test.family_size]
f,ax=plt.subplots(1,2,figsize=(18,6))

sns.factorplot('family_size','Survived',data=data_train,ax=ax[0])

ax[0].set_title('family_size vs Survived')

sns.factorplot('is_alone','Survived',data=data_train,ax=ax[1])

ax[1].set_title('is_alone vs Survived')

plt.close(2)

plt.close(3)

plt.show()
def family_group(size):

    a = ''

    if (size <= 1):

        a = 'loner'

    elif (size <= 4):

        a = 'small'

    else:

        a = 'large'

    return a

# gen fanily group by its size

data_train['family_group'] = data_train['family_size'].map(family_group)

data_test['family_group'] = data_test['family_size'].map(family_group)
sns.factorplot('is_alone','Survived',data=data_train,hue='Sex',col='Pclass')

plt.show()
#age group

plt.subplots(figsize = (22,10),)

sns.distplot(data_train.Age, bins = 100, kde = True, rug = False, norm_hist=False);
def age_group_fun(age):

    a= ''

    if age <= 10:

        a = "child"

    elif age <= 22:

        a = "teenager"

    elif age <= 33:

        a = "yong_Adult"

    elif age <= 45:

        a = "middle_age"

    else:

        a = "old"

    return a



data_train['age_group'] = data_train['Age'].map(age_group_fun)

data_test['age_group'] = data_test['Age'].map(age_group_fun)
#fare group

data_train['Fare_Range']=pd.qcut(data_train['Fare'],4)

data_train.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
def fare_group_fun(fare):

    a= ''

    if fare <= 7.896:

        a = "low"

    elif fare <= 14.454:

        a = "normal"

    elif fare <= 30.696:

        a = "middle"

    else:

        a = "high"

    return a



data_train['fare_group'] = data_train['Fare'].map(fare_group_fun)

data_test['fare_group'] = data_test['Fare'].map(fare_group_fun)
# prepare for onehot encoding

data_train['Pclass'].astype(str)

data_train['Sex'].astype(str)

data_train['is_alone'].astype(str)

data_test['Pclass'].astype(str)

data_test['Sex'].astype(str)

data_test['is_alone'].astype(str)
# onehot encoding & drop unused variables

data_train = pd.get_dummies(data_train, columns=['Title',"Pclass", 'Sex','is_alone','Cabin','age_group','Embarked','family_group', 'fare_group'], drop_first=False)

data_test = pd.get_dummies(data_test, columns=['Title',"Pclass", 'Sex','is_alone','Cabin','age_group','Embarked','family_group', 'fare_group'], drop_first=False)

data_train.drop(['family_size','Name','PassengerId','Ticket','Fare_Range'], axis=1, inplace=True)

passengerid = data_test['PassengerId'].values

data_test.drop(['family_size','Name','PassengerId','Ticket'], axis=1, inplace=True)
data_train.head(10)
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

age_fare = data_train[['Age','Fare']] # get age and fare features

age_fare = mm.fit_transform(age_fare)

age_fare_df = pd.DataFrame(age_fare, columns=['Age','Fare']) # scaling data

data_train.drop(['Age','Fare'], axis=1, inplace=True)

data_train = data_train.reset_index(drop=True)

data_train = pd.concat([data_train, age_fare_df],axis=1) # merge the scaling data back to train data set



age_fare = data_test[['Age','Fare']] #same for test

age_fare = mm.fit_transform(age_fare)

age_fare_df = pd.DataFrame(age_fare, columns=['Age','Fare']) # scaling data

data_test.drop(['Age','Fare'], axis=1, inplace=True)

data_test = data_test.reset_index(drop=True)

data_test = pd.concat([data_test, age_fare_df],axis=1)
data_train
data_test
# Split the training data into test and train datasets

X = data_train.drop(['Survived'], axis = 1)

y = data_train["Survived"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .33, random_state=0)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 ) 

column_names = X.columns

X = X.values



# use grid search to get the best parameters

C_vals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,17.5,18] #alpla of lasso and ridge 

penalties = ['l1','l2'] # Choosing penalties(Lasso(l1) or Ridge(l2))

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25) # Choose a cross validation strategy. 

param = {'penalty': penalties, 'C': C_vals} # setting param for param_grid in GridSearchCV. 

logreg = LogisticRegression(solver='liblinear')

## Calling on GridSearchCV object. 

grid = GridSearchCV(estimator=LogisticRegression(), 

                           param_grid = param,

                           scoring = 'accuracy',

                            n_jobs =-1,

                     cv = cv

                          )

## Fitting the model

grid.fit(X, y)



# get accuracy

logreg_grid = grid.best_estimator_

logreg_grid.score(X,y)
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1,31)

weights_options=['uniform','distance']

param = {'n_neighbors':k_range, 'weights':weights_options}

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)

grid.fit(X,y)

knn_grid= grid.best_estimator_

knn_grid.score(X,y)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(X, y)

y_pred = gaussian.predict(X_test)

gaussian_accy = round(accuracy_score(y_pred, y_test), 3)

print(gaussian_accy)
from sklearn.svm import SVC

Cs = [0.001, 0.01, 0.1, 1,1.5,2,2.5,3,4,5, 10] ## penalty parameter C for the error term. 

gammas = [0.0001,0.001, 0.01, 0.1, 1]

param_grid = {'C': Cs, 'gamma' : gammas}

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

grid_search = GridSearchCV(SVC(kernel = 'rbf', probability=True), param_grid, cv=cv) ## 'rbf' stands for gaussian kernel

grid_search.fit(X,y)

svm_grid = grid_search.best_estimator_

svm_grid.score(X,y)
from sklearn.tree import DecisionTreeClassifier

max_depth = range(1,30)

max_feature = [21,22,23,24,25,26,28,29,30,'auto']

criterion=["entropy", "gini"]



param = {'max_depth':max_depth, 

         'max_features':max_feature, 

         'criterion': criterion}

grid = GridSearchCV(DecisionTreeClassifier(), 

                                param_grid = param, 

                                 verbose=False, 

                                 cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),

                                n_jobs = -1)

grid.fit(X, y) 

dectree_grid = grid.best_estimator_

dectree_grid.score(X,y)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier

n_estimators = [140,145,150,155,160];

max_depth = range(1,10);

criterions = ['gini', 'entropy'];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)





parameters = {'n_estimators':n_estimators,

              'max_depth':max_depth,

              'criterion': criterions

              

        }

grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

grid.fit(X,y) 

rf_grid = grid.best_estimator_

rf_grid.score(X,y)
from sklearn.ensemble import BaggingClassifier

n_estimators = [10,30,50,70,80,150,160, 170,175,180,185];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)



parameters = {'n_estimators':n_estimators,

              

        }

grid = GridSearchCV(BaggingClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.

                                      bootstrap_features=False),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

grid.fit(X,y) 

bagging_grid = grid.best_estimator_

bagging_grid.score(X,y)
from sklearn.ensemble import AdaBoostClassifier

n_estimators = [100,140,145,150,160, 170,175,180,185];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

learning_r = [0.1,1,0.01,0.5]



parameters = {'n_estimators':n_estimators,

              'learning_rate':learning_r

              

        }

grid = GridSearchCV(AdaBoostClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.

                                     ),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

grid.fit(X,y) 

adaBoost_grid = grid.best_estimator_

adaBoost_grid.score(X,y)
from sklearn.ensemble import GradientBoostingClassifier



gradient_boost = GradientBoostingClassifier()

gradient_boost.fit(X, y)

y_pred = gradient_boost.predict(X_test)

gradient_accy = round(accuracy_score(y_pred, y_test), 3)

print(gradient_accy)
from sklearn.ensemble import ExtraTreesClassifier

ExtraTreesClassifier = ExtraTreesClassifier()

ExtraTreesClassifier.fit(X, y)

y_pred = ExtraTreesClassifier.predict(X_test)

extraTree_accy = round(accuracy_score(y_pred, y_test), 3)

print(extraTree_accy)
from sklearn.gaussian_process import GaussianProcessClassifier

GaussianProcessClassifier = GaussianProcessClassifier()

GaussianProcessClassifier.fit(X, y)

y_pred = GaussianProcessClassifier.predict(X_test)

gau_pro_accy = round(accuracy_score(y_pred, y_test), 3)

print(gau_pro_accy)
from sklearn.ensemble import VotingClassifier



voting_classifier = VotingClassifier(estimators=[

    ('lr_grid', logreg_grid),

    ('svc', svm_grid),

    ('random_forest', rf_grid),

    ('gradient_boosting', gradient_boost),

    ('decision_tree_grid',dectree_grid),

    ('knn_classifier', knn_grid),

    #('XGB_Classifier', XGBClassifier),

    ('bagging_classifier', bagging_grid),

    ('adaBoost_classifier',adaBoost_grid),

    ('ExtraTrees_Classifier', ExtraTreesClassifier),

    ('gaussian_classifier',gaussian),

    ('gaussian_process_classifier', GaussianProcessClassifier)

],voting='hard')



#voting_classifier = voting_classifier.fit(train_x,train_y)

voting_classifier = voting_classifier.fit(X,y)



y_pred = voting_classifier.predict(X_test)

voting_accy = round(accuracy_score(y_pred, y_test), 3)

print(voting_accy)
all_models = [logreg_grid,

              knn_grid, 

              svm_grid,

              dectree_grid,

              rf_grid,

              bagging_grid,

              adaBoost_grid,

              voting_classifier]



c = {}

for i in all_models:

    a = i.predict(X_test)

    b = accuracy_score(a, y_test)

    c[i] = b
test_prediction = (max(c, key=c.get)).predict(data_test)

submission = pd.DataFrame({

        "PassengerId": passengerid,

        "Survived": test_prediction})



submission.PassengerId = submission.PassengerId.astype(int)

submission.Survived = submission.Survived.astype(int)



submission.to_csv("titanic2_submission.csv", index=False)