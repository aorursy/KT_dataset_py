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
import warnings

warnings.filterwarnings('ignore')
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic.head()

## checking the head of our data set
titanic.info()

## checking info of all columns
titanic.shape

## checking shape of data set
titanic.describe()

## statistical information about numerical variable
round(100*(titanic.isnull().sum()/len(titanic)),2)

## checking missing value percentage in all columns
titanic.drop('Cabin',axis=1,inplace=True)

## cabin almost have 77% of missing values hence remove this column from data set
age_median = titanic['Age'].median(skipna=True)

titanic['Age'].fillna(age_median,inplace=True)

## as there is 19% of missing values in age column hence it is not a good idea to remove this row wise or column wise hence impute those missing values with the median of age 

titanic = titanic[titanic['Embarked'].isnull()!=True]

## as embarked has a very small amount of missing values hence remove those rows which have missing values in embarked column 

titanic.shape

## checking shape after removing null values
titanic_dub = titanic.copy()

## creating copy of the data frame to check duplicate values
titanic_dub.shape

## comparing shapes of two data frames
titanic.shape

## shape of original data frame
import seaborn as sns

import matplotlib.pyplot as plt

## importing libraries for data visualitation
plt.figure(figsize=(15,5), dpi=80)

plt.subplot(1,4,1)

sns.boxplot(y=titanic['Age'])

plt.title("Outliers in 'Age'")



plt.subplot(1,4,2)

ax = sns.boxplot(y=titanic['Fare'])

ax.set_yscale('log')

plt.title("Outliers in 'Fare'")



plt.subplot(1,4,3)

sns.boxplot(y=titanic['SibSp'])

plt.title("Outliers in 'SibSp'")





plt.subplot(1,4,4)

sns.boxplot(y=titanic['Parch'])

plt.title("Outliers in 'Parch'")

#ax.set_yscale('log')

plt.tight_layout()

plt.show()



## plotting all four variables to check for outliers

## it clearly shows that all four variables has some outliers




sns.catplot(x="SibSp", col = 'Survived', data=titanic, kind = 'count', palette='pastel')

sns.catplot(x="Parch", col = 'Survived', data=titanic, kind = 'count', palette='pastel')

plt.tight_layout()

plt.show()



## plotting of sibsp and parch in basis of survived and not survived
def alone(x):

    if (x['SibSp']+x['Parch']>0):

        return (1)

    else:

        return (0)

titanic['Alone'] = titanic.apply(alone,axis=1)

## creating a function to make one variable which tells us whether a person is single or accompanied by some on the ship
sns.catplot(x="Alone", col = 'Survived', data=titanic, kind = 'count', palette='pastel')

plt.show()
## drop parch and sibsp

titanic = titanic.drop(['Parch','SibSp'],axis=1)

titanic.head()

sns.distplot(titanic['Fare'])

plt.show()
sns.catplot(x="Sex", y="Survived", col="Pclass", data=titanic, saturation=.5, kind="bar", ci=None, aspect=0.8, palette='deep')

sns.catplot(x="Sex", y="Survived", col="Embarked", data=titanic, saturation=.5, kind="bar", ci=None, aspect=0.8, palette='deep')

plt.show()



## plotting of survive on basis of pclass
survived_0 = titanic[titanic['Survived']==0]

survived_1 = titanic[titanic['Survived']==1]

## divided our dataset into survived or not survived to check the distribution of age in both the cases 
survived_0.shape

## checking shape of the data set that contains the data of passengers who not survived
survived_1.shape

## checking shape of the data set that contains the data of passengers who survived
sns.distplot(survived_0['Age'])

plt.show()

## checking distribution of age in not survived data set
sns.distplot(survived_1['Age'])

plt.show()

## checking distribution of age in survived dataset
sns.boxplot(x='Survived',y='Fare',data=titanic)

plt.show()

## checking survival rate on basis of fare
Pclass_dummy = pd.get_dummies(titanic['Pclass'],prefix='Pclass',drop_first=True)

Pclass_dummy.head()

## creating dummy variables for pclass



## joing dummy variables

titanic = pd.concat([titanic,Pclass_dummy],axis=1)

titanic.head()
titanic.drop('Pclass',axis=1,inplace=True)

## as there is no use of pclass after joining the columns that contains dummy variables  for pclass
Embarked_dummy = pd.get_dummies(titanic['Embarked'],drop_first=True)

Embarked_dummy.head()

## creating dummy variables for embarked and dropping first column
titanic = pd.concat([titanic,Embarked_dummy],axis=1)

titanic.drop('Embarked',axis=1,inplace=True)

## joining dummy variables
titanic.head()

## checking head of the data set after joining dummy variables
def sex_map(x):

    if x == 'male':

        return (1)

    elif x == 'female':

        return (0)

titanic['Sex'] = titanic['Sex'].apply(lambda x:sex_map(x))



## creating function for convert sex into binary values
titanic = titanic[['Survived','Sex','Age','Fare','Alone','Pclass_2','Pclass_3','Q','S']]
## let's plot one heatmap to check the corelations 

plt.figure(figsize=(10,10))

sns.heatmap(titanic.corr(),annot=True)

plt.show()
## read test data 



titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
## check shape and info of titanic_test



titanic_test.shape
titanic_test.info()
## imputing missing values of age 

titanic_test['Age'].fillna(age_median,inplace=True)

## create alone column

titanic_test['Alone'] = titanic_test.apply(alone,axis=1)

## create sex column with binary value 0 for 'female' and 1 for 'male'

titanic_test['Sex'] = titanic_test['Sex'].apply(lambda x:sex_map(x))
## compute dummy encoding in test data 



Pclass_dummy_test = pd.get_dummies(titanic_test['Pclass'],prefix='Pclass',drop_first=True)



titanic_test = pd.concat([titanic_test,Pclass_dummy_test],axis=1)
## compute dummy encoding in test data 





Embarked_dummy_test = pd.get_dummies(titanic_test['Embarked'],drop_first=True)



titanic_test = pd.concat([titanic_test,Embarked_dummy_test],axis=1)
## let's seperate the target variable and divide train data into train and test (validation data) our actual test data is already seperated



target = titanic.pop('Survived')



## define train and test to from train to evaluate our result



## import libraries 

import sklearn

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(titanic,target,random_state=42,stratify=target)



## 'stratify' makes sure that both train and test contains same percentage of target '0' and '1'
import xgboost as xgb



xgb_clf = xgb.XGBClassifier(objective='binary:logistic',seed=42)

xgb_clf.fit(X_train,y_train,verbose=True,early_stopping_rounds=10,eval_metric='aucpr',eval_set=[(X_test,y_test)])
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(xgb_clf,X_test,y_test,values_format='d',display_labels=["Not Survived","Survived"])

plt.show()



## plot confusion matrix on seperated test set of the train data 
## define my parameters



from sklearn.model_selection import GridSearchCV



##ROUND1



param_grid = {

               'max_depth':[2,3,4],

               'learning_rate':[0.1,0.01,0.05],

               'gamma':[0,0.25,1.0],

               'reg_lambda':[0,1.0,10.0]

    

}



optimal_model1 = GridSearchCV(estimator=xgb.XGBClassifier(objctive='binary:logistic',seed=42,subsample=0.9,colsample_bytree=0.6),

                              param_grid=param_grid,

                              scoring='roc_auc',

                              verbose=0,

                              n_jobs=10,

                              cv=5

                             ).fit(

                                   X_train,y_train,early_stopping_rounds=10,eval_metric='auc',eval_set=[(X_test,y_test)]

)
optimal_model1.best_params_
##ROUND2

param_grid = {

               'max_depth':[5,6,7],

               'learning_rate':[0.001,0.003,0.005],

               'gamma':[2.0,5.0,10.0],

               'reg_lambda':[0,1.0,10.0]

    

}



optimal_model2 = GridSearchCV(estimator=xgb.XGBClassifier(objctive='binary:logistic',seed=42,subsample=0.9,colsample_bytree=0.6),

                              param_grid=param_grid,

                              scoring='roc_auc',

                              verbose=0,

                              n_jobs=10,

                              cv=5

                             ).fit(

                                   X_train,y_train,early_stopping_rounds=10,eval_metric='auc',eval_set=[(X_test,y_test)]

)
optimal_model2.best_params_
##ROUND3

param_grid = {

               'max_depth':[8,9,10],

               'learning_rate':[0.0001,0.0003,0.0005],

               'gamma':[1.25,1.50,1.75],

               'reg_lambda':[0,1.0,10.0]

    

}



optimal_model3 = GridSearchCV(estimator=xgb.XGBClassifier(objctive='binary:logistic',seed=42,subsample=0.9,colsample_bytree=0.6),

                              param_grid=param_grid,

                              scoring='roc_auc',

                              verbose=0,

                              n_jobs=10,

                              cv=5

                             ).fit(

                                   X_train,y_train,early_stopping_rounds=10,eval_metric='auc',eval_set=[(X_test,y_test)]

)
optimal_model3.best_params_
optimal_model2.best_estimator_
## hence pick our final params {'gamma': 2.0, 'learning_rate': 0.03, 'max_depth': 7, 'reg_lambda': 0}



xgb_final = optimal_model2.best_estimator_



titanic_test['Survived'] = xgb_final.predict(titanic_test[['Sex', 'Age', 'Fare', 'Alone', 'Pclass_2', 'Pclass_3', 'Q', 'S']])