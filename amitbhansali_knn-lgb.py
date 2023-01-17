import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

%matplotlib inline

import os

import warnings

warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
train_data.head()

test_data.head()
titanic = pd.concat([train_data,test_data],axis=0,sort=False) # join train and test into titanic

titanic.shape 

titanic.isna().sum() # verifying na
titanic.drop('Cabin', 1, inplace = True)
titanic['title'] = titanic['Name'].apply(lambda x: x.split(", ")[1].split(".")[0])
#Replace title with more common one



titanic['title'] = titanic['title'].map({

           'Mlle': 'Miss', 

           'Ms': 'Miss', 

           'Mme': 'Mrs',

           'Major': 'Other', 

           'Col': 'Other', 

           'Dr' : 'Other', 

           'Rev' : 'Other',

           'Capt': 'Other', 

           'Jonkheer': 'Royal',

           'Sir': 'Royal', 

           'Lady': 'Royal',

           'Dona': 'Royal',

           'Don': 'Royal',

           'the Countess': 'Royal', 

           'Dona': 'Royal',

           'Miss': 'Miss',

           'Mrs': 'Mrs',

           'Master': 'Master',

           'Mr': 'Mr'})



#titanic['title'] = titanic['title'].astype('category') # converting into category

titanic['title'].value_counts(ascending=True)

titanic.info()
titanic.groupby('title', sort=False)['Age'].mean()
def impute_age(cols):

    Age = cols[0]

    title = cols[1]

    

    if pd.isnull(Age):



        if title == 'Mr':

            return 32.25



        elif title == 'Mrs':

            return 36.91

        

        elif title == 'Miss':

            return 21.82

        

        elif title == 'Master':

            return 5.48

        

        elif title == 'Royal':

            return 41.16

        

        elif title == 'Other':

            return 46.27

    else:

        return Age

    

titanic['Age'] = titanic[['Age','title']].apply(impute_age,axis=1)
sns.distplot(titanic['Age'])
## Show the categories of Age and Fare

print(pd.qcut(titanic['Fare'].fillna(titanic['Fare'].median()), 5).unique())

print(pd.qcut(titanic['Age'].fillna(titanic['Age'].median()), 5).unique())
# Age



def age_Cateogry(cols):

    Age = cols[0]



    if Age < 16:

        return 1

    

    elif Age < 32 and Age >=16:

        return 2

    

    elif Age < 48 and Age >=32:

        return 3

    

    elif Age < 64 and Age >=48:

        return 4



    else:

        return 5

    

titanic['Age_Code'] = titanic[['Age']].apply(age_Cateogry,axis=1)
# Fare



def fare_Cateogry(cols):

    fare = cols[0]



    if fare < 7.854:

        return 1

    

    elif fare < 10.5 and fare >=7.854:

        return 2

    

    elif fare < 21.558 and fare >=10.5:

        return 3

    

    elif fare < 41.579 and fare >=21.558:

        return 4

    

    else:

        return 5

    

titanic['Fare_Code'] = titanic[['Fare']].apply(fare_Cateogry,axis=1)
# title



def title_Cateogry(cols):

    title = cols[0]



    if title == 'Mr':

            return 1



    elif title == 'Mrs':

            return 2

        

    elif title == 'Miss':

            return 3

        

    elif title == 'Master':

            return 4

        

    elif title == 'Royal':

            return 5



    elif title == 'Other':

            return 6

     



titanic['Title_Code'] = titanic[['title']].apply(title_Cateogry,axis=1)
#FamilySize = SibSp + Parch

titanic['FamSize'] = titanic['SibSp'] + titanic['Parch'] + 1

titanic['Embarked'] = titanic['Embarked'].map({'S':0,'C':1,'Q':2})

titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})
titanic.info()
titanic['Fare'].mean()
titanic["Fare"].fillna("33.29", inplace = True) 
titanic.info()
TrainData = titanic.iloc[:891, :]

TestData = titanic.iloc[891:, :]



TrainData.dropna(inplace=True)

X_train_Kaggel = TrainData[['Pclass', 'FamSize', 'Age_Code', 'Fare_Code','Title_Code','Sex','Embarked']]

Y_train_Kaggle = TrainData[['Survived']]

X_test_kaggle =  TestData[['Pclass', 'FamSize', 'Age_Code', 'Fare_Code','Title_Code','Sex','Embarked']]



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X_train_Kaggel,Y_train_Kaggle,test_size=0.3)



#Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



## transforming "train_x"

X_train = scaler.fit_transform(X_train)

## transforming "test_x"

X_test = scaler.transform(X_test)



## transforming "The testset"

#X_test_kaggle = scaler.transform(X_test_kaggle)
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=2)

## Search for an optimal value of k for KNN.

k_range = range(1,31)

k_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X_train,Y_train, cv = cv, scoring = 'accuracy')

    k_scores.append(scores.mean())

print("Accuracy scores are: {}\n".format(k_scores))

print ("Mean accuracy score: {}".format(np.mean(k_scores)))
knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train,Y_train)

predictions = knn.predict(X_test)

                           

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(Y_test, predictions))

print(classification_report(Y_test, predictions))

print(cross_val_score(knn, X_test,predictions, cv = cv, scoring = 'accuracy'))


X_train,X_test,Y_train,Y_test = train_test_split(X_train_Kaggel,Y_train_Kaggle,test_size=0.3)





import lightgbm as lgbm



from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV



# Stats

import scipy.stats as ss

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform



random_state = 42

    

fit_params = {"early_stopping_rounds" : 100, 

             "eval_metric" : 'auc', 

             "eval_set" : [(X_train,Y_train)],

             'eval_names': ['valid'],

             'verbose': 0,

             'categorical_feature': 'auto'}



param_test = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],

              'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],

              'num_leaves': sp_randint(6, 50), 

              'min_child_samples': sp_randint(100, 500), 

              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

              'subsample': sp_uniform(loc=0.2, scale=0.8), 

              'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],

              'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),

              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}



#number of combinations

n_iter = 500 



#intializing lgbm and lunching the search

lgbm_clf = lgbm.LGBMClassifier(random_state=random_state, silent=True, metric='None', n_jobs=4)

grid_search = RandomizedSearchCV(

    estimator=lgbm_clf, param_distributions=param_test, 

    n_iter=n_iter,

    scoring='accuracy',

    cv=5,

    refit=True,

    random_state=random_state,

    verbose=True)



grid_search.fit(X_train_Kaggel, Y_train_Kaggle, **fit_params)

print('Best score reached: {} with params: {} '.format(grid_search.best_score_, grid_search.best_params_))



opt_parameters =  grid_search.best_params_
%%time

lgbm_clf = lgbm.LGBMClassifier(**opt_parameters)

lgbm_clf.fit(X_train, Y_train)

y_pred = lgbm_clf.predict(X_test_kaggle)
y_pred = y_pred.astype(int)

print (y_pred)
#predictionsTest = knn.predict(X_test_kaggle)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")