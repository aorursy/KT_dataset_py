# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from pandas import Series, DataFrame

import pandas as pd

import numpy as np

import seaborn

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
train =  pd.read_csv('../input/train.csv')

test =  pd.read_csv('../input/test.csv')

train.shape, test.shape
train_passengerid = train['PassengerId']

train_survived = train['Survived']

test_passengerid = test['PassengerId']

train.drop(['PassengerId', 'Survived'], axis = 1, inplace = True)

test.drop('PassengerId', axis = 1, inplace = True)
total = pd.concat([train, test])
def get_title():

    

    global total

    total['title'] = total['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    title_dict = { "Capt":       "Officer",

               "Col":        "Officer",

               "Major":      "Officer",

               "Jonkheer":   "Royalty",

               "Don":        "Royalty",

               "Sir" :       "Royalty",

               "Dr":         "Officer",

               "Rev":        "Officer",

               "the Countess":"Royalty",

               "Dona":       "Royalty",

               "Mme":        "Mrs",

               "Mlle":       "Miss",

               "Ms":         "Mrs",

               "Mr" :        "Mr",

               "Mrs" :       "Mrs",

               "Miss" :      "Miss",

               "Master" :    "Master",

               "Lady" :      "Royalty"}

    total['title'] = total['title'].map(title_dict)

    

get_title()
total.drop('Name', axis = 1, inplace = True)

total.head(2)
grouped = total.groupby(['Sex', 'Pclass', 'title'])

grouped.median()
def fill_age(df):

    

    if df['Sex' == 'female']:

        

        if df['Pclass' == 1]:

            

            if df['title' == 'Miss']:

                return 30

            elif df['title' == 'Mrs']:

                return 45

            elif df['title' == 'Officer']:

                return 49

            elif df['title' == 'Royalty']:

                return 39

            

        elif df['Pclass' == 2]:

            

            if df['title' == 'Miss']:

                return 20

            elif df['title' == 'Mrs']:

                return 30

            

        elif df['Pclass' == 3]:

            

            if df['title' == 'Miss']:

                return 18

            elif df['title' == 'Mrs']:

                return 31

        

    if df['Sex' == 'male']:

        

        if df['Pclass' == 1]:

            

            if df['title' == 'Master']:

                return 6

            elif df['title' == 'Mr']:

                return 41.5

            elif df['title' == 'Officer']:

                return 52

            elif df['title' == 'Royalty']:

                return 40

            

        elif df['Pclass' == 2]:

            

            if df['title' == 'Master']:

                return 2

            elif df['title' == 'Mr']:

                return 30

            elif df['title' == 'Officer']:

                return 41.5

            

            

        elif df['Pclass' == 3]:

            

            if df['title' == 'Master']:

                return 6

            elif df['title' == 'Mr']:

                return 26

   

total.Age = total.apply(lambda r : fill_age(r) if np.isnan(r['Age']) else r['Age'], axis=1)
sex_dict = {'male' : 1, 'female' : 0}

total['Sex'] = total['Sex'].map(sex_dict)
total['family_size'] = total['SibSp'] + total['Parch'] + 1 

total['Single'] = total['family_size'].map(lambda s : 1 if s==1 else 0)

total['small_family'] = total['family_size'].map(lambda s : 1 if 1<s<5 else 0)

total['large_family'] = total['family_size'].map(lambda s : 1 if 4<s else 0)
total['Cabin'].fillna('U', inplace = True)

total['Cabin'] = total['Cabin'].map(lambda s: s[0])
total['Embarked'].fillna('S', inplace = True)
total.info()
total['Fare'].fillna(total.Fare.mean(), inplace = True)
total.drop('Ticket', axis = 1, inplace = True)
total = pd.get_dummies(total, columns=['Cabin', 'Embarked', 'title', 'Pclass'])
# Assembling the  & spliting datsets for model prepartion & validation

# Assembling the  & spliting datsets for model prepartion & validation



Train = total[:train.shape[0]]

Test = total[train.shape[0]:]



print ("new Train shape:", Train.shape)

print ("new Test shape:", Test.shape)
Train = Train.apply(lambda x: x/x.max(), axis = 0)
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score
target = train_survived

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=200)

clf = clf.fit(Train, target)
features = pd.DataFrame()

features['feature'] = Train.columns

features['importance'] = clf.feature_importances_
features.sort(['importance'],ascending=False)
model = SelectFromModel(clf, prefit=True)

Train_new = model.transform(Train)

Train_new.shape



Test_new = model.transform(Test)

Test_new.shape
from sklearn.cross_validation import cross_val_score

from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import StratifiedKFold



forest = RandomForestClassifier(max_features='sqrt')



parameter_grid = {

                 'max_depth' : [1,2,3,4,5,6,7,8],

                 'n_estimators': [50,100,200,500],

                 'criterion': ['gini','entropy']

                 }



cross_validation = StratifiedKFold(target, n_folds=10, random_state = 42) # Used stratified random sampling for cross validation



grid_search = GridSearchCV(forest,

                           param_grid=parameter_grid,

                           cv=cross_validation)



grid_search.fit(Train, target)
print ('Best Score: {}'.format(grid_search.best_score_))

print ('Best Parameters: {}'.format(grid_search.best_params_))

print ('Best estimator: {}'.format(grid_search.best_estimator_))
Test_pred_class =grid_search.predict(Test) 
Disaster_test = pd.DataFrame(Test_pred_class)

submission = pd.DataFrame({"PassengerId":test_passengerid, "Survived": Disaster_test[0]})

submission.to_csv("submission.csv", index=False)
print(check_output(["ls", "../input"]).decode("utf8"))