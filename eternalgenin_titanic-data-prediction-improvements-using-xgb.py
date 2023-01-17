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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
## Checking the dataset for completeness



train.info()



## Remove Cabin

## Substitute missing Age

## Remove row with missing Embarked
## Extracting the salutation from name as a variable



train['Salute'] = train['Name'].apply(lambda x:x.split()[1])
pd.value_counts(train['Salute']).head()
## Grouping the minor salutations as others



def Salute_group(col):

    

    if col[0] in ['Mr.', 'Miss.', 'Mrs.', 'Master.']:

        return col[0]

    else:

        return 'Others'
train['Salute_Grp'] = train[['Salute']].apply(Salute_group, axis =1)
sns.set_style('whitegrid')

sns.countplot(x='Salute_Grp', data = train, hue = 'Survived')
## Creating the first two characters of ticket as a variable



train['Ticket_First'] = train['Ticket'].apply(lambda x:x.replace('.','').replace('/','').split()[0][:2])

## Checking the survival frequency across Ticket_First variable



ticket_freq = train[['Ticket_First','Survived']].groupby(['Ticket_First']).agg([('Nos people', 'count'), ('Nos survived', 'sum')])

ticket_freq.columns = ticket_freq.columns.get_level_values(1)

ticket_freq = ticket_freq.reset_index(level = [0])

ticket_freq['Survival %'] = round(ticket_freq['Nos survived']*100/ticket_freq['Nos people'])

ticket_freq.sort_values(by = ['Nos people'], ascending = False)



## It does seem like there are too many variables with too little observations to reliably decide survival. So, grouping the 

## Ticket_First where # observations =< 10
## Grouping the minor Ticket_First as others



def Ticket_Grp(col):

    

    if col[0] in ticket_freq[ticket_freq['Nos people'] > 10]['Ticket_First'].to_list():

        return col[0]

    else:

        return 'Others'
train['Ticket_Grp'] = train[['Ticket_First']].apply(Ticket_Grp, axis =1)
train['Ticket_Grp'].value_counts()
##Missing Values



sns.heatmap(train.isnull())
# Treat Age



# Substitute missing values with medians by Sex X Pclass

sns.boxplot (x='Sex', y='Age', data = train, hue = 'Pclass')

# Calculating medians



PclassXSex_med = train[['Sex','Age','Pclass']].groupby(['Sex','Pclass']).median()
# Defining a function to impute median (using median since the data is skewed) for each PclassXSex.



## MUCH MORE EFFICIENT WAY TO WRITING FUNCTION THAN BEFORE ##



def age_PclassSex(cols):

    age = cols[0]

    Pclass = cols[1]

    Sex = cols[2]

    

    if pd.isnull(age) == True:

        return PclassXSex_med.loc[Sex].loc[Pclass][0]

    else:

        return age

train['Age_PclXSex'] = train[['Age', 'Pclass', 'Sex']].apply(age_PclassSex, axis = 1)
# Removing the Age (as we have already created a different Age variable) and NA-dominated columns



train.drop(['Age', 'Cabin'], axis =1 , inplace = True)
# Drop the na rows



train.dropna(inplace = True)
# Check if all the null values are gone



sns.heatmap(pd.isnull(train))
## Now creating dummy variables for Sex and Embarked





Sex_Dumm = pd.get_dummies(train['Sex'], drop_first = True)

Embarked_Dumm = pd.get_dummies(train['Embarked'], drop_first = True)

Ticket_Grp = pd.get_dummies(train['Ticket_Grp'], drop_first = True, prefix = 'Ticket')

Salute_Group = pd.get_dummies(train['Salute_Grp'], drop_first = True)
train = pd.concat([train, Sex_Dumm, Embarked_Dumm, Ticket_Grp, Salute_Group], axis = 1)

train.head()

train.columns
## Creating test train dataset from 'train' dataframe only as we don't have the 'y' for test.

from sklearn.model_selection import train_test_split



y = train['Survived']



X_train, X_test, y_train, y_test = train_test_split(train[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_PclXSex', 'male', 'Q', 'S',  'Ticket_13', 'Ticket_17',

       'Ticket_19', 'Ticket_23', 'Ticket_24', 'Ticket_25', 'Ticket_26',

       'Ticket_28', 'Ticket_29', 'Ticket_31', 'Ticket_33', 'Ticket_34',

       'Ticket_35', 'Ticket_36', 'Ticket_37', 'Ticket_A5', 'Ticket_CA',

       'Ticket_Others', 'Ticket_PC', 'Ticket_SC', 'Ticket_SO', 'Ticket_ST',

        'Miss.', 'Mr.', 'Mrs.', 'Others']], y, test_size = 0.3, random_state = 143)

from xgboost import XGBClassifier
## Building base model for benchmark



xgb_base = XGBClassifier(random_state = 105)

xgb_base.fit(X_train, y_train)

xgb_base
pred = xgb_base.predict(X_test)

print(accuracy_score(y_test, pred))
## Randomly selecting best set of parameters for Xtreme Gradient boosting model



from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV



## Defining parameter grid



param_grid = {"n_est" : [10, 20, 25, 30, 40, 50, 100,150, 200],

         "learning_rate" : [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

         "max_depth" : [ 2, 3, 4, 5, 6, 8, 10, 12, 15],

         "min_child_weight" : [ 1, 3, 5, 7 ],

         "gamma" : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

         "colsample_bytree" : [ 0.1, 0.2, 0.3, 0.4, 0.5 , 0.7 ] }



## Using Randomized Search CV for finding best set of parameters

xgb_cl = XGBClassifier(random_state = 105)

xgb_random = RandomizedSearchCV(xgb_cl, param_grid, n_iter = 500, cv = 5, verbose =2 , 

                                scoring = 'accuracy', random_state = 143, n_jobs = -1)

xgb_random.fit(X_train, y_train)
xgb_random.best_params_
pred2 = xgb_random.best_estimator_.predict(X_test)
print(accuracy_score(y_test, pred2))
## Prepare the test dataset in the same way



test = pd.read_csv('/kaggle/input/titanic/test.csv')
## Extracting the salutation from name as a variable



test['Salute'] = test['Name'].apply(lambda x:x.split()[1])
## Grouping the minor salutations as others



def Salute_group(col):

    

    if col[0] in ['Mr.', 'Miss.', 'Mrs.', 'Master.']:

        return col[0]

    else:

        return 'Others'
test['Salute_Grp'] = test[['Salute']].apply(Salute_group, axis =1)



test['Salute_Grp'].value_counts()
## Creating the first two characters of ticket as a variable



test['Ticket_First'] = test['Ticket'].apply(lambda x:x.replace('.','').replace('/','').split()[0][:2])

## Grouping the minor Ticket_First as others



def Ticket_Grp2(col):

    

    if col[0] in ['A5', 'PC', 'ST', '11', '37', '33', '17', '34', '23',

       '35', '24', '26', '19', 'CA', 'SC', '31', '29', '36', 'SO', '25',

       '28', '13']:

        return col[0]

    else:

        return 'Others'
test['Ticket_Grp'] = test[['Ticket_First']].apply(Ticket_Grp2, axis =1)
PclassXSex_med = test[['Sex','Age','Pclass']].groupby(['Sex','Pclass']).median()

PclassXSex_med
test['Age_PclXSex'] = test[['Age', 'Pclass', 'Sex']].apply(age_PclassSex, axis = 1)
# Removing the unneeded and NA-dominated columns



test.drop(['Cabin', 'Age'], axis =1 , inplace = True)
Fare_med = test[['Pclass','Fare','Sex', 'Embarked']].groupby(['Pclass','Sex', 'Embarked']).agg(['count', 'mean'])



Fare_med
test['Fare'].fillna(12.718, inplace = True)
## Now creating dummy variables for Sex and Embarked





Sex_Dumm = pd.get_dummies(test['Sex'], drop_first = True)

Embarked_Dumm = pd.get_dummies(test['Embarked'], drop_first = True)

Ticket_Grp = pd.get_dummies(test['Ticket_Grp'], drop_first = True, prefix = 'Ticket')

Salute_Group = pd.get_dummies(test['Salute_Grp'], drop_first = True)
test = pd.concat([test, Sex_Dumm, Embarked_Dumm, Ticket_Grp, Salute_Group], axis = 1)

test.columns
# Now using all the train dataset to fit the model and then predicting the test data



X = train[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_PclXSex', 'male', 'Q', 'S',  'Ticket_13', 'Ticket_17',

       'Ticket_19', 'Ticket_23', 'Ticket_24', 'Ticket_25', 'Ticket_26',

       'Ticket_28', 'Ticket_29', 'Ticket_31', 'Ticket_33', 'Ticket_34',

       'Ticket_35', 'Ticket_36', 'Ticket_37', 'Ticket_A5', 'Ticket_CA',

       'Ticket_Others', 'Ticket_PC', 'Ticket_SC', 'Ticket_SO', 'Ticket_ST',

        'Miss.', 'Mr.', 'Mrs.', 'Others']]

y = train['Survived']

## Predicting using base model first

xgb_fin_base = XGBClassifier()

xgb_fin_base.fit(X,y)
test.set_index('PassengerId', inplace = True)
test_fin =test[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_PclXSex', 'male', 'Q', 'S',  'Ticket_13', 'Ticket_17',

       'Ticket_19', 'Ticket_23', 'Ticket_24', 'Ticket_25', 'Ticket_26',

       'Ticket_28', 'Ticket_29', 'Ticket_31', 'Ticket_33', 'Ticket_34',

       'Ticket_35', 'Ticket_36', 'Ticket_37', 'Ticket_A5', 'Ticket_CA',

       'Ticket_Others', 'Ticket_PC', 'Ticket_SC', 'Ticket_SO', 'Ticket_ST',

        'Miss.', 'Mr.', 'Mrs.', 'Others']]



test_fin
pred_fin_base = xgb_fin_base.predict(test_fin)





pred_base_df = pd.DataFrame(pred_fin_base, columns = ['Survived'],index = test_fin.index)

pred_base_df
# Output Result

pred_base_df['Survived'].to_csv('My_Titanic_Predictions.csv', index = True, header = True)
## Predicting using tuned model

xgb_tuned = xgb_random.best_estimator_
xgb_tuned.fit(X,y)
pred_fin_tuned = xgb_tuned.predict(test_fin)





pred_tuned = pd.DataFrame(pred_fin_tuned, columns = ['Survived'],index = test_fin.index)

pred_tuned
# Output Result

pred_tuned['Survived'].to_csv('My_Titanic_Predictions2.csv', index = True, header = True)