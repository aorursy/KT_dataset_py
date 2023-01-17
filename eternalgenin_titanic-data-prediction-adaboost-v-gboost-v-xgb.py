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

## Remove column with missing Embarked
## Creating the first letter of ticket as a variable



train['Ticket_First'] = train['Ticket'].apply(lambda x:x.split()[0][:1])



pd.value_counts(train['Ticket_First'])
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
# Removing the unneeded and NA-dominated columns



train.drop(['Age', 'Cabin'], axis =1 , inplace = True)
# Drop the na rows



train.dropna(inplace = True)
# Check if all the null values are gone



sns.heatmap(pd.isnull(train))
## Now creating dummy variables for Sex and Embarked





Sex_Dumm = pd.get_dummies(train['Sex'], drop_first = True)

Embarked_Dumm = pd.get_dummies(train['Embarked'], drop_first = True)

Ticket_First = pd.get_dummies(train['Ticket_First'], drop_first = True, prefix = 'Ticket')

Salute_Group = pd.get_dummies(train['Salute_Grp'], drop_first = True)
train = pd.concat([train, Sex_Dumm, Embarked_Dumm, Ticket_First, Salute_Group], axis = 1)

train.head()
train.columns
## Creating test train dataset from 'train' dataframe only as we don't have the 'y' for test.



from sklearn.model_selection import train_test_split



y = train['Survived']



X_train, X_test, y_train, y_test = train_test_split(train[['Pclass', 'SibSp', 'Parch', 'Fare',

       'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W',

        'Miss.', 'Mr.', 'Mrs.', 'Others']], y, test_size = 0.3, random_state = 143)
## Fitting into the base AdaBoost model for different n_estimators



from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



for n_est in [50,100,150,200]:

    

    ad_cl = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), n_estimators = n_est, random_state = 105)

    ad_cl.fit(X_train,y_train)

    pred = ad_cl.predict(X_test)

    print('No of estimators = ', n_est)

    print('Accuracy Score = ',accuracy_score(y_test, pred))

    print('\n')

    

## So, 150 seems to be the best no. of estimators with Accuracy Score of ~0.85
## Fitting into the base Gradient boosting model



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score



for n_est in [20, 50,100,150,200]:

    

    gd_cl = GradientBoostingClassifier(n_estimators = n_est, random_state = 105)

    gd_cl.fit(X_train,y_train)

    pred = gd_cl.predict(X_test)

    print('No of estimators = ', n_est)

    print('Accuracy Score = ',accuracy_score(y_test, pred))

    print('\n')



## So, 100 seems to be the best no. of estimators with Accuracy Score of ~0.82 without tuning other parameters
from xgboost import XGBClassifier
## Fitting into the base Xtreme Gradient boosting model



from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score



for n_est in [20, 50,100,150,200]:

    

    xgb_cl = XGBClassifier(n_estimators = n_est, random_state = 105)

    xgb_cl.fit(X_train,y_train)

    pred = xgb_cl.predict(X_test)

    print('No of estimators = ', n_est)

    print('Accuracy Score = ',accuracy_score(y_test, pred))

    print('\n')



## So, 100 seems to be the best no. of estimators with Accuracy Score of ~0.82 without tuning other parameters
## Prepare the test dataset in the same way



test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.info()
test['Ticket_First'] = test['Ticket'].apply(lambda x:x.split()[0][:1])



test['Ticket_First'].unique()
test['Salute'] = test['Name'].apply(lambda x:x.split()[1])
test['Salute'] = test['Name'].apply(lambda x:x.split()[1])

def Salute_group(col):

    

    if col[0] in ['Mr.', 'Miss.', 'Mrs.', 'Master.']:

        return col[0]

    else:

        return 'Others'
test['Salute_Grp'] = test[['Salute']].apply(Salute_group, axis =1)


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

Ticket_First = pd.get_dummies(test['Ticket_First'], drop_first = True, prefix = 'Ticket')

Salute_Group = pd.get_dummies(test['Salute_Grp'], drop_first = True)
test = pd.concat([test, Sex_Dumm, Embarked_Dumm, Ticket_First, Salute_Group], axis = 1)

test.head()
## Adding these two variables as these were not present in test data 



test['Ticket_5']=0

test['Ticket_8']=0
# Now using all the train dataset to fit the model and then predicting the test data



X = train[['Pclass', 'SibSp', 'Parch', 'Fare',

                                  'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_5', 'Ticket_6', 'Ticket_7', 'Ticket_8', 'Ticket_9', 'Ticket_A',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W', 'Miss.', 'Mr.', 'Mrs.', 'Others']]

y = train['Survived']
mdl_fin = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1),n_estimators = 150, random_state = 105)
mdl_fin.fit(X,y)



test.set_index('PassengerId', inplace = True)
test_fin =test[['Pclass', 'SibSp', 'Parch', 'Fare',

                                  'Age_PclXSex', 'male', 'Q', 'S', 'Ticket_2', 'Ticket_3', 'Ticket_4',

       'Ticket_6', 'Ticket_7', 'Ticket_9', 'Ticket_A', 'Ticket_5', 'Ticket_8',

       'Ticket_C', 'Ticket_F', 'Ticket_L', 'Ticket_P', 'Ticket_S', 'Ticket_W', 'Miss.', 'Mr.', 'Mrs.', 'Others']]



test_fin
pred_fin = mdl_fin.predict(test_fin)





pred_df = pd.DataFrame(pred_fin, columns = ['Survived'],index = test_fin.index)

pred_df
# Output Result

pred_df['Survived'].to_csv('My_Titanic_Predictions.csv', index = True, header = True)