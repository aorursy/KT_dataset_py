import numpy as np

import pandas as pd

import re
test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

train.info()
# define the target vector

target = train["Survived"].values



# concatenate the datasets for feature engineering

full = pd.concat([train, test], axis=0, sort=False)
# get the title from the name

full["Title"] = full["Name"].apply(lambda x: re.search(' ([A-Za-z]+)\.',x).group(1))

title_mapping = {

    "Mr": 1,

    "Miss": 2,

    "Mrs": 3,

    "Master": 4,

    "Dr": 5,

    "Rev": 6,

    "Major": 7,

    "Col": 7,

    "Mlle": 2,

    "Mme": 3,

    "Don": 9,

    "Dona": 9,

    "Lady": 10,

    "Countess": 10,

    "Jonkheer": 10,

    "Sir": 9,

    "Capt": 7,

    "Ms": 2,

}

# map the values into categories

full["TitleCat"] = full.loc[:,'Title'].map(title_mapping)
# get family size by combining the number of siblings, parents and themself

full["FamilySize"] = full["SibSp"] + full["Parch"] + 1

# bin the values into different groups

full["FamilySize"] = pd.cut(full["FamilySize"], bins=[0,1,4,20], labels=[0,1,2])
# get the full lenght of the name

full["NameLength"] = full["Name"].apply(lambda x: len(x))
# transform the embarked column into categorical

full["Embarked"] = pd.Categorical(full.Embarked).codes
# fill the fare null values with the median fare

median_fare = test.Fare.median()

full["Fare"] = full["Fare"].fillna(median_fare)
# one-hot encode the sex column

full = pd.concat([full,pd.get_dummies(full['Sex'])],axis=1)
# fill the na values in the cabin column and save it to a new column

full['CabinCat'] = full.Cabin.fillna('0')

# apply lambda function to retain only the cabin letter

full['CabinCat'] = full.CabinCat.apply(lambda x: x[0])

# transform the column into categorical

full['CabinCat'] = pd.Categorical(full.CabinCat).codes
# function to check if the cabin is even/odd/null 

def get_type_cabine(cabine):

    # Use a regular expression to search for number

    cabine_search = re.search('\d+', cabine)

    # If the number exists, extract and return it.

    if cabine_search:

        num = cabine_search.group(0)

        if np.float64(num) % 2 == 0:

            return '2'

        else:

            return '1'

    return '0'



# fill the na values in the cabin column and save it to a new column

full["CabinType"] = full.Cabin.fillna(" ")

# apply the funciton

full["CabinType"] = full.CabinType.apply(get_type_cabine)
# get the person type from age and sex

def get_person_type(passenger):

    age, sex = passenger

    if (age < 18):

        return 'child'

    elif (sex == 'female'):

        return 'female_adult'

    else:

        return 'male_adult'

    

person = pd.DataFrame(full[['Age', 'Sex']].apply(get_person_type, axis=1), columns=['person'])

# concatenate the person series with the full dataset

full = pd.concat([full, person], axis=1)

# one-hot enconde the person column

full = pd.concat([full,pd.get_dummies(full['person'])],axis=1)
# count the number of members for each ticket

ticket_table = pd.DataFrame(full.Ticket.value_counts())

ticket_table.rename(columns={'Ticket':'Ticket_Members'}, inplace=True)



# create boolean series

isFemale = full.female_adult == 1

isMale = full.male_adult == 1

survived = full.Survived == 1

perished = full.Survived == 0

hasFamily = (full.Parch > 0) | (full.SibSp > 0)



# get the number of women that perished for each ticket 

ticket_table['Ticket_perishing_women'] = full.Ticket[

    isFemale & perished & hasFamily

].value_counts()

# fill nan values with 0

ticket_table['Ticket_perishing_women'] = ticket_table.Ticket_perishing_women.fillna(0)

# transform into a boolean series encoding any value greater than 0 as 1

hasPerishedWomen = ticket_table.Ticket_perishing_women > 0

ticket_table.loc[hasPerishedWomen, 'Ticket_perishing_women'] = 1.0 



# get the number of men that survived for each ticket

ticket_table['Ticket_surviving_men'] = full.Ticket[

    isMale & survived & hasFamily

].value_counts()

# fill nan values with 0

ticket_table['Ticket_surviving_men'] = ticket_table.Ticket_surviving_men.fillna(0)

# transform into a boolean series encoding any value greater than 0 as 1

hasSurvivingMan = ticket_table.Ticket_surviving_men > 0

ticket_table.loc[hasSurvivingMan, 'Ticket_surviving_men'] = 1.0 



# crate a Ticket_Id categorical column

ticket_table['Ticket_Id'] = pd.Categorical(ticket_table.index).codes

# assign tickets with less than 3 members the code -1

hasLessThan3Members = ticket_table.Ticket_Members < 3

ticket_table.loc[hasLessThan3Members, 'Ticket_Id'] = -1

# bin the members values into different groups

ticket_table['Ticket_Members'] = pd.cut(ticket_table['Ticket_Members'], bins=[0,1,4,20], labels=[0,1,2])



# merge the ticket_table dataframe with the full dataframe

full = pd.merge(

    full,

    ticket_table,

    left_on="Ticket",

    right_index=True,

    how='left', 

    sort=False

)
# get the surname from the name

full['surname'] = full["Name"].apply(lambda x: x.split(',')[0].lower())



# count the number of members for each surname

surname_table = pd.DataFrame(full['surname'].value_counts())

surname_table.rename(columns={'surname':'Surname_Members'}, inplace=True)



# create boolean series

isFemale = full.female_adult == 1

isMale = full.male_adult == 1

survived = full.Survived == 1

perished = full.Survived == 0

hasFamily = (full.Parch > 0) | (full.SibSp > 0)



# get the number of women that perished for each surname 

surname_table['Surname_perishing_women'] = full.surname[

    isFemale & perished & hasFamily

].value_counts()

# fill nan values with 0

surname_table['Surname_perishing_women'] = surname_table.Surname_perishing_women.fillna(0)

# transform into a boolean series encoding any value greater than 0 as 1

hasPerishedWomen = surname_table.Surname_perishing_women > 0

surname_table.loc[hasPerishedWomen, 'Surname_perishing_women'] = 1.0 



# get the number of men that survived for each surname

surname_table['Surname_surviving_men'] = full.surname[

    isMale & survived & hasFamily

].value_counts()

# fill nan values with 0

surname_table['Surname_surviving_men'] = surname_table.Surname_surviving_men.fillna(0)

# transform into a boolean series encoding any value greater than 0 as 1

hasSurvivingMan = surname_table.Surname_surviving_men > 0

surname_table.loc[hasSurvivingMan, 'Surname_surviving_men'] = 1.0 



# crate a Surname_Id categorical column

surname_table['Surname_Id'] = pd.Categorical(surname_table.index).codes

# assign surnames with less than 3 members the code -1

hasLessThan3Members = surname_table.Surname_Members < 3

surname_table.loc[hasLessThan3Members, 'Surname_Id'] = -1

# bin the members values into different groups

surname_table['Surname_Members'] = pd.cut(surname_table['Surname_Members'], bins=[0,1,4,20], labels=[0,1,2])



# merge the surname_table dataframe with the full dataframe

full = pd.merge(

    full,

    surname_table,

    left_on="surname",

    right_index=True,

    how='left', 

    sort=False

)
print('Number of missing ages: ', full.Age.isnull().sum())

print('Median age: ', full.Age.median())

print('Mean age: ', full.Age.mean())

print('Std age: ', full.Age.std())

print('Min age: ', full.Age.min())

print('Max age: ', full.Age.max())
from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor



features = ['Fare','Parch','Pclass','SibSp','TitleCat', 

'CabinCat','female','male', 'Embarked', 'FamilySize', 'NameLength','Ticket_Members','Ticket_Id']



# build the train sets using the rows without the missing age

X_train = full[features][full['Age'].notnull()]

y_train = full['Age'][full['Age'].notnull()]



# build the text set with the rows with missing ages

X_test = full[features][full['Age'].isnull()]



# create and fit the model

model = ExtraTreesRegressor(n_estimators=500)

model.fit(X_train, y_train)



# make predictions on the train set

y_preds_train = model.predict(X_train)



# calculate the MSE of the predictions

print('MSE: ', mean_absolute_error(y_train, y_preds_train))
# make predictions on the test set (missing ages)

y_preds_test = model.predict(X_test)



# fill the missing ages with the predictions

full.loc[full.Age.isnull(), 'Age'] = y_preds_test
print('Number of missing ages: ', full.Age.isnull().sum())

print('Median age: ', full.Age.median())

print('Mean age: ', full.Age.mean())

print('Std age: ', full.Age.std())

print('Min age: ', full.Age.min())

print('Max age: ', full.Age.max())
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



from sklearn.ensemble import RandomForestClassifier



import xgboost as xgb

from xgboost import XGBClassifier



import time
# split back the full dataset into train and test sets

train = full[0:891].copy()

test = full[891:].copy()
# specify the features to be used in the model

features = ['female','male','Age','male_adult','female_adult', 'child','TitleCat', 'Pclass',

'Pclass','Ticket_Id','NameLength','CabinType','CabinCat', 'SibSp', 'Parch',

'Fare','Embarked','Surname_Members','Ticket_Members','FamilySize',

'Ticket_perishing_women','Ticket_surviving_men',

'Surname_perishing_women','Surname_surviving_men']
# build the train and test sets

X_train = train[features]

y_train = train['Survived']



X_test = test[features]
seed = 42



# create the default model

rfc = RandomForestClassifier(

#     min_samples_split=4,

    class_weight={0:0.745,1:0.255}, 

    random_state=seed

)



# Create the grid search parameter grid and scoring funcitons

param_grid = {

    "max_depth": np.linspace(1, 32, 2),

    "n_estimators": np.arange(100, 1000, 100),

    "criterion": ["gini","entropy"],

    "max_leaf_nodes": [16, 64, 128, 256],

    "oob_score": [True],

}

scoring = {

    'AUC': 'roc_auc', 

    'Accuracy': make_scorer(accuracy_score)

}



# create the Kfold object

num_folds = 10

kfold = StratifiedKFold(n_splits=num_folds, random_state=seed)



# create the grid search object with the full pipeline as estimator

n_iter=50

grid = RandomizedSearchCV(

    estimator=rfc, 

    param_distributions=param_grid,

    cv=kfold,

    scoring=scoring,

    n_jobs=-1,

    n_iter=n_iter,

    refit="AUC"

)





# fit grid search

%time best_model = grid.fit(X_train,y_train)
print(f'Best score: {best_model.best_score_}')

print(f'Best params: {best_model.best_params_}')
# metrics of the  best model

pred_train = best_model.predict(X_train)



print('Train Accuracy: ', accuracy_score(y_train, pred_train))

print("Out-of-Bag Accuracy: ", best_model.best_estimator_.oob_score_)

print('\nClassification Report:')

print(classification_report(y_train,pred_train))
# make the predictions on the test set

predictions = best_model.predict(X_test)
# save the predictions to the outputfile

PassengerId = np.array(test['PassengerId']).astype(int)

predictions = predictions.astype(int)



my_predictions = pd.DataFrame({

    'PassengerId': PassengerId,

    'Survived': predictions

})

my_predictions.to_csv("my_predictions.csv", index=False)
my_predictions.Survived.value_counts()
my_predictions.info()