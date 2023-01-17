# Import libs

import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import seaborn as sns

import re





%matplotlib inline 
# Open files

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
# Check the columns

train.columns
# Which gender survived the most?

sns.countplot(x = 'Survived', hue = 'Sex', data = train)
# Which class survived more?

sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
# Plot the ages

train['Age'].hist().plot()
train['Age'].mean()
train['Fare'].hist().plot(bins = 20, figsize = (10, 5))

sns.countplot( x = 'SibSp', hue='Survived', data = train)
train.info()
train.isnull().sum()
sns.heatmap(train.isnull(), cmap = 'viridis')
sns.boxplot(x = 'Pclass', y = 'Age', data = train)
# Drop the Survived from the train data and concat 

X_full = pd.concat([train.drop('Survived', axis = 1), test], axis = 0)



# Remove the passenger ID

X_full.drop('PassengerId', axis = 1, inplace=True)



X_full.head()
# Map the letter & number for the cabin

X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0] 

cabin_dict = {k:i for i, k in enumerate(X_full.Cabin_mapped.unique())} 

X_full.loc[:, 'Cabin_mapped'] = X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)





X_full['Cabin_number'] = X_full['Cabin'].str.extract('(\d+)')

X_full['Cabin_number'].fillna(value=-99, inplace=True)

X_full['Cabin_number'] = pd.to_numeric(X_full['Cabin_number'])



X_full.drop('Cabin', axis = 1, inplace=True)
# Below 13 doesn't really matter the sex, can be treated the same

def expand_sex(sex, age):

    if age < 13:

        return 'kid'

    else:

        return sex



X_full['Sex'] = list(map(expand_sex, X_full['Sex'], X_full['Age']))
def father(sex, age, parch):

    if sex == 'male' and age > 16 and parch > 0:

        return 1

    else:

        return 0

        

        

def mother(sex, age, parch):

    if sex == 'female' and age > 16 and parch > 0:

        return 1

    else:

        return 0

        

        

def parent(sex, age, parch):

    if mother(sex, age, parch) == 1 or father(sex, age, parch) == 1:

        return 1

    else:

        return 0





# Family features

X_full['FamilySize'] = X_full['SibSp'] + X_full['Parch']

X_full['Father'] = list(map(father, X_full.Sex, X_full.Age, X_full.Parch))

X_full['Mother'] = list(map(mother, X_full.Sex, X_full.Age, X_full.Parch))

X_full['Parent'] = list(map(parent, X_full.Sex, X_full.Age, X_full.Parch))

X_full['has_parents_or_kids'] = X_full.Parch.apply(lambda x: 1 if x > 0 else 0)

def extract_maritial(name):

    """ extract the person's title, and bin it to Mr. Miss. and Mrs.

    assuming a Miss, Lady or Countess has more change to survive than a regular married woman."""

    

    re_maritial = r' ([A-Za-z]+\.) '   # use regular expressions to extract the persons title

    found = re.findall(re_maritial, name)[0]

    replace = [['Dr.','Sir.'],

               ['Rev.','Sir.'],

               ['Major.','Officer.'],

               ['Mlle.','Miss.'],

               ['Col.','Officer.'],

               ['Master.','Sir.'],

               ['Jonkheer.','Sir.'],

               ['Sir.','Sir.'],

               ['Don.','Sir.'],

               ['Countess.','High.'],

               ['Capt.','Officer.'],

               ['Ms.','High.'],

               ['Mme.','High.'],

               ['Dona.','High.'],

               ['Lady.','High.']]

                

    for i in range(0,len(replace)):

        if found == replace[i][0]:

            found = replace[i][1]

            break

    return found



X_full['Title'] = list(map(extract_maritial, X_full['Name']))



X_full.drop('Name', axis = 1, inplace=True)
# Replace the missing fares with the mean

fare_mean = X_full[X_full.Pclass == 3].Fare.mean()

X_full['Fare'].fillna(fare_mean, inplace = True)



X_full['FareBin'] = pd.cut(X_full.Fare, bins=(-1000,0,8.67,16.11,32,350,1000))



X_full.drop('Fare', axis = 1, inplace=True)
# Fill the gaps in age by the mean of the title

age_by_title = X_full.groupby('Title', as_index=False)['Age'].mean()



for t in age_by_title['Title']:

    ft = (X_full['Title'] == t)

    age_median = age_by_title.loc[age_by_title.Title == t, 'Age'].values[0]

    X_full.loc[ft, 'Age'] = X_full.loc[ft,'Age'].fillna(age_median)



X_full['AgeBin'] = pd.cut(X_full.Age, bins=(0,13,26,39,60,90))



X_full.drop('Age', axis = 1, inplace=True)
# Replace the missing embarked witht he most common

X_full['Embarked'].fillna('S', inplace = True)


X_full.drop('Ticket', axis = 1, inplace=True)
# Transform categorical values

X_dummies = pd.get_dummies(X_full, columns = ['Sex','Embarked', 'FareBin', 'AgeBin', 'Title'], drop_first= True)



display(X_dummies)

X_dummies.dtypes
regex = re.compile(r"\[|\]|<", re.IGNORECASE)



X_dummies.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_dummies.columns.values]



display(X_dummies)
X = X_dummies[:len(train)]

new_X = X_dummies[len(train):]

y = train.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size = .3,

                                                    random_state = 5,

                                                    stratify = y)

from xgboost import XGBClassifier

from xgboost import plot_importance

from numpy import sort

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score

from xgboost import plot_tree





xgb = XGBClassifier()

xgb.fit(X_train, y_train)

xgb.score(X_test, y_test)



#print(xgb.feature_importances_)

#plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)

#plt.show()



#plot_importance(xgb)

#plt.show()



plot_tree(xgb)

plt.show()





thresholds = sort(xgb.feature_importances_)

for thresh in thresholds:

    # select features using threshold

    selection = SelectFromModel(xgb, threshold=thresh, prefit=True)

    select_X_train = selection.transform(X_train)

    # train model

    selection_model = XGBClassifier()

    selection_model.fit(select_X_train, y_train)

    # eval model

    select_X_test = selection.transform(X_test)

    y_pred = selection_model.predict(select_X_test)

    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)

    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

print("Best parameters found: ", xgb_random.best_params_)

print("Best accuracy found: ", xgb_random.best_score_)

import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV



# Create the parameter grid: gbm_param_grid 

gbm_param_grid = {

    'n_estimators': range(8, 20),

    'max_depth': range(6, 10),

    'learning_rate': [.4, .45, .5, .55, .6],

    'colsample_bytree': [.6, .7, .8, .9, 1]

}



# Instantiate the regressor: gbm

gbm = XGBClassifier(n_estimators=10)



# Perform random search: grid_mse

xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 

                                    estimator = gbm, scoring = "accuracy", 

                                    verbose = 1, n_iter = 50, cv = 4)





# Fit randomized_mse to the data

xgb_random.fit(X, y)



# Print the best parameters and lowest RMSE

print("Best parameters found: ", xgb_random.best_params_)

print("Best accuracy found: ", xgb_random.best_score_)





xgb_pred = xgb_random.predict(new_X)



submission = pd.concat([test.PassengerId, pd.DataFrame(xgb_pred)], axis = 'columns')

submission.columns = ["PassengerId", "Survived"]

submission.to_csv('my_prediction.csv', header = True, index = False)
