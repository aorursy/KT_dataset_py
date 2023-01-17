import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots

import eli5

from eli5.sklearn import PermutationImportance

import shap
# Read in csv files

titanic_original = pd.read_csv('../input/train.csv')

titanic_validation  = pd.read_csv('../input/test.csv')



# Makes an original copy that isn't a pointer

titanic = titanic_original.copy(deep = True) 

titanic_val = titanic_validation.copy(deep = True)



# Clean up number of lines to work on both sets

data_sets = [titanic,titanic_val] 



titanic.head()
titanic.describe()
for dataset in data_sets:

    for col in dataset.columns:

    #     print(titanic[col].isnull().any())

        if dataset[col].isnull().any() == True:

            print(col + ' has {} empty values'.format(dataset[col].isnull().sum()))
# Fill in missing values with median/mode

for dataset in data_sets:

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



# Delete columns we aren't interested in

for dataset in data_sets:

    del_cols = ['PassengerId','Ticket','Cabin']

    dataset.drop(del_cols, axis=1, inplace = True)



# Check to see if all missing values have been imputed

for dataset in data_sets:

    print(dataset.isnull().sum())

for dataset in data_sets:

    # Family Size + 1 to include the passenger

    dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    # Alone

    dataset['Alone'] = 0 # Initially set them to not be alone

    dataset['Alone'].loc[dataset['Family_Size'] == 1] = 1 # Set to alone if family = 0



    # Titles

    dataset['Title'] = dataset['Name'].str.split(', ', expand = True)[1].str.split('.', expand = True)[0]



    print(dataset.Title.value_counts())
for dataset in data_sets:

    useable_titles = (dataset['Title'].value_counts() < 10)

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Rare' 

                                              if useable_titles.loc[x] == True else x)

    # You can also use .replace() for dataframes

    print(dataset.Title.value_counts())
for dataset in data_sets:

    

#     # Cutting the Fare values into 4 groups

#     dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)



#     # Cutting Age into 5 bins while also converting them from float to int

#     dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

    

    # Or replacing them with chosen values using quartiles from above as ranges

    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3

    

    # We also convert ages to ranges

    dataset.loc[dataset['Age'] <= 20.125, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 20.125) & (dataset['Age'] <= 28), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 38), 'Age'] = 2

    dataset.loc[dataset['Age'] > 38, 'Age'] = 3

    

    dataset['Fare'] = dataset['Fare'].astype(int)

    dataset['Age'] = dataset['Age'].astype(int)

#     dataset.Fare.astype(int)

#     dataset.Age.astype(int)

titanic.head()
label = LabelEncoder()



# Change categorical variables into dummy variables

for dataset in data_sets:

    dataset['Sex_Var'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])



print(titanic.columns.tolist())

titanic.head()
X = titanic.drop(['Survived','Name','Sex','Embarked','Title'],axis=1)

y = titanic['Survived']

X.head()
# This is for one data set to split, we already have them split up

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# X_train = X

# y_train = y

# X_test = titanic_val.drop(['Survived','Name','Sex','Embarked','Title'],axis=1)
X_train.count()
# Logreg

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(X_train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
random_forest = RandomForestClassifier(n_estimators=100)

first_model = random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
titanic_validation['Survived'] = random_forest.predict(titanic_val[X_train.columns.tolist()])

titanic_validation.head()
#submit file

submit = titanic_validation[['PassengerId','Survived']]

submit.to_csv("../working/submit.csv", index=False)