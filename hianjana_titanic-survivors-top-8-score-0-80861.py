# Import libraries required for EDA, Data pre-processing

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

# Import all libraries for algorithms

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Import libraries for model evaluation

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
# Load the train and test datasets

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
# Check the dataset to get a hang of it
df_train.head()
# Store PassengerId before dropping it

df_test_passengers = df_test['PassengerId']

# Drop PassengerId from train and test datasets as it is just a continuous number

df_train = df_train.drop('PassengerId',axis=1)
df_test = df_test.drop('PassengerId',axis=1)
# Checking all the columns, datatypes
df_train.info()
# Function which returns columns having NULLs
def missing_vals(dataset):
    null_columns=dataset.columns[dataset.isnull().any()]
    print(dataset[null_columns].isnull().sum())
# Check for missing values in columns in train 
missing_vals(df_train)
# Check for missing values in columns in test
missing_vals(df_test)
########## Fill missing values for: Age ##########
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp.fit(df_train[['Age']])
df_train['Age'] = imp.transform(df_train[['Age']])

df_test['Age'] = imp.transform(df_test[['Age']])

# Convert Age to integer
data = [df_train, df_test]

for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
########## Fill missing values for: Embarked ##########
# Check for most frequent value in train dataset
df_train.groupby(['Embarked']).size()
# Replace missing values in train dataset for the column 'Embarked' with the most frequent value of train: 'S'
df_train['Embarked'].fillna('S',inplace=True)
########## Fill missing values for: Fare ##########
# Replace missing values in test for 'Fare' with 0
df_test['Fare'].fillna(0,inplace=True)
########## Fill missing values for: Cabin ##########
# Replace missing values for Cabin with 'U' for 'Unknown'
df_train['Cabin'].fillna('U',inplace=True)
df_test['Cabin'].fillna('U',inplace=True)
# Check again if there are any missing values in train
missing_vals(df_train)
# Check again if there are any missing values in test
missing_vals(df_test)
#### Analyse columns: SibSp, Parch, Fare
# Create new column 'FamilySize' which is SibSp + Parch + 1 (1 is added to represent oneself)
# Drop the columns: SibSp, Parch

dataset = [df_train, df_test]
for data in dataset:
    data['FamilySize'] = ''
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
df_train = df_train.drop(['SibSp', 'Parch'], axis=1)
df_test = df_test.drop(['SibSp', 'Parch'], axis=1)        
# Convert Fare to integer
data = [df_train, df_test]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].astype(int)
# Create new column 'FarePP' which is FarePerPerson by dividing Fare by FamilySize to get individual person's fare
dataset=[df_train, df_test]
for data in dataset:
    data['FarePP'] = ''
    data['FarePP'] = data['Fare']/ data['FamilySize']
# Convert FarePP to integer
data = [df_train, df_test]
for dataset in data:
    dataset['FarePP'] = dataset['FarePP'].astype(int)
###### Analyse column: Sex ######

sns.countplot(x="Sex", hue="Survived", data=df_train);
# Map 'male' to 0 and 'female' to 1 in both train and test
# Convert the datatype to int
data = [df_train, df_test]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1})
    dataset['Sex'] = dataset['Sex'].astype(int)
##### Analyse column: Pclass #####

sns.countplot(x="Pclass", hue="Survived", data=df_train);
##### Analyse column: Embarked #####

sns.countplot(x='Embarked', hue='Survived', data=df_train)
# Analyse column: FamilySize

sns.countplot(x='FamilySize', hue='Survived', data=df_train)
# Create 3 groups of of family and store it in a new column: 'Family'
dataset = [df_train, df_test]
for data in dataset:
    data['Family'] = ''
    data.loc[data['FamilySize'] == 1, 'Family'] = 1
    data.loc[(data['FamilySize'] > 1) & (data['FamilySize'] <= 4), 'Family'] = 2
    data.loc[data['FamilySize'] > 4, 'Family' ] = 3 
##### Analyse column: Cabin #####

df_train['Cabin'].unique()
# Extract Deck from the column Cabin as 'Deck' and drop Cabin
df_train['Deck'] = df_train['Cabin'].str.slice(0,1)
df_test['Deck'] = df_test['Cabin'].str.slice(0,1)
# Check how Deck and survival are correlated
sns.countplot(x='Deck', hue='Survived', data=df_train)
# Count number of people per Deck in train
df_train.groupby(['Deck']).size()
# Count number of people per Deck in test
df_test.groupby(['Deck']).size()
# Replace 'T' with 'U' for Deck in train as there is only one person
df_train.loc[df_train['Deck'] == 'T', 'Deck'] = 'U'
# Check again
df_train.groupby(['Deck']).size()
# Now the Decks are same in train and test
# Drop the column Cabin
df_train = df_train.drop(['Cabin'], axis=1)
df_test = df_test.drop(['Cabin'], axis=1)
##### Analyse column: Age #####

age_xt = pd.crosstab(df_train['Age'], df_train['Survived'])
age_xt_pct = age_xt.div(age_xt.sum(1).astype(float), axis=0)

age_xt_pct.plot(kind='bar', 
                  stacked=True, 
                  title='Age & Survival')
plt.xlabel('Age')
plt.ylabel('Survival Rate')
plt.rcParams["figure.figsize"] = (35, 3)
# Binning for the column 'Age' into 4 bins and store in AgeBin

df_train['AgeBin'] = pd.qcut(df_train['Age'], 4)
df_train['AgeBin'].unique()
# Binning for the column 'Age' in test dataset by referencing the bins formed in train dataset
bins = [-0.001, 22.0, 24.0, 35.0, 80.0]
df_test['AgeBin'] = pd.cut(df_test['Age'], bins)
# Cross check whether the bins formed in train and test are the same
df_test['AgeBin'].unique()
# Use LabelEncoder to form codes for the bins formed
label = LabelEncoder()
df_train['AgeBinCode'] = label.fit_transform(df_train['AgeBin'])
df_test['AgeBinCode'] = label.transform(df_test['AgeBin'])
# Check for the unique age bin codes in train
df_train['AgeBinCode'].unique()
# Check for the unique age bin codes in test
df_test['AgeBinCode'].unique()
# Check how survivors fall under the newly created age bins
sns.countplot(x='AgeBinCode', hue='Survived', data=df_train)
# Drop the column: Age, AgeBin
df_train = df_train.drop(['Age', 'AgeBin'], axis=1)
df_test = df_test.drop(['Age', 'AgeBin'], axis=1)
##### Analyse column: FarePP (FarePerPerson)#####
# Sort dataframe by 'FarePP' descending order to see it's relation with 'Ticket'
df_train.sort_values(by='FarePP', ascending=False).head(10)

# Those with the same fare have the same Ticket Number. 
# Hence,the column 'Ticket' doesn't seem to give any significant information
# To see how FarePP and Survival are correlated
farepp_xt = pd.crosstab(df_train['FarePP'], df_train['Survived'])
farepp_xt_pct = farepp_xt.div(farepp_xt.sum(1).astype(float), axis=0)

farepp_xt_pct.plot(kind='bar', 
                  stacked=True, 
                  title='FarePP & Survival')
plt.xlabel('FarePP')
plt.ylabel('Survival Rate')
plt.rcParams["figure.figsize"] = (28, 5)
# Binning for FarePP 
df_train['FarePPBin'] = pd.qcut(df_train['FarePP'], 4)
# Check for unique bins created in train dataset
df_train['FarePPBin'].unique()
# Binning for the column 'FarePP' in test dataset by referencing the bins formed in train dataset
bins = [-0.001, 7.0, 8.0, 23.0, 512.0]
df_test['FarePPBin'] = pd.cut(df_test['FarePP'], bins)
# Check for unique bins created in test to see train and test bins are in sync
df_test['FarePPBin'].unique()
# Create fareBinCode using LabelEncoder
farelabel = LabelEncoder()
df_train['FareBinCode'] = farelabel.fit_transform(df_train['FarePPBin'])
df_test['FareBinCode'] = farelabel.transform(df_test['FarePPBin'])
# Crosscheck train and test for FareBinCodes created
df_train['FareBinCode'].unique()
# Crosscheck train and test for FareBinCodes created
df_test['FareBinCode'].unique()
# Check how survivors are falling under the newly created FareBinCodes
sns.countplot(x='FareBinCode', hue='Survived', data=df_train)
# Drop columns: Fare, FamilySize, FarePPBin, FarePP
df_train = df_train.drop(['Fare', 'FamilySize', 'FarePPBin', 'FarePP'], axis=1)
df_test = df_test.drop(['Fare', 'FamilySize', 'FarePPBin', 'FarePP'], axis=1)
###### Analyse column: Ticket ######
df_train['Ticket'].unique()
# Drop the column Ticket as it is not having any significant information
df_train = df_train.drop('Ticket', axis=1)
df_test = df_test.drop('Ticket', axis=1)
 ##### Analyse the column: Name #####
# Create a new column: Title
df_train['Title'] = ''
df_test['Title'] = ''

# Extract titles into this column
df_train['Title'] = df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df_test['Title'] = df_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# Check for NULLs
print(df_train['Title'].isnull().sum())
print(df_test['Title'].isnull().sum())
# Check for NULLs in test
print(df_test['Title'].isnull().sum())
# Check how Title and Survival are correlated

title_xt = pd.crosstab(df_train['Title'], df_train['Survived'])
title_xt_pct = title_xt.div(title_xt.sum(1).astype(float), axis=0)

title_xt_pct.plot(kind='bar', 
                  stacked=True, 
                  title='Survival Rate by title')
plt.xlabel('Title')
plt.ylabel('Survival Rate')
plt.rcParams["figure.figsize"] = (15,2)
# Mapping of each title to specific groups
data = [df_train, df_test]

for dataset in data:
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Mrs')      
    dataset['Title'] = dataset['Title'].replace('Mrs', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')     
    dataset['Title'] = dataset['Title'].replace('Miss', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Master', 'Master')
    dataset['Title'] = dataset['Title'].replace('Mr', 'Mr')
    dataset['Title'] = dataset['Title'].replace('Capt', 'Officer')
    dataset['Title'] = dataset['Title'].replace('Major', 'Officer')
    dataset['Title'] = dataset['Title'].replace('Dr', 'Officer')
    dataset['Title'] = dataset['Title'].replace('Col', 'Officer')
    dataset['Title'] = dataset['Title'].replace('Rev', 'Officer') 
    dataset['Title'] = dataset['Title'].replace('Jonkheer', 'Royalty')    
    dataset['Title'] = dataset['Title'].replace('Don', 'Royalty')
    dataset['Title'] = dataset['Title'].replace('Dona', 'Royalty')
    dataset['Title'] = dataset['Title'].replace('Countess', 'Royalty')
    dataset['Title'] = dataset['Title'].replace('Lady', 'Royalty')
    dataset['Title'] = dataset['Title'].replace('Sir', 'Royalty')  
# Cross check train and test for the newly created column Title
df_train['Title'].unique()
# Cross check train and test for the newly created column Title
df_test['Title'].unique()
# Check the Title column again after the mapping

title_xt = pd.crosstab(df_train['Title'], df_train['Survived'])
title_xt_pct = title_xt.div(title_xt.sum(1).astype(float), axis=0)

title_xt_pct.plot(kind='bar', 
                  stacked=True, 
                  title='Survival Rate by title')
plt.xlabel('Title')
plt.ylabel('Survival Rate')
plt.rcParams["figure.figsize"] = (15,2)
# Drop the column: Name
df_train = df_train.drop(['Name'], axis=1)
df_test = df_test.drop(['Name'], axis=1)
# Rename the column 'Survived' to 'Target'
df_train = df_train.rename(columns={'Survived': 'Target'})

####### One hot code: Embarked #######
embarked_one_hot = pd.get_dummies(df_train['Embarked'], prefix='Embarked')
embarked_one_hot_test = pd.get_dummies(df_test['Embarked'], prefix='Embarked')

df_train = df_train.join(embarked_one_hot)
df_test = df_test.join(embarked_one_hot_test)

# Drop the column Embarked
df_train = df_train.drop(['Embarked'], axis=1)
df_test = df_test.drop(['Embarked'], axis=1)

####### One hot code: Title #######
title_one_hot = pd.get_dummies(df_train['Title'], prefix='Title')
title_one_hot_test = pd.get_dummies(df_test['Title'], prefix='Title')

df_train = df_train.join(title_one_hot)
df_test = df_test.join(title_one_hot_test)

# Drop the column Title
df_train = df_train.drop(['Title'], axis=1)
df_test = df_test.drop(['Title'], axis=1)

####### One hot code: Pclass #######
pclass_one_hot = pd.get_dummies(df_train['Pclass'], prefix='Pclass')
pclass_one_hot_test = pd.get_dummies(df_test['Pclass'], prefix='Pclass')

df_train = df_train.join(pclass_one_hot)
df_test = df_test.join(pclass_one_hot_test)

# Drop the column: Pclass
df_train = df_train.drop(['Pclass'], axis=1)
df_test = df_test.drop(['Pclass'], axis=1)

####### One hot code: Deck #######
deck_one_hot = pd.get_dummies(df_train['Deck'], prefix='Deck')
deck_one_hot_test = pd.get_dummies(df_test['Deck'], prefix='Deck')

df_train = df_train.join(deck_one_hot)
df_test = df_test.join(deck_one_hot_test)

# Drop the column: Deck
df_train = df_train.drop(['Deck'], axis=1)
df_test = df_test.drop(['Deck'], axis=1)

####### One hot code: Family #######
family_one_hot = pd.get_dummies(df_train['Family'], prefix='Family')
family_one_hot_test = pd.get_dummies(df_test['Family'], prefix='Family')

df_train = df_train.join(family_one_hot)
df_test = df_test.join(family_one_hot_test)

# Drop the column: Family
df_train = df_train.drop(['Family'], axis=1)
df_test = df_test.drop(['Family'], axis=1)

####### One hot code: AgeBinCode #######
age_one_hot = pd.get_dummies(df_train['AgeBinCode'], prefix='AgeBinCode')
age_one_hot_test = pd.get_dummies(df_test['AgeBinCode'], prefix='AgeBinCode')

df_train = df_train.join(age_one_hot)
df_test = df_test.join(age_one_hot_test)

# Drop the column: AgeBinCode
df_train = df_train.drop(['AgeBinCode'], axis=1)
df_test = df_test.drop(['AgeBinCode'], axis=1)

####### One hot code: FareBinCode #######
fare_one_hot = pd.get_dummies(df_train['FareBinCode'], prefix='FareBinCode')
fare_one_hot_test = pd.get_dummies(df_test['FareBinCode'], prefix='FareBinCode')

df_train = df_train.join(fare_one_hot)
df_test = df_test.join(fare_one_hot_test)

# Drop the column: FareBinCode
df_train = df_train.drop(['FareBinCode'], axis=1)
df_test = df_test.drop(['FareBinCode'], axis=1)
#### Check all columns once again in train and test ####
df_train.head(5)
df_test.head(5)
##################### X & y creation #####################

y = df_train['Target']
X = df_train.drop('Target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
### Cross validate various models ###

num_folds = 5
seed = 2
scoring = 'accuracy'
models = []
scores = []
names = []

models.append(('LR', LogisticRegression()))
models.append(('SVC', SVC()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('GBC', GradientBoostingClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))

for name, model in models:
    kfold = StratifiedKFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    scores.append(cv_results)
    names.append(name)
    msg = "%s %f %f " % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# LogisticRegression, GradientBoostingClassifier, and RandomForestClassifier are selected    
##### Logistic Regression with GridSearch #####

c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

LR = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
lr_gs = GridSearchCV(LR, param_grid=param_grid, cv=5)

# Fit model to the data
lr_gs.fit(X_train, y_train)

parameters = lr_gs.best_params_

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(lr_gs.best_params_)) 
print("Best score is {}".format(lr_gs.best_score_))
y_pred_lr = lr_gs.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(accuracy_lr)
##### Random Forest with Grid Search #####

RF = RandomForestClassifier()
param_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 100],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'criterion': ['gini', 'entropy']
                 }
rf_gs = GridSearchCV(RF,
                           scoring='accuracy',
                           param_grid=param_grid,
                           cv=5,
                           verbose=1
                           )

rf_gs.fit(X_train, y_train)

parameters = rf_gs.best_params_

print('Best score: {}'.format(rf_gs.best_score_))
print('Best parameters: {}'.format(rf_gs.best_params_))
y_pred_rf = rf_gs.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(accuracy_rf)
##### Gradient Boost Classifier #####

GBC = GradientBoostingClassifier(n_estimators=100)
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gbc_gs = GridSearchCV(GBC, param_grid = gb_param_grid, cv=kfold, scoring='accuracy', n_jobs= 4, verbose = 1)

gbc_gs.fit(X_train, y_train)
y_pred_gbc = gbc_gs.predict(X_test)
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
print(accuracy_gbc)
############## Model evaluation ##############
confusion_matrix(y_test, y_pred_lr)
confusion_matrix(y_test, y_pred_rf)
confusion_matrix(y_test, y_pred_gbc)
# Scores of Logistic Regression

print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:",recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))
# Scores of GBC

print("Precision:", precision_score(y_test, y_pred_gbc))
print("Recall:",recall_score(y_test, y_pred_gbc))
print("F1 Score:", f1_score(y_test, y_pred_gbc))
# Score of RF

print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:",recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
##################### Final Prediction using GBC #####################

y_pred_final = gbc_gs.predict(df_test)

final_submission = pd.DataFrame({
        "PassengerId": df_test_passengers,
        "Survived": y_pred_final
    })

filename = 'titanic_submission.csv'
final_submission.to_csv(filename, sep=',', index = False)