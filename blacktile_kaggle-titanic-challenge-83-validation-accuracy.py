from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import learning_curve

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import KFold

from matplotlib import pyplot as plt

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import shap

%matplotlib inline
import warnings

warnings.simplefilter('ignore')
# read the csv train and test files 

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
# number of rows and columns in train and test set

train.shape, test.shape
train.head()
test.head()
#identify numerical and categorical variables

train.dtypes
train.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)

test.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
def get_titles(data):

    """ To get the title from each name in the dataset and to add it to a new feature column 'Title.

    

        Args:



            data (dataframe) - Dataset from which we need to get and set the titles in the name.



        Returns:



            data (dataframe) - Dataset after the update is returned.

        

    """

    # extract the title from each name

    data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    return data
train = get_titles(train)

test = get_titles(test)
# remove the varibale 'Name' from both the test and train set

train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
# find unique titles in train

train.Title.unique()
# find unique titles in test

test.Title.unique()
title_dict = {

    "Capt": "Rare",

    "Col": "Rare",

    "Major": "Rare",

    "Jonkheer": "Rare",

    "Don": "Rare",

    "Dona": "Miss",

    "Sir" : "Rare",

    "Dr": "Rare",

    "Rev": "Rare",

    "the Countess":"Rare",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Rare"

}

# map each title to its class or category

train.Title = train.Title.map(title_dict)

test.Title = test.Title.map(title_dict)
train.Title.unique()
test.Title.unique()
train.isna().sum()
test.isna().sum()
train.drop(['Cabin'], axis=1, inplace=True)

test.drop(['Cabin'], axis=1, inplace=True)
# groupby and find the median of age for train

grouped_median_train = train.groupby(['Sex','Pclass','Title']).median()

grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
grouped_median_train.head()
def fill_age(row):

    """ To assign value for the missing values of based on the group.



            If a title is missing then the age will be assigned based on sex and class.

            

        Args: 

            

            row (dataframe) - A row in the dataframe.

        

        Returns:

        

            The values of the computed age in the dataframe.

    """

    condition = (

        (grouped_median_train['Sex'] == row['Sex']) & 

        (grouped_median_train['Title'] == row['Title']) & 

        (grouped_median_train['Pclass'] == row['Pclass'])

    ) 

    if np.isnan(grouped_median_train[condition]['Age'].values[0]):

        print('true')

        condition = (

            (grouped_median_train['Sex'] == row['Sex']) & 

            (grouped_median_train['Pclass'] == row['Pclass'])

        )



    return grouped_median_train[condition]['Age'].values[0]





def process_age(data):

    """ To fill the missing values of the Age variable

        

        Args:

        

            data (dataframe) - The dataset into which we need to fill the missing values of the age.

            

        Returns:

        

            data (dataframe) - Processed dataset

    """

    data.Age = data.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

    return data
# replace missing values of Embarked in train with its most frequent data

train.Embarked.fillna('S', inplace=True)

# fill the missing values of age in train with the median age of the group

train = process_age(train)

test = process_age(test)

# assign mean of the variable Fare in test to the missing value of fare 

test.Fare = test['Fare'].fillna(train['Fare'].median())

# to convert the variable Age to dataframe which is currently series 

train = pd.DataFrame(train)

test = pd.DataFrame(test)
train.isna().sum()
test.isna().sum()
train.head()
test.head()
plt.figure(figsize=[10, 8])

plt.hist(x=train.Age, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)

plt.xlabel('Value',fontsize=15)

plt.ylabel('Frequency',fontsize=15)

plt.title('Data Distribution Histogram - Age',fontsize=15)

plt.show()
plt.figure(figsize=[10, 8])

plt.hist(x=train.Fare, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)

plt.xlabel('Value',fontsize=15)

plt.ylabel('Frequency',fontsize=15)

plt.title('Data Distribution Histogram - Fare',fontsize=15)

plt.show()
# find frequency of each label in the variable

Pclass_counts = train.Pclass.value_counts()

Sex_counts = train.Sex.value_counts()

SibSp_counts = train.SibSp.value_counts()

Parch_counts = train.Parch.value_counts()

Embarked_counts = train.Embarked.value_counts()

Title_counts = train.Title.value_counts()

Survived_counts = train.Survived.value_counts()
fig, axs = plt.subplots(3, 2, figsize=[15, 20])



axs[0,0].bar(Pclass_counts.index, Pclass_counts, alpha=0.7)

axs[0,0].set_title('Categorical Distribution - Pclass')



axs[0,1].bar(Sex_counts.index, Sex_counts,alpha=0.7)

axs[0,1].set_title('Categorical Distribution - Sex')



axs[1,0].bar(SibSp_counts.index, SibSp_counts,alpha=0.7)

axs[1,0].set_title('Categorical Distribution - SibSp')



axs[1,1].bar(Parch_counts.index, Parch_counts,alpha=0.7)

axs[1,1].set_title('Categorical Distribution - Parch')



axs[2,0].bar(Embarked_counts.index, Embarked_counts,alpha=0.7)

axs[2,0].set_title('Categorical Distribution - Embarked')



axs[2,1].bar(Title_counts.index, Title_counts,alpha=0.7)

axs[2,1].set_title('Categorical Distribution - Title')



for ax in axs.flat:

    ax.set(xlabel='Category', ylabel='Frequency')
plt.figure(figsize=[10, 8])

p = plt.bar(Survived_counts.index, Survived_counts, width=.4)

plt.xlabel('Category',fontsize=15)

plt.ylabel('Frequency',fontsize=15)

plt.title('Categorical Distribution - Survival',fontsize=15)

plt.show()
data = train[['Survived', 'Sex']]

data['Died'] = 1 - data['Survived']
data.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar',stacked=True, figsize=[10, 6])
plt.figure(figsize=[10, 6])

sns.violinplot(x='Sex', y='Age', hue='Survived', data=train, split=True)
plt.figure(figsize=[10, 6])

sns.violinplot(x='Sex', y='Fare', hue='Survived', data=train, split=True)
fig, ax = plt.subplots(figsize=[10, 8])

train.boxplot(column='Fare', by='Survived', ax=ax, grid=False, fontsize=15)
plt.figure(figsize=[10, 6])

sns.violinplot(x='Sex', y='Pclass', hue='Survived', data=train, split=True)
data['Title'] = train['Title']
data.groupby('Title').agg('sum')[['Survived','Died']].plot(kind='bar',stacked=True, figsize=[10, 6])
train.head()
test.head()
def process_family(data):

    # the size of families (including the passenger)

    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1 

    return data
train = process_family(train)

test = process_family(test)
sex_map = {

    'male' : 0,

    'female' : 1

}



embarked_map = {

    'Q' : 0,

    'S' : 1,

    'C' : 2

}



title_map = {

             "Mr": 1, 

             "Master": 2, 

             "Mrs": 3, 

             "Miss": 4, 

             "Rare": 5

            }
# label encode the categorical variables

train.Sex = train.Sex.map(sex_map)

test.Sex = test.Sex.map(sex_map)

train.Embarked = train.Embarked.map(embarked_map)

test.Embarked = test.Embarked.map(embarked_map)

train.Title = train.Title.map(title_map)

test.Title = test.Title.map(title_map)
train.head()
test.head()
# drop the variable 'SibSp' as we have already created a similar variable FamilySize

train = train.drop(['SibSp'], axis = 1)

test  = test.drop(['SibSp'], axis = 1)
# seperate the feature set and the target set

X_train = train.loc[:, train.columns!='Survived']

y_train = train['Survived']

X_test = test
#split the train data into train and valid set

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10, random_state=111)
X_train.shape, X_valid.shape
model = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, max_features=0.5, n_jobs=-1)

model.fit(X_train, y_train)

model.score(X_train, y_train)
train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.01, 1.0, 10), scoring='accuracy')



# Create means of training set scores

train_mean = np.mean(train_scores, axis=1)



# Create means of test set scores

test_mean = np.mean(test_scores, axis=1)
# Create plot

plt.plot(train_sizes, train_mean, label="Training")

plt.plot(train_sizes, test_mean, label="Validation")

plt.title("Learning Curve")

plt.xlabel("Training Set Size")

plt.ylabel("Accuracy Score")

plt.legend()

plt.show()
predict = model.predict(X_valid)

accuracy_score(y_valid, predict)
print(classification_report(y_valid, predict))
# find the variable or feature importance

shap_values = shap.TreeExplainer(model).shap_values(X_train)

shap.summary_plot(shap_values, X_train)
# Desired number of Cross Validation folds

cv = KFold(n_splits=10)            

accuracies = list()

max_attributes = len(list(test))

depth_range = range(1, max_attributes + 1)



# Testing max_depths from 1 to max attributes

# Uncomment prints for details about each Cross Validation pass

for depth in depth_range:

    fold_accuracy = []

    tree_model = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, max_features=0.5, n_jobs=-1, max_depth=depth)

    # print("Current max depth: ", depth, "\n")

    for train_fold, valid_fold in cv.split(train):

        f_train = train.loc[train_fold] # Extract train data with cv indices

        f_valid = train.loc[valid_fold] # Extract valid data with cv indices



        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), 

                               y = f_train["Survived"]) # We fit the model with the fold train data

        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), 

                                y = f_valid["Survived"])# We calculate accuracy with the fold validation data

        fold_accuracy.append(valid_acc)



    avg = sum(fold_accuracy)/len(fold_accuracy)

    accuracies.append(avg)

    # print("Accuracy per fold: ", fold_accuracy, "\n")

    # print("Average accuracy: ", avg)

    # print("\n")

    

# Just to show results conveniently

df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})

df = df[["Max Depth", "Average Accuracy"]]

print(df.to_string(index=False))
print(np.mean(accuracies))
params = {



    'max_depth' : 5,

    'n_estimators': 5,

    'gamma': 6,

    'objective' : 'binary:logistic',

    'eval_metric' : ["error", "logloss"],

    'n_gpus' : 0

}
model = xgb.XGBClassifier(**params)
evallist = [(X_train, y_train), (X_valid, y_valid)]
model.fit(X_train, y_train, eval_set=evallist, eval_metric=["error", "logloss"], verbose=False)
predictions = model.predict(X_valid)
print(classification_report(y_valid, predictions))
# retrieve performance metrics

results = model.evals_result_

epochs = len(results['validation_0']['error'])

x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(10,8))

plt.plot(x_axis, results['validation_0']['error'], label = 'Train')

plt.plot(x_axis, results['validation_1']['error'], label = 'Test')

ax.legend()

plt.ylabel('Classification Loss')

plt.title('XGBoost Classification Loss')

plt.show()
# retrieve performance metrics

results = model.evals_result_

epochs = len(results['validation_0']['logloss'])

x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(10,8))

plt.plot(x_axis, results['validation_0']['logloss'], label = 'Train')

plt.plot(x_axis, results['validation_1']['logloss'], label = 'Test')

ax.legend()

plt.ylabel('Classification Loss')

plt.title('XGBoost Classification Loss')

plt.show()
# find the variable or feature importance

shap_values = shap.TreeExplainer(model).shap_values(X_train)

shap.summary_plot(shap_values, X_train)
params = {



    'n_estimators': 5,

    'gamma': 6,

    'objective' : 'binary:logistic',

    'eval_metric' : ["error", "logloss"],

    'n_gpus' : 0

}
# Desired number of Cross Validation folds

cv = KFold(n_splits=10)            

accuracies = list()

max_attributes = len(list(test))

depth_range = range(1, max_attributes + 1)



# Testing max_depths from 1 to max attributes

# Uncomment prints for details about each Cross Validation pass

for depth in depth_range:

    fold_accuracy = []

    tree_model = xgb.XGBClassifier(**params, max_depth=depth)

    # print("Current max depth: ", depth, "\n")

    for train_fold, valid_fold in cv.split(train):

        f_train = train.loc[train_fold] # Extract train data with cv indices

        f_valid = train.loc[valid_fold] # Extract valid data with cv indices



        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), 

                               y = f_train["Survived"]) # We fit the model with the fold train data

        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), 

                                y = f_valid["Survived"])# We calculate accuracy with the fold validation data

        fold_accuracy.append(valid_acc)



    avg = sum(fold_accuracy)/len(fold_accuracy)

    accuracies.append(avg)

    # print("Accuracy per fold: ", fold_accuracy, "\n")

    # print("Average accuracy: ", avg)

    # print("\n")

    

# Just to show results conveniently

df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})

df = df[["Max Depth", "Average Accuracy"]]

print(df.to_string(index=False))
print(np.mean(accuracies))
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
numerical_columns = ['Age', 'Fare']
scaler = StandardScaler()

train[numerical_columns] = scaler.fit_transform(train[numerical_columns])

test[numerical_columns] = scaler.fit_transform(test[numerical_columns])
X_train = train.loc[:, train.columns!='Survived']

y_train = train['Survived']

X_test = test

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10)
# Instantiate our model

logreg = LogisticRegression()

# Fit our model to the training data

logreg.fit(X_train, y_train)

# Predict on the test data

logreg_predictions = logreg.predict(X_valid)

print(classification_report(y_valid, logreg_predictions))
model=LogisticRegression()

predicted = cross_val_predict(model, X_train, y_train, cv=10)

print(classification_report(y_train, predicted) )

print(accuracy_score(y_train, predicted))
# may use which ever model is the best

output = model.predict(X_test)
test_set = pd.read_csv('../input/titanic/test.csv')

output_csv = pd.DataFrame()

output_csv['PassengerId'] = test_set['PassengerId']

output_csv['Survived'] = output

output_csv[['PassengerId','Survived']].to_csv('../output/random_forest_predictions.csv', index=False)