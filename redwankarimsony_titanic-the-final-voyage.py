!pip install seaborn==0.11.0



# General Data Manipulation Library

import pandas as pd

import numpy as np

import re





# Plotting LIbraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

from IPython.display import clear_output

warnings.filterwarnings('ignore')

import random



random.seed(1455)

np.random.seed(1455)



from sklearn.metrics import accuracy_score



sns.set_theme()

clear_output()
# Load in the train and test datasets

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



# Store our passenger ID for easy access

PassengerId = test['PassengerId']



test.head(5)
# Dealing with the Ticket

train['Ticket_type'] = train['Ticket'].apply(lambda x: x[0:4])

train['Ticket_type'] = train['Ticket_type'].astype('category')

train['Ticket_type'] = train['Ticket_type'].cat.codes



test['Ticket_type'] = test['Ticket'].apply(lambda x: x[0:4])

test['Ticket_type'] = test['Ticket_type'].astype('category')

test['Ticket_type'] = test['Ticket_type'].cat.codes



train['Ticket_type'].value_counts().plot.bar(figsize=(30, 5), title = 'Ticket Code Conversion', xlabel = 'Ticket Code', ylabel = 'People Count')
train['Words_Count'] = train['Name'].apply(lambda x: len(x.split()))

test['Words_Count'] = test['Name'].apply(lambda x: len(x.split()))



fig = sns.displot(data=train, x="Words_Count", hue="Survived", multiple="stack", kde = True)

fig.set(title='Words_Count feature vs Survived')
# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



fig = sns.displot(data=train, x="Has_Cabin", hue="Survived", multiple="stack", kde = True)

fig.set(title='Has_Cabin feature vs Survived')
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1



fig = sns.displot(data=train, x="FamilySize", hue="Survived", multiple="stack", kde = True)

fig.set(title='FamilySize feature vs Survived')
train['IsAlone'] = 0

train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1



test['IsAlone'] = 0

test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Fare column and create a new feature CategoricalFare

train['Fare'] = train['Fare'].fillna(train['Fare'].median())

test['Fare'] = test['Fare'].fillna(train['Fare'].median())



train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
whole_data  = [train, test]



for dataset in whole_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)









fig = sns.displot(data=train, x="Age", hue="Survived", multiple="stack", kde = True)

fig.set(title='Age vs Survived')







# Mapping Age

dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;



fig = sns.displot(data=train, x="Age", hue="Survived", multiple="stack", kde = True)

fig.set(title='Age Mapped vs Survived')
# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



# Create a new feature Title, containing the titles of passenger names

for dataset in whole_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

    

# Group all non-common titles into one single grouping "Rare"

for dataset in whole_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



    

for dataset in whole_data:

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

fig = sns.displot(data=train, x="Title", hue="Survived", multiple="stack", kde = True)

fig.set(title='Title Mapped vs Survived')
for dataset in whole_data:

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

# Mapping Sex [Male: 1, Female: 0]

for dataset in whole_data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



fig = sns.displot(data=train, x="Sex", hue="Survived", multiple="stack", kde = True)

fig.set(title='Sex vs Survived')





fig = sns.displot(data=train, x="Fare", hue="Survived", multiple="stack", kde = True)

fig.set(title='Fare vs Survived')
# Taking Care of the Missing Values

train['Embarked'] = train['Embarked'].fillna('S')

test['Embarked'] = test['Embarked'].fillna('S')

    

# Mapping Embarked

train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



fig = sns.displot(data=train, x="Embarked", hue="Survived", multiple="stack", kde = True)

fig.set(title='Embarked feature vs Survived')
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)

train.head(5)
plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.3,vmax=1.0, 

            square=True, cmap='PiYG', linecolor='white', annot=True)
plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=12)

sns.heatmap(train.astype(float).corr(method = 'spearman'),linewidths=0.3,vmax=1.0, 

            square=True, cmap='PiYG', linecolor='white', annot=True)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize= (30, 12))

sns.heatmap(train.astype(float).corr(method = 'pearson'),linewidths=0.3,vmax=1.0, 

            square=True, cmap='PiYG', linecolor='white', annot=True, ax = ax[0])



sns.heatmap(train.astype(float).corr(method = 'spearman'),linewidths=0.3,vmax=1.0, 

            square=True, cmap='PiYG', linecolor='white', annot=True, ax = ax[1])



ax[0].title.set_text('Peason Correlation of Features')

ax[1].title.set_text('Spearman Correlation of Features')



fig.show()
y_train = train['Survived'].ravel() # Creates an array of the train labels

train = train.drop(['Survived'], axis=1)

x_train = train.values # Creates an array of the train data

x_test = test.values # Creates an array of the test labels
test_data_with_labels = pd.read_csv("https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv")

test_data = pd.read_csv('../input/titanic/test.csv')



for i, name in enumerate(test_data_with_labels['name']):

    if '"' in name:

        test_data_with_labels['name'][i] = re.sub('"', '', name)

        

for i, name in enumerate(test_data['Name']):

    if '"' in name:

        test_data['Name'][i] = re.sub('"', '', name)

        

survived = []



for name in test_data['Name']:

    survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))





# Ground Label Evaluation

y_true = np.array(survived)
import xgboost as xgb







gbm = xgb.XGBClassifier(n_estimators= 100, 

                        max_depth = 4,

                        gamma = 0.9,

                        nthread = -1,

                        scale_pos_weight=1,

                       random_state = 3101)





gbm.fit(x_train, y_train)

xgb_predictions = gbm.predict(x_test)



score_gbm = gbm.score(x_train, y_train)

print(f'Random Forest Classifier score (Train Accuracy): {score_gbm}')



test_acc_gbm = accuracy_score(y_true, xgb_predictions)

print(f'Random Forest Classifier score (Test Accuracy): {test_acc_gbm}')





# Making Submission DataFrame

xgb_submission = pd.DataFrame({ 'PassengerId': PassengerId,

                                   'Survived': xgb_predictions })



# Writing to csv file

xgb_submission.head()

xgb_submission.to_csv('xgb_submission.csv', index = False)
xgb.plot_importance(gbm)
xgb.to_graphviz(gbm, num_trees=2)
#random forest classifier

from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators = 4, 

                            max_features  = 5,

                            random_state = 216)  

rfc.fit(x_train, y_train)

score_rfc = rfc.score(x_train, y_train)

out_rfc = rfc.predict(x_test)

print(f'Random Forest Classifier score (Train Accuracy): {score_rfc}')



test_acc_rfc = accuracy_score(y_true, out_rfc)

print(f'Random Forest Classifier score (Test Accuracy): {test_acc_rfc}')







# Making Submission DataFrame

rfc_submission = pd.DataFrame({ 'PassengerId': PassengerId,

                                   'Survived': out_rfc })



# Writing to csv file

rfc_submission.head()

rfc_submission.to_csv('rfc_submission.csv', index = False)
from sklearn.neighbors import KNeighborsClassifier



#knn classifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

score_knn = knn.score(x_train, y_train)

out_knn = knn.predict(x_test)

 



print(f'K- Nearest Neighbour ClassifierClassifier score (Train Accuracy): {score_knn}')

test_acc_knn = accuracy_score(y_true, out_knn)

print(f'K- Nearest Neighbour Classifier Classifier score (Test Accuracy): {test_acc_knn}')
from sklearn.svm import SVC



#SVM

# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

svc = SVC(C = 5, kernel = 'linear', random_state = 8)

svc.fit(x_train, y_train)

score_svc = svc.score(x_train, y_train)

out_svc = svc.predict(x_test)    

print(f'Support Vector Machine Classifier score (Train Accuracy): {score_svc}')

test_acc_svc = accuracy_score(y_true, out_svc)

print(f'K- Support Vector Machine  Classifier Classifier score (Test Accuracy): {test_acc_svc}')
classifier = ['XGBoost', 	'RandomForest', 'KNN', 'SVC']

train_acc  = [score_gbm,	score_rfc,	score_knn, score_svc]

test_acc   = [test_acc_gbm, 	test_acc_rfc, 	test_acc_knn, test_acc_svc]



score_df = pd.DataFrame({'classifier': classifier, 'train_acc': train_acc, 'test_acc': test_acc})

score_df