# Data analysis
import numpy as np
import pandas as pd
# Load test and train data
train = pd.read_csv('/kaggle/input/titanic/train.csv')  # train set
test  = pd.read_csv('/kaggle/input/titanic/test.csv')   # test  set
combine  = [train, test]
# Clean cabin data
for dataset in combine:
    dataset['Deck'] = dataset['Cabin'].str[0]
    dataset.loc[dataset['Cabin'].isnull()==False, 'Deck'] = 'known'
    dataset.loc[dataset['Cabin'].isnull(), 'Deck'] = 'unknown'
    #dataset.loc[dataset['Cabin'] == 'T', 'Deck'] = 'U'
# TODO: figure out hoe to leverage these two variables
# Remove ticket and cabin
#train.drop(['Ticket'], axis=1, inplace=True)
#test.drop(['Ticket'], axis=1, inplace=True)


# Add title
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    #Group titles
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Dona'],'Royalty')
    dataset['Title'] = dataset['Title'].replace(['Mme'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Major','Rev'], 'Officer')
    dataset['Title'] = dataset['Title'].replace(['Jonkheer', 'Don','Sir'], 'Royalty')
    dataset.loc[(dataset['Sex'] == 'male')   & (dataset['Title'] == 'Dr'),'Title'] = 'Mr'
    dataset.loc[(dataset['Sex'] == 'female') & (dataset['Title'] == 'Dr'),'Title'] = 'Mrs'


# Fill missing age values and group in bands
labels = ['baby', 'child', 'teen', 'young_adult', 'adult', 'senior']
bins = [0, 3, 10, 20, 35, 60, 80]
for dataset in combine:
    dataset['Age'].loc[dataset['Age'].isnull()] = dataset.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    dataset['Age_Cat'] = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)
# Create a column with family size
for dataset in combine:
    dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch']
    dataset['isAlone'] = 1
    dataset.loc[(dataset['Family_Size'] > 2), 'isAlone'] = 0
# Create group survival rate dictionary
for dataset in combine:
    dataset['Family_Group'] = dataset['Name'].str.extract('^(.+?),', expand=False) + '_' + dataset["Ticket"].str[:-1]
    dataset.loc[dataset['Family_Size'] < 2, 'Family_Group'] = 'small_family'

group_rates = train.groupby('Family_Group')['Survived'].mean()
group_rates_women_children = train.loc[(train['Sex'] == 'female') | (train['Age']) <= 10].groupby('Family_Group')['Survived'].mean()

# Add group survival rate to dataset
for dataset in combine:
    dataset['Group_Rate'] = dataset['Family_Group'].map(group_rates)
    dataset.loc[dataset['Group_Rate'].isnull(), 'Group_Rate'] = dataset.loc[dataset['Family_Group'] != 'small_family', 'Group_Rate'].mean()
    dataset['Group_Rate_Band'] = pd.cut(dataset['Group_Rate'], 3)

    dataset['Group_Rate_WC'] = dataset['Family_Group'].map(group_rates_women_children)
    dataset.loc[dataset['Group_Rate_WC'].isnull(), 'Group_Rate_WC'] = dataset.loc[dataset['Family_Group'] != 'small_family', 'Group_Rate_WC'].mean()
    dataset['Group_Rate_WC_Band'] = pd.cut(dataset['Group_Rate_WC'], 3)
# Fill values for embarked
for dataset in combine:
    dataset['Embarked'].loc[dataset['Embarked'].isnull()] = "S"
# Fill values for embarked on test set and add fare bands
test['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)
for dataset in combine:
    dataset['Fare_Band'] = pd.qcut(dataset['Fare'], 3)
# Select features and turn to numerical values
feature_list = ['Sex', 'Title', 'Pclass', 'Age_Cat', 'Deck', 'Fare_Band', 'Embarked', 'isAlone', 'Group_Rate_Band', 'Group_Rate_WC_Band']
X_train = pd.get_dummies(train[feature_list])
y_train = train['Survived']
X_test = pd.get_dummies(test[feature_list])

test_col = X_test.columns
train_col = X_train.columns
# Random Forrest classifier model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, min_samples_split=5, max_depth=15, random_state=1)
score = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
print('scores: ',score)
print('min: ', score.min())
print('mean: ',score.mean())
print('median: ',np.median(score))
# Grid search for parameter optimization
# from sklearn.model_selection import GridSearchCV
# n_estimators = [100, 300, 500, 800, 1200]
# max_depth = [5, 8, 15, 25, 30]
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10] 

# hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
#               min_samples_split = min_samples_split, 
#              min_samples_leaf = min_samples_leaf)

# gridF = GridSearchCV(model, hyperF, cv = 3, verbose = 1, 
#                       n_jobs = -1)
# bestF = gridF.fit(X_train, y_train)
# bestF.best_estimator_
model.fit(X_train, y_train)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)