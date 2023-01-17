# might come in handy when dealing with arrays and general math

import numpy as np

# regular pandas also for data manipulation

import pandas as pd



import matplotlib.pyplot as plt

# import matplotlib as plt

%matplotlib inline

# As in microcourses there was used seaborn, then I will use this as well.

import seaborn as sns



#I start with RandomForest as this is used in ML intro course in Kaggle. And since Titanic is classification problem, then I will try to 

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder

# I add some other things later



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score



from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, learning_curve
#train_file_path = './data/titanic_train.csv'

#test_file_path = './data/titanic_test.csv'



train_file_path = "../input/titanic/train.csv"

test_file_path = "../input/titanic/test.csv"



train_data = pd.read_csv(train_file_path)

test_data = pd.read_csv(test_file_path)
# train_data.describe(include='all')

train_data.info()
test_data.info()
train_data.describe(include='all')
test_data.describe(include='all')
train_data.groupby(by=['Pclass']).count()
sns.countplot(x=train_data['Pclass'], hue=train_data['Survived']);

# this has a little different version too, just for learning again.

# sns.countplot(x = "Pclass", hue = "Survived", data = train_data, palette = 'RdPu');
fig, ax = plt.subplots(figsize = (10,6))

ax = sns.countplot(x = 'Survived', hue = 'Pclass', data = train_data, palette = 'YlOrRd')

ax.set_xlabel('Survived')

ax.set_title('Survival Rate for Passenger Classes', fontsize = 14, fontweight='bold');



# This graph learned here: https://www.kaggle.com/alenavorushilova/data-analysis-and-data-visualization-seaborn
ax = sns.catplot(x="Pclass", hue="Sex", col="Survived",

                data=train_data, kind="count",

                height=4, aspect=.7, palette = 'OrRd');

# This also learned from here: https://www.kaggle.com/alenavorushilova/data-analysis-and-data-visualization-seaborn
sns.barplot(x="Pclass", y="Survived", data= train_data, palette = 'BuGn');
perc = train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

perc*100

# This line is just to get the percentages of survival
sns.factorplot(x='Pclass', y='Survived', hue = 'Sex', data = train_data, palette = 'PRGn');

# And one last graph from this kernel again: https://www.kaggle.com/alenavorushilova/data-analysis-and-data-visualization-seaborn
def extract_title(name):

    """This just gets the title from name column"""

    for string in name.split():

        if '.' in string:

            return string[:-1]

    # return name.split(',')[1].split('.')[0] ## My own code, still learning python too. This code taken from other kernel



train_data['Title'] = train_data['Name'].apply(lambda n: extract_title(n))

test_data['Title'] = test_data['Name'].apply(lambda n: extract_title(n))

print(test_data['Title'].value_counts(),'\n\n',train_data['Title'].value_counts())

# print(train_data['Title'].value_counts())
# Lets try to organize those titles that are less than 10 and others also

for dataframe in [train_data, test_data]:

    dataframe['Title'] = dataframe['Title'].replace('Mlle', 'Miss')

    dataframe['Title'] = dataframe['Title'].replace('Ms', 'Miss')

    dataframe['Title'] = dataframe['Title'].replace('Mme', 'Mrs')



    dataframe['Title'] = dataframe['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 

                                             'Major', 'Rev', 'Sir', 'Dona', 'Countess', 'Jonkheer'], 'Other')



    # Lets drop names too

    dataframe.drop('Name', axis=1, inplace=True)

    
plt.figure(figsize=[10,5])

sns.countplot(x='Sex', hue='Survived', data=train_data)

plt.xticks(rotation=20);
sns.barplot(x='Sex', y='Survived', data=train_data, palette=('RdPu'));
print('% of survived females:', train_data['Survived'][train_data['Sex'] == 'female'].value_counts(normalize = True)[1]*100)

print('% of survived males:', train_data['Survived'][train_data['Sex'] == 'male'].value_counts(normalize = True)[1]*100)

# train_data['Survived'][train_data['Sex'] == 'female'].value_counts(normalize = True)
# Another way to do this, using groupby:

train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']

test_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']

# Now that we have family size, we can create feature IsAlone.

train_data['IsAlone'] = train_data['FamilySize'].apply(lambda fs: 1 if fs == 0 else 0)

test_data['IsAlone'] = test_data['FamilySize'].apply(lambda fs: 1 if fs == 0 else 0)
sns.countplot(x='IsAlone', hue='Survived', data=train_data);
sns.countplot(x='FamilySize', hue='Survived', data=train_data);
train_data[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data.drop(['Parch', 'SibSp'], axis=1, inplace=True)

test_data.drop(['Parch', 'SibSp'], axis=1, inplace=True)
train_data[['Ticket', 'PassengerId']].groupby('Ticket', as_index=False).count().sort_values('PassengerId', ascending=False)
train_data['TicketGroupSize'] = train_data.groupby(['Ticket'])['PassengerId'].transform('count') 

test_data['TicketGroupSize'] = test_data.groupby(['Ticket'])['PassengerId'].transform('count') 
sns.countplot(x='TicketGroupSize', hue='Survived', data=train_data);
# Lets drop ticket for now

train_data.drop('Ticket', axis=1, inplace=True)

test_data.drop('Ticket', axis=1, inplace=True)
fare_median = test_data['Fare'].median()

test_data['Fare'] = test_data['Fare'].fillna(fare_median)
train_data['HasCabin'] = train_data['Cabin'].notnull().astype('int')

test_data['HasCabin'] = test_data['Cabin'].notnull().astype('int')
sns.countplot(x='HasCabin', hue='Survived', data=train_data);
train_data.drop('Cabin', axis=1, inplace=True)

test_data.drop('Cabin', axis=1, inplace=True)
train_data['Embarked'].value_counts()
train_data['Embarked'] = train_data['Embarked'].fillna('S')

sns.countplot(x='Embarked', hue='Survived', data=train_data);
# I will concatenate datasets to get better age cover

full_data = pd.concat([train_data.drop('Survived', axis=1), test_data], ignore_index=True)
full_data[['Age', 'Pclass', 'Sex', 'Embarked']].groupby(['Pclass', 'Sex', 'Embarked'])['Age'].mean()
train_data[['Age', 'Pclass', 'Sex', 'Embarked']].groupby(['Pclass', 'Sex', 'Embarked'])['Age'].mean()
# To see how empty values count differs on training and full data set, use these lines

# full_data[full_data['Age'].isnull()][['Pclass', 'Sex', 'Embarked', 'PassengerId']].groupby(['Pclass', 'Sex', 'Embarked'])['PassengerId'].count()

# train_data[train_data['Age'].isnull()][['Pclass', 'Sex', 'Embarked', 'PassengerId']].groupby(['Pclass', 'Sex', 'Embarked'])['PassengerId'].count()



# Same to see how filled values count differ on training and full set

# full_data[full_data['Age'].notnull()][['Pclass', 'Sex', 'Embarked', 'PassengerId']].groupby(['Pclass', 'Sex', 'Embarked'])['PassengerId'].count()

# train_data[train_data['Age'].notnull()][['Pclass', 'Sex', 'Embarked', 'PassengerId']].groupby(['Pclass', 'Sex', 'Embarked'])['PassengerId'].count()
fig, axes = plt.subplots(1, 3, figsize=(20,5))



sns.distplot(a=train_data.Age.dropna(), kde=False, ax=axes[0], bins=30);

axes[0].set_title('Training data distribution')



sns.distplot(a=test_data.Age.dropna(), kde=False, ax=axes[1], bins=30);

axes[1].set_title('Test data distribution')



axes[2].set_title('Full data distribution')

sns.distplot(a=full_data.Age.dropna(), kde=False, ax=axes[2], bins=30);
def get_age(element):

    age = element[0]

    pclass = element[1]

    sex = element[2]

    embarked = element[3]

    if (pd.isnull(age)):

        temp_data = full_data[(full_data['Pclass'] == pclass) & (full_data['Sex'] == sex) & (full_data['Embarked'] == embarked)]

        mean_age = temp_data['Age'].mean()

        # temp_data[['Age', 'Pclass', 'Sex', 'Embarked']].groupby(['Pclass', 'Sex', 'Embarked'])['Age'].mean()

        # print(pclass, sex, embarked, mean_age)

        return mean_age

    return age

train_data['Age'] = train_data[['Age', 'Pclass', 'Sex', 'Embarked']].apply(get_age, axis=1)

test_data['Age'] = test_data[['Age', 'Pclass', 'Sex', 'Embarked']].apply(get_age, axis=1)
# Last view over distribution

full_data = pd.concat([train_data.drop('Survived', axis=1), test_data], ignore_index=True)

sns.distplot(a=full_data.Age, kde=False, bins=40);
# Lets one hot encode on my own these sex values...



sex = pd.get_dummies(train_data['Sex'], prefix='Sex', drop_first=True)

embarked = pd.get_dummies(train_data['Embarked'], prefix='Embarked', drop_first=True)

# pclass = pd.get_dummies(train_data['Pclass'], prefix='Pclass', drop_first=True)

title = pd.get_dummies(train_data['Title'], prefix='Title', drop_first=True)



train_data.drop(['Sex', 'Title', 'Embarked'], axis=1, inplace=True)



train_data = pd.concat([train_data, sex, embarked, title], axis=1)
train_data.head()
# Lets do same for test data



sex = pd.get_dummies(test_data['Sex'], prefix='Sex', drop_first=True)

embarked = pd.get_dummies(test_data['Embarked'], prefix='Embarked', drop_first=True)

# pclass = pd.get_dummies(test_data['Pclass'], prefix='Pclass', drop_first=True)

title = pd.get_dummies(test_data['Title'], prefix='Title', drop_first=True)



test_data.drop(['Sex', 'Title', 'Embarked'], axis=1, inplace=True)



test_data = pd.concat([test_data, sex, embarked, title], axis=1)
test_data.head()
# Prepare data for modelling

y_train = train_data['Survived']

X_train = train_data.drop(['Survived', 'Fare'], axis=1)

test_data = test_data.drop(['Fare'], axis=1)

# X_train, X_test, Y_train, Y_test = train_test_split(train_data_without_y, y, test_size=0.3, random_state=13)
kfold = StratifiedKFold(n_splits=5)



random_state = 13
random_forest_classifier = RandomForestClassifier(random_state=random_state)



cv_result = cross_val_score(random_forest_classifier, X_train, y_train, cv=kfold, scoring='accuracy')

print("CV result mean: ", cv_result.mean(), "CV result std: ", cv_result.std())

cv_result
xgbClassifier = XGBClassifier(random_state=random_state)



cv_result = cross_val_score(xgbClassifier, X_train, y_train, cv=kfold, scoring='accuracy')



print("CV result mean: ", cv_result.mean(), "CV result std: ", cv_result.std())

cv_result
rf_param_grid = {'max_depth'        :[1, 2, 3, 4, 5, 6],

                 'max_features'     :[2, 4, 6, 8, 10],

                 'min_samples_split':[2, 4, 6, 8, 10],

                 'bootstrap'        :[False, True],

                 'n_estimators'     :[50, 100, 200, 500],

                 'criterion'        :['gini']}



grid_search_random_forest_classifier = GridSearchCV(random_forest_classifier, param_grid=rf_param_grid, cv=kfold, scoring='accuracy', n_jobs=4, verbose=1)

grid_search_random_forest_classifier.fit(X_train, y_train)



random_forest_best = grid_search_random_forest_classifier.best_estimator_



grid_search_random_forest_classifier.best_score_
grid_search_random_forest_classifier.best_params_
xgb_param_grid={'colsample_bylevel':[0.1, 0.9, 1],

                'colsample_bytree' :[0.2, 0.8, 1],

                'gamma'            :[0.99, 9, 99],

                'max_depth'        :[2, 4, 6, 8, 10],

                'min_child_weight' :[1, 2, 4, 6, 8, 10],

                'n_estimators'     :[10, 20, 50, 70, 100, 200, 500, 1000]}

                # 'nthread'          :[1,2,3,4],

                # 'silent'           :[True]}



grid_search_xgboost_classifier = GridSearchCV(xgbClassifier, param_grid=xgb_param_grid, cv=kfold, scoring='accuracy', n_jobs=4, verbose=1)

grid_search_xgboost_classifier.fit(X_train, y_train)



xgb_best = grid_search_xgboost_classifier.best_estimator_



grid_search_xgboost_classifier.best_score_
grid_search_xgboost_classifier.best_params_
x_tr, x_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=random_state)
random_forest_classifier = RandomForestClassifier(n_estimators=200, max_depth=3, max_features=8, min_samples_split=2, bootstrap=True, random_state=random_state)

random_forest_classifier.fit(x_tr, y_tr)

predicted_test = random_forest_classifier.predict(x_val)



accuracy_random_forest = accuracy_score(y_val, predicted_test)

f1_random_forest = f1_score(y_val, predicted_test)

print("Random forest scores: ", accuracy_random_forest, f1_random_forest)
xgbClassifier = XGBClassifier(max_depth=4,n_estimators=20,gamma=9,colsample_bylevel=0.9,random_state=random_state)

#{'colsample_bylevel': 0.9, 'colsample_bytree': 1, 'gamma': 9, 'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 20}

xgbClassifier.fit(x_tr, y_tr)

predicted_test = xgbClassifier.predict(x_val)



accuracy_xgb = accuracy_score(y_val, predicted_test)

f1_xgb = f1_score(y_val, predicted_test)

print("XGBooster scores: ", accuracy_xgb, f1_xgb)
# random_forest_classifier.fit(train_data_without_y, y)

xgbClassifier.fit(X_train, y_train)

# test_X = test_data.drop(labels=['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=False)

# test_X['Fare'].fillna(method='ffill',inplace=True )



# submission_prediction = random_forest_classifier.predict(test_data)

submission_prediction = xgbClassifier.predict(test_data)



submission = pd.DataFrame({

    'PassengerId': test_data['PassengerId'],

    'Survived': submission_prediction

})

# submission_file_path = './data/titanic_submission.csv'

submission_file_path = './titanic_submission.csv'

submission.to_csv(submission_file_path, index=False)