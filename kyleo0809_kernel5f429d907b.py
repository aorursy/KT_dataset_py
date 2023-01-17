import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set(style = 'whitegrid', font_scale= 1.5)
train = pd.read_csv('titanic/train.csv')

test_titanic = pd.read_csv('titanic/test.csv')
train.head()
test_titanic.head()
train.shape, test_titanic.shape
train.info()
train.isna().sum() 
survive = train[train['Survived'] == 1]

dead = train[train['Survived'] == 0]



print(f'Survived: {len(survive)} ({len(survive)/len(train) * 100:.2f}%)')

print(f'Deceased: {len(dead)} ({len(dead)/len(train) * 100:.2f}%)')

print(f'Total Passengers: {len(train)}')
train['Survived'].value_counts(normalize=True).plot(kind = 'bar', color = ['darkblue', 'gold'], 

                                                    title = 'Fatality Rate')

plt.xticks(rotation = 0);
pd.crosstab(train['Sex'], train['Survived'])
sns.countplot(x = 'Sex', data = train);
sns.barplot(x = 'Sex', y = 'Survived', data = train);
# Age Distribution of those who survived and those who died

g = sns.FacetGrid(train, col = 'Sex', hue = 'Survived',

                  height = 5, aspect = 2)

g.map(plt.hist, 'Age', bins = 25, alpha = 0.6)

g.add_legend();
train['Pclass'].value_counts()
pd.crosstab(train['Pclass'], train['Survived'])
train[['Pclass', 'Survived']].groupby(['Pclass']).mean()
sns.barplot(x = 'Pclass', y = 'Survived', data = train);
g = sns.catplot(x = "Pclass", y = "Survived", hue = "Sex",

                   data = train, kind = "bar",

                   height = 6, aspect = 2, palette = "deep",

                   legend = True)

g.despine(left=True)

#plt.legend(loc='upper right')

g.set_ylabels("Survival Probability");
sns.barplot(x = 'Embarked', y = 'Survived', data = train);
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()
pd.crosstab(train['Parch'], train['Survived'])
sns.barplot(x = 'Parch', y = 'Survived', data = train, ci = None);
pd.crosstab(train['SibSp'], train['Survived'])
sns.barplot(x = 'SibSp', y = 'Survived', data = train, ci = None);
# Finding a correlation

corr_matrix = train.drop('PassengerId', axis = 1).corr()

fig, ax = plt.subplots(figsize = (10,6))

ax = sns.heatmap(corr_matrix, 

                 annot = True,

                 lw = 0.5,

                 fmt = '.2f',

                 cmap = 'brg')
train.isna().sum()
test_titanic.isna().sum()
train['ageRange'] = pd.cut(train['Age'], 5)

train[['ageRange', 'Survived']].groupby(['ageRange'], as_index = False).mean()
train['CabinLetter'] = train['Cabin'].str[0]

train[['CabinLetter', 'Survived']].groupby(['CabinLetter']).mean().sort_values(by = 'Survived', ascending = False)
train['familySize'] = np.where((train['SibSp'] + train['Parch'] != 0), train['SibSp'] + train['Parch'], 0)

train[['familySize', 'Survived']].groupby(['familySize'], as_index = False).mean()
train = pd.read_csv('titanic/train.csv')

test_titanic = pd.read_csv('titanic/test.csv')
# Function that will change both the training set and test set

def change_test(model):

    # Creating a column that combines SibSp and Parch column into one column

    model['familySize'] = np.where((model['SibSp'] + model['Parch'] != 0), model['SibSp'] + model['Parch'], 0)

        

    # Fills in NaN data in the model with pandas method

    model['Age'].fillna(model['Age'].median(), inplace = True)

    model['Embarked'].fillna('S', inplace = True)

    model['Fare'].fillna(model['Fare'].median(), inplace = True)

        

    # Turns categorical dtypes into numerical

    model['Sex'] = np.where(model['Sex'] == 'male', 0, 1)

    model['Embarked'] = np.where(model['Embarked'] == 'C', 3,

                                     np.where(model['Embarked'] == 'Q', 2, 1))

        

    # Getting the first letter of cabin category and making it into a numerical column

    model['CabinLetter'] = model['Cabin'].str[0]

    cabin_map = {'A': 1, 'G': 2, 'C': 3, 'F': 4, 'B': 5, 'E': 6, 'D': 7, 'T': 0}

    model['CabinLetter'] = model['CabinLetter'].map(cabin_map)

    model['CabinLetter'].fillna(8, inplace = True)

    

    # Extract abbreviations from name and change it to numerical

    model['Title'] = model['Name'].str.extract('([a-zA-Z]+)\.')

    model['Title'] = model['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 

                                                 'Major', 'Mlle', 'Mme', 'Ms', 'Rev', 'Sir'], 'Other')

    map_title = {'Mr': 1, 'Master': 2, 'Miss': 3, 'Mrs': 4, 'Other': 5}

    model['Title'] = model['Title'].map(map_title)

    model['Title'].fillna(0, inplace = True)

    

    # Age determination

    # Putting age into a numerical category called ageRange

    model.loc[model['Age'] <= 16, 'ageRange'] = 1

    model.loc[(model['Age'] > 16) & (model['Age'] <= 32), 'ageRange'] = 2

    model.loc[(model['Age'] > 32) & (model['Age'] <= 48), 'ageRange'] = 3

    model.loc[(model['Age'] > 48) & (model['Age'] <= 64), 'ageRange'] = 4

    model.loc[model['Age'] > 64, 'ageRange'] = 5

    model.loc[model['Age'].isna(), 'ageRange'] = 6

        

        

    # Returns the new dataframe

    return model
new_train = change_test(train)

new_test = change_test(test_titanic)
new_train.drop(['Name','Ticket','Cabin'], axis = 1, inplace = True)

new_test.drop(['Name','Ticket','Cabin'], axis = 1, inplace = True)
new_train.head()
new_test.head()
new_matrix = new_train.drop('PassengerId', axis = 1).corr()

fig, ax = plt.subplots(figsize = (14,9))

ax = sns.heatmap(new_matrix, 

                 annot = True,

                 lw = 0.7,

                 fmt = '.2f',

                 cmap = 'brg')
from sklearn.model_selection import train_test_split



# Splitting up the training data

X = new_train.drop(['PassengerId', 'Survived'], axis = 1)

y = new_train['Survived']

new_test = new_test.drop('PassengerId', axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
new_test.shape
# Models to try

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC



from sklearn.model_selection import cross_val_score
# Creating a dictionary of model instances for classification

classification_models = {

    'KNeighborsClassifier': KNeighborsClassifier(),

    'GradientBoostingClassifier': GradientBoostingClassifier(),

    'RandomForestClassifier': RandomForestClassifier(),

    'DecisionTreeClassifier': DecisionTreeClassifier(),

    'LogisticRegression': LogisticRegression(max_iter = 1000),

    'LinearSVC': LinearSVC(max_iter=10000)

}



# An empyy dictionary of the regression results

classification_results = {}
np.random.seed(21)

for model_name, model in classification_models.items():

    

    # Putting the score of the model into the empty classification dictionary

    classification_results[model_name] = np.mean(cross_val_score(model, X, y))
classification_results
gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

gbc.score(X_test, y_test)
np.mean(cross_val_score(gbc, X, y))
gbc.fit(X, y)

y_preds = gbc.predict(new_test)

y_preds
# Visualize the importance feaetures

feature_dict = dict(zip(X.columns, gbc.feature_importances_))

importance_df = pd.DataFrame(feature_dict, index = [0])

importance_df.T.plot.bar(title = 'Feature Importance', legend = False);
classification_results
#param = ['SibSp', 'Parch', 'ageRange', 'Embarked']

param = ['SibSp', 'Parch', 'Age', 'Embarked']

newX = X.drop(param, axis = 1)

newer_test = new_test.drop(param, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size = 0.2, random_state = 42)
classification_models = {

    'KNeighborsClassifier': KNeighborsClassifier(),

    'GradientBoostingClassifier': GradientBoostingClassifier(),

    'RandomForestClassifier': RandomForestClassifier(),

    'DecisionTreeClassifier': DecisionTreeClassifier(),

    'LogisticRegression': LogisticRegression(max_iter = 1000)

}

# An empyy dictionary of the regression results

classification_results = {}
np.random.seed(42)

for model_name, model in classification_models.items():

      

    # Putting the score of the model into the empty classification dictionary

    classification_results[model_name] = np.mean(cross_val_score(model, newX, y))
classification_results
gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

gbc.score(X_test, y_test)
np.mean(cross_val_score(gbc, newX, y))
gbc.fit(newX,y)

y_preds = gbc.predict(newer_test)

y_preds
feature_dict = dict(zip(newX.columns, (gbc.feature_importances_)))

importance_df = pd.DataFrame(feature_dict, index = [0])

importance_df.T.plot.bar(title = 'Feature Importance', legend = False);
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
gb_rs = {'max_depth': [5,6,7,8],

        'min_samples_split': np.arange(110, 121),

        'min_samples_leaf': np.arange(5, 16),

        'n_estimators': [69,100,150]

        }



gb_grid = {'max_depth': [4,5,6],

        'min_samples_split': [110, 112, 120],

        'min_samples_leaf': [5,6,7],

        }



def random_search(random_dict, model):

    searchCV = RandomizedSearchCV(estimator = model,

                                 param_distributions = random_dict,

                                 n_iter = 15,

                                 cv = 5,

                                 scoring = 'accuracy')

    return searchCV



def grid_search(random_dict, model):

    searchCV = GridSearchCV(estimator = model,

                           param_grid = random_dict,

                           cv = 8,

                           verbose = 0)

    return searchCV
gb_random_tune = random_search(gb_rs, GradientBoostingClassifier())

gb_random_tune.fit(X_train, y_train)

gb_random_tune.score(X_test, y_test)
gb_grid_tune = grid_search(gb_grid, GradientBoostingClassifier())

gb_grid_tune.fit(X_train, y_train)

gb_grid_tune.score(X_test, y_test)
gb_random_tune.best_params_
gb_grid_tune.best_params_
# Utilizing a combination of GridSearch and slight manual hyperparameter tuning

#gbc = GradientBoostingClassifier(max_depth = 6, min_samples_leaf = 6, min_samples_split = 118, n_estimators = 69)

gbc = GradientBoostingClassifier(min_samples_split = 112, min_samples_leaf = 6, max_depth = 5)

np.mean(cross_val_score(gbc, newX, y))
gbc.fit(newX, y)

y_preds = gbc.predict(newer_test)

y_preds
submit = pd.DataFrame({

    'PassengerId': test_titanic['PassengerId'],

    'Survived': y_preds

})



submit.to_csv('Titanic-challenge.csv', index = False)