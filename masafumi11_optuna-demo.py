# Install optuna

!pip install optuna
# Load libraries

import pandas as pd

import numpy as np

import re

import sklearn



import warnings

warnings.filterwarnings('ignore')



from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.svm import SVC



from sklearn.model_selection import cross_val_score



import optuna
# load in the train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# store our passenger ID for easy access

PassengerId = test['PassengerId']



train.head(3)
full_data = [train, test]



# Some features of my own that I have added in

# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)

train.head(3)
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_train = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

x_train = train.values # Creates an array of the train data

x_test = test.values # Creats an array of the test data
class ClassificationObjective(object):

    def __init__(self, x, y):

        self.x = x

        self.y = y



    def __call__(self, trial):

        # data used by train and evaluation

        x, y = self.x, self.y



        # hyperparameters to be tuned

        n_estimators = trial.suggest_int('n_estimators', 10, 100)

        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

        max_depth = trial.suggest_int('max_depth', 1, 6)

        min_samples_split = trial.suggest_int('min_samples_split', 2, 16)

        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])



        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,

                                    max_depth=max_depth, min_samples_split=min_samples_split,

                                    max_features=max_features)





        # evaluate by mean accuracy of 3-fold cv

        score = cross_val_score(clf, x, y, n_jobs=-1, cv=3, scoring="accuracy")

        score = score.mean()

        return score
# define objective. see above.

objective = ClassificationObjective(x_train, y_train)



# create study object

study = optuna.create_study(direction='maximize')

# optimize

study.optimize(objective, n_trials=100)

# get best hyperparameters

hyperparams = study.best_params
# first 10 iteration's summary

df = study.trials_dataframe()

df.head(10)
df.plot( y=[('params', 'max_depth'), ('params', 'min_samples_split'), ('params', 'n_estimators')], figsize=(16,4), alpha=0.5)
df.plot(x=('params', 'min_samples_split'), y= ('params', 'n_estimators'), kind='scatter')
# best params

hyperparams
# modify object class

class ClassificationObjective(object):

    def __init__(self, models, x, y):

        self.models = models

        self.x = x

        self.y = y



    def __call__(self, trial):

        models, x, y = self.models, self.x, self.y

        

        classifier_name = trial.suggest_categorical('classifier', models)



        if classifier_name == "RandomForest": # Random Forest

            n_estimators = trial.suggest_int('n_estimators', 10, 100)

            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

            max_depth = trial.suggest_int('max_depth', 1, 6)

            min_samples_split = trial.suggest_int('min_samples_split', 2, 16)

            max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])

            

            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,

                                         max_depth=max_depth, min_samples_split=min_samples_split,

                                        max_features=max_features)

        elif classifier_name == "SVM": # SVM

            C = trial.suggest_loguniform('C', 0.1, 10)

            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']) #  'poly' 遅いので除外

            gamma = trial.suggest_categorical('gamma', ["scale", "auto"])

            clf = SVC(C=C, kernel=kernel, gamma=gamma)

        elif classifier_name == "MLPClassifier": # MLPClassifier

            hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 10, 100)

            activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])

            solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])

            nn_learning_rate_init =  trial.suggest_loguniform('learning_rate_init', 10e-4, 10e-2)

            clf = MLPClassifier(hidden_layer_sizes= (hidden_layer_sizes,),

                                activation=activation, solver=solver, learning_rate_init=nn_learning_rate_init)

        elif classifier_name == "XGBClassifier": # XGBClassifier

            xgb_max_depth = trial.suggest_int('max_depth', 1, 6)

            xgb_n_estimators = trial.suggest_int('n_estimators', 10, 100)

            min_child_weight = trial.suggest_int('min_samples_split', 1, 20)

            colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.3, 1.0)

            subsample = trial.suggest_uniform('colsample_bytree', 0.3, 1.0)

            clf = XGBClassifier(xgb_max_depth=xgb_max_depth, n_estimators=xgb_n_estimators,

                                min_child_weight=min_child_weight, colsample_bytree=colsample_bytree,

                                subsample=subsample)



        # evaluate by mean accuracy of 3-fold cv

        score = cross_val_score(clf, x, y, n_jobs=-1, cv=3, scoring="accuracy")

        score = score.mean()

        return score
# optimize among 4 models

models = ["RandomForest", "SVM", "MLPClassifier", "XGBClassifier"]

objective = ClassificationObjective(models, x_train, y_train)

study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=100) # this n_trials may be too few.

best_params = study.best_params
# first 10 iteration's summary

df = study.trials_dataframe()

df.head(10)
# show best params

best_params
# show histgram

df[('params', 'classifier')].value_counts().plot(kind="bar")
# select classifier that scored the best evaluate accuracy.

classifier = best_params.pop('classifier')



if classifier == "RandomForest":

    BestClassifier = RandomForestClassifier

elif classifier == "SVM":

    BestClassifier = SVC

elif classifier_name == "MLPClassifier":

    BestClassifier = MLPClassifier

elif classifier_name == "XGBClassifier":

    BestClassifier = XGBClassifier



model = BestClassifier(**best_params)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

#Generate Submission File 

Submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

Submission.to_csv("Submission.csv", index=False)