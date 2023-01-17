import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit, GridSearchCV, StratifiedKFold 
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split

from hyperopt import hp
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import xgboost as xgb

from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
def transform_dataset(dataset):
    # Age
    # Filling missing values based on sample distribution from Pclass
    age_mean = dataset['Age'].groupby(dataset['Pclass']).mean().astype(int)
    age_std =  dataset['Age'].groupby(dataset['Pclass']).std().astype(int)
    age_null_count = dataset['Age'].isnull().groupby(dataset['Pclass']).sum().astype(int)
    pclass_values = set(dataset['Pclass'].values)
    
    for m, s, c, p in zip(age_mean, age_std, age_null_count, pclass_values):
        age_null_random_list = np.random.randint(m - s, m + s, size=c)
        dataset['Age'][(dataset['Pclass']==p) & (dataset['Age'].isnull())] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)
    
    # Pclass
    # Creating dummy variables
    dataset = pd.concat([dataset.drop('Pclass', axis=1), pd.get_dummies(dataset['Pclass'], prefix='Pclass_')], axis=1)
    
    # Name
    # Extracting title from Name column
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset = dataset.drop('Name', axis=1)
    # Regrouping categories
    rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 
                  'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms', 'Miss'], 'Miss')
    dataset['Title'] = dataset['Title'].replace(['Mme', 'Mrs'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(rare_title, 'Rare')
    # Creating dummy variables
    dataset = pd.concat([dataset.drop('Title', axis=1), pd.get_dummies(dataset['Title'], prefix='Title_')], axis=1)
    
    # Sex
    # Creating dummy variables
    dataset = pd.concat([dataset.drop('Sex', axis=1), pd.get_dummies(dataset['Sex'])], axis=1)
    
    # SibSp & Parch
    dataset['Is_Alone'] = 0
    dataset['Is_Alone'][(dataset['SibSp'] == 0) & (dataset['Parch'] == 0)] = 1
    dataset = dataset.drop(['SibSp', 'Parch'], axis=1)
    
    # Ticket
    dataset['Has_Ticket'] = 1
    dataset['Has_Ticket'][dataset['Ticket'].isnull()] = 0
    dataset = dataset.drop('Ticket', axis=1)
    
    # Fare
    # Filling missing values with median
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
    
    # Cabin
    dataset['Has_Cabin'] = 1
    dataset['Has_Cabin'][dataset['Cabin'].isnull()] = 0
    dataset = dataset.drop('Cabin', axis=1)
    
    # Embarked
    # Filling missing values with mode
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode())
    # Creating dummy variables
    dataset = pd.concat([dataset.drop('Embarked', axis=1), pd.get_dummies(dataset['Embarked'], prefix='Embarked_')], axis=1)
    
    return dataset
train_transformed = transform_dataset(train)
train_transformed.head()
test_transformed = transform_dataset(test)
test_transformed.head()
# Rearranging columns, scaling
target = train_transformed['Survived'].values
train_transformed = train_transformed.drop(['PassengerId', 'Survived'], axis=1)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_transformed)

PassengerId = np.array(test_transformed['PassengerId']).astype(int)
test_transformed = test_transformed.drop(['PassengerId'], axis=1)
test_scaled = scaler.transform(test_transformed)
# Feature Selection

# hyperoptimalization for RandomForestClassifier
h_param = [{'max_depth': [2, 5, 8, 10],
            'n_estimators': [50, 100, 150, 200]}]

clf = GridSearchCV(RandomForestClassifier(), h_param, cv=5, scoring='accuracy')
clf.fit(train_scaled, target)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()

estimator = RandomForestClassifier(**clf.best_params_)
rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(3), scoring='accuracy')
rfecv.fit(train_scaled, target)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

new_features = rfecv.transform(train_scaled)
new_test_features = rfecv.transform(test_scaled)

print(new_features.shape, new_test_features.shape)
model = xgb.XGBClassifier(
    max_depth = 12,
    learning_rate = 0.2,
    n_estimators = 50,
    silent = 0,
    objective = 'reg:linear',
    nthread = -1,
    # gamma = 5290.,
    # min_child_weight = 4.2922,
    subsample = 0.7,
    colsample_bytree = 0.6,
    seed = 0
)
nfolds = 5
folds = KFold(len(target), n_folds=nfolds, shuffle = True, random_state = 0)

for num_iter, (train_index, test_index) in enumerate(folds):
    X_train, y_train = new_features[train_index], target[train_index]
    X_test, y_test   = new_features[test_index], target[test_index]
    
    model.fit(X_train, y_train, eval_metric='error',
              eval_set=[(new_features[train_index], target[train_index]), 
                         (new_features[test_index], target[test_index])],
              verbose=True)
    
    y_pred = model.predict(X_test)

    score = accuracy_score(y_test, y_pred)
    print("Fold{0}, score={1}".format(num_iter+1, score))
    
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.show()
# prediction
my_prediction = model.predict(new_test_features)
    
# saving solution file
PassengerId = np.array(test['PassengerId']).astype(int)
my_solution = pd.DataFrame({'PassengerId':PassengerId, "Survived":my_prediction})
my_solution.to_csv('XGBoost.csv', index = False)