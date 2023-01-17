import pandas as pd

import re

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
# Load in the train and test datasets

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



# Store our passenger ID for easy access

PassengerId = test['PassengerId']



train.head(3)
full_data = [train, test]



# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



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

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)
train.head(3)
test.head(3)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare',

                        u'Embarked', u'FamilySize', u'Title']],hue='Survived', 

                 diag_kind='hist', palette = 'seismic', size=1.2, plot_kws=dict(s=10))

g.set(xticklabels=[])
from sklearn.model_selection import (GridSearchCV,

                                     train_test_split,

                                     StratifiedKFold)



from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression



from xgboost import XGBClassifier



# параметры базовых алгоритмов

gbc_params = {'learning_rate': np.arange(0.1, 0.8, 0.5)} # GradientBoostingClassifier



rfc_params = {'n_estimators': range(10, 100, 10), # RandomForestClassifier

              'min_samples_leaf': range(1, 5)}



svc_params = {'kernel': ['linear', 'rbf'], # SVC

              'C': np.arange(0.1, 1, 0.2)}



lr_params = {'C': np.arange(0.5, 1, 0.1)}



skf = StratifiedKFold(n_splits=5, random_state=17)
from sklearn.metrics import accuracy_score



train_targets = train['Survived']

train_values = train.drop(columns='Survived')



x_train, x_test, y_train, y_test = train_test_split(train_values, 

                                                    train_targets,

                                                    test_size=0.3,

                                                    random_state=17)
gbc = GradientBoostingClassifier(random_state=17)

grid_gbc = GridSearchCV(gbc, param_grid=gbc_params, cv=skf)

model_gbc = grid_gbc.fit(x_train, y_train)

model_gbc.best_params_
rfc = RandomForestClassifier(random_state=17)

grid_rfc = GridSearchCV(rfc, param_grid=rfc_params, cv=skf)

model_rfc = grid_rfc.fit(x_train, y_train)

model_rfc.best_params_
svc = SVC(random_state=17)

grid_svc = GridSearchCV(svc, param_grid=svc_params, cv=skf)

model_svc = grid_svc.fit(x_train, y_train)

model_svc.best_params_
lr = LogisticRegression(random_state=17)

grid_lr = GridSearchCV(lr, param_grid=lr_params, cv=skf)

model_lr = grid_lr.fit(x_train, y_train)

model_lr.best_params_
gbc_base = GradientBoostingClassifier(random_state=17, learning_rate=0.1)

model_gbc_base = gbc_base.fit(x_train, y_train)

gbc_base_predict = model_gbc_base.predict(x_test)

gbc_accuracy =accuracy_score(gbc_base_predict, y_test)

print(f'GradientBoostingClassifier accuracy: {gbc_accuracy}')
rfc_base = RandomForestClassifier(random_state=17, min_samples_leaf=3, n_estimators=10)

model_rfc_base = rfc_base.fit(x_train, y_train)

rfc_base_predict = model_rfc_base.predict(x_test)

rfc_accuracy = accuracy_score(rfc_base_predict, y_test)

print(f'RandomForestClassifier accuracy: {rfc_accuracy}')
svc_base = SVC(random_state=17, C=0.9000000000000001, kernel='rbf')

model_svc_base = svc_base.fit(x_train, y_train)

svc_base_predict = model_svc_base.predict(x_test)

svc_accuracy = accuracy_score(svc_base_predict, y_test)

print(f'SVC accuracy: {svc_accuracy}')
lr_base = LogisticRegression(random_state=17, C=0.5)

model_lr_base = lr_base.fit(x_train, y_train)

lr_base_predict = model_lr_base.predict(x_test)

lr_accuracy = accuracy_score(lr_base_predict, y_test)

print(f'LogisticRegression accuracy: {lr_accuracy}')
xgb_params = {'n_estimators': range(10, 100, 5),

              'eta': np.arange(0.1, 1., .1),

              'min_child_weight': range(1, 10, 1),

              'subsample': np.arange(0.1, 1., 0.2)}
meta_mtrx = np.empty((y_test.shape[0], 4))

meta_mtrx[:,0] = gbc_base_predict 

meta_mtrx[:,1] = rfc_base_predict

meta_mtrx[:,2] = svc_base_predict

meta_mtrx[:,3] = lr_base_predict
xgb = XGBClassifier(random_state=17, cv=5)

grid_xgb = GridSearchCV(xgb, param_grid=xgb_params, cv=skf)

model_xgb = grid_xgb.fit(meta_mtrx, y_test)

model_xgb.best_params_
from sklearn.model_selection import cross_val_predict



gbc_base = GradientBoostingClassifier(random_state=17, learning_rate=0.1)

rfc_base = RandomForestClassifier(random_state=17, min_samples_leaf=3, n_estimators=10)

svc_base = SVC(random_state=17, C=0.9000000000000001, kernel='rbf')

lr_base = LogisticRegression(random_state=17, C=0.5)



meta_alg = XGBClassifier(random_state=17, cv=5, eta=0.1, min_child_weight=1, n_estimators=55, subsample=0.30000000000000004)



models = [gbc_base, rfc_base, svc_base, lr_base]



meta_mtrx = np.empty((train_targets.shape[0], len(models)))

models_base = []

        

for n, model in enumerate(models):

    meta_mtrx[:, n] = cross_val_predict(model, train_values.values, train_targets.values, cv=5, method='predict')

    model_base = model.fit(train_values, train_targets);

    models_base.append(model_base)

        

model_meta = meta_alg.fit(meta_mtrx, train_targets)

        

meta_mtrx_test = np.empty((test.shape[0], len(models)))         

for n, model in enumerate(models_base):

    meta_mtrx_test[:, n] = model.predict(test)

            

meta_predict = model_meta.predict(meta_mtrx_test)  
def write_to_submission_file(predictions, PassengerID, out_file='Submission.csv', columns=['PassengerID', 'Survived']):

    predicted_df = pd.DataFrame(np.array([PassengerId, predictions]).T, columns=columns)

    predicted_df.to_csv(out_file, index=False)
write_to_submission_file(meta_predict, PassengerId)