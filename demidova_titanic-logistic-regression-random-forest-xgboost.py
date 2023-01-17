import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# path to train dataset
train_path = '../input/titanic/train.csv'
# path to test dataset
test_path = '../input/titanic/test.csv'

# Read a comma-separated values (csv) file into pandas DataFrame
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# shape of tha data
print('Train shape: ', train_data.shape)
print('Test shape: ', test_data.shape)
# Passengers with wrong number of siblings and parch
train_data.loc[train_data['PassengerId'] == 69, ['SibSp', 'Parch']] = [0,0]
test_data.loc[test_data['PassengerId'] == 1106, ['SibSp', 'Parch']] = [0,0]

# Age outlier 
train_data.loc[train_data['PassengerId'] == 631, 'Age'] = 48
# check data for NA values
train_NA = train_data.isna().sum()
test_NA = test_data.isna().sum()
pd.concat([train_NA, test_NA], axis=1, sort = False, keys = ['Train NA', 'Test NA'])
plt.figure(figsize = (16, 7))

plt.subplot(1,2,1)
sns.heatmap(train_data.isnull(), cbar=False)
plt.xticks(rotation = 35,     horizontalalignment='right',
    fontweight='light'  )
plt.title('Training dataset missing values')

plt.subplot(1,2,2)
sns.heatmap(test_data.isnull(), cbar=False)
plt.xticks(rotation=35,     horizontalalignment='right',
    fontweight='light'  )
plt.title('Test dataset missing values')

plt.tight_layout()
# Add new variable Age_NA indicates that there is no age in the original data.
train_data.loc[train_data['Age'].isna(), 'Age_NA'] = 1     # 1 for missing Age value
train_data.loc[train_data['Age_NA'].isna(), 'Age_NA'] = 0  # 0 if Age value is not null
test_data.loc[test_data['Age'].isna(), 'Age_NA'] = 1       
test_data.loc[test_data['Age_NA'].isna(), 'Age_NA'] = 0

# titles categories dict
title_dict = {  'Mr':     'Mr',
                'Mrs':    'Mrs',
                'Miss':   'Miss',
                'Master': 'Master',              
                'Ms':     'Miss',
                'Mme':    'Mrs',
                'Mlle':   'Miss',
                'Capt':   'military',
                'Col':    'military',
                'Major':  'military',
                'Dr':     'Dr',
                'Rev':    'Rev',                  
                'Sir':    'honor',
                'the Countess': 'honor',
                'Lady':   'honor',
                'Jonkheer': 'honor',
                'Don':    'honor',
                'Dona':   'honor' }

# add title variable
train_data['Title'] = train_data['Name'].str.split(',', expand = True)[1].str.split('.', expand = True)[0].str.strip(' ')
test_data['Title'] = test_data['Name'].str.split(',', expand = True)[1].str.split('.', expand = True)[0].str.strip(' ')

# map titles to category
train_data['Title_category'] = train_data['Title'].map(title_dict)
test_data['Title_category'] = test_data['Title'].map(title_dict)

# delete Title variable
del train_data['Title']
del test_data['Title']

# Filling the missing values in Age with the medians of Sex and Pclass, Title groups
train_data['Age'] = train_data.groupby(['Pclass', 'Sex', 'Title_category'])['Age'].apply(lambda x: x.fillna(x.median()))
test_data['Age'] = test_data.groupby(['Pclass', 'Sex', 'Title_category'])['Age'].apply(lambda x: x.fillna(x.median()))
train_data[train_data['Embarked'].isna()]
mode_emb = train_data[(train_data['Fare'] > 77) & (train_data['Fare'] < 82)& (train_data['Pclass']==1)]['Embarked'].mode()
train_data.loc[train_data['Embarked'].isna(), 'Embarked'] = mode_emb[0]
# Filling the missing values in Age with the medians of Sex and Pclass, Title groups
test_data['Fare'] = test_data.groupby(['Pclass', 'Sex', 'Title_category', 'Parch'])['Fare'].apply(lambda x: x.fillna(x.median()))
def feature_generator (data, train = False):
    
    features_data = data
    
    # Deck
    # Extract deck letter from cabin number
    features_data['deck'] = features_data['Cabin'].str.split('', expand = True)[1]
    # If cabin is NA - deck = U
    features_data.loc[features_data['deck'].isna(), 'deck'] = 'U'
    # If cabin is T - change to A (see EDA)
    features_data.loc[features_data['deck'] == 'T', 'deck'] = 'A'
    # Create dummy variables with prefix 'deck'
    features_data = pd.concat([features_data,
                               pd.get_dummies(features_data['deck'], prefix = 'deck')], 
                               axis=1)
    
    
    # titles dummy
    features_data = pd.concat([features_data, 
                               pd.get_dummies(features_data['Title_category'],
                                              prefix = 'title')], axis=1)

    # family size
    features_data['Family_size'] = features_data['SibSp'] + features_data['Parch'] + 1
    features_data['Family_size_group'] = features_data['Family_size'].map(
                                            lambda x: 'f_single' if x == 1 
                                                    else ('f_usual' if 5 > x >= 2 
                                                          else ('f_big' if 8 > x >= 5 
                                                               else 'f_large' )))
    features_data = pd.concat([features_data, 
                               pd.get_dummies(features_data['Family_size_group'], 
                                              prefix = 'family')], axis=1)     
    
    
    # Sex to number
    features_data['Sex'] = features_data['Sex'].map({'female': 1, 'male': 0}).astype(int)
    
    # embarked dummy
    features_data = pd.concat([features_data, 
                               pd.get_dummies(features_data['Embarked'], 
                                              prefix = 'embarked')], axis=1)
    
    # zero fare feature
    features_data['zero_fare'] = features_data['Fare'].map(lambda x: 1 if x == 0 else (0))
    
    # from numeric to categorical
    features_data['SibSp'] = features_data['SibSp'].map(lambda x: 1 if x > 0 else (0))
    features_data['Parch'] = features_data['Parch'].map(lambda x: 1 if x > 0 else (0))
    
    # delete variables we are not going to use anymore
    del features_data['PassengerId']
    del features_data['Ticket']
    del features_data['Cabin']
    del features_data['deck']    
    del features_data['Title_category']
    del features_data['Name']
    del features_data['Family_size']
    del features_data['Family_size_group'] 
    del features_data['Embarked']    
    
    return features_data     
# Extract target variable (label) from training dataset
all_train_label = train_data['Survived']
del train_data['Survived']

# Generate features from training dataset
all_train_features = feature_generator(train_data)
# Generate features from test dataset
all_test_features = feature_generator(test_data)
plt.figure(figsize=(12,10))
cor = all_train_features.corr()
sns.heatmap(cor)
# set model. max_iter - Maximum number of iterations taken for the solvers to converge.
lg_model = LogisticRegression(random_state = 64, max_iter = 1000)

# set parameters values we are going to check
optimization_dict = {'class_weight':['balanced', None],
                     'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                     'C': [0.01, 0.05, 0.07, 0.1, 0.5, 1, 2, 4, 5, 10, 15, 20]
                     }
# set GridSearchCV parameters
model = GridSearchCV(lg_model, optimization_dict, 
                     scoring='accuracy', n_jobs = -1, cv = 10)

# use training features
model.fit(all_train_features, all_train_label)

# print result
print(model.best_score_)
print(model.best_params_)
# set best parameters to the model
lg_tuned_model =  LogisticRegression(solver = 'newton-cg',
                                     C = 0.5,
                                     random_state = 64,
                                     n_jobs = -1)
# train our model with training data
lg_tuned_model.fit(all_train_features, all_train_label)

# calculate importances based on coefficients.
importances = abs(lg_tuned_model.coef_[0])
importances = 100.0 * (importances / importances.max())
# sort 
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [all_train_features.columns[i] for i in indices]

# visualize
plt.figure(figsize = (12, 5))
sns.set_style("whitegrid")
chart = sns.barplot(x = names, y = importances[indices])
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light'  
)
plt.title('Logistic regression. Feature importance')
plt.tight_layout()
# set model
rf_model = RandomForestClassifier(oob_score = True, n_jobs = -1, random_state = 64)
# create a dictionary of parameters values we want to try
optimization_dict = {'criterion':['gini', 'entropy'],
                     'n_estimators': [100, 500, 1000, 1700],
                     'max_depth': [7, 10, 11, 12],
                     'min_samples_split': [6, 7, 8, 10],
                     'min_samples_leaf': [3, 4, 5]
                     }

# set GridSearchCV parameters
model = GridSearchCV(rf_model, optimization_dict, 
                     scoring='accuracy', verbose = 1, n_jobs = -1, cv = 5)

# use training data
model.fit(all_train_features, all_train_label)

# print best score and best parameters combination
print(model.best_score_)
print(model.best_params_)
# set best parameters to the model
rf_tuned_model =  RandomForestClassifier(criterion = 'gini',
                                       n_estimators = 100,
                                       max_depth = 12,
                                       min_samples_split = 6,
                                       min_samples_leaf = 4,
                                       max_features = 'auto',
                                       oob_score = True,
                                       random_state = 64,
                                       n_jobs = -1)
# train model using training dataset
rf_tuned_model.fit(all_train_features, all_train_label)

# Calculate feature importances
importances = rf_tuned_model.feature_importances_

# Visualize Feature Importance
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [all_train_features.columns[i] for i in indices]

plt.figure(figsize = (12, 5))
sns.set_style("whitegrid")
chart = sns.barplot(x = names, y=importances[indices])
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light'  
)
plt.title('Random forest. Feature importance')
plt.tight_layout()
# set model
xgb_model = XGBClassifier(random_state = 64)
# create a dictionary of parameters values we want to try
optimization_dict = {'n_estimators': [200, 1000, 1700, 2000],
                     'max_depth': [4, 6, 8, 10],
                     'learning_rate': [0.001, 0.01, 0.1, 0.5],
                     'gamma': [0, 1, 5],
                     'min_child_weight':[3, 6, 10],
                     'subsample': [0.5, 0.8, 0.9]
                     }
# set GridSearchCV parameters
model = GridSearchCV(xgb_model, optimization_dict, 
                     scoring='accuracy', verbose = 1, n_jobs = -1, cv = 5)

# use training data
model.fit(all_train_features, all_train_label)
print(model.best_score_)
print(model.best_params_)
# set model with best parameters
xgb_tuned_model =  XGBClassifier(n_estimators = 200,
                               max_depth = 8,
                               learning_rate = 0.5,
                               gamma = 1,
                               min_child_weight = 6,
                               subsample = 0.9,
                               random_state = 64)
# train model with training dataset
xgb_tuned_model.fit(all_train_features, all_train_label)

# Calculate feature importances
importances = xgb_tuned_model.feature_importances_

# Visualize Feature Importance
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [all_train_features.columns[i] for i in indices]

plt.figure(figsize = (12, 5))
sns.set_style("whitegrid")
chart = sns.barplot(x = names, y=importances[indices])
plt.xticks(rotation=45, horizontalalignment='right', fontweight='light')
plt.title('XGBoost. Feature importance')
plt.tight_layout()
models = []
# add our tuned models into list
models.append(('Logistic Regression', lg_tuned_model))
models.append(('Random Forest', rf_tuned_model))
models.append(('XGBoost', xgb_tuned_model))

results = []
names = []

# evaluate each model in turn
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, shuffle = True, random_state = 64)
    cv_results = model_selection.cross_val_score(model, all_train_features, 
                                                 all_train_label, 
                                                 cv = 10, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    # print mean accuracy and standard deviation
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

fig = plt.figure(figsize=(6,4))
plt.boxplot(results)
plt.title('Algorithm Comparison')
plt.xticks([1,2,3], names)
plt.show()
# train chosen model on training dataset
rf_tuned_model.fit(all_train_features, all_train_label)

# get predictions on test dataset
predictions = rf_tuned_model.predict(all_test_features)

# Save results in the required format
output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': predictions})
output.to_csv('submission_xgb.csv', index=False)
output.head()