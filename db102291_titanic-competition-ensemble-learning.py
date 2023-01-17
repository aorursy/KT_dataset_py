from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import category_encoders as ce
import seaborn as sns
import numpy as np
import pandas as pd 
import random as rand
#Import training and testing data
train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")
#Which columns have missing values?
display(train_data.isnull().sum().sort_values(ascending=False))
display(test_data.isnull().sum().sort_values(ascending=False))
#Cabin letter
train_data['Cabin_new'] = train_data['Cabin'].str[0]
test_data['Cabin_new'] = test_data['Cabin'].str[0]

#Family size
train_data['Fam_size'] = train_data['SibSp'] + train_data['Parch']
test_data['Fam_size'] = test_data['SibSp'] + test_data['Parch']

#Title
train_data['Title']=train_data.Name.str.extract('([A-Za-z]+)\.')
test_data['Title']=test_data.Name.str.extract('([A-Za-z]+)\.')

#Sex-Class
train_data['Sex_class'] = train_data['Sex'] + "_" + str(train_data['Pclass'])
test_data['Sex_class'] = test_data['Sex'] + "_" + str(test_data['Pclass'])
#Fare
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
test_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)

#Age
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

#Cabin_new
train_data['Cabin_new'].fillna('X', inplace=True)
test_data['Cabin_new'].fillna('X', inplace=True)

#Embarked
train_data['Embarked'].fillna('S', inplace=True) #most common value
#Setup for model
y = train_data["Survived"]
features = ["Pclass", "Sex", "Parch", "SibSp", "Fam_size", "Embarked", "Fare", "Age", "Cabin_new", "Title", "Sex_class"]
cat_features = ['Sex', 'Embarked', 'Pclass', "Cabin_new", "Title", "Sex_class"]

X = train_data[features]
X_test = test_data[features]
# Create the count encoder
count_enc = ce.CountEncoder(cols=cat_features)

# Learn encoding from the training set
count_enc.fit(X[cat_features])

# Apply encoding to the train and validation sets as new columns
# Make sure to add `_count` as a suffix to the new columns
train_encoded = X.join(count_enc.transform(X[cat_features]).add_suffix('_count'))
test_encoded = X_test.join(count_enc.transform(X_test[cat_features]).add_suffix('_count'))
# Create the CatBoost encoder
cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=126)

# Learn encoding from the training set
cb_enc.fit(X[cat_features], y)

# Apply encoding to the train and validation sets as new columns
# Make sure to add `_cb` as a suffix to the new columns
train_encoded = X.join(cb_enc.transform(X[cat_features]).add_suffix('_cb'))
test_encoded = X_test.join(cb_enc.transform(X_test[cat_features]).add_suffix('_cb'))
#Update X for Count Encoding
#features_encoded = ["Pclass_count", "Sex_count", "Fam_size", "Embarked_count", "Fare", "Age", "Sex_class_count"]
#X = train_encoded[features_encoded]
#X_test = test_encoded[features_encoded]

#Update X for CatBoost Encoding
features_encoded = ["Pclass_cb", "Sex_cb", "Embarked_cb", "Fam_size", "Fare", "Age", "Sex_class_cb"]
X = train_encoded[features_encoded]
X_test = test_encoded[features_encoded]
scaler = StandardScaler() #to normalize data for neural net

model_rf = RandomForestClassifier(random_state=126)
model_xgb = XGBClassifier(random_state=126)
model_mlp = MLPClassifier(random_state=126)

param_grid_rf = {
    'model__n_estimators': [90, 110, 130],
    'model__max_depth': [7, 10, 13],
    'model__criterion': ['gini', 'entropy']}

param_grid_xgb = {
    'model__n_estimators': [150, 180, 210],
    'model__max_depth': [5, 8],
    'model__learning_rate': [0.08, 0.1, 0.12]}

param_grid_mlp = {'model__hidden_layer_sizes': [(25, 25, 25), (50,50,50)],
                  'model__activation': ['tanh', 'relu'],
                  'model__solver': ['sgd', 'adam'],
                  'model__alpha': [0.0001, 0.00001],
                  'model__learning_rate': ['constant','adaptive']}
my_pipeline = Pipeline(steps=[('model', model_rf)])

search = GridSearchCV(my_pipeline, param_grid_rf, n_jobs=-1, verbose=10, cv=5)
search.fit(X, y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
my_pipeline = Pipeline(steps=[('model', model_xgb)])

search = GridSearchCV(my_pipeline, param_grid_xgb, n_jobs=-1, verbose=10, cv=5)
search.fit(X, y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
#MLP 
my_pipeline = Pipeline(steps=[('preprocess', scaler), 
                              ('model', model_mlp)])

search = GridSearchCV(my_pipeline, param_grid_mlp, n_jobs=-1, verbose=10, cv=5)
search.fit(X, y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
#Building a voting classifier 
model_rf = RandomForestClassifier(random_state=126, criterion='entropy', max_depth=10, n_estimators=110)
model_xgb = XGBClassifier(random_state=126, learning_rate=0.1, max_depth=5, n_estimators=180)
model_mlp = MLPClassifier(random_state=126, activation='tanh', alpha=1e-04, hidden_layer_sizes=(25, 25, 25), learning_rate='constant', solver='adam')

model_vote = VotingClassifier(estimators=[('RF', model_rf), ('XGB', model_xgb), ('MLP', model_mlp)], voting='hard')
model_vote.fit(X, y)
predictions = model_vote.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")