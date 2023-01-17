!pip install xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os 
print(os.listdir('../input'))
#read in the dataset
df = pd.read_csv('../input/diabetes_data.csv')

#take a look at the data
df.head(20)
#check dataset size
df.shape
#split data into inputs and targets
X = df.drop(columns = ['diabetes'])
y = df['diabetes']
print(y.shape)
print("first 10 labels")
print(y[:10])
#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#create new a knn model
knn = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
params_knn = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gs = GridSearchCV(knn, params_knn, cv=5)
#fit model to training data
knn_gs.fit(X_train, y_train)
#save best model
knn_best = knn_gs.best_estimator_
print(knn_best)
#check best n_neigbors value
print(knn_gs.best_params_)
#create a new rf classifier
rf = RandomForestClassifier(random_state=1)

#create a dictionary of all values we want to test for n_estimators
# params_rf = {'n_estimators': [50, 100, 200, 300, 400, 500], 'max_depth': [4, 5, 6, 7, 8], 'min_samples_split': [2, 4, 6, 8, 10]}
params_rf = {'n_estimators': [100], 'max_depth': [8], 'min_samples_split': [10]}

#use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=5)

#fit model to training data
rf_gs.fit(X_train, y_train)
#save best model
rf_best = rf_gs.best_estimator_
print(rf_best)
#check best n_estimators value
print(rf_gs.best_params_)
#create a new logistic regression model
log_reg = LogisticRegression(solver='lbfgs',random_state=1)

#fit the model to the training data
log_reg.fit(X_train, y_train)
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_split=10)
et.fit(X_train, y_train)
from xgboost import XGBClassifier
xgb_boost = XGBClassifier()
xgb_boost.fit(X_train, y_train)
#test the three models with the test data and print their accuracy scores

print('knn: {}'.format(knn_best.score(X_test, y_test)))
print('rf: {}'.format(rf_best.score(X_test, y_test)))
print('log_reg: {}'.format(log_reg.score(X_test, y_test)))
print('et: {}'.format(et.score(X_test, y_test)))
print('xgb: {}'.format(xgb_boost.score(X_test, y_test)))
#create a dictionary of our models
estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg), ('et', et), ('xgb', xgb_boost)]

#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='hard')

#fit model to training data
ensemble.fit(X_train, y_train)

#test our model on the test data
ensemble.score(X_test, y_test)
#create a dictionary of our models
estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg), ('et', et), ('xgb', xgb_boost)]

#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='soft')

#fit model to training data
ensemble.fit(X_train, y_train)

#test our model on the test data
ensemble.score(X_test, y_test)
#create a dictionary of our models
estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg), ('et', et), ('xgb', xgb_boost)]

#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='soft', weights=[2, 4, 1, 2, 5],flatten_transform=True)

#fit model to training data
ensemble.fit(X_train, y_train)

#test our model on the test data
ensemble.score(X_test, y_test)