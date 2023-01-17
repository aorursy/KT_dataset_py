

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np

import pandas as pd



from sklearn import preprocessing



from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



db = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')

db.head()
db.info() #no missings
db = db.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)
pd.options.display.max_rows = 200

db.groupby('Exited').describe().T
from sklearn.preprocessing import OrdinalEncoder



ordinal_encoder = OrdinalEncoder()

db_cat = db[['Gender', 'Geography']] 

db_cat_encoded = ordinal_encoder.fit_transform(db_cat)
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

db_cat_encoded_1hot = cat_encoder.fit_transform(db_cat_encoded)

db_cat_encoded_1hot
array = db_cat_encoded_1hot.toarray()



db_cat_encoded_1hot_df = pd.DataFrame(array)

db_cat_encoded_1hot_df.head()
db_cat_encoded_1hot_df.columns = ['M', 'F','France', 'Germany', 'Spain' ]
db_final = db.merge(db_cat_encoded_1hot_df, left_index = True, right_index = True)

db_final.head()
#making bins to be able to split correctly, but using in regression/forest the original variable

db_final['Credit_Score'] = pd.cut(db_final['CreditScore'], bins = [0,500,650,720,np.inf], labels = [1,2,3,4])



db_final['Estimated_Salary'] = pd.cut(db_final['EstimatedSalary'], bins = [0,50000,100000,150000,np.inf], labels = [1,2,3,4])



db_final['Age_new'] = pd.cut(db_final['Age'], bins = [0,30,35,45,60,np.inf], labels = [1,2,3,4,5])
db_final.head()
train, test = train_test_split(db_final, test_size = 0.2,

                               stratify = db_final[['Estimated_Salary','Credit_Score',

                                                    'M','F', 'France','Germany', 'Spain','Exited']])
train_label = train['Exited']

train = train.drop(['Exited','Gender', 'Geography'], axis = 1)



test_label = test['Exited']

test = test.drop(['Exited','Gender', 'Geography'], axis = 1)
param_grid = {

    'C': [0.01, 0.05,0.1,0.15,0.3,0.5,0.8,1,5,3,5,5,10,20,50,100,150,200,300,400,1000],

    'max_iter': [50, 100, 150]

    }



log_reg = LogisticRegression()

grid_search = GridSearchCV(estimator = log_reg , param_grid = param_grid, 

                          cv = 10)

grid_search.fit(train,train_label)

grid_search.best_params_
best_grid = grid_search.best_estimator_

best_grid
predict_label = best_grid.predict(test)



from sklearn.metrics import classification_report

classification_report(test_label, predict_label)

target_names = ['0','1']

print(classification_report(test_label, predict_label, target_names=target_names))
from sklearn.metrics import accuracy_score

accuracy_score(test_label, predict_label) #an accuracy of 0.79 ( for the first run)
from pprint import pprint 



from sklearn.model_selection import RandomizedSearchCV



n_estimators = [int(x) for x in np.linspace(start = 5, stop = 800, num = 30)]



max_features = ['auto', 'sqrt']



max_depth = [int(x) for x in np.linspace(5, 2000, num = 30)]

max_depth.append(None)



min_samples_split = [2, 4, 6, 10]



min_samples_leaf = [1,2, 4, 6, 10]





bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)
forest_reg = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = forest_reg, param_distributions = random_grid, n_iter = 100, cv = 3, 

                               verbose=2, random_state=42, n_jobs = -1)



rf_random.fit(train,train_label)
rf_random.best_params_
from sklearn.model_selection import GridSearchCV



param_grid = {

    'bootstrap': [True],

    'max_depth': [1500,2000, 2200],

    'max_features': ['sqrt'],

    'min_samples_leaf': [8,10,12],

    'min_samples_split': [4,6,8],

    'n_estimators': [360,550,600,650,700]

}


rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search.fit(train,train_label)

grid_search.best_params_
best_grid = grid_search.best_estimator_



best_grid
predict_label_2 = best_grid.predict(test)



from sklearn.metrics import classification_report

classification_report(test_label, predict_label_2)

target_names = ['0','1']

print(classification_report(test_label, predict_label_2, target_names=target_names))
from sklearn.metrics import accuracy_score

accuracy_score(test_label, predict_label_2) # nice, 0.86 accuracy for the first run :)  