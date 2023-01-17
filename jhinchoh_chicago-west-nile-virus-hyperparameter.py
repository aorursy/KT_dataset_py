from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score



import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



scoring = 'accuracy'

seed=8

models = []

scores = []



df = pd.read_csv('/kaggle/input/west-nile-virus-wnv-mosquito-test-results.csv')
# For this iteration, just drop when location information is missing

df = df.dropna()



# Scale numerical values

scaler = MinMaxScaler(feature_range=(0, 1))

lst_scaler = ['Wards','Census Tracts', 'Zip Codes', 'Community Areas','Historical Wards 2003-2015']

df[lst_scaler] = scaler.fit_transform(df[lst_scaler])



# One-hot encode categorical values

lst_onehot = ['SEASON YEAR','WEEK','SPECIES','TRAP_TYPE']

df_s = df[lst_onehot]

df_o = pd.get_dummies(df_s)

df = df.drop(lst_onehot,axis = 1)

df = pd.concat([df,df_o], axis=1)



# Remove outliers

df = df[df['NUMBER OF MOSQUITOES'] < 50] 



# Convert target to numerical values

df['RESULT'] = df['RESULT'].map({'positive': 1,'negative': 0})



y = df['RESULT']

X = df.drop(['TEST ID','BLOCK','TRAP','TEST DATE','RESULT','LOCATION'], axis=1)



X_train,X_test,Y_train,Y_test= train_test_split(X,y,random_state=seed,test_size=0.3)

def evaluate(model, name):

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    models.append(name) 

    scores.append(accuracy_score(Y_test, Y_pred)) 



def search_grid(model):

    model.fit(X_train, Y_train)

    model.best_params_

    return(model.best_estimator_)
def model_knc(name):

    model_default = KNeighborsClassifier(n_jobs = -1)

    param_grid = {

        'metric': ['euclidean','manhattan'],

        'weights': ['uniform', 'distance'],

        'n_neighbors': [100, 200, 300]

    }

    evaluate(model_default, 'Default ' + name + ' Model')

    

    best_random_model = search_grid(RandomizedSearchCV(model_default, param_grid, cv=2, n_jobs = -1))

    evaluate(best_random_model, 'Best ' + name + ' Random Model')



def model_gbc(name):

    model_default = GradientBoostingClassifier()

    param_grid = {

        "learning_rate": [0.075, 0.1, 0.15, 0.2],

        'min_samples_leaf': [3, 4, 5],

        'min_samples_split': [8, 10, 12],

        "max_depth": [3,5,8],

        "subsample": [0.5, 0.8, 0.9, 1.0],

        'n_estimators': [100, 200, 300]

    }

    evaluate(model_default, 'Default ' + name + ' Model')

    

    best_random_model = search_grid(RandomizedSearchCV(model_default, param_grid, cv=2, n_jobs = -1))

    evaluate(best_random_model, 'Best ' + name + ' Random Model')



def model_rfc(name):

    model_default = RandomForestClassifier(random_state = seed, n_jobs = -1, verbose = 0)

    param_grid = {

        'bootstrap': [True],

        'max_depth': [80, 90, 100],

        'max_features': [2, 3],

        'min_samples_leaf': [3, 4, 5],

        'min_samples_split': [8, 10, 12],

        'n_estimators': [100, 200, 300]

    }

    evaluate(model_default, 'Default ' + name + ' Model')

    

    best_random_model = search_grid(RandomizedSearchCV(model_default, param_grid, cv=2, n_jobs = -1))

    evaluate(best_random_model, 'Best ' + name + ' Random Model')

    

def model_xgbc(name):

    model_default = XGBClassifier(random_state = seed,n_jobs = -1, verbose = 0)

    param_grid = {

        'max_depth': [3, 4, 5],

        'subsample': [0.9, 1.0],

        'colsample_bytree': [0.9, 1.0],

        'learning_rate': [0.05, 0.1, 0.5]

    }

    evaluate(model_default, 'Default ' + name + ' Model')

    

    best_random_model = search_grid(RandomizedSearchCV(model_default, param_grid, cv=2, n_jobs = -1))

    evaluate(best_random_model, 'Best ' + name + ' Random Model')
def run_models():

    model_knc('KNC')

    model_gbc('GBC')

    model_rfc('RFC')

    model_xgbc('XGBC')

    

run_models()

pd.DataFrame({"model":models, "score":scores})