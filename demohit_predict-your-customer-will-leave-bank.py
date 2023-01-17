import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
import matplotlib.pyplot as plt
# Import supplementary visualizations code visuals.py


# Pretty display for notebooks
%matplotlib inline

dataframe=pd.read_csv('../input/Churn_Modelling.csv')
def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

def fit_model(X, y):
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    regressor = DecisionTreeRegressor()
    params = {'max_depth':[1,10]}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, params, scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_
def predectiveAnalysis():
    dataframe['Geography'].replace('France',1,inplace=True)
    dataframe['Geography'].replace('Spain',2,inplace=True)
    dataframe['Geography'].replace('Germany',3,inplace=True)
    dataframe['Gender'].replace('Female',0,inplace=True)
    dataframe['Gender'].replace('Male',1,inplace=True)
    y = dataframe['Exited']
    X = dataframe.drop(['Exited','Surname','IsActiveMember'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    reg = fit_model(X_train, y_train)
    print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
#   Here you can provide fresh client's data , i have dropped three freature exited,surname and
#   IsActiveMember because IsActiveMember is a dominant variable which predicts output based on
#   Its value , here i am using DecisionTreeRegressor to compete between features
    client_data=[[1231, 19632322 ,619 ,1  ,0  ,40 ,2,0.00,1 ,0 ,324323248.88 ]]
    print('Will leave bank soon' if reg.predict(client_data)[0]==1 else 'Will Not Leave')
   
predectiveAnalysis()
