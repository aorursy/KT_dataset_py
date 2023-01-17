import pandas as pd

import numpy as np



from sklearn import ensemble

from sklearn import metrics

from sklearn import model_selection

from sklearn import preprocessing

from sklearn import decomposition

from sklearn import pipeline
df = pd.read_csv('../input/mobile-price-classification/train.csv')

X  = df.drop('price_range', axis = 1).values

y  = df['price_range'].values
scl = preprocessing.StandardScaler()

pca = decomposition.PCA()
#n_jobs=-1 so that it can use all the cores of the system

rf = ensemble.RandomForestClassifier(n_jobs=-1)
classifier = pipeline.Pipeline([("scaling", scl),("pca", pca),("rf", rf)])
param_grid = {

    "pca__n_components": np.arange(5, 10),

    "rf__n_estimators": np.arange(100, 1500, 100), #100 to 1500 with 100 step_size

    "rf__max_depth": np.arange(1, 20),

    "rf__criterion": ["gini", "entropy"],

}
model = model_selection.RandomizedSearchCV(

    estimator = classifier,

    param_distributions = param_grid,

    n_iter = 5,

    scoring = "accuracy",

    verbose = 10,  #max_value

    n_jobs = 1,

    cv = 5,   

    #stratified fold is recomended

    # if we dont specify cv = 5 but it is still going to use cv = 5

    # if we have categoris as target or its binalry then is it going to use stratified k-fold     

)
model.fit(X,y)
print(model.best_score_)

print(model.best_estimator_.get_params())
param_grid = {

    "pca__n_components": [5, 10],

    "rf__n_estimators": [100, 200, 300, 400],

    "rf__max_depth": [1, 3, 5, 7],

    "rf__criterion": ["gini", "entropy"]

}
model = model_selection.GridSearchCV(

    estimator = classifier,

    param_grid = param_grid,

    scoring = "accuracy",

    verbose = 10,  #max_value

    n_jobs = 1,

    

    #stratified fold is recomended

    # if we dont specify cv = 5 but it is still going to use cv = 5

    # if we have categoris as target or its binalry then is it going to use stratified k-fold 

    cv = 5   

)
model.fit(X,y)
print(model.best_score_)

print(model.best_estimator_.get_params())