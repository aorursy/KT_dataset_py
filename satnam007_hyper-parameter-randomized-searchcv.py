import pandas as pd

import numpy as np

from sklearn import ensemble

from sklearn import metrics

from sklearn import model_selection

from sklearn.model_selection import cross_val_score
df = pd.read_csv('../input/mobile-price-classification/train.csv')

X  = df.drop('price_range', axis = 1).values

y  = df['price_range'].values



#n_jobs=-1 so that it can use all the cores of the system

classifier = ensemble.RandomForestClassifier(n_jobs=-1)
param_grid = {

    "n_estimators": np.arange(100, 1500, 100), #100 to 1500 with 100 step_size

    "max_depth": np.arange(1, 20),

    "criterion": ["gini", "entropy"],

}
model = model_selection.RandomizedSearchCV(

    estimator = classifier,

    param_distributions = param_grid,

    n_iter = 10,

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
classifier = ensemble.RandomForestClassifier(criterion ='entropy', max_depth = 15, n_estimators = 900,

                                             n_jobs=-1)
score = cross_val_score(classifier,X,y, cv=10)

print('scores\n',score)

print('\ncv values', score.shape)

print('\nScore_Mean', score.mean())