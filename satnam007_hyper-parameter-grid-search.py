import pandas as pd

import numpy as np

from sklearn import ensemble

from sklearn import metrics

from sklearn import model_selection
df = pd.read_csv('../input/mobile-price-classification/train.csv')

df.columns

df.head(2)
# if __name__ == __main__:

df = pd.read_csv('../input/mobile-price-classification/train.csv')

X  = df.drop('price_range', axis = 1).values

y  = df['price_range'].values
#n_jobs=-1 so that it can use all the cores of the system

classifier = ensemble.RandomForestClassifier(n_jobs=-1)
param_grid = {

    "n_estimators": [100, 200, 300, 400],

    "max_depth": [1, 3, 5, 7],

    "criterion": ["gini", "entropy"]

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
# here we need to pay attention to only three parameters

# 'criterion': 'entropy',

# 'max_depth': 7,

# 'n_estimators': 400
classifier = ensemble.RandomForestClassifier(criterion ='entropy', max_depth = 7, n_estimators = 400,

                                             n_jobs=-1)
from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier,X,y)
print('scores\n',score)

print('\n cv values', score.shape)

# so as we said earlier cv=5 is defult values
score.mean()
score = cross_val_score(classifier,X,y, cv=10)

print('scores\n',score)

print('\ncv values', score.shape)

print('\nScore_Mean', score.mean())