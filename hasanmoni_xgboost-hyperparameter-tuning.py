import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost

import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("../input/drug-classification/drug200.csv")
data.head()
data.info()
data['Na_to_K_gre_fifteen'] = [1 if i >15 else 0 for i in data.Na_to_K]
data.head()
cholesterol = {"HIGH":1, "NORMAL":0}
bp = {"HIGH":2, "LOW":0, "NORMAL":1}
sex = {"F":1, "M":0}
drug = {"drugA":0, "drugB":1, "drugC":2, "drugX":3, "DrugY":4}
# using map
data.Cholesterol = data.Cholesterol.map(cholesterol)
data.BP = data.BP.map(bp)
data.Sex = data.Sex.map(sex)
data.Drug = data.Drug.map(drug)
data.head()
X = data.drop('Drug', axis=1)
y = data['Drug']
Xtrain, xtest, Ytrain,ytest = train_test_split(X,y, test_size=.2, random_state=42, shuffle=True)
Ytrain = Ytrain.values.reshape(-1,1)
ytest = ytest.values.reshape(-1,1)
print("Shap of the Xtrain is :", Xtrain.shape)
print("Shape of the xtest is :", xtest.shape)
print("Shape of the Ytrain is :", Ytrain.shape)
print("Shape of the ytest is :", ytest.shape)
classifier=xgboost.XGBClassifier()
hyperparameter_grid = {
    'n_estimators': [100, 500, 900, 1100, 1500],
    'max_depth':[2, 3, 5, 10, 15],
    'learning_rate':[0.05,0.1,0.15,0.20],
    'min_child_weight':[1,2,3,4],
    'booster':['gbtree','gblinear'],
    'base_score':[0.25,0.5,0.75,1]
    }
random_cv = RandomizedSearchCV(estimator=classifier,
            param_distributions=hyperparameter_grid,
            cv=5, 
            n_iter=50,
            scoring = 'neg_mean_absolute_error',
            n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
random_cv.fit(Xtrain,Ytrain)
random_cv.best_estimator_
classifier=xgboost.XGBClassifier(base_score=0.25, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=2,
              min_child_weight=1,
              n_estimators=900, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
classifier.fit(Xtrain,Ytrain)
predict = classifier.predict(xtest)
predict
#cross_val_score in train data
cvs = cross_val_score(classifier, Xtrain,Ytrain,cv=5)
cvs
cvs.mean()
# save the model to disk
filename = 'Drug_Classification.sav'
pickle.dump(classifier, open(filename, 'wb'))