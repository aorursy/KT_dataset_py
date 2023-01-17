path = '../input/older-dataset-for-dont-overfit-ii-challenge/'
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from yellowbrick.classifier import ROCAUC

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import RandomizedSearchCV
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
submission = pd.read_csv(path+'sample_submission.csv')
print(train.shape)
print(test.shape)
print(submission.shape)
X, y = train.drop(['id','target'], axis=1), train['target']
test = test.drop(['id'], axis=1) 
print(X.shape, y.shape)
print(test.shape)
scaler  = StandardScaler()
X  = scaler.fit_transform(X)
test = scaler.transform(test)
# Hyperparameters for Logistic Regression
LRparams = {
    "class_weight":["balanced"],
    "penalty" : ["l2","l1"],
    "tol" : [0.0001,0.0002,0.0003],
    "max_iter": [100,200,300],
    "C" :[0.001,0.01, 0.1, 1, 10, 100],
    "intercept_scaling": [1, 2, 3, 4],
    "solver":["liblinear"],
}
model = LogisticRegression(random_state=42)
rs=RandomizedSearchCV(
    model ,
    param_distributions = LRparams,
    verbose=0, 
    n_jobs=-1, 
    scoring='roc_auc',
    cv = 25, 
    n_iter=100, 
    random_state=42
)
rs.fit(X,y)
print('Model name:{:20}\nBest Score: {:.4f}\nBest Param: {}'.format('Logistic Regression',rs.best_score_, rs.best_params_))
clf = rs.best_estimator_
clf.fit(X,y)

selector = RFE(clf, 25, step=1)
selector.fit(X,y)
visualizer = ROCAUC(selector, classes=["0", "1"])

visualizer.fit(X, y)
visualizer.score(X, y) 
visualizer.show()    
pred = selector.predict_proba(test)[:,1]
pred
submission['target'] = pred
submission.head()
submission.to_csv('submission.csv', index=False)