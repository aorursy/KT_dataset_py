import pandas as pd
%matplotlib inline
train = pd.read_csv("../input/train.csv");
print(train.head())
trainnp = train.values
X_train = trainnp[:,1:] 
y_train = trainnp[:,0] 
from mlxtend.plotting import plot_decision_regions # convenient library for plotting decision regions
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier
knclassifier = KNeighborsClassifier(n_neighbors=3, p=2)

# train the classifier
knclassifier.fit(X_train, y_train)
plot_decision_regions(X_train, y_train.astype(int), knclassifier)
import xgboost as xgb

params = {
    'booster':'gbtree',
    'colsample_bylevel':1,
    'colsample_bytree':1,
    'gamma':0, 
    'learning_rate':0.1, 
    'max_delta_step':0,
    'max_depth':3,
    'min_child_weight':1,
    'n_estimators':100,
    'objective':'multi:softprob',
    'random_state':2018,
    'reg_alpha':0, 
    'reg_lambda':1,
    'seed':2018,
    'subsample':1}

xgb_clf = xgb.XGBClassifier(**params)
xgb_clf = xgb_clf.fit(X_train, y_train, verbose=True)
xgb_clf = xgb_clf.fit(X_train, y_train)

plot_decision_regions(X_train, y_train.astype(int), xgb_clf)
