#Load ML libraries
%matplotlib inline
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.feature_selection import SelectKBest
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('/content/drive/My Drive/MachineHack/ForestCoverData/train.csv')
train_data.head()
train_data["Cover_Type"].hist()
train_data['Cover_Type'].value_counts()
test_data = pd.read_csv('/content/drive/My Drive/MachineHack/ForestCoverData/test.csv')
test_data.head()
train_data.isnull().sum()
train_data.skew()
train_data.shape, test_data.shape
X = train_data.drop('Cover_Type',1)
y = train_data['Cover_Type']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
clf = RandomForestClassifier(n_estimators=300,class_weight='balanced',n_jobs=2,random_state=42)
clf.fit(X_train,y_train)
print("Accuracy:",clf.score(X_test,y_test))
etr = ExtraTreesClassifier(n_estimators=300,class_weight='balanced',n_jobs=2,random_state=42)
etr.fit(X_train,y_train)
print("Accuracy:",etr.score(X_test,y_test))
lgr = LogisticRegression()
lgr.fit(X_train,y_train)
print("Accuracy:",lgr.score(X_test,y_test))
predict_probabs = etr.predict_proba(test_data)
print(predict_probabs)
len(predict_probabs)
predict_probabs[0]
prediction = pd.DataFrame(np.round(predict_probabs, decimals=2),columns=['1', '2', '3', '4', '5', '6', '7'])

#results = pd.concat([id_,prediction],axis=1)
prediction.head()
prediction.to_csv('/content/drive/My Drive/MachineHack/ForestCoverData/Submission_file.csv', index= False)
etc_para = [{'n_estimators':[10,20,30,50,100,200,300,500], 'max_depth':[5,10,15,20], 'max_features':[0.1,0.2,0.3]}] 
# Default number of features is sqrt(n)
# Default number of min_samples_leaf is 1
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
ETC = RandomizedSearchCV(ExtraTreesClassifier(),param_distributions=etc_para, cv=5, scoring= 'neg_log_loss',n_jobs=-1)
ETC.fit(X_train, y_train)
print(ETC.best_score_, ETC.best_estimator_)
random_etr = ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=20, max_features=0.2,
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=None,
                     oob_score=False, random_state=None, verbose=0,
                     warm_start=False)
random_etr.fit(X_train,y_train)
print("Accuracy:",random_etr.score(X_test,y_test))
new_probabs = random_etr.predict_proba(test_data)
random_prediction = pd.DataFrame(np.round(predict_probabs, decimals=2),columns=['1', '2', '3', '4', '5', '6', '7'])

#results = pd.concat([id_,prediction],axis=1)
random_prediction.head()
from pprint import pprint
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
rf_random = RandomizedSearchCV(estimator = clf , param_distributions = random_grid, scoring='neg_log_loss',cv = 10, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
print(rf_random.best_score_, rf_random.best_estimator_)
random_rf = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight='balanced',
                       criterion='gini', max_depth=50, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=2000,
                       n_jobs=2, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
random_rf.fit(X_train, y_train)
random_rf.score(X_train, y_train)
random_rf.score(X_test, y_test)
rf_probabs = random_rf.predict_proba(test_data)
rf_predictions = pd.DataFrame(np.round(rf_probabs, decimals=2),columns=['1', '2', '3', '4', '5', '6', '7'])

#results = pd.concat([id_,prediction],axis=1)
rf_predictions.head()
rf_predictions.to_csv('/content/drive/My Drive/MachineHack/ForestCoverData/RFModel_Submission_file.csv', index= False)