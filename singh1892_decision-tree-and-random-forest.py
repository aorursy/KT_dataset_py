import pandas as pd
data_1 = pd.read_csv('../input/column_2C_weka.csv')
data_1.head()
data_randamoized = data_1.sample(frac=1.0)
import matplotlib.pyplot as plt
import seaborn as sns
data_1.info()
data_1.isnull().sum()
data_1.describe()
sns.countplot(x="class", data=data_1)
data_1.loc[:,'class'].value_counts()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.metrics import cohen_kappa_score
X_train, X_test, y_train, y_test = train_test_split(data_randamoized.iloc[:,:-1], data_randamoized['class'],test_size =0.2,random_state = 123)
treeclassifier = DecisionTreeClassifier()
from sklearn.metrics import make_scorer
def kappa(y, y_pred, **kargs):
    return cohen_kappa_score(y,y_pred)

kappa = make_scorer(kappa)
params = {'criterion':['entropy','gini'],'max_depth':[3,5,7,9]}
best_tree = GridSearchCV(treeclassifier,param_grid=params,scoring= kappa,cv=5)
best_tree.fit(X=X_train, y=y_train)
best_tree.best_params_
cross_val_score(best_tree.best_estimator_,X=X_train,y=y_train , cv=5)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
param_grid_rf = {'n_estimators':[100,200,500], 'max_depth':[3,5,7,9]}
best_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, scoring=kappa,  cv=5)
best_rf.fit(X=X_train, y=y_train)
cross_val_score(best_rf.best_estimator_, X=X_train, y= y_train,cv=5)
#we see that it gives better result than Decision tree.
data_2 = pd.read_csv('../input/column_3C_weka.csv.csv')
data_randomzied2 = data_2.sample(frac=1.0)
data_randomzied2.info()
data_randomzied2.isnull().sum()
sns.countplot(data_randomzied2['class'],data=data_randomzied2)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(data_randomzied2.iloc[:,:-1], data_randomzied2['class'],test_size =0.2,random_state = 123)
treeclassifier1 = DecisionTreeClassifier()
best_tree1 = GridSearchCV(treeclassifier1,param_grid=params,scoring= kappa,cv=5)
best_tree1.fit(X=X_train_1, y=y_train_1)
best_tree1.best_params_
cross_val_score(best_tree.best_estimator_,X=X_train_1,y=y_train_1 , cv=5)
rf1 = RandomForestClassifier()
best_rf1 = GridSearchCV(estimator=rf1, param_grid=param_grid_rf, scoring=kappa,  cv=5)
best_rf1.fit(X=X_train_1, y=y_train_1)
cross_val_score(best_rf1.best_estimator_, X=X_train_1, y= y_train_1,cv=5)
# here we see a dramatic change in the scores, from dececison tree from random forest.

