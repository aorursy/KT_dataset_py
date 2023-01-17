# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/minor-project-2020/train.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
%matplotlib inline
train = pd.read_csv("/kaggle/input/minor-project-2020/train.csv",index_col='id')
test = pd.read_csv("/kaggle/input/minor-project-2020/test.csv",index_col='id')
train.head()
test.head()
train.describe()
for i in test.dtypes:
    if i not in ['int64','float64']:
        print(i)
a=pd.concat([train.dtypes],axis=1)
a.columns=['dtype']
numerical_features=list(pd.concat([a[a['dtype']=='float64']]).index)
temp=['label']
for i in numerical_features:
#     print(i)
    if np.array(train[i].unique()).shape[0]==1:
        train=train.drop(columns=[i])
        test=test.drop(columns=[i])
        temp.append(i)
numerical_features=[i for i in numerical_features if i not in temp]
corr=train.corr()


pipeline = Pipeline(
    [
     ('selector',SelectKBest(f_regression)),
     ('model',LinearRegression())
    ]
)
search = GridSearchCV(
    estimator = pipeline,
    param_grid = {'selector__k':[3,4,5,6,7,8,9,10]},
    n_jobs=-1,
    scoring="neg_mean_squared_error",
    cv=5,
    verbose=3

    )
#search.fit(train,test)
train.head()
#train.drop(['col_1'],axis=1,inplace=True)
#train.des
Y=train['target']
#

#train_Y= train.
#train.head()
#print(train_X)
#from sklearn.feature_selection import mutual_info_classif
#train_mutual_information = mutual_info_classif(train_X, train_Y)
train.drop(['target'],axis=1,inplace=True)
X=train
from sklearn.feature_selection import GenericUnivariateSelect

trans = GenericUnivariateSelect(score_func=lambda X, y: X.mean(axis=0), mode='percentile', param=50)
X_trans = trans.fit_transform(X, Y)
print("We started with {0} pixels but retained only {1} of them!".format(X.shape[1], X_trans.shape[1]))
X=X_trans
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=121)
from sklearn.feature_selection import mutual_info_classif
mutual_information = mutual_info_classif(X, Y)

plt.subplots(1, figsize=(26, 1))
sns.heatmap(mutual_information[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)
plt.yticks([], [])
plt.gca().set_xticklabels(train.columns[1:], rotation=45, ha='right', fontsize=12)
plt.suptitle("Kepler Variable Importance (mutual_info_classif)", fontsize=18, y=1.2)
plt.gcf().subplots_adjust(wspace=0.2)
pass
trans = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=50)
X_trans = trans.fit_transform(X, Y)
X_trans.shape
train
columns_retained_Select = train.iloc[:, 1:].columns[trans.get_support()].values
pd.DataFrame(X_trans, columns=columns_retained_Select).head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=121)
print(len(X_train), len(X_test))
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scaled_X_train = scalar.fit_transform(X_train)
scaled_X_test = scalar.transform(X_test)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
dt = DecisionTreeClassifier()
dt.fit(scaled_X_train, y_train)
y_pred = dt.predict(scaled_X_test)
print("Accuracy is : {}".format(dt.score(scaled_X_test, y_test)))
column_names = X_train.columns
feature_importances = pd.DataFrame(dt.feature_importances_,
                                   index = column_names,
                                    columns=['importance'])
feature_importances.sort_values(by='importance', ascending=False).head(60)
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
#Z=train
Z=train[('col_1','col_2')]
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(dt, scaled_X_test, y_test, cmap = plt.cm.Blues)
print("Classification Report: ")
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-pastel')

FPR, TPR, _ = roc_curve(y_test, y_pred)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Target', fontsize= 18)
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
train = pd.read_csv("/kaggle/input/minor-project-2020/train.csv",index_col='id')
train
train.describe()
train['target'].value_counts()
pd.set_option('max_columns',100)
sns.heatmap(train.corr())
from sklearn.model_selection import StratifiedShuffleSplit
X = train.drop(['target'],axis = 1)
X.shape

y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=101)
# Random forest classifier to fit basic model and find importance of each feature
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# importance = rfc.feature_importances_
# # summarize feature importance
# for i,v in enumerate(importance):
#     print('Feature: %s, Score: %.5f' % (X_train.columns[i],v))
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, rfc_pred))
print(accuracy_score(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))
test = pd.read_csv("/kaggle/input/minor-project-2020/test.csv",index_col='id')
rfc_pred = rfc.predict(test.drop(['id'],axis=1))
# print(confusion_matrix(y_test, rfc_pred))
# print(accuracy_score(y_test, rfc_pred))
# print(classification_report(y_test, rfc_pred))
test
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scaled_X_train = scalar.fit_transform(X_train)
scaled_X_test = scalar.transform(X_test)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
dt = DecisionTreeClassifier()
dt.fit(scaled_X_train, y_train)
y_pred = dt.predict(scaled_X_test)
print("Accuracy is : {}".format(dt.score(scaled_X_test, y_test)))
column_names = X_train.columns
feature_importances = pd.DataFrame(dt.feature_importances_,
                                   index = column_names,
                                    columns=['importance'])
feature_importances.sort_values(by='importance', ascending=False).head(60)
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(dt, scaled_X_test, y_test, cmap = plt.cm.Blues)
print("Classification Report: ")
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-pastel')

FPR, TPR, _ = roc_curve(y_test, y_pred)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Target', fontsize= 18)
plt.show()