import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sas
sas.set(style='whitegrid')
pd.options.display.max_columns= None # To view all the columns
pd.options.display.max_rows= None

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head(3)
df.describe()
round((((df.corr()['Class']**2).sort_values(ascending=False))), 5).head(10)
df['Class'].value_counts(normalize= True)*100
plt.figure(figsize=(20,5))
sas.heatmap(data= df.isnull(), yticklabels=False, cmap="rocket", linecolor='black',
    cbar=False, linewidths=0.01)

# We see there is no missing data
plt.figure(figsize=(14,5))
sas.distplot(df['Time'])
Fraud, Normal = df[df['Class']==1], df[df['Class']==0]
print(Fraud.Amount.describe(), '\n\n', Normal.Amount.describe())
# Not much to gain from this:(
X= df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]

y= df['Class']
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
scaler1= StandardScaler().fit_transform(X)
pca= PCA(n_components=3).fit_transform(scaler1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
pca.shape
from sklearn.pipeline import Pipeline # Import only if you're using it
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# pipe = Pipeline([("classifier", RandomForestClassifier())])
# 
# grid_param = [
#                 {"classifier": [RandomForestClassifier()],
#                  "classifier__n_estimators": [10,50,100,250],
#                  "classifier__max_depth": [2, 4, 8, 16, None]
#                  },
#                 {"classifier": [GradientBoostingClassifier()],
#                  "classifier__n_estimators": [5, 50, 250, 500],
#                  "classifier__max_depth": [1, 3, 5, 7, 9],
#                  "classifier__learning_rate":[0.01, 0.1, 1, 10, 100]
#                  },
#                 {"classifier": [LogisticRegression()],
#                  "classifier__C":[0.001, 0.01, 0.1, 1, 10, 100]}]
# gridsearch = GridSearchCV(pipe, grid_param, cv=3)
# best_model = gridsearch.fit(X_train,y_train)

# To choose the best parameters among the best classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

sas.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_pred,y_test), '\n')
print(confusion_matrix(y_pred,y_test))
y_pred_prob = logreg.predict_proba(X_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
# USE GRIDCV TO FIND THE BEST PARAMETERS

from sklearn.ensemble import RandomForestClassifier
rand_f = RandomForestClassifier()
rand_f.fit(X_train, y_train)
y_pred = rand_f.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
sas.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

print(classification_report(y_pred,y_test))
print(roc_auc_score(y_pred,y_test), '\n')
y_pred_prob = rand_f.predict_proba(X_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
# from xgboost import XGBRegressor
# xgb= XGBRegressor()
# parameters= {"learning_rate"    : [0.01, 0.10, 0.20, 0.30] ,
#  "max_depth"        : [ 5, 10, 12, 15, 30, 50],
#  "n_estimators" : [50, 100, 1000, 5000, 10000 ]}
 
# grid_search= GridSearchCV(xgb, parameters)
# results= grid_search.fit(X_train,y_train)

# ALWAYS USE GRIDCV TO FIND THE BEST PARAMETERS
# This will take a while

import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
sas.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

print(classification_report(y_test, y_pred),'\n')
print(roc_auc_score(y_pred,y_test))
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn.utils.testing import all_estimators
from sklearn import base

estimators = all_estimators()

for name, class_ in estimators:
    if issubclass(class_, base.ClassifierMixin):
        print(name)
