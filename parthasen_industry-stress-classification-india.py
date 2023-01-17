# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
excel_data= pandas.read_excel('/kaggle/input/industry-stress-india/ml_data.xlsx', sheet_name='data')
y=excel_data.Stress_class
X=excel_data.drop('Stress_class',axis=1)
features = {"f1":u"Debt Ratio",
"f2":u"Earning Ratio",
"f3":u"Current Ratio",
"f4":u"PAT margin",
"f5":u"PAT",
"f6":u"Inventory Turnover",
"f7":u"Total Liabilities",
"f8":u"Total Assets",
"f9":u"Gross Sales",
"f10":u"Free Cash flow",
"f11":u"Exports - FOB Value",
"f12":u"Imports - CIF Value",
"f13":u"Capacity Utilized"}
forest = RandomForestClassifier(n_estimators=11, max_features=5,random_state=0)
forest.fit(X, y)
importances=forest.feature_importances_
indices = np.argsort(importances)[::-1]
# Plot the feature importancies of the forest
num_to_plot = 13
feature_indices = [ind+1 for ind in indices[:num_to_plot]]

# Print the feature ranking
print("Feature ranking:")
  
for f in range(num_to_plot):
    print("%d. %s %f " % (f + 1,features["f"+str(feature_indices[f])], 
            importances[indices[f]]))
plt.figure(figsize=(15,5))
plt.title(u"Feature Importance")
bars = plt.bar(range(num_to_plot), 
               importances[indices[:num_to_plot]],
       color=([str(i/float(num_to_plot+1)) 
               for i in range(num_to_plot)]),
               align="center")
ticks = plt.xticks(range(num_to_plot), 
                   feature_indices)
plt.xlim([-1, num_to_plot])
plt.legend(bars, [u''.join(features["f"+str(i)]) 
                  for i in feature_indices]);
y=excel_data.Stress_class
X=excel_data.drop(['Stress_class','CU'],axis=1) # dropped the CU variable
X.columns
#preprocessing
from sklearn import preprocessing
X_scaled=preprocessing.scale(X)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.05, random_state=1)
# fit the model
model = LogisticRegression(solver='newton-cg')
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.05, random_state=1)
# fit the model
model = RandomForestClassifier(n_estimators=1000, max_features=9,max_depth=3)
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.05, random_state=1)
# fit the model
model = XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=1000,objective='binary:logitraw', booster='gbtree',gamma=0,base_score=0.57)
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
# make predictions for test data
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.05, random_state=1)
# fit the model
model = RandomForestClassifier(n_estimators=1000, max_features=9,max_depth=3)
model.fit(X_train, y_train)
tree_list = model.estimators_
feat=["Debt Ratio","Earning Ratio","Current Ratio",
"PAT margin","PAT","Inventory Turnover","Total Liabilities","Total Assets","Gross Sales","Free Cash flow","Exports - FOB Value",
"Imports - CIF Value", "Capacity Utilized"]
plt.figure(figsize=(16,6))
tree.plot_tree(tree_list[0], filled=True, feature_names=feat,class_names=['low','High'], node_ids=True);
plt.figure(figsize=(16,6))
tree.plot_tree(tree_list[1], filled=True, feature_names=feat,class_names=['low','High'], node_ids=True);
plt.figure(figsize=(16,6))
tree.plot_tree(tree_list[2], filled=True, feature_names=feat,class_names=['low','High'], node_ids=True);
