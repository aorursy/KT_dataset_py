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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.shape
df.head()
df.isnull().values.any()
df.drop_duplicates(inplace=True) # dropping duplicate records
df.duplicated().sum()
df["Class"].value_counts(normalize=True)*100
X = df.drop(columns=['Class'],axis=1)
y = df.pop('Class')
# splitting data into training and test set for independent attributes
from sklearn.model_selection import train_test_split

X_train, X_test, train_labels, test_labels = train_test_split(X, y, test_size=.30, random_state=1)
param_grid_rf = {'model__n_estimators': [75]}
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, matthews_corrcoef,accuracy_score
from sklearn.pipeline import Pipeline
len(X_train.columns)

param_grid = {
    'max_depth': [7,10],
    'max_features': [5,8,10],
    'min_samples_leaf': [2000, 5000],
    'min_samples_split': [6000, 15000],
    'n_estimators': [40,50,100]
}
rfcl = RandomForestClassifier()
grid_rf = GridSearchCV(estimator=rfcl, param_grid=param_grid, n_jobs=-1, pre_dispatch='2*n_jobs', cv=3, verbose=1, return_train_score=False)
grid_rf.fit(X_train, train_labels)
best_grid = grid_rf.best_estimator_
grid_rf.best_params_
# x=pd.DataFrame(best_grid.feature_importances_*100,index=X_train.columns).sort_values(by=0,ascending=False)
# plt.figure(figsize=(12,7))
# sns.barplot(x[0],x.index,palette='dark')
# plt.xlabel('Feature Importance in %')
# plt.ylabel('Features')
# plt.title('Feature Importance using RF')
# plt.show()
ytrain_predict = best_grid.predict(X_train)
ytest_predict = best_grid.predict(X_test)
# AUC and ROC for the training data

# predict probabilities
probs = best_grid.predict_proba(X_train)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(train_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(train_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()
# AUC and ROC for the test data

# predict probabilities
probs = best_grid.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()
# Model accuracy
accuracy_score(train_labels,ytrain_predict)
from sklearn.metrics import confusion_matrix,classification_report
sns.heatmap(confusion_matrix(test_labels,ytest_predict),annot=True, fmt='d', cbar=False,cmap='YlGnBu')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()
print(classification_report(train_labels,ytrain_predict))
print(classification_report(test_labels,ytest_predict))