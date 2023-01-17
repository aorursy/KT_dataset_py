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
# loading data
data = pd.read_csv('/kaggle/input/breastcancer.csv')
data.head()
# data information
data.info()
# checking for missing values
data.isna().sum()
# to check number of attributes
data.columns
# drop unneccesary attributes
data.drop(columns= ['id','Unnamed: 32'], axis = 1 , inplace = True)
data.columns
import matplotlib.pyplot as plt
import seaborn as sns
# checking data distributions
data.select_dtypes(include= np.float).hist(figsize = (15,15))
plt.tight_layout()
plt.show()
# pairplot for scatter and hists based on diagnosis
sns.pairplot(data, hue = 'diagnosis')
plt.tight_layout()
plt.show()
# checking for class balance with bar chart
data['diagnosis'].value_counts().plot(kind = 'bar')
plt.show()
# checking unique values in target class
data['diagnosis'].unique()
# converting target class into binary values 
data['diagnosis'] = data['diagnosis'].apply(lambda x : 1 if x == 'M' else 0)
data['diagnosis'].unique()
# split the data into X and y
X = data.iloc[:,1:].values
y = data.iloc[:, 0].values
# train and test split of data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)
# scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# model building and necessary packages 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier
# calling all classification models
classication_lr = LogisticRegression()
classication_lr.fit(X_train,y_train)

classication_dt = DecisionTreeClassifier()
classication_dt.fit(X_train,y_train)

classication_rf = RandomForestClassifier()
classication_rf.fit(X_train,y_train)

classication_svc = SVC()
classication_svc.fit(X_train,y_train)

classication_knn = KNeighborsClassifier()
classication_knn.fit(X_train,y_train)

classication_xgb = XGBClassifier()
classication_xgb.fit(X_train,y_train)

classication_xgbrf = XGBRFClassifier()
classication_xgbrf.fit(X_train,y_train)
# each classification model score
print('training score : {}, test score : {}'.format(classication_lr.score(X_train, y_train), classication_lr.score(X_test, y_test)))
print('training score : {}, test score : {}'.format(classication_dt.score(X_train, y_train), classication_dt.score(X_test, y_test)))
print('training score : {}, test score : {}'.format(classication_rf.score(X_train, y_train), classication_rf.score(X_test, y_test)))
print('training score : {}, test score : {}'.format(classication_svc.score(X_train, y_train), classication_svc.score(X_test, y_test)))
print('training score : {}, test score : {}'.format(classication_knn.score(X_train, y_train), classication_knn.score(X_test, y_test)))
print('training score : {}, test score : {}'.format(classication_xgb.score(X_train, y_train), classication_xgb.score(X_test, y_test)))
print('training score : {}, test score : {}'.format(classication_xgbrf.score(X_train, y_train), classication_xgbrf.score(X_test, y_test)))

# getting predictions for all models 
y_pred_lr = classication_lr.predict(X_test)
y_pred_dt = classication_dt.predict(X_test)
y_pred_rf = classication_rf.predict(X_test)
y_pred_svc = classication_svc.predict(X_test)
y_pred_knn = classication_knn.predict(X_test)
y_pred_xgb = classication_xgb.predict(X_test)
y_pred_xgbrf = classication_xgbrf.predict(X_test)

# error metrics for classification problem
from sklearn.metrics import confusion_matrix, classification_report
cm_lr = confusion_matrix(y_test, y_pred_lr)
print('logistic reg: {}'.format(cm_lr), '\n')

cm_dt = confusion_matrix(y_test, y_pred_dt)
print('decison tree classif: {}'.format(cm_dt), '\n')

cm_rf = confusion_matrix(y_test, y_pred_rf)
print('randomforest classif: {}'.format(cm_rf), '\n')

cm_svc = confusion_matrix(y_test, y_pred_svc)
print('svc classif: {}'.format(cm_svc), '\n')

cm_knn = confusion_matrix(y_test, y_pred_knn)
print('knn classif: {}'.format(cm_knn), '\n')

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print('xgboost tree classif: {}'.format(cm_xgb), '\n')

cm_xgbrf = confusion_matrix(y_test, y_pred_xgbrf)
print('xgboost randomforest classif: {}'.format(cm_xgbrf))
# classification report by each model
print('---------- logistic reg', '\n')
print(classification_report(y_test, y_pred_lr),'\n')
print('---------- decison tree', '\n')
print(classification_report(y_test, y_pred_dt),'\n')
print('---------- randomforest', '\n')
print(classification_report(y_test, y_pred_rf),'\n')
print('---------- svc', '\n')
print(classification_report(y_test, y_pred_svc),'\n')
print('---------- knn', '\n')
print(classification_report(y_test, y_pred_knn),'\n')
print('---------- xgboost', '\n')
print(classification_report(y_test, y_pred_xgb),'\n')
print('---------- xgboost with randomforest', '\n')
print(classification_report(y_test, y_pred_xgbrf))
