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
#from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

%matplotlib inline

import matplotlib.pyplot as plt  

import seaborn as sns
train = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/train.csv", header = None)

trainLabels = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/trainLabels.csv", header = None)

test = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/test.csv", header = None)
train.describe()
train.head()
train.dtypes
cor =train.corr()

cor
plt.subplots(figsize=(10,10))

sns.heatmap(cor)
X=train

y = trainLabels

x_train,x_val,y_train,y_val=train_test_split(X,y)
model = XGBClassifier()

model.fit(x_train,y_train)

y_pred = model.predict(x_val)

acc_model=accuracy_score(y_val, y_pred)*100

print(accuracy_score(y_val, y_pred)*100)
X_output = test



final_test = model.predict(X_output)

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model = LogisticRegression()

log_model.fit(x_train, y_train)

y_pred = log_model.predict(x_val)

acc_log=accuracy_score(y_pred, y_val) * 100

print(acc_log)
from sklearn.svm import SVC

svm_model =SVC()

svm_model.fit(x_train,y_train)

y_pred =svm_model.predict(x_val)

acc_svc =accuracy_score(y_pred,y_val)*100

print(acc_svc)
from sklearn.tree import DecisionTreeClassifier



decisiontree_model =DecisionTreeClassifier()

decisiontree_model.fit(x_train,y_train)

y_pred =decisiontree_model.predict(x_val)



acc_decisiontree_model=accuracy_score(y_pred, y_val)*100

print(acc_decisiontree_model)
from sklearn.ensemble import RandomForestClassifier



randomforest_model = RandomForestClassifier()

randomforest_model.fit(x_train, y_train)

y_pred = randomforest_model.predict(x_val)

acc_randomforest =accuracy_score(y_pred, y_val) * 100

print(acc_randomforest)
from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier()

knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_val)

acc_knn_model = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn_model)
from sklearn.ensemble import GradientBoostingClassifier



gbk_model = GradientBoostingClassifier()

gbk_model.fit(x_train, y_train)

y_pred = gbk_model.predict(x_val)

acc_gbk_model= accuracy_score(y_pred, y_val) * 100

print(acc_gbk_model)
compare =pd.DataFrame({

    'model':['XGBC','Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 

              'Decision Tree', 'Gradient Boosting Classifier'],

    'Score': [acc_model,acc_svc, acc_knn_model, acc_log, 

              acc_randomforest, acc_decisiontree_model,

               acc_gbk_model]

})
compare
X_output = test



final_test = svm_model.predict(X_output)
submission = pd.DataFrame(final_test)

print(submission.shape)

submission.columns = ['Solution']

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission
filename = 'output.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)