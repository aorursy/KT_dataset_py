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
train=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/train_data.csv")
test=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/test_data.csv")
submission=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/sample_submission.csv")
train.head()
train.columns
train.info()
y = train['price_range']
x = train.drop('price_range', axis = 1)
y.unique()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size = 0.2, random_state = 101)
print(x_train.shape)
print(x_test.shape)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,12))
sns.heatmap(train.corr(),annot=True, cmap="GnBu")
plt.show()
from sklearn.linear_model import LogisticRegression
lr =LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr=lr.predict(x_test)
y_pred_lr
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import accuracy_score
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_lr)
confusion_matrix
print(metrics.classification_report(y_test, y_pred_lr))
acc_lr = metrics.accuracy_score(y_test, y_pred_lr)
acc_lr
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=101)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_dt)
confusion_matrix
print(metrics.classification_report(y_test, y_pred_dt))
acc_dt = metrics.accuracy_score(y_test, y_pred_dt)
acc_dt
from sklearn.ensemble import RandomForestClassifier
rf = DecisionTreeClassifier(random_state=0)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_rf)
confusion_matrix
print(metrics.classification_report(y_test, y_pred_rf))
acc_rf = metrics.accuracy_score(y_test, y_pred_rf)
acc_rf
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred_svc = svc.predict(x_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_svc)
confusion_matrix
print(metrics.classification_report(y_test, y_pred_svc))
acc_svc = metrics.accuracy_score(y_test, y_pred_svc)
acc_svc
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=3)  
model_knn.fit(x_train, y_train)
y_pred_knn = model_knn.predict(x_test)
print(metrics.confusion_matrix(y_test, y_pred_knn))
print(accuracy_score(y_test, y_pred_knn))
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors':np.arange(1,30)}
knn = KNeighborsClassifier()

model = GridSearchCV(knn, parameters, cv=5)
model.fit(x_train, y_train)
model.best_params_
knn = KNeighborsClassifier(n_neighbors=26)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_knn)
confusion_matrix
print(metrics.classification_report(y_test, y_pred_knn))
acc_knn = metrics.accuracy_score(y_test, y_pred_knn)
acc_knn
models = ['logistic regression', 'decision tree', 'random forest', 'support vector machine','knn']
acc_scores = [acc_lr,acc_dt,acc_rf,acc_svc,acc_knn]
print(acc_scores)
plt.bar(models, acc_scores, color=['lightblue', 'pink', 'lightgrey', 'cyan','lightgreen'])
plt.ylabel("accuracy scores")
plt.title("Which model is the most accurate?")
plt.show()
predicted_price=svc.predict(test)
predicted_price
test['price_range']=predicted_price
data={'Id':test['id'],'price_range':predicted_price}
result=pd.DataFrame(data)
result.to_csv('/kaggle/working/prediction.csv',index=False)