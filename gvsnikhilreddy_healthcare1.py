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
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve,auc,classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pylab import rcParams
import warnings
def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_predict)
    auc = roc_auc_score(y_test, y_predict)
    plt.plot(fpr,tpr,label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc=4)
    plt.show()
    pass
%matplotlib inline
rcParams['figure.figsize'] = 10, 6
warnings.filterwarnings('ignore')
sns.set(style="darkgrid")
data= pd.read_csv(r'../input/healthcare/Project 2/Healthcare - Diabetes/health care diabetes.csv')
data.head()
data.shape
data.describe() 
data.median()
data.columns
l=[]
for j in data.drop(['Outcome'],axis=1).columns:
   l.append(len([i for i in data[j] if i == 0]))
zero=np.array(l)
zero
fig = go.Figure(data=[go.Pie(labels=data.drop(['Outcome'],axis=1).columns, values=zero)])
fig.update_layout(title_text='missing values')
fig.show()
sns.barplot(x=data.drop(['Outcome'],axis=1).columns,y=zero)
fig = plt.gcf()
fig.set_size_inches(16, 8)
data.drop(['Outcome'],axis=1).hist()
fig = plt.gcf()
fig.set_size_inches(16, 8)
data.drop(['Outcome'],axis=1).hist()
fig = plt.gcf()
fig.set_size_inches(16, 8)
data=data.fillna(data.median())
data
for i in data.drop(['Outcome'],axis=1):
 sns.catplot(x=i, col='Outcome',
                data=data, kind="count")
outcome_1 = data[data['Outcome']==1]
outcome_0 = data[data['Outcome']==0]
print(outcome_1.shape,outcome_0.shape)
sns.catplot(x='Outcome',
                data=data, kind="count")
sns.pairplot(data,hue='Outcome')
data_corr=data.corr()
sns.heatmap(data_corr)
fig = plt.gcf()
fig.set_size_inches(16, 8)
x=data.drop(['Outcome'],axis=1)
y=data['Outcome']
scl=StandardScaler()
x_train, x_test,y_train,y_test =train_test_split(x,y,test_size=0.33,random_state=10)
x_train = scl.fit_transform(x_train)
x_test  =scl.fit_transform(x_test)
smk = SMOTE()
X_res,y_res=smk.fit_sample(x_train,y_train)
X_res.shape,y_res.shape
from collections import Counter
print('Original dataset shape {}'.format(Counter(y_train)))
print('Resampled dataset shape {}'.format(Counter(y_res)))
import xgboost
#xgboot
model_x=xgboost.XGBClassifier(max_depth =2,
                             subsample =0.8,
                             n_estimators =200,
                             learning_rate=0.05,
                             min_child_weight=2,
                             random_state=5
)
model_x.fit(x_train,y_train)
y_predict = model_x.predict(x_test)
y_predict_xroc =model_x.predict_proba(x_test)
print(classification_report(y_test,y_predict))
print(accuracy_score(y_test,y_predict)*100)
#knn
model_knn= KNeighborsClassifier()
model_knn.fit(X_res,y_res)
y_predict_knn = model_knn.predict(x_test)
y_predict_knnroc =model_knn.predict_proba(x_test)
print(classification_report(y_test,y_predict_knn))
print('knn',accuracy_score(y_test,y_predict_knn)*100)
model_svm=SVC()
model_svm.fit(X_res,y_res)
y_predict_svm = model_svm.predict(x_test)
y_predict_svmroc =model_svm.decision_function(x_test)
print(classification_report(y_test,y_predict_svm))
print(accuracy_score(y_test,y_predict_svm)*100)
model_log=LogisticRegression()
model_log.fit(X_res,y_res)
y_predict_log= model_log.predict(x_test)
y_logroc_predict =model_log.decision_function(x_test)
print(classification_report(y_test,y_predict_log))
print(accuracy_score(y_test,y_predict_log)*100)
log_fpr,log_tpr,t=roc_curve(y_test,y_logroc_predict )
auc_log=auc(log_fpr,log_tpr)
svm_fpr,svm_tpr,t=roc_curve(y_test,y_predict_svmroc)
auc_svm=auc(svm_fpr,svm_tpr)
x_fpr,x_tpr,t=roc_curve(y_test,y_predict_xroc[:,1])
auc_x=auc(x_fpr,x_tpr)
knn_fpr,knn_tpr,t=roc_curve(y_test,y_predict_knnroc[:,1] )
auc_knn=auc(knn_fpr,knn_tpr)
plt.plot(log_fpr,log_tpr,label="logical(auc=%0.3f)"%auc_log)
plt.plot(svm_fpr,svm_tpr,label="svm(auc=%0.3f)"%auc_svm)
plt.plot(knn_fpr,knn_tpr,label="knn(auc=%0.3f)"%auc_knn)
plt.plot(x_fpr,x_tpr,label="x(auc=%0.3f)"%auc_x)
plt.plot()
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.legend()


