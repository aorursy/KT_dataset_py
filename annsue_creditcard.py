import numpy as np



import pandas as pd

from pandas import Series,DataFrame



import matplotlib.pyplot as plt

%matplotlib inline



# from imblearn.over_sampling import SMOTE
credit = pd.read_csv('../input/creditcard.csv')

credit.head()
credit.shape
credit.info()
counts = credit['Class'].value_counts()

counts
plt.figure(figsize=(2*7,1*7))

ax = plt.subplot(1,2,1)

counts.plot(kind='pie',ax=ax,autopct='%.2f%%')



ax = plt.subplot(1,2,2)

counts.plot(kind='bar',ax=ax)
credit.tail()
credit['Time'] = credit['Time'].map(lambda x: divmod(x,3600)[0])

credit.tail()
# 研究V1的分布

cond0 = credit['Class'] ==0

cond1 = credit['Class'] ==1
credit['V1'][cond0]
v1_0 = credit.loc[cond0,'V1']
v1_1 = credit.loc[cond1,'V1']
# 画直方图，反映数据的分布

v1_0.plot(kind='hist',bins=500,density=True)

v1_1.plot(kind='hist',bins=50,density=True)
credit.columns
columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
plt.figure(figsize=(2*8,14*8))

for i,col in enumerate(columns):

    ax = plt.subplot(14,2,i+1)

    v_0 = credit.loc[cond0,col]

    v_1 = credit.loc[cond1,col]

    v_0.plot(kind='hist',bins=500,density=True)

    v_1.plot(kind='hist',bins=50,density=True)

    ax.set_title(col,fontdict=dict(fontsize=20,color='r'))

    
credit.drop(columns=['V8', 'V13', 'V15', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'],inplace=True)
credit.head()
credit['Amount'].min()
credit['Amount'].max()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
credit['Amount'] = scaler.fit_transform(credit[['Amount']])
credit['Amount'].min()
credit['Amount'].max()
credit['Amount'].std()
credit['Amount'].mean()
credit['Time'] = scaler.fit_transform(credit[['Time']])
credit.shape
from sklearn.ensemble import GradientBoostingClassifier
data = credit.iloc[:,0:-1].values

target = credit['Class'].values
gbdt = GradientBoostingClassifier()

%time gbdt.fit(data,target)
importances = gbdt.feature_importances_

importances
index = np.argsort(importances)[::-1]

importances[index]
credit.columns
columns = np.array(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11',

       'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'Amount'])
plt.figure(figsize=(10,8))

plt.bar(np.arange(0,18),importances[index])

plt.xticks(np.arange(0,18),columns[index])
credit.drop(columns=['V7','V5','V4','V19','V11','V1','Amount'],inplace=True)
credit.shape
from imblearn.over_sampling import SMOTE
smote = SMOTE()
data = credit.iloc[:,0:-1].values

target = credit['Class'].values
data_resampled,target_resampled = smote.fit_sample(data,target)
(target_resampled == 1).sum()
# 画图方法

# 绘制真实值和预测值对比情况

import itertools

def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > threshold else "black")#若对应格子上面的数量不超过阈值则，上面的字体为白色，为了方便查看



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(data_resampled,target_resampled,test_size=0.2)
logistic = LogisticRegression()

logistic.fit(X_train,y_train)
X_train.shape
logistic.score(X_train,y_train)
y_ = logistic.predict(X_test)
y_
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_).T

cm
pd.crosstab(index=y_,columns=y_test,margins=True,rownames=['预测值'],colnames=['真实值'])
recall = cm[1,1]/ (cm[0,1] + cm[1,1])

recall
plot_confusion_matrix(cm,[0,1])
from sklearn.model_selection import GridSearchCV
logistic = LogisticRegression()

param_grid={

    'C':[0.1,1,10],

    'penalty':['l1','l2'],

    'tol':[0.00001,0.0001,0.01]

}





gv = GridSearchCV(estimator=logistic,param_grid=param_grid,n_jobs=4)

%time gv.fit(X_train,y_train)
gv.best_params_
gv.best_score_
estimator = gv.best_estimator_
y_ = estimator.predict(X_test)
estimator.score(X_test,y_test)
plot_confusion_matrix(cm,[0,1])
cm = confusion_matrix(y_test,y_).T

plot_confusion_matrix(cm,[0,1])
# 逻辑回归本身就是一个优化的非常好的算法，优化空间不大
from sklearn.metrics import auc,roc_curve
thresholds = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

recalls = []

aucs = []

accuracies = []

precisions = []

for threshold in thresholds:

    logistic = LogisticRegression()

    logistic.fit(X_train,y_train)

    

#     返回概率

    y_proba = logistic.predict_proba(X_test)

    y_  = y_proba[:,1] >= threshold

    cm = confusion_matrix(y_test,y_).T

    fpr,tpr,_ = roc_curve(y_test,y_)

    auc_ = auc(fpr,tpr)

    aucs.append(auc_)

    recall = cm[1,1]/(cm[0,1]+cm[1,1])

    recalls.append(recall)

    accuracy= (cm[1,1]+cm[0,0])/cm.sum()

    accuracies.append(accuracy)

    precision = cm[1,1]/(cm[1,0]+cm[1,1])

    precisions.append(precision)
plt.plot(thresholds,recalls,label='recall')

plt.plot(thresholds,aucs,label='auc')

plt.plot(thresholds,accuracies,label='accuracy')

plt.plot(thresholds,precisions,label='precision')

plt.xticks(thresholds)

plt.grid()

plt.legend()
