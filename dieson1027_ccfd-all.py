# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#数据加载

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import time

import xgboost as xgb



data = pd.read_csv("../input/creditcard.csv")
data.head()
data.describe()
#热力图

correlation_matrix = data.corr()

fig = plt.figure(figsize=(12,9))

plt.title("The Heatmap of CreditCard Data",fontsize=16)

sns.heatmap(correlation_matrix,vmax=0.8,square = True)

plt.show()
#数据来源

'''

数据集包含了2013年9月欧洲持卡人通过信用卡进行的交易。这个数据集显示了两天内发生的交易，

在284,807个交易中，我们有492个欺诈。数据集高度不平衡，阳性类(舞弊)占所有事务的0.172%。

'''

#数据非欺诈与欺诈类分布

bar_x = [0,1]

bar_y = [data['Class'][data['Class']==0].count(),data['Class'][data['Class']==1].count()]

bar_color = ['blue','red']

x_label = ['Normal','Fraud']

plt.xticks(bar_x,x_label)

plt.text(bar_x[0],bar_y[0],bar_y[0],ha='center',va='bottom',fontsize=10)

plt.text(bar_x[1],bar_y[1],bar_y[1],ha='center',va='bottom',fontsize=10)

plt.title("Normal and Fraud",fontsize=16)

plt.ylim(0,300000)

plt.grid()

plt.bar(bar_x,bar_y,color=bar_color,alpha=0.8)

plt.show()



print("非欺诈数据有",bar_y[0],"个，占总数据的",round(bar_y[0]/(sum(bar_y))*100.0,2),"%")

print("欺诈数据有",bar_y[1],"个，占总数据的",round(bar_y[1]/(sum(bar_y))*100.0,2),"%")
#Amount的分布

fig,axis = plt.subplots(1,1,figsize=(8,3))



plt.ylabel('Frequency', fontsize=14)

plt.xlabel('Amount', fontsize=14)

sns.distplot(data['Amount'].values, ax=axis, color='b',axlabel="Amount")

axis.set_title('Distribution of Transaction Amount', fontsize=16)

axis.set_xlim([min(data['Amount'].values), max(data['Amount'].values)/2])
#Amount和Time的分布

fig,axis = plt.subplots(1,1,figsize=(8,3))



plt.ylabel('Frequency', fontsize=14)

plt.xlabel('Time', fontsize=14)

sns.distplot(data['Time'].values, ax=axis, color='r',label='Time')

axis.set_title('Distribution of Transaction Time', fontsize=16)

axis.set_xlim([min(data['Time'].values), max(data['Time'].values)])

from sklearn.preprocessing import RobustScaler



#归一化数据，去除time列

rb_scaler = RobustScaler()

data['scaled_amount'] = rb_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

#data['scaled_time'] = rb_scaler.fit_transform(data['Time'].values.reshape(-1,1))

data.drop(labels=['Time','Amount'],axis=1,inplace=True)



#分割xy数据集

X = data.drop(['Class'], axis=1)

y = data['Class']
#数据分割

from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=20190506)

#转化为numpy类型



X_train = Xtrain.values

X_test = Xtest.values

y_train = ytrain.values

y_test = ytest.values



# See if both the train and test label distribution are similarly distributed

train_unique_label, train_counts_label = np.unique(y_train, return_counts=True)

test_unique_label, test_counts_label = np.unique(y_test, return_counts=True)

print('-' * 100)



#检查数据量比例

print('Label Distributions: \n')

print(train_counts_label/ len(y_train))

print(test_counts_label/ len(y_test))
#SMOTE

from imblearn.over_sampling import SMOTE

from collections import Counter

X_train_smote, y_train_smote = SMOTE().fit_sample(X_train,y_train)

 

sorted(Counter(y_train_smote).items())
#oversampling+smote+逻辑回归

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV,StratifiedShuffleSplit

from imblearn.pipeline import make_pipeline

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score,roc_auc_score

from imblearn.over_sampling import SMOTE



# List to append the score and then find the average

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []



sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],"solver":['liblinear']}

rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4,cv=5)



count = 1

t1 = time.time()

for train, test in sss.split(X_train, y_train):

    print("第",count,"轮")

    count+=1

    pipeline = make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..

    model = pipeline.fit(X_train[train], y_train[train])

    log_reg_est = rand_log_reg.best_estimator_

    prediction = log_reg_est.predict(X_train[test])

    

    accuracy_lst.append(pipeline.score(X_train[test], y_train[test]))

    precision_lst.append(precision_score(y_train[test], prediction))

    recall_lst.append(recall_score(y_train[test], prediction))

    f1_lst.append(f1_score(y_train[test], prediction))

    auc_lst.append(roc_auc_score(y_train[test], prediction))

t2 = time.time()



print("Run Time:",round((t2-t1)/60),"min",round((t2-t1)%60),"s")

print('---' * 45)

print('')

print("accuracy: {}".format(np.mean(accuracy_lst)))

print("precision: {}".format(np.mean(precision_lst)))

print("recall: {}".format(np.mean(recall_lst)))

print("f1: {}".format(np.mean(f1_lst)))

print("roc_auc: {}".format(np.mean(auc_lst)))

print('---' * 45)
from sklearn.metrics import classification_report

labels = ['No Fraud', 'Fraud']

print("训练集")

smote_prediction = log_reg_est.predict(X_train)

print(classification_report(y_train, smote_prediction, target_names=labels))



print("测试集")

smote_prediction = log_reg_est.predict(X_test)

print(classification_report(y_test, smote_prediction, target_names=labels))
#heatmap

from sklearn.metrics import confusion_matrix

fig, ax = plt.subplots(1, 1,figsize=(8,5))

log_reg_cf = confusion_matrix(y_test, smote_prediction)

sns.heatmap(log_reg_cf, annot=True, cmap=plt.cm.copper)

ax.set_title("Logistic Regression \n Confusion Matrix", fontsize=16)

ax.set_xticklabels(['', ''], fontsize=14, rotation=90)

ax.set_yticklabels(['', ''], fontsize=14, rotation=360)
#ROC曲线

from sklearn import metrics

# 计算欺诈交易的概率值，用于生成ROC曲线的数据

y_score = log_reg_est.decision_function(X_test)

fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)

# 计算AUC的值

roc_auc = metrics.auc(fpr,tpr)



# 绘制面积图

plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')

# 添加边际线

plt.plot(fpr, tpr, color='black', lw = 1)

# 添加对角线

plt.plot([0,1],[0,1], color = 'red', linestyle = '--')

# 添加文本信息

plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)

plt.title("Logistic Regression ROC curve ",fontsize=16)

# 添加x轴与y轴标签

plt.xlabel('1-Specificity')

plt.ylabel('Sensitivity')

# 显示图形

plt.show()
#保存ROC曲线变量

fpr_lr,tpr_lr = fpr,tpr

roc_auc_lr = roc_auc
#PR曲线

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

y_score = log_reg_est.decision_function(X_test)

average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))



precision, recall, _ = precision_recall_curve(y_test, y_score)



fig = plt.figure(figsize=(12,6))

plt.step(recall, precision, color='b', alpha=0.2,where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,color='#078B00')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('OverSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(

          average_precision), fontsize=16)
#保存PR曲线变量

y_score_lr = y_score

average_precision_lr = average_precision

precision_lr, recall_lr = precision, recall
import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy
n_inputs = X_train.shape[1]



ann_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(22, activation='relu'),

    Dense(2, activation='softmax')

])



ann_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

ann_model.fit(X_train, y_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)
from sklearn.metrics import classification_report,f1_score,confusion_matrix



y_pred = ann_model.predict_classes(X_test, batch_size=200, verbose=0)

labels = ['No Fraud', 'Fraud']

print(classification_report(y_test, y_pred, target_names=labels))

print("F1-Score",f1_score(y_test,y_pred))



fig, ax = plt.subplots(1, 1,figsize=(8,5))

ann_cf = confusion_matrix(y_test, y_pred)

sns.heatmap(ann_cf,ax=ax,annot=True, cmap=plt.cm.copper)



ax.set_title(" ANN \n Confusion Matrix", fontsize=16)

ax.set_xticklabels(['', ''], fontsize=14, rotation=90)

ax.set_yticklabels(['', ''], fontsize=14, rotation=360)
#ROC

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve,auc

print("roc_auc_score:",roc_auc_score(y_test,y_pred))

y_score = ann_model.predict(X_test)[:,1]

fpr,tpr,thresholds=roc_curve(y_test,y_score)

# 计算AUC的值

roc_auc = auc(fpr,tpr)



# 绘制面积图

plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')

# 添加边际线

plt.plot(fpr, tpr, color='black', lw = 1)

# 添加对角线

plt.plot([0,1],[0,1], color = 'red', linestyle = '--')

# 添加文本信息

plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)

plt.title("ANN ROC curve ",fontsize=16)

# 添加x轴与y轴标签

plt.xlabel('1-Specificity')

plt.ylabel('Sensitivity')

# 显示图形

plt.show()
#保存ROC曲线变量

fpr_ann,tpr_ann = fpr,tpr

roc_auc_ann = roc_auc
#PR曲线

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

y_score = ann_model.predict(X_test)[:,1]

average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))



precision, recall, _ = precision_recall_curve(y_test, y_score)



fig = plt.figure(figsize=(12,6))

plt.step(recall, precision, color='b', alpha=0.2,where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,color='#078B00')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('OverSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(

          average_precision), fontsize=16)
#保存PR曲线变量

y_score_ann = y_score

average_precision_ann = average_precision

precision_ann, recall_ann = precision, recall
from sklearn.model_selection import StratifiedShuffleSplit

from imblearn.pipeline import make_pipeline

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score,roc_auc_score

# List to append the score and then find the average

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []



sss = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=41)



xgb_model = xgb.XGBClassifier(max_depth=3, n_estimators=200, learn_rate=0.01)

t1 = time.time()

count = 1

labels = ['No Fraud', 'Fraud']

for train, test in sss.split(X_train, y_train):

    print(count,"-----")

    count += 1

    xgb_model.fit(X_train[train], y_train[train])  

    test_score = xgb_model.score(X_test, y_test)

    y_test_pred = xgb_model.predict(X_train[test])

    print(classification_report(y_train[test], y_test_pred, target_names=labels))

    

    accuracy_lst.append(xgb_model.score(X_train[test], y_train[test]))

    precision_lst.append(precision_score(y_train[test], y_test_pred))

    recall_lst.append(recall_score(y_train[test], y_test_pred))

    f1_lst.append(f1_score(y_train[test], y_test_pred))

    auc_lst.append(roc_auc_score(y_train[test], y_test_pred))



t2= time.time()

print("Run Time:",round((t2-t1)/60),"min",round((t2-t1)%60),"s")

print('---' * 45)

print('')

print("accuracy: {}".format(np.mean(accuracy_lst)))

print("precision: {}".format(np.mean(precision_lst)))

print("recall: {}".format(np.mean(recall_lst)))

print("f1: {}".format(np.mean(f1_lst)))

print("roc_auc: {}".format(np.mean(auc_lst)))

print('---' * 45)
y_pred = xgb_model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=labels))

print("F1-Score",f1_score(y_test,y_pred))
#heatmap

fig, ax = plt.subplots(1, 1,figsize=(8,5))

xgb_cf = confusion_matrix(y_test, y_pred)





sns.heatmap(xgb_cf,ax=ax,annot=True, cmap=plt.cm.copper)

ax.set_title(" XGBoost \n Confusion Matrix", fontsize=16)

ax.set_xticklabels(['', ''], fontsize=14, rotation=90)

ax.set_yticklabels(['', ''], fontsize=14, rotation=360)
#ROC曲线

from sklearn import metrics

# 计算欺诈交易的概率值，用于生成ROC曲线的数据

y_score = xgb_model.predict_proba(np.array(X_test))[:,1]

fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)

# 计算AUC的值

roc_auc = metrics.auc(fpr,tpr)



# 绘制面积图

plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')

# 添加边际线

plt.plot(fpr, tpr, color='black', lw = 1)

# 添加对角线

plt.plot([0,1],[0,1], color = 'red', linestyle = '--')

# 添加文本信息

plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)

plt.title("XGBoost ROC curve ",fontsize=16)

# 添加x轴与y轴标签

plt.xlabel('1-Specificity')

plt.ylabel('Sensitivity')

# 显示图形

plt.show()
#保存ROC曲线变量

fpr_xgb,tpr_xgb = fpr,tpr

roc_auc_xgb = roc_auc
#PR曲线

y_score = xgb_model.predict_proba(np.array(X_test))[:,1]

average_precision = metrics.average_precision_score(y_test, y_score)

precision, recall,threshold = metrics.precision_recall_curve(y_test, y_score)



plt.step(recall, precision, color='b', alpha=0.2,where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,color='#078B00')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title(' Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(

          average_precision), fontsize=16)
#保存PR曲线变量

y_score_xgb = y_score

average_precision_xgb = average_precision

precision_xgb, recall_xgb = precision, recall
fig, ax = plt.subplots(3,1 ,figsize=(8,20))



sns.heatmap(log_reg_cf, ax=ax[0], annot=True, cmap=plt.cm.copper)

ax[0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)

ax[0].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(ann_cf,ax=ax[1],annot=True, cmap=plt.cm.copper)

ax[1].set_title(" ANN \n Confusion Matrix", fontsize=14)

ax[1].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[1].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(xgb_cf,ax=ax[2],annot=True, cmap=plt.cm.copper)

ax[2].set_title(" XGBoost \n Confusion Matrix", fontsize=14)

ax[2].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[2].set_yticklabels(['', ''], fontsize=14, rotation=360)

#fig.savefig("test.png")
print(log_reg_cf,'\n',ann_cf,'\n',xgb_cf)
#ROC曲线



#LR

#fpr_lr,tpr_lr

#roc_auc_lr

#ann

#fpr_ann,tpr_ann

#roc_auc_ann

#xgb

#fpr_xgb,tpr_xgb

#roc_auc_xgb



error = 0

fig = plt.figure(figsize=(10,8))

# 绘制面积图

#plt.stackplot(fpr_lr, tpr_lr, color='red', alpha = 0.5, edgecolor = 'black')

#plt.stackplot(fpr_ann, tpr_ann, color='blue', alpha = 0.5, edgecolor = 'black')

#plt.stackplot(fpr_xgb, tpr_xgb, color='green', alpha = 0.5, edgecolor = 'black')

# 添加边际线

l1 = plt.plot(fpr_lr, tpr_lr, color='red', lw = 1)

l2 = plt.plot(fpr_ann, tpr_ann, color='blue', lw = 1,linestyle = '--')

l3 = plt.plot(fpr_xgb, tpr_xgb, color='green', lw = 1,linestyle = '-.')

# 添加对角线

plt.plot([0,1],[0,1], color = 'black', linestyle = '--')

# 添加文本信息

#plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % error)

plt.title("ROC curve ",fontsize=24)

# 添加x轴与y轴标签

plt.xlabel('1-Specificity',fontsize=20)

plt.ylabel('Sensitivity',fontsize=20)

#图例

str_roc_auc = "-AUROC:"

line_label = ["LR"+str_roc_auc+str(round(roc_auc_lr,4)),"ANN"+str_roc_auc+str(round(roc_auc_ann,4)),"XGB"+str_roc_auc+str(round(roc_auc_xgb,4))]

plt.legend([l1,l2,l3],labels = line_label,fontsize=15)

# 显示图形

plt.show()
'''

y_score_xgb

average_precision_xgb

precision_xgb, recall_xgb

'''

#PR曲线

plt.figure(figsize=(10,8))

l1 = plt.plot(recall_lr, precision_lr, color='red', lw = 1)

l2 = plt.plot(recall_ann, precision_ann, color='blue', lw = 1,linestyle = '--')

l3 = plt.plot(recall_xgb, precision_xgb, color='green', lw = 1,linestyle = '-.')



plt.xlabel('Recall',fontsize=20)

plt.ylabel('Precision',fontsize=20)

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title(' Precision-Recall curve', fontsize=24)

str_pr_auc =  "-AUPRC:"

line_label = ["LR"+str_pr_auc+str(round(average_precision_lr,4)),"ANN"+str_pr_auc+str(round(average_precision_ann,4)),"XGB"+str_pr_auc+str(round(average_precision_xgb,4))]

plt.legend([l1,l2,l3],labels = line_label,fontsize=15)