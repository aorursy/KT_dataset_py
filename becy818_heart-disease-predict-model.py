# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, KFold

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, f1_score, classification_report, roc_curve, auc, confusion_matrix

from sklearn.preprocessing import MinMaxScaler, StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
## 载入数据。

data = pd.read_csv("../input/heart.csv")

data.head(10)
## 将sex中的0 赋值成 female；1 赋值成 male

##data['sex'][data['sex'] == 0] = 'female'

##data['sex'][data['sex']==1] = 'male'
## 获取sex中不同取值的统计个数

data['sex'].value_counts()
## 展示所有数值类型字段的统计信息 均值 方差 均方差 分位数等

data.describe()
## 数据集准备及预处理

print("Before drop target : {}".format(data.shape))

y = data['target']

y.value_counts() # 展示数据的统计值

data.drop(columns=['target'], inplace=True)

print("After drop target : {}".format(data.shape))

'''data_dummies = pd.get_dummies(data) ## one_hot 编码 ，虚拟变量.只对分类变量有效，对数值型连续变量无效

x = data_dummies.values

print("Features after get_dummies: \n", list(data_dummies.columns))

'''

x = data.values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)



## 数据缩放

##scaler = MinMaxScaler()

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
## 梯度提升回归树  GBRT

gbrt = GradientBoostingClassifier(random_state=0)



## 网格搜索算法查询最优参数（n_estimators 树个数；learning_rate 学习率； max_depth 树深度 ）

params_gbrt = [{

    'n_estimators':[30,50,40,20],

    'learning_rate':[0.01,0.1,1,0.001],

    'max_depth':[5,2,3,4]

}]



gbrt_grid = GridSearchCV(gbrt, params_gbrt, cv=6)

gbrt_grid.fit(X_train, y_train)

#gbrt.fit(X_train, y_train)

print("Best cross-validation score: {:.2f}".format(gbrt_grid.best_score_))

print("Best parameters: ", gbrt_grid.best_params_)

print("Accuracy on training set:{:.3f}".format(gbrt_grid.score(X_train, y_train)))

print("Accuracy on test set:{:.3f}".format(gbrt_grid.score(X_test, y_test)))
## 使用gbrt的最优参数构建模型 并评估模型性能

best_params_gbrt = gbrt_grid.best_params_

gbrt_model = GradientBoostingClassifier(**best_params_gbrt)

gbrt_model.fit(X_train, y_train)

print("Accuracy on test set:{:.3f}".format(gbrt_model.score(X_test, y_test)))



gbrt_pred = gbrt_model.predict(X_test)

## f1 score

print("f1 score of GBRT: {:.2f}".format(f1_score(y_test,gbrt_pred)))

## 模型评估报告

print("Classification report of GBRT: \n{}".format(classification_report(y_test, gbrt_pred, 

                                        target_names=["no disease", "disease"])))

gbrt_proba = gbrt_model.predict_proba(X_test)

## 计算GBRT的准确度和召回率

precision_gbrt, recall_gbrt,thresholds_gbrt = precision_recall_curve(y_test, gbrt_proba[:,1])

plt.plot(precision_gbrt, recall_gbrt, label="GBRT")

plt.xlabel("Precision")

plt.ylabel("Recall")

plt.legend(loc=1)

plt.show()
from sklearn.metrics import confusion_matrix

## 计算混淆矩阵

confusion_gbrt = confusion_matrix(y_test, gbrt_pred)

print("Confusion matrix of GBRT:\n {}".format(confusion_gbrt))
## 随机森林算法建模

rf = RandomForestClassifier()



rf.fit(X_train, y_train)

print("Accuracy of random forest model is {:.2f}".format(rf.score(X_test, y_test)))

rf.feature_importances_

proba = rf.predict_proba(X_test)

#log_proba = rf.predict_log_proba(X_test)

print(rf.predict(X_test))

pred_rf = rf.predict(X_test)

print((pred_rf == y_test).sum() / y_test.size)



## 计算随机森林的准确度和召回率

precision_rf, recall_rf,thresholds_rf = precision_recall_curve(y_test, proba[:,1])

plt.plot(precision_rf, recall_rf, label="rf")

plt.xlabel("precision")

plt.ylabel("recall")

plt.legend(loc=1)



## f1 score

print("f1 score : {:.2f}".format(f1_score(y_test,pred_rf)))

## 模型评估报告

print("classification report: \n{}".format(classification_report(y_test, pred_rf, 

                                        target_names=["no disease", "disease"])))
## 使用网格搜索 查找最优参数

param_grid = [{'n_estimators':[20,60,100],

               'max_features':[4,6,8,10],

               'max_depth':[6,10,20,30]}]

#kfold = KFold(n_splits=4, shuffle=True, random_state=66)

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5) ## cv=kfold

grid_search.fit(X_train, y_train)

print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

print("Best parameters: ", grid_search.best_params_)



print("Test score: {:.2f}".format(grid_search.score(X_test, y_test)))



proba_grid = grid_search.predict_proba(X_test)

## 计算随机森林的准确度和召回率

precision_rf, recall_rf,thresholds_rf = precision_recall_curve(y_test, proba_grid[:,1])

plt.plot(precision_rf, recall_rf, label="rf")

plt.xlabel("precision")

plt.ylabel("recall")

plt.legend(loc=1)

pred_grid = grid_search.predict(X_test)

## f1 score

print("f1 score : {:.2f}".format(f1_score(y_test,pred_grid)))

## 模型评估报告

print("classification report: \n{}".format(classification_report(y_test, pred_grid, 

                                        target_names=["no disease", "disease"])))
# shape属性 返回一个元组 记录数据的行数 列数 

data.shape

# 取数据的行数 shape[0]

# 取数据的列数（即数据特征个数） shape[1]



##print("数据的")

##data.columns
## 计算特征重要性

'''

data 是DataFrame 类型的源数据

model 是模型实例

'''

def plot_feature_importances(data, model):

    n_features = data.shape[1]

    plt.barh(range(n_features),model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), data.columns)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

params = grid_search.best_params_ ##使用网格搜索的最优参数

model = RandomForestClassifier().set_params(**params)

model.fit(X_train, y_train)

pred_grid = grid_search.predict(X_test)

## f1 score

print("f1 score : {:.2f}".format(f1_score(y_test,pred_grid)))

## 模型评估报告

print("classification report: \n{}".format(classification_report(y_test, pred_grid, 

                                        target_names=["no disease", "disease"])))

plot_feature_importances(data, model)
##绘制ROC曲线 计算AUC

from sklearn.metrics import roc_curve,auc,roc_auc_score

import numpy as np



proba_rf =  model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, proba_rf)



#model.predict_proba(X_test)[:,1]

#y_test.values

fig, ax = plt.subplots()

#ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".4")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for heart disease classifier of Random Forest')



plt.plot(fpr, tpr, label='ROC Curve')

plt.xlabel("FPR")

plt.ylabel("TPR(recall)")

plt.legend(loc=4)

plt.grid(True)

close_default_rf = np.argmin(np.abs(thresholds - 0.5))

plt.plot(fpr[close_default_rf], tpr[close_default_rf], 'o',markersize=10, 

         label='Thresholds 0.5 RF', fillstyle="none", c='k',mew=2)

plt.legend(loc=4)

rf_auc = roc_auc_score(y_test,  proba_rf)

print("AUC of Random Forest: {:.3f}".format(rf_auc))

## print(auc(fpr, tpr)) ## 使用auc函数亦可得到roc曲线下面积