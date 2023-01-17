import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
# Loading train data and test data into Pandas.
train = pd.read_csv("../input/train.csv", header=0)
test = pd.read_csv("../input/test.csv", header=0)
#训练集D0
del_col=[0,55]                                   #删除Id列和Cover_Type列
train_data=train.drop(train.columns[del_col],axis=1,inplace=False)
train_label=train['Cover_Type']
del_col.pop()
test_data=test.drop(test.columns[del_col],axis=1,inplace=False)
#训练集D1
from sklearn.decomposition import PCA
pca = PCA(n_components= 25 )
pca.fit(train_data)
train_data_pca = pca.transform(train_data)
print(train_data_pca.shape)
test_data_pca = pca.transform(test_data)
#训练集D2
del_col = range(14,54)
train_data_new = train_data.drop(train_data.columns[del_col],axis=1,inplace=False)
test_data_new = test_data.drop(test_data.columns[del_col],axis=1,inplace=False)
#设定随机森林分类模型
from sklearn import ensemble
#在原始数据上处理
clf1 = ensemble.RandomForestClassifier(100) #设定包含100个决策树
clf1.fit(train_data, train_label)           #拟合模型
sub = pd.DataFrame({"Id": test['Id'],"Cover_Type": clf1.predict(test_data)})  #预测
sub.to_csv("result_1.csv", index=False)


#支持向量机分类器
from sklearn.svm import SVC
clf2 = SVC()

clf2.fit(train_data, train_label)           #拟合模型
sub = pd.DataFrame({"Id": test['Id'],"Cover_Type": clf2.predict(test_data)})  #预测
sub.to_csv("result_2_1.csv", index=False)

clf2.fit(train_data_pca, train_label)           #拟合模型
sub = pd.DataFrame({"Id": test['Id'],"Cover_Type": clf2.predict(test_data_pca)})  #预测
sub.to_csv("result_2_2.csv", index=False)

clf2.fit(train_data_new, train_label)           #拟合模型
sub = pd.DataFrame({"Id": test['Id'],"Cover_Type": clf2.predict(test_data_new)})  #预测
sub.to_csv("result_2_3.csv", index=False)

