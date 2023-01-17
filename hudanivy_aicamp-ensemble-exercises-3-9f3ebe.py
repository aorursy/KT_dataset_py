# necessary imports
%matplotlib inline
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

np.random.seed(19)
# TODO 读取肿瘤数据
data = pd.read_csv("../input/breastCancer.csv",sep=',')
# 打印部分数据
data.head()
# TODO
data.diagnosis.value_counts()
# TODO
data.diagnosis[data.diagnosis=='M']=1
data.diagnosis[data.diagnosis=='B']=-1
data.diagnosis.value_counts()
data.describe()
#drop掉unnamed 那一列，确保其他列没有缺失项
data.drop(columns=['id','Unnamed: 32'],inplace=True)
data.isnull().any()
# 柱状图
import seaborn as sns
plt.hist(data.area_se[data.diagnosis==1],alpha=0.5)
plt.hist(data.area_se[data.diagnosis==-1],alpha=0.5)
plt.hist(data.radius_mean[data.diagnosis==1],alpha=0.5)
plt.hist(data.radius_mean[data.diagnosis==-1],alpha=0.5)
plt.subplots(figsize=(20,15))

sns.heatmap(data.corr())
features=data.columns
features=features.drop(['diagnosis','area_mean','radius_worst','area_worst'])
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(data[features],data.iloc[:,0].astype('int'))
pd.DataFrame(dt.feature_importances_, columns=['importance'])
#pd.DataFrame(trainX.columns,columns=['variables'])
tmp=pd.concat([pd.DataFrame(data.columns,columns=['variables']),pd.DataFrame(dt.feature_importances_, columns=['importance'])],axis=1).sort_values(by='importance',ascending=False)[:10]
print(tmp)
# TODO
features = tmp.variables.ravel()
target =data.diagnosis
features
%matplotlib inline
i = 0
for feature in features:

    bins = 25
    # 将特征的直方图画出来
    # TODO
    plt.subplots(figsize=(15,10))
    plt.xlabel(feature)
    plt.ylabel('Amount of count')
    plt.hist(data[feature][data.diagnosis==1],bins=bins,alpha=0.5)
    plt.hist(data[feature][data.diagnosis==-1],bins=bins,alpha=0.5)
   # plt.legend()
    plt.show()

np.delete(features,[2,5])
from sklearn.model_selection import train_test_split
# TODO 训练集和测试集
train_data, test_data =train_test_split(data,test_size=0.3,random_state=2)
# TODO 数据和标签
trainX, trainY = train_data[features],train_data.iloc[:,0].astype('int')
testX, testY = test_data[features],test_data.iloc[:,0].astype('int')
# TODO logistic 模型的表现
logistic_model = LogisticRegression()
print("Logistic Regression performance: %f" % (cross_val_score(logistic_model, trainX, trainY, cv=8).mean()))
# TODO
tree_model = DecisionTreeClassifier(criterion='entropy')
print("Decision Tree performance: %f" % (cross_val_score(tree_model, trainX, trainY, cv=8).mean()))
ada_model = AdaBoostClassifier(n_estimators=100)
print("Adaboost performance: %f" % (cross_val_score(ada_model, trainX, trainY, cv=8).mean()))
logistic_model = LogisticRegression()
logistic_model.fit(trainX,trainY)
logistic_model.score(testX,testY)
# TODO
tree_model = DecisionTreeClassifier()
tree_model.fit(trainX,trainY)
tree_model.score(testX,testY)
# TODO
ada_model = AdaBoostClassifier(n_estimators=200)
ada_model.fit(trainX,trainY)
ada_model.score(testX,testY)
# TODO
from sklearn.base import BaseEstimator
class Adaboost(BaseEstimator):
    
    def __init__(self, M):
        # TODO
        self.M=M
    def fit(self, X, Y):
        # TODO
        self.models = []
        self.model_weights = []
        
        N, _ = X.shape
        alpha = np.array([1/N]*N)
        
        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=2,criterion='entropy')
            tree.fit(X, Y, sample_weight=alpha)
            prediction = tree.predict(X)
            
            # 计算加权错误
            weighted_error =alpha.dot(prediction!=Y)
            
            # 计算当前模型的权重
            model_weight = 0.5*(np.log(1-weighted_error)-np.log(weighted_error))
            
            # 更新数据的权重
            alpha =alpha*(np.exp(-model_weight*Y*prediction))
            
            # 数据权重normalize
            alpha = alpha/alpha.sum()
            
            self.models.append(tree)
            self.model_weights.append(model_weight)
            
    def predict(self, X):
        # TODO
        pred=np.zeros(len(X))
        for model, model_weight in zip(self.models, self.model_weights):
            pred += model.predict(X)*model_weight
         
        return np.sign(pred)

    
    def score(self, X, Y):
        # TODO
        pred=self.predict(X)
       
        return np.mean(pred==Y)
# TODO
adamodel = Adaboost(200)
print("Adaboost model performance: %f" % (cross_val_score(adamodel, trainX.as_matrix().astype(np.float64), trainY.as_matrix().astype(np.float64), cv=8).mean()))
adamodel.fit(testX.as_matrix().astype(np.float64), testY.as_matrix().astype(np.float64))
print("Adaboost model test performance: %f" % adamodel.score(testX.as_matrix().astype(np.float64), testY.as_matrix().astype(np.float64)))



