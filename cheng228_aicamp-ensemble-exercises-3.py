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
data_folder = "../input"
#data_folder = "data"
data = pd.read_csv(os.path.join(data_folder, "breastCancer.csv"))
# 打印部分数据
data.head()
# TODO
data['diagnosis'].value_counts()
# TODO
data.drop('id', axis=1, inplace=True)
data.drop('Unnamed: 32', axis = 1, inplace=True)
# TODO
data['diagnosis'] = data['diagnosis'].apply(lambda x: +1 if x=="M" else -1)
# TODO
data.describe()
data.info()
# 柱状图
import seaborn as sns
# TODO
sns.countplot(data['diagnosis'])
# TODO
features = data.columns[1:7]
target = 'diagnosis'
features
i = 0
for feature in features:

    bins = 25
    # 将特征的直方图画出来
    # TODO
    plt.hist(data[feature][data[target]==-1], bins=bins, color = 'lightblue', label= 'B-healthy', alpha=1)
    plt.hist(data[feature][data[target]==+1], bins=bins, color = 'k', label= 'M-bad', alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Amount of count')
    
    plt.legend()
    
    plt.show()
from sklearn.model_selection import train_test_split
# TODO 训练集和测试集
train_data, test_data = train_test_split(data, test_size = 0.3)
# TODO 数据和标签
trainX, trainY = train_data.iloc[:, 1:], train_data[target]
testX, testY = test_data.iloc[:, 1:], test_data[target]
# TODO logistic 模型的表现
logistic_model = LogisticRegression()
print("Logistic Regression performance: %f" % (cross_val_score(logistic_model, trainX, trainY, cv=8).mean()))
# TODO
tree_model = DecisionTreeClassifier()
print("Logistic Regression performance: %f" % (cross_val_score(tree_model, trainX, trainY, cv=8).mean()))
# TODO
ada_model = AdaBoostClassifier(n_estimators=200)
print("Logistic Regression performance: %f" % (cross_val_score(ada_model, trainX, trainY, cv=8).mean()))
logistic_model = LogisticRegression()
# TODO
logistic_model.fit(trainX, trainY)
print("Logistic Regression test performance: %f" % logistic_model.score(testX, testY))
tree_model = DecisionTreeClassifier()
# TODO
tree_model.fit(trainX, trainY)
print("Decision Tree test performance: %f" % tree_model.score(testX, testY))
ada_model = AdaBoostClassifier(n_estimators=200)
# TODO
ada_model.fit(trainX, trainY)
print("Adaboost test performance: %f" % ada_model.score(testX, testY))
from sklearn.base import BaseEstimator
class Adaboost(BaseEstimator):
    
    def __init__(self, M):
        # TODO
        self.M = M
        
    def fit(self, X, Y):
        # TODO
        self.models = []
        self.model_weights = []
        
        N, _ = X.shape
        alpha = np.ones(N) / N
        
        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=2)
            tree.fit(X, Y, sample_weight=alpha)
            prediction = tree.predict(X)
            
            # 计算加权错误
            weighted_error = alpha.dot(prediction != Y)
            
            # 计算当前模型的权重
            model_weight = 0.5 * (np.log(1- weighted_error) - np.log(weighted_error))
            
            # 更新数据的权重
            alpha *= np.exp(-model_weight * prediction * Y)
            
            # 数据权重normalize
            alpha = alpha/alpha.sum()
            
            self.models.append(tree)
            self.model_weights.append(model_weight)
            
    def predict(self, X):
        # TODO
        result = np.zeros(len(X))
        for model in self.models:
            result += model.predict(X)
        return np.sign(result)
    
    def score(self, X, Y):
        pred = self.predict(X)
        return (pred == Y).mean()
        # TODO
# TODO
adamodel = Adaboost(200)
print("Adaboost model performance: %f" % (cross_val_score(adamodel, trainX.as_matrix().astype(np.float64), trainY.as_matrix().astype(np.float64), cv=8).mean()))
adamodel.fit(trainX.as_matrix().astype(np.float64), trainY.as_matrix().astype(np.float64))
print("Adaboost model test performance: %f" % adamodel.score(testX.as_matrix().astype(np.float64), testY.as_matrix().astype(np.float64)))
