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

# after set seed, 每次产生的random num 一样，这样方便比较
np.random.seed(19)
data_folder = "../input"
#data_folder = "data"
data = pd.read_csv(os.path.join(data_folder, "breastCancer.csv"))
data.head()
data['diagnosis'].value_counts() # data['column'].value_counts() 打印出每种类别的个数
data.drop('id',axis=1,inplace=True) # remove 'id' column
data.head()
# last column is: 'Unnamed: 32', delete last column
data.drop('Unnamed: 32',axis=1,inplace=True)
data.head()
data['diagnosis'] = data['diagnosis'].apply(lambda x : +1 if x=='M' else -1)
data.describe()
data.info() # 每一行的not-null 数据 表示，那个feature有多少个non-null 数据
# seaborn.countplot column class
import seaborn as sns
sns.countplot(data['diagnosis'])
# 1,2,3,4,5,6
features = data.columns[1:7]
target = 'diagnosis'
features
i = 0
for feature in features:

    bins = 25
    # 每种类别一个颜色
    plt.hist(data[feature][data[target] == -1], bins=bins, color='lightblue', label= 'B-healthy', alpha=1)
    plt.hist(data[feature][data[target] == 1], bins=bins, color='k', label='M-bad', alpha=0.5)
    
    plt.xlabel(feature)
    plt.ylabel('Amount of count')
    
    plt.legend()
    
    plt.show()
data.head()
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.3)
train_data.head()
# 上面定义过了：target = 'diagnosis'
trainX, trainY = train_data[data.columns[1:]], train_data[target]
testX, testY = test_data[data.columns[1:]], test_data[target]
logistic_model = LogisticRegression()
print("Logistic Regression performance: %f" % (cross_val_score(logistic_model, trainX, trainY, cv=8).mean()))
tree_model = DecisionTreeClassifier()
print("Decision Tree performance: %f" % (cross_val_score(tree_model, trainX, trainY, cv=8).mean()))
ada_model = AdaBoostClassifier(n_estimators=200)
print("Decision Tree performance: %f" % (cross_val_score(ada_model, trainX, trainY, cv=8).mean()))
logistic_model = LogisticRegression()
logistic_model.fit(trainX, trainY)
print("Logistic Regression test performance: %f" % logistic_model.score(testX, testY))
tree_model = DecisionTreeClassifier()
tree_model.fit(trainX, trainY)
print("Decision Tree test performance: %f" % tree_model.score(testX, testY))
ada_model = AdaBoostClassifier(n_estimators=200)
ada_model.fit(trainX, trainY)
print("Adaboost test performance: %f" % ada_model.score(testX, testY))
from sklearn.base import BaseEstimator
class Adaboost(BaseEstimator):
    
    def __init__(self, M):
        self.M = M
        
    def fit(self, X, Y):
        self.models = []
        self.model_weights = []
        
        N, _ = X.shape
        
        # 初始化每个数据点的权重一样的
        # 譬如：N = 5, np.ones(5) -> array[1,1,1,1,1] 
        #      alpha = np.ones(N) / N -> array([0.2, 0.2, 0.2, 0.2, 0.2])
        alpha = np.ones(N) / N
        
        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=2)
            tree.fit(X, Y, sample_weight=alpha)
            prediction = tree.predict(X)
            
            
            # weighted_error: Week6.Session1.Ensemble_Theory.pdf page 92 和 page 114公式
            # page 114里， weighted_error 公式里， I(^y!=label) 这里的I 相当于一个函数，一样就是1， 不一样就是0
            # 由于所有sample data的权重和为1 所以就省略掉分母. 否则分母应该为所有data 的权重之和，分子就是（prediction != Y）那些sample data 的权重之和
            weighted_error = alpha.dot(prediction != Y)
            
            # 计算当前模型的权重 Week6.Session1.Ensemble_Theory.pdf page 97 algorithm
            model_weight = 0.5 * (np.log(1 - weighted_error) - np.log(weighted_error))
            
            # 更新数据的权重
            # Week6.Session1.Ensemble_Theory.pdf page 101 algorithm 
            # Y=1：  预测对了 prediction =1     alpha * np.exp(-model_weight） 
            # Y=1：  预测错了 prediction =-1    alpha * np.exp(model_weight）
            
            # Y=-1： 预测对了 prediction =-1   alpha * np.exp(-model_weight） 
            # Y=-1： 预测错了 prediction =1    alpha * np.exp (model_weight）
            alpha = alpha * np.exp(-model_weight * Y * prediction)
            
            # 数据权重normalize,确保所有数据点权重加在一起的和仍然是1
            alpha = alpha / alpha.sum()
            
            self.models.append(tree)
            self.model_weights.append(model_weight)
            
    # Week6.Session1.Ensemble_Theory.pdf page 112公式 貌似和这里实现不一样
    def predict(self, X):
        N, _ = X.shape
        result = np.zeros(N)
        for wt, tree in zip(self.model_weights, self.models):
            result += wt * tree.predict(X)
        # sign 范围 -1 ， 1
        return np.sign(result)
    
    def score(self, X, Y):
        prediction = self.predict(X)
        return np.mean(prediction == Y)

adamodel = Adaboost(200)
print("Adaboost model performance: %f" % (cross_val_score(adamodel, trainX.as_matrix().astype(np.float64), trainY.as_matrix().astype(np.float64), cv=8).mean()))
adamodel.fit(trainX.as_matrix().astype(np.float64), trainY.as_matrix().astype(np.float64))
print("Adaboost model test performance: %f" % adamodel.score(testX.as_matrix().astype(np.float64), testY.as_matrix().astype(np.float64)))
