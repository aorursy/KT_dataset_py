# necessary imports
%matplotlib inline
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

np.random.seed(19)
data_folder = "../input"
data = pd.read_csv(os.path.join(data_folder, "breastCancer.csv"))
data.head()
data['diagnosis'].value_counts()
data.drop('id',axis=1,inplace=True)
data.drop('Unnamed: 32',axis=1,inplace=True)
data['diagnosis'] = data['diagnosis'].apply(lambda x: +1 if x=='M' else -1)
data.describe()
data.info()
import seaborn as sns
sns.countplot(data['diagnosis'])
features = data.columns[1:7]
target = 'diagnosis'
features
i = 0
for feature in features:

    bins = 25
    # 将特征的直方图画出来
    plt.hist(data[feature][data[target]==-1], bins=bins, color='lightblue', label='Healthy', alpha=1)
    plt.hist(data[feature][data[target]==+1], bins=bins, color='k', label='Bad', alpha=0.5)
    
    plt.xlabel(feature)
    plt.ylabel('Amount of count')
    
    plt.legend()
    
    plt.show() #重叠的越少，这个feature越有用
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.3)
trainX, trainY = train_data[data.columns[1:]], train_data[target]
testX, testY = test_data[data.columns[1:]], test_data[target]
logistic_model = LogisticRegression()
print("Logistic Regression performance: %f" % (cross_val_score(logistic_model, trainX, trainY, cv=8).mean()))
tree_model = DecisionTreeClassifier()
print("Decision Tree performance: %f" % (cross_val_score(tree_model, trainX, trainY, cv=8).mean()))
ada_model = AdaBoostClassifier(n_estimators=200)#n_estimators--基学习器的个数 
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
        alpha = np.ones(N)/N # 1/N---把alpha这个N维的向量每个元素设为1/N
        
        for m in range(self.M):
            #max_depth可以设为别的值，超参数本来就是要调的 
            tree = DecisionTreeClassifier(max_depth=2)#弱学习器，max_depth也要限制一下
            tree.fit(X,Y,sample_weight=alpha)#sample_weight输入数据的权重
            prediction = tree.predict(X)
            
            # 计算加权错误
            weighted_error = alpha.dot(prediction !=Y) #点积
            
            # 计算当前模型的权重
            model_weight = 0.5*(np.log(1-weighted_error) - np.log(weighted_error))
            
            # 更新数据的权重
            alpha = alpha * np.exp(-model_weight * Y * prediction)
            
            # 数据权重normalize
            alpha = alpha / alpha.sum()#alpha的计算是为了最小化损失函数 
            
            self.models.append(tree)
            self.model_weights.append(model_weight)
            
    def predict(self, X):
        N, _ = X.shape
        result = np.zeros(N)
        for wt, tree in zip(self.model_weights, self.models):
            result += wt*tree.predict(X)#加权和
        
        return np.sign(result) # sign: -1 if input < 0 and 1 if input > 0
    
    def score(self, X, Y):
        prediction = self.predict(X)
        return np.mean(prediction == Y)
adamodel = Adaboost(200)
print("Adaboost model performance: %f" % (cross_val_score(adamodel, trainX.as_matrix().astype(np.float64), trainY.as_matrix().astype(np.float64), cv=8).mean()))
adamodel.fit(trainX.as_matrix().astype(np.float64), trainY.as_matrix().astype(np.float64))
print("Adaboost model test performance: %f" % adamodel.score(testX.as_matrix().astype(np.float64), testY.as_matrix().astype(np.float64)))
