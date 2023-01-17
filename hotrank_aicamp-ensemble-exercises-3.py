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
data.diagnosis.value_counts()
# TODO
data.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
# TODO
data.diagnosis = data.diagnosis.apply(lambda x: 1 if x=='M' else '-1')
# data.diagnosis
# TODO
data.describe()
data.info()
# 柱状图
import seaborn as sns
# TODO
sns.countplot(data.diagnosis)
data.diagnosis = data.diagnosis.astype(int)
figure = plt.subplots(figsize = (20, 15))
sns.heatmap(data.corr(), square = True, annot = True, fmt = '.2f')

# features with correlation > 0.7 radius_mean, perimeter_mean, area_mean, concavity_mean, concave points_mean, radius_worst, perimeter_worst, area_worst, concave_points_worst
# TODO
features = ['radius_mean', 'concavity_mean', 'radius_se', 'compactness_mean', 'texture_mean', 'symmetry_mean']
target = 'diagnosis'
features
# add a temp target label with value 0 and 1, so that malignancy rate can be calculated using .mean() during plotting in the next cell
data['temp_target'] = data.diagnosis.apply(lambda x: 0 if x==-1 else 1)
i = 0
for feature in features:

    bins1 = 25
    bins2 = 10
    # 将特征的直方图画出来
    # TODO
    plt.subplots(1,2,figsize = (16,5))
    plt.subplot(1,2,1)
    data[feature][data[target] == -1].hist(bins=bins1, color='lightblue', label= 'Benign', alpha=1)
    data[feature][data[target] == 1].hist(bins=bins1, color='k', label='Malignant', alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Amount of count')
    plt.legend()
    plt.subplot(1,2,2)
    data.groupby(pd.qcut(data[feature], bins2))['temp_target'].mean().plot(kind = 'bar')
    plt.ylabel('malignancy rate')
    

    
#     plt.legend()
    
    plt.show()
from sklearn.model_selection import train_test_split
# TODO 训练集和测试集
train_data, test_data = train_test_split(data, test_size = 0.3, random_state = 1)
# TODO 数据和标签
trainX, trainY = train_data[data.columns[1:-1]], train_data[target]
testX, testY = test_data[data.columns[1:-1]], test_data[target]

# TODO logistic 模型的表现
logistic_model = LogisticRegression(solver = 'lbfgs', max_iter = 100 )
print("Logistic Regression performance: %f" % (cross_val_score(logistic_model, trainX, trainY, cv=8).mean()))
# TODO
tree_model = DecisionTreeClassifier()
print('Decision Tree Classifier performance: %f' % (cross_val_score(tree_model, trainX, trainY, cv = 8).mean()))

# TODO
adaModel = AdaBoostClassifier(n_estimators= 200)
print('Adaboost Classifier performance: %f' % (cross_val_score(adaModel, trainX, trainY, cv = 8).mean()))
logistic_model = LogisticRegression(solver = 'lbfgs')
logistic_model.fit(trainX, trainY)
logistic_model.score(testX, testY)

# TODO
tree_model = DecisionTreeClassifier(max_depth = 10)
tree_model.fit(trainX, trainY)
tree_model.score(testX, testY)
# TODO
ada_model = AdaBoostClassifier(n_estimators=200)
# TODO
ada_model.fit(trainX, trainY)
ada_model.score(testX, testY)
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
        alpha = np.ones(N)/N
        
        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth = 3)
            tree.fit(X, Y, sample_weight=alpha)
            prediction = tree.predict(X)
            
            # 计算加权错误
            weighted_error = alpha.dot(prediction != Y)
            
            # 计算当前模型的权重
            model_weight = 0.5 * (np.log(1-weighted_error) - np.log(weighted_error))
            
            # 更新数据的权重
            alpha = alpha * (np.exp(-Y * prediction * model_weight)) 
            
            # 数据权重normalize
            alpha = alpha/alpha.sum()
            
            self.models.append(tree)
            self.model_weights.append(model_weight)
            
    def predict(self, X):
        # TODO
        result = np.zeros(len(X))
        for weight, tree in zip(self.model_weights, self.models):
            result += weight * tree.predict(X)
        
        return np.sign(result)

    
    def score(self, X, Y):
        # TODO
        pred = self.predict(X)
        return (pred == Y).mean()
# TODO
adamodel = Adaboost(200)
print("Adaboost model performance: %f" % (cross_val_score(adamodel, trainX.values.astype(np.float64), trainY.values.astype(np.float64), cv=8).mean()))
adamodel.fit(trainX.values.astype(np.float64), trainY.values.astype(np.float64))
print("Adaboost model test performance: %f" % adamodel.score(testX.values.astype(np.float64), testY.values.astype(np.float64)))
