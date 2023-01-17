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
data['diagnosis'].value_counts()
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else -1)
data.describe()
data.info()
# 柱状图

import seaborn as sns



sns.countplot(data['diagnosis']);
# TODO

features = data.columns.tolist()[1:7]

target = ['diagnosis']

features
i = 0

for feature in features:



    bins = 25

    # 将特征的直方图画出来

    plt.hist(data[data['diagnosis'] == +1][feature], color='c', alpha=0.4, label='Benign')

    plt.hist(data[data['diagnosis'] == -1][feature], color='k', alpha=0.4, label='Malignant')

    

    plt.xlabel(feature)

    plt.ylabel('Amount of count')

    

    plt.legend()

    

    plt.show()
from sklearn.model_selection import train_test_split

# TODO 训练集和测试集

train_data, test_data = train_test_split(data, test_size=0.3, random_state=1)
# TODO 数据和标签

trainX, trainY = train_data[features], np.ravel(train_data[target])

testX, testY = test_data[features], np.ravel(test_data[target])
# TODO logistic 模型的表现

logistic_model = LogisticRegression(solver='liblinear')

print("Logistic Regression performance: %f" % (cross_val_score(logistic_model, trainX, trainY, cv=8).mean()))
tree_model = DecisionTreeClassifier()

print("Decision Tree performance: %f" % (cross_val_score(tree_model, trainX, trainY, cv=8).mean()))
ada_model = AdaBoostClassifier(n_estimators=200)

print("Adaboost performance: %f" % (cross_val_score(ada_model, trainX, trainY, cv=8).mean()))
logistic_model = LogisticRegression(solver='liblinear')

logistic_model.fit(trainX, trainY)

print("Logistic Regression test performance: %f" % (logistic_model.score(testX, testY)))
tree_model = DecisionTreeClassifier()

tree_model.fit(trainX, trainY)

print("Decision Tree test performance: %f" % (tree_model.score(testX, testY)))
ada_model = AdaBoostClassifier(n_estimators=200)

ada_model.fit(trainX, trainY)

print("Adaboost test performance: %f" % (ada_model.score(testX, testY)))
from sklearn.base import BaseEstimator

class Adaboost(BaseEstimator):

    

    def __init__(self, M):

        self.M = M

        

    def fit(self, X, Y):

        self.models = list()

        self.model_weights = list()

        

        N, _ = X.shape

        alpha = np.ones(N) / N * 1.0

        

        for m in range(self.M):

            tree = DecisionTreeClassifier(max_depth=2)

            tree.fit(X, Y, sample_weight=alpha)

            prediction = tree.predict(X)

            

            # 计算加权错误

            weighted_error = np.dot(alpha, (Y != prediction))

            

            # 计算当前模型的权重

            model_weight = 0.5 * np.log((1. - weighted_error) / weighted_error)

            

            # 更新数据的权重

            alpha = alpha * np.exp(-model_weight * Y * prediction)

            

            # 数据权重normalize

            alpha = alpha / alpha.sum()

            

            self.models.append(tree)

            self.model_weights.append(model_weight)

            

    def predict(self, X):

        result = np.zeros(len(X))

        for w, model in zip(self.model_weights, self.models):

            result += w * model.predict(X)

        return np.sign(result)

   

    def score(self, X, Y):

        pred = self.predict(X)

        return np.mean(pred == Y)
# TODO

adamodel = Adaboost(M=400)

print("Adaboost model performance: %f" % (cross_val_score(adamodel, np.asmatrix(trainX).astype(np.float64), trainY.astype(np.float64), cv=8).mean()))
adamodel.fit(np.asmatrix(trainX).astype(np.float64), trainY.astype(np.float64))

print("Adaboost model test performance: %f" % adamodel.score(np.asmatrix(testX).astype(np.float64), testY.astype(np.float64)))