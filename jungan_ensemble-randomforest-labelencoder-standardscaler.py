
%matplotlib inline
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

np.random.seed(19)
data_folder = "../input"
#data_folder = "data"
data = pd.read_csv(os.path.join(data_folder, "mushrooms.csv"), header=None)

data.head()
data[0] = data.apply(lambda row: 0 if row[0] == 'e' else 1, axis=1)

# 1 - 22 features, 每一列如果有null，用"missing"代替
#Note: data.loc['column name'], i.e. 必须是column name, 不可以用数字， 这里之所以可以，因为正好column name 就是数字
cols = np.arange(1,23)
for col in cols:
    if np.any(data[col].isnull()):
        # data[col].isnull() rerturn true / false
        data.loc[data[col].isnull(), col] = 'missing'
labelEncoders = dict()

# 对每一列进行 on hot encoding 
for col in cols:
    encoder = LabelEncoder()
    values = data[col].tolist()
    values.append('missing') # 加入missing这种值
    encoder.fit(values)
    labelEncoders[col] = encoder # 每一列的encoder 加入到labelEncoders list
    
# 计算和查看一下one hot encoding之后的列数, 一共161列了，下面定义新的data的shape需要用到
dimensionality = 0
for col, encoder in labelEncoders.items():
    # encoder.classes_ list of new columns (transformed from one column)
    dimensionality += len(encoder.classes_)
print("dimensionality:  %d" % (dimensionality))
# 用于测试数据的变换
def transform(df):
    N, _ = df.shape # just need row number here, no need for column number
    X = np.zeros((N, dimensionality)) # new shape of data i.e. (N, dimensionality) dimensionality = 161
    
    i = 0
    for col ,encoder in labelEncoders.items():
        k = len(encoder.classes_)
        # 每次encode 一个原始的column data (包含所有行), 等号右边的1， 就是用于one hot encoding里显示的1，如果定义为2， 就会显示2
        X[np.arange(N), encoder.transform(df[col]) + i] = 1
        # 沿着y轴 横向的，一块一块的拼接  [new col1_from col1, new col2_from col1 | new col3_from col2， new col4_from col2 ]
        i += k
    return X
# 准备数据和标签 , X is numpy type which doesn't have head() function
X = transform(data)
X.shape
X[:2]
data[0].shape
data[0][:5] # OR data[0].head()
Y = data[0].as_matrix()
Y
Y.shape
logistic_model = LogisticRegression()
print("logistic Regression performance: %f" % (cross_val_score(logistic_model, X, Y, cv=8).mean()))
tree_model = DecisionTreeClassifier()
print("Decision Tree performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))
forest = RandomForestClassifier(n_estimators=20)
print("Random Forest performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))
from sklearn.base import BaseEstimator
from sklearn.base import BaseEstimator
class FakeRandomForest(BaseEstimator):
    
    def __init__(self, M):
        self.M = M
        
    def fit(self, X, Y, n_features=None):
        N,D = X.shape
        if n_features is None:
            # 特征的个数 (np.sqrt, 开根号). 参考 Week6.Session1.Ensemble_Theory.pdf page60
            n_features = int(np.sqrt(D))
        
        self.models = [] # 袋子，用来装models
        self.features = []   #特征
        
        # M 表示模型个数，M个Decision Tree Model
        # fake random forest: 1. 在每个训练数据集扇面随机抽取特征 2.用tree训练每个数据集
        for m in range(self.M):
            tree = DecisionTreeClassifier()
            
            #有放回的随机抽取N个数据. N: 输入数据的行数. 这里从N行里， 抽取N次
            idx = np.random.choice(N, size=N, replace=True)
            X_current = X[idx]
            Y_current = Y[idx]
            
            # 随机抽取n_features个特征 （原始输入数据 D columns）
            features = np.random.choice(D, size=n_features, replace=False)
            
            #训练当前的决策树模型, X_current[:, features] 表示X的所有行，和 抽取的列
            tree.fit(X_current[:, features], Y_current)
            self.features.append(features)
            self.models.append(tree)
            
    # 这里是二分类问题，输出的算法参考 Week6.Session1.Ensemble_Theory.pdf page36
    # 对于多分类问题，输出，参考Week6.Session1.Ensemble_Theory.pdf page 33- 36
    def predict(self, X):
        N = len(X)
        # 二分类 仅需要一位N维数组（N就是training data 个数）去存储结果
        # 多分类 则需要一位（N, k）维数组（N就是training data 个数,K是类别个数）去存储结果
        results = np.zeros(N)
        
        for features, tree in zip(self.features, self.models): # 多个tree models,注意 fit function last two statment
            results += tree.predict(X[:, features])  #多个model 预测的result 加在一起
            
        # 多个模型预测结果的和 除以 模型个数, __init__ 函数传进来的
        # np.round(results/ self.M)由于np.round取整，要不是1， 要不是0， 达到了分两类的目的
        print(results)
        print("====")
        return np.round(results/ self.M)
    
    def score(self, X, Y):
        prediction = self.predict(X)
        return np.mean(prediction == Y)

# 与FakeRandomForest的区别，这里针对每个模型，都是用所有的features, 而FakeRandomForest里，每个model,都只是用一部分随机抽取的features
class BaggedTreeClassifier(BaseEstimator):
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y):
        N = len(X)
        self.models = []
        for m in range(self.M):
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]
            model = DecisionTreeClassifier(max_depth=2)
            model.fit(Xb, Yb)
            self.models.append(model)

    def predict(self, X):
        # no need to keep a dictionary since we are doing binary classification
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return np.round(predictions / self.M)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)
baggedtc = BaggedTreeClassifier(20)
cross_val_score(baggedtc, X, Y, cv=8).mean()
# tree number 增加，有助于提提高正确率
fakerf = FakeRandomForest(6)
# cv number 增加，有助于提提高正确率
cross_val_score(fakerf, X, Y, cv=3).mean()
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
house_data = pd.read_csv(os.path.join(data_folder, "kc_house_data.csv"))
house_data.head()
house_data.columns
# price is the target
NUMERICAL_COLS = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above','sqft_basement',
                 'sqft_living15', 'sqft_lot15']
house_data['price'].as_matrix()
house_data['price'].as_matrix().astype(np.float64).reshape(-1,1)
# fit 每一列数据的scaler
scalers = dict()
for col in NUMERICAL_COLS:
    # 对离散类型变量，用 LabelEncoder, StandardScaler 一般apply到连续性的数字feature 列
    # from sklearn.preprocessing import LabelEncoder, StandardScaler
    # 去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。
    # 标准正态分布 期望值μ=0，即曲线图象对称轴为Y轴，标准差σ=1条件下的正态分布，记为N(0，1)
    scaler = StandardScaler()
    # house_data[col].as_matrix(),一列 变成 一个 一行的array
    # numpy allow us to give one of new shape parameter as -1 (eg: (2,-1) or (-1,3) but not (-1, -1)). It simply means that it is an unknown dimension and we want numpy to figure it out.
    # scaler fit first, then later, we will call transform 
    # The data used to compute the mean and standard deviation used for later scaling along the features axis.
    scaler.fit(house_data[col].as_matrix().astype(np.float64).reshape(-1,1)) # -1 not sure how many rows
    scalers[col] = scaler
    
    
def transform_numerical(df):
    N, _ = df.shape
    D = len(NUMERICAL_COLS)
    result = np.zeros((N,D))
    i = 0
    for col, scaler in scalers.items():
        # ：means all rows, i, 表示一列一列的处理
        result[:, i] = scaler.transform(df[col].as_matrix().astype(np.float64).reshape(1,-1)) # fit in previous step
        i += 1
    return result    

from sklearn.model_selection import train_test_split
# hdata is numpy array type
hdata = transform_numerical(house_data)
hdata[:2,:]
train_data, test_data = train_test_split(hdata, test_size=0.2)
# 第一列price是 label 
trainX, trainY = train_data[:,1:], train_data[:, 0]
testX, testY = test_data[:, 1:], test_data[:, 0]
rfregressor = RandomForestRegressor(n_estimators=100)
rfregressor.fit(trainX, trainY)
predictions = rfregressor.predict(testX)
plt.scatter(testY, predictions)
plt.xlabel("target")
plt.ylabel("prediction")
ymin = np.round(min(min(testY), min(predictions)))
ymax = np.ceil(max(max(testY), max(predictions)))
r = range(int(ymin), int(ymax) + 1)
plt.plot(r,r)
plt.show()
plt.plot(testY, label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()
lr = LinearRegression()
print("linear regression performance: %f" % (cross_val_score(lr, trainX, trainY).mean()))
print("random forest regressor performance: %f" % (cross_val_score(rfregressor, trainX, trainY).mean()))
lr.fit(trainX, trainY)
print("linear regression test score: %f" % (lr.score(testX, testY)))
rfregressor.fit(trainX, trainY)
print("random forest regressor test score: %f" % (rfregressor.score(testX, testY)))
