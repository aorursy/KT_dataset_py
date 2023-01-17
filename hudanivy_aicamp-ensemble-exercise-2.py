
%matplotlib inline
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# TODO 
# you need to import LabelEncoder, StandardScaler
# you need to import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder


np.random.seed(19)

#data_folder = "data"
data = pd.read_csv("../input/mushrooms.csv", header=None)
data.columns=data.iloc[0,:]
data.drop(index=0,inplace=True)
for col in data.columns:
    new_col=col.replace('-','_')
    data.rename(columns={col:new_col},inplace=True)
data.rename(columns={'class':'class_'},inplace=True)
#data.reset_index(drop=True)
data.head()

data['class_'] = data['class_'].apply(lambda x: 1 if x=='p' else 0)# TODO
data.head()

#print("Decision Tree performance: %f" % (cross_val_score(tree_model, tmp_x, tmp_y, cv=8).mean()))
#data.isnull().any()
# 每一列如果有null，用"missing"代替

"""
labelEncoders = []#TODO

# TODO 对每一列进行label encoding 
for col in cols:
    # TODO
    le=LabelEncoder()
    le=fit(data[col])
    labeldata[col]=fit

# 计算label encoding之后的列数
dimensionality = 0
for col, encoder in labelEncoders.items():
    dimensionality = #TODO
print("dimensionality:  %d" % (dimensionality))
"""

from patsy import dmatrices
tmp=''
for col in data.columns:
    tmp += 'C('+col+')+'
tmp='class_~'+tmp[10:-1]
print (tmp)

print(data.columns)
Y, X = dmatrices(tmp,data,return_type='dataframe')
print(X.shape)
#columns=X.columns
Y=np.ravel(Y)
X=np.asmatrix(X)
# 用于测试数据的变换
#def transform(df):
    #TODO
# 准备数据和标签
#X = #TODO
#Y = data['class_'].as_matrix()
logistic_model = LogisticRegression()
print("logistic Regression performance: %f" % (cross_val_score(logistic_model, X, Y, cv=8).mean()))
tree_model = DecisionTreeClassifier()
print("Decision Tree performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))
forest = RandomForestClassifier(n_estimators=10)
print("Random Forest performance: %f" % (cross_val_score(forest, X, Y, cv=8).mean()))


from sklearn.base import BaseEstimator
class FakeRandomForest(BaseEstimator):
    
    def __init__(self, M):
        self.M = M
        
        
    def fit(self, X, Y, n_features=None):
        N,D = X.shape
        if n_features is None:
            # TODO 特征的个数
            n_features = 20
        
        # TODO袋子。。
        self.models =[]
        
        # TODO特征
        self.features =[]
        
        for m in range(self.M):
            tree = DecisionTreeClassifier()
            idx=np.random.choice(N,size=N,replace=True)
            X_current=X[idx]
            Y_current=Y[idx]
            #TODO 有放回的随机抽取N个数据
            
            
            #TODO 随机抽取n_features个特征
            features = np.random.choice(D,size=n_features,replace=False)
            #训练当前的决策树模型
            tree.fit(X_current[:, features], Y_current)
            self.features.append(features)
            self.models.append(tree)
      
            
    
    def predict(self, X):
        # TODO
        N = len(X)
        results = np.zeros(N)
        for features, tree in zip(self.features, self.models):
           
            results += tree.predict(X[:,features])
        return np.round(results/ self.M)
    
    def score(self, X, Y):
        # TODO
        prediction = self.predict(X)
        return np.mean(prediction == Y)
        
            
            
            

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
fakerf = FakeRandomForest(20)
fakerf.fit(X,Y)
fakerf.score(X,Y)
cross_val_score(fakerf, X, Y, cv=8).mean()
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
house_data = pd.read_csv("../input/kc_house_data.csv",sep=',')
house_data.head()
house_data.columns
# price is the target
NUMERICAL_COLS = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above','sqft_basement',
                 'sqft_living15', 'sqft_lot15']
# 直接fit_transform
scalers = dict()
for col in NUMERICAL_COLS:
    scaler = StandardScaler()
    house_data[col]=scaler.fit_transform(house_data[col].astype(np.float64).as_matrix().reshape(-1,1))
    
print(house_data.head())
    
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(house_data[NUMERICAL_COLS],test_size=0.3)
train_data.head()
trainX, trainY =train_data.iloc[:,1:],train_data.price # train data and train target
testX, testY = test_data.iloc[:,1:].reset_index(drop=True),test_data.price.reset_index(drop=True)# test data and test target
print(testX.shape,testY.shape)
rfregressor = RandomForestRegressor(n_estimators=20)
rfregressor.fit(trainX, trainY)
predictions = rfregressor.predict(testX)
print(predictions.shape)
# TODO
plt.scatter(testY, predictions)
plt.xlabel("target")
plt.ylabel("prediction")
ymin = np.round(min(min(testY), min(predictions)))
ymax = np.ceil(max(max(testY), max(predictions)))
r = range(int(ymin), int(ymax) + 1)
plt.plot(r,r)
plt.show()

plt.plot(testY,label='predictions')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()
lr = LinearRegression()

print("linear regression performance: %f" % cross_val_score(lr, trainX, trainY).mean())
print("random forest regressor performance: %f" %cross_val_score(rfregressor, trainX, trainY).mean())
lr.fit(trainX, trainY)
print("linear regression test score: %f" % lr.score(testX,testY))
rfregressor.fit(trainX, trainY)
print("random forest regressor test score: %f" % rfregressor.score(testX,testY))
#linear regressor score is R2 score, maximum number is 1.0, can be negative. The larger the better. It describes how well this
#linear model fits the data.



