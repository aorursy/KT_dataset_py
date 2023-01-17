

%matplotlib inline

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier



np.random.seed(19)
data_folder = "../input"

#data_folder = "data"

data = pd.read_csv(os.path.join(data_folder, "mushrooms.csv"), header=0)

data.head()
data['class'] = data['class'].apply(lambda x: 1 if x == 'p' else 0)
# 每一列如果有null，用"missing"代替

cols = data.columns.tolist()[1:]

for col in cols:

    if data[col].isnull().any():

        data.loc[data[col].isnull(), col] = 'missing'
labelEncoders = dict()



# TODO 对每一列进行label encoding 

for col in cols:

    labelEncoders[col] = LabelEncoder().fit(data[col])



# 计算label encoding之后的列数

dimensionality = 0

for col, encoder in labelEncoders.items():

    dimensionality += len(encoder.classes_)

print("dimensionality:  %d" % (dimensionality))
# 用于测试数据的变换

def transform(df):

    N, _ = df.shape

    X = np.zeros((N, dimensionality))

    

    idx = 0

    for col, encoder in labelEncoders.items():

        new_cols = len(encoder.classes_)

        X[np.arange(N), encoder.transform(data[col]) + idx] = 1

        idx += new_cols

    return X
# 准备数据和标签

X = transform(data)

Y = np.ravel(data['class'])
logistic_model = LogisticRegression(solver='liblinear')

print("logistic Regression performance: %f" % (cross_val_score(logistic_model, X, Y, cv=8).mean()))
tree_model = DecisionTreeClassifier()

print("Decision Tree performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))
forest = RandomForestClassifier(n_estimators=20)

print("Random Forest performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))
from sklearn.base import BaseEstimator

class FakeRandomForest(BaseEstimator):

    

    def __init__(self, M):

        self.M = M

        

        

    def fit(self, X, Y, n_features=None):

        N,D = X.shape

        if n_features is None:

            # 特征的个数

            n_features = int(np.sqrt(D))

        

        # 袋子。。

        self.models = list() 

        

        # 特征

        self.features = list()

        

        for m in range(self.M):

            tree = DecisionTreeClassifier()

            

            # 有放回的随机抽取N个数据

            idx = np.random.choice(N, N, replace=True)

            X_current = X[idx]

            Y_current = Y[idx]

            # 随机抽取n_features个特征

            features = np.random.choice(D, n_features, replace=False)

            

            #训练当前的决策树模型

            tree.fit(X_current[:, features], Y_current)

            self.features.append(features)

            self.models.append(tree)

            

    

    def predict(self, X):

        N = X.shape[0]

        results = np.zeros(N)

        for features, tree in zip(self.features, self.models):

            results += tree.predict(X[:, features]) 

        return np.round(results/ self.M)

    

    def score(self, X, Y):

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
cross_val_score(fakerf, X, Y, cv=8).mean()
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
house_data = pd.read_csv(os.path.join(data_folder, "kc_house_data.csv"))
house_data.head()
house_data.columns
# price is the target

NUMERICAL_COLS = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above','sqft_basement',

                 'sqft_living15', 'sqft_lot15']
# fit 每一列数据的scaler

scalers = dict()

for col in NUMERICAL_COLS:

    scalers[col] = StandardScaler().fit(np.ravel(house_data[col]).reshape(-1,1))

    
def transform_numerical(df):

    N, _ = df.shape

    D = len(NUMERICAL_COLS)

    result = np.zeros((N,D))

    i = 0

    for col, scaler in scalers.items():

        result[:, i] = np.ravel(scaler.transform(np.ravel(df[col]).reshape(-1,1)))

        i += 1

    return result    

from sklearn.model_selection import train_test_split
hdata = transform_numerical(house_data)
train_data, test_data = train_test_split(hdata, test_size=0.2, random_state=2)
trainX, trainY = train_data[:, 1:], train_data[:, 0]

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

print("linear regression performance: %f" % (cross_val_score(lr, trainX, trainY, cv=5).mean()))
print("random forest regressor performance: %f" % (cross_val_score(rfregressor, trainX, trainY, cv=5).mean()))
lr.fit(trainX, trainY)

print("linear regression test score: %f" % (lr.score(testX, testY)))
rfregressor.fit(trainX, trainY)

print("random forest regressor test score: %f" % (rfregressor.score(testX, testY)))