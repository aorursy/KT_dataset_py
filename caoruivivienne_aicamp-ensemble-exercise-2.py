

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



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier



np.random.seed(19)
data_folder = "../input"

#data_folder = "data"

data = pd.read_csv(os.path.join(data_folder, "mushrooms.csv"), header=None)

data.head()
data[0] = data.apply(lambda row : 0 if row[0] == 'e' else 1, axis=1) # TODO
# 每一列如果有null，用"missing"代替

cols = np.arange(1,23)

for col in cols:

    #TODO

    if np.any(data[col].isnull()):

        data.loc[data[col].isnull(), col] = 'missing'

labelEncoders = dict()#TODO



# TODO 对每一列进行label encoding 

for col in cols:

    # TODO

    encoder = LabelEncoder()

    values = data[col].tolist()

    values.append('missing')

    encoder.fit(values)

    labelEncoders[col] = encoder 

    



# 计算label encoding之后的列数

dimensionality = 0

for col, encoder in labelEncoders.items():

    dimensionality +=  len(encoder.classes_)#TODO

print("dimensionality:  %d" % (dimensionality))
# 用于测试数据的变换

def transform(df):

    #TODO

    N = df.shape[0]

    X = np.zeros((N, dimensionality))

    i = 0

    for col, encoder in labelEncoders.items():

        k = len(encoder.classes_)

        X[np.arange(N), encoder.transform(df[col]) + i] = 1

        i += k

    return X
# 准备数据和标签

X = transform(data)#TODO

Y = data[0].as_matrix()
logistic_model = LogisticRegression(solver='liblinear')#TODO

print("logistic Regression performance: %f" % (cross_val_score(logistic_model, X, Y, cv=8).mean()))
tree_model = DecisionTreeClassifier()#TODO

print("Decision Tree performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))
forest = RandomForestClassifier(n_estimators=20)#TODO

print("Random Forest performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))
from sklearn.base import BaseEstimator

class FakeRandomForest(BaseEstimator):

    

    def __init__(self, M):

        self.M = M

        

        

    def fit(self, X, Y, n_features=None):

        N,D = X.shape

        if n_features is None:

            # TODO 特征的个数

            n_features = int(np.sqrt(D))

        

        # TODO袋子。。

        self.models = []

        

        # TODO特征

        self.features = []

        

        for m in range(self.M):

            tree = DecisionTreeClassifier()

            

            #TODO 有放回的随机抽取N个数据

            idx = np.random.choice(N, size=N, replace=True)

            X_current = X[idx]

            Y_current = Y[idx]



            

            #TODO 随机抽取n_features个特征

            features = np.random.choice(D, size=n_features, replace=False)

            

            #训练当前的决策树模型

            tree.fit(X_current[:, features], Y_current)

            self.features.append(features)

            self.models.append(tree)

            

    

    def predict(self, X):

        # TODO

        N = len(X)

        results = np.zeros(N)

        for features, tree in zip(self.features, self.models):

            results += tree.predict(X[:, features])

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

    # TODO

    scaler = StandardScaler()

    scaler.fit(house_data[col].as_matrix().astype(np.float64).reshape(-1, 1))

    scalers[col]= scaler
def transform_numerical(df):

    N, _ = df.shape

    D = len(NUMERICAL_COLS)

    result = np.zeros((N,D))

    i = 0

    for col, scaler in scalers.items():

        # TODO

        result[:, i] = scaler.transform(df[col].as_matrix().astype(np.float64).reshape(1, -1))

        i += 1

    return result    

from sklearn.model_selection import train_test_split
hdata = transform_numerical(house_data)# TODO  transform your data
train_data, test_data = train_test_split(hdata, test_size=0.2)# train test split
trainX, trainY = train_data[:, 1:], train_data[:, 0]# train data and train target

testX, testY = test_data[:,1:], test_data[:,0] # test data and test target

rfregressor = RandomForestRegressor(n_estimators=100)# TODO

rfregressor.fit(trainX, trainY)

predictions = rfregressor.predict(testX)
# TODO

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
lr = LinearRegression()# TODO

print("linear regression performance: %f" % (cross_val_score(lr, trainX, trainY).mean()))
print("random forest regressor performance: %f" % (cross_val_score(rfregressor, trainX, trainY).mean()))
lr.fit(trainX, trainY)

print("linear regression test score: %f" % (lr.score(testX, testY)))
rfregressor.fit(trainX, trainY)

print("random forest regressor test score: %f" % (rfregressor.score(testX, testY)))