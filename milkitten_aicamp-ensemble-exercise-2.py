

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

from sklearn.preprocessing import StandardScaler, LabelEncoder

np.random.seed(19)
data_folder = "../input"

#data_folder = "data"

data = pd.read_csv(os.path.join(data_folder, "mushrooms.csv"), header=None)

print(data.shape)

data.head()
data = pd.DataFrame(data.iloc[1:, :].values,columns=data.iloc[0,:])

data.head()
data.iloc[:,0] = data.iloc[:,0].apply(lambda x: 1 if x=='p' else 0) 

data.head()
# 每一列如果有null，用"missing"代替

for col in data.columns[1:]:

    #TODO

    if data[col].isnull().values.any():

        print(col)

        data[data[col].isnull(),col] = 'missing'        
labelEncoders = {}



# TODO 对每一列进行label encoding 

# for col in cols:

#     # TODO

#     encoder = LabelEncoder()

#     encoder.fit(data[col].tolist())

#     labelEncoders[col]= encoder

#     print(encoder.classes_)



for col in data.columns[1:]:

    encoder = LabelEncoder()

    values = data[col].tolist()

    values.append('missing')  #加入missing这种值

    encoder.fit(values)

    labelEncoders[col] = encoder

    print(encoder.classes_)

    print(encoder.transform(data[col])) 

    

# 计算label encoding之后的列数

dimensionality = 0

for col, encoder in labelEncoders.items():

    dimensionality += len(encoder.classes_)

print("dimensionality:  %d" % (dimensionality))
# 用于测试数据的变换

def transform(df):

    #TODO

    N, _ = df.shape

    X = np.zeros((N, dimensionality))

    i = 0

    for col, encoder in labelEncoders.items():

        k = len(encoder.classes_)

        X[np.arange(N), encoder.transform(df[col])+i] = 1

        i+=k

    return X    
# 准备数据和标签

X_LE = transform(data)

Y_LE = data['class'].values

print(X_LE.shape)

X = pd.get_dummies(data, columns=data.columns[1:], drop_first=False)

Y = data.iloc[:,0]

print(X.shape)
X.head()
X = X.iloc[:,1:].values

Y = Y.values

print(X.shape)

logistic_model = LogisticRegression()

print("logistic Regression performance: %f" % (cross_val_score(logistic_model, X, Y, cv=8).mean()))

print("logistic Regression performance with empty missing column added: %f" % (cross_val_score(logistic_model, X_LE, Y_LE, cv=8).mean()))
tree_model = DecisionTreeClassifier()

print("Decision Tree performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))

print("Decision Tree performance with empty missing column added: %f" % (cross_val_score(tree_model, X_LE, Y_LE, cv=8).mean()))
forest = RandomForestClassifier(n_estimators=20)

print("Random Forest performance: %f" % (cross_val_score(forest, X, Y, cv=8).mean()))

print("Random Forest performance with empty missing column added: %f" % (cross_val_score(forest, X_LE, Y_LE, cv=8).mean()))
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
print(cross_val_score(baggedtc, X, Y, cv=8).mean())

print(cross_val_score(baggedtc, X_LE, Y_LE, cv=8).mean())
fakerf = FakeRandomForest(20)
print(cross_val_score(fakerf, X, Y, cv=8).mean())

print(cross_val_score(fakerf, X_LE, Y_LE, cv=8).mean())
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

    standar = StandardScaler()

    standar.fit(house_data[col].values.reshape(-1,1))

    scalers[col]=standar

    
def transform_numerical(df):

    N, _ = df.shape

    D = len(NUMERICAL_COLS)

    result = np.zeros((N,D))

    i = 0

    for col, scaler in scalers.items():

        result[:,i]=scaler.transform(df[col].values.reshape(1,-1))

        i+=1

    return result    

X = transform_numerical(house_data)



X_P = StandardScaler().fit_transform(house_data[NUMERICAL_COLS])



print((X_P!=X).any())
from sklearn.model_selection import train_test_split
hdata = StandardScaler().fit_transform(house_data[NUMERICAL_COLS])# TODO  transform your data
train_data, test_data = train_test_split(hdata, test_size=0.2,random_state=0)# train test split
trainX, trainY = train_data[:,1:], train_data[:,0]# train data and train target

testX, testY = test_data[:,1:], test_data[:,0]# test data and test target

rfregressor = RandomForestRegressor(100)# TODO

rfregressor.fit(trainX, trainY)

predictions = rfregressor.predict(testX)
# TODO

plt.scatter(testY,predictions)

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

print("linear regression performance: %f" % cross_val_score(lr, hdata[:,1:], hdata[:,0], cv=5).mean())#TODO
print("random forest regressor performance: %f" % cross_val_score(rfregressor, hdata[:,1:], hdata[:,0], cv=5).mean())#TODO
lr.fit(trainX, trainY)

print("linear regression test score: %f" % lr.score(testX,testY))
rfregressor.fit(trainX, trainY)

print("random forest regressor test score: %f" % rfregressor.score(testX,testY))#TODO