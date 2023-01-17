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

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

np.random.seed(19)
data_folder = "../input"
#data_folder = "data"
data = pd.read_csv(os.path.join(data_folder, "mushrooms.csv"), header=None)

data.head()
# remove the first row of data and save for records
features = data.loc[0]
data_raw = data.drop(0, axis = 0)
data_raw.head()
data_raw[0].value_counts()
Y = data_raw[0].apply(lambda x: 1 if x == 'p' else 0)
Y.head() # TODO

# 每一列如果有null，用"missing"代替
X_raw = data_raw.loc[:,1:]
cols = X_raw.columns
for col in cols:
    print(col, X_raw[col].isnull().values.any())

# There is no missing data.

# if there is missing data, do the following:

for col in cols:
    X_raw.loc[X_raw[col].isnull(), col] = 'missing'

labelEncoders = dict() #TODO

# TODO 对每一列进行label encoding 
for col in cols:
    encoder = LabelEncoder()
    values = X_raw[col].tolist()
    values.append('missing')  
    encoder.fit(values)
    labelEncoders[col] = encoder   # TODO

# 计算label encoding之后的列数
dimensionality = 0
for col, encoder in labelEncoders.items():
    dimensionality += len(encoder.classes_)  #TODO
print("dimensionality:  %d" % (dimensionality))
# 用于测试数据的变换
def transform(df):
    N = len(df)
    X = np.zeros((N, dimensionality))
    i = 0
    for col, encoder in labelEncoders.items():
        X[np.arange(N) ,encoder.transform(X_raw[col]) + i] = 1
        i += len(encoder.classes_)
    return X
    #TODO
# 准备数据和标签
X = transform(X_raw) #TODO
Y = Y.values
X.shape
test = pd.DataFrame(X)
test.head()
logistic_model = LogisticRegression(solver = 'lbfgs') #TODO
print("logistic Regression performance: %f" % (cross_val_score(logistic_model, X, Y, cv=8).mean()))
tree_model = DecisionTreeClassifier(max_depth = 100) #TODO
print("Decision Tree performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))
forest = RandomForestClassifier(n_estimators= 30, max_depth= 5) #TODO
print("Random Forest performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))


type(Y)
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
            tree = DecisionTreeClassifier(max_depth = 5)
            
            #TODO 有放回的随机抽取N个数据
            rows = np.random.choice(N, size = N, replace = True)
            X_current = X[rows]
            Y_current = Y[rows]
            
            #TODO 随机抽取n_features个特征
            features = np.random.choice(D, n_features, replace = False)
            
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
cross_val_score(fakerf, X, Y, cv=8).mean()
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
house_data = pd.read_csv(os.path.join(data_folder, "kc_house_data.csv"))
house_data.head()
house_data.columns
# price is the target
NUMERICAL_COLS = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above','sqft_basement',
                 'sqft_living15', 'sqft_lot15']
test = house_data['price'].values
test.shape
# fit 每一列数据的scaler
scalers = dict()
for col in NUMERICAL_COLS:
    scaler = StandardScaler()
    scaler.fit(house_data[col].values.astype(np.float64).reshape(-1,1))
    scalers[col] = scaler
    # TODO
    
def transform_numerical(df):
    N, _ = df.shape
    D = len(NUMERICAL_COLS)
    result = np.zeros((N,D))
    i = 0
    for col, scaler in scalers.items():
        result[:,i] = scaler.transform(df[col].values.astype(np.float64).reshape(-1,1)).ravel()
        i +=1
        # TODO
    return result    

from sklearn.model_selection import train_test_split
hdata = transform_numerical(house_data[NUMERICAL_COLS]) # TODO  transform your data
train_data, test_data = train_test_split(hdata, test_size = 0.3, random_state = 0) # train test split
trainX, trainY = train_data[:,1:], train_data[:,0] # train data and train target
testX, testY = test_data[:, 1:], test_data[:,0] # test data and test target

rfregressor = RandomForestRegressor(n_estimators= 20, max_depth = 5) # TODO
rfregressor.fit(trainX, trainY)
predictions = rfregressor.predict(testX)
# TODO
plt.xlabel("target")
plt.ylabel("prediction")
ymin = np.round(min(min(testY), min(predictions)))
ymax = np.ceil(max(max(testY), max(predictions)))
r = range(int(ymin), int(ymax) + 1)
plt.plot(r,r)
plt.scatter(testY, predictions)
plt.show()
plt.plot(testY, label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()
rfregressor.score(testX, testY)
lr = LinearRegression() # TODO
print("linear regression performance: %f" % (cross_val_score(lr, trainX, trainY, cv = 10).mean())) 
print("random forest regressor performance: %f" % (cross_val_score(rfregressor, trainX, trainY, cv = 10).mean()))
lr.fit(trainX, trainY)
print("linear regression test score: %f" % (lr.score(testX, testY)))
rfregressor.fit(trainX, trainY)
print("random forest regressor test score: %f" % (rfregressor.score(testX, testY)))
