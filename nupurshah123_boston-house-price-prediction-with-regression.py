import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.pipeline import make_pipeline
train_df = pd.read_csv("../input/boston-housing-dataset/house-price-prediction-with-boston-housing-dataset/train1.csv")
train_df.head()
train_df.info()
train_df["CRIM"] = train_df["CRIM"].replace(np.nan,train_df["CRIM"].median())
train_df["ZN"] = train_df["ZN"].replace(np.nan,train_df["ZN"].median())
train_df["INDUS"] = train_df["INDUS"].replace(np.nan,train_df["INDUS"].mean())
train_df["CHAS"] = train_df["CHAS"].replace(np.nan,train_df["CHAS"].median())
train_df["AGE"] = train_df["AGE"].replace(np.nan,train_df["AGE"].median())
train_df["LSTAT"] = train_df["LSTAT"].replace(np.nan,train_df["LSTAT"].mean())
train_df.head()
corrmat = train_df.corr()
corrmat.head()
plt.figure(figsize=(12,9))
sns.heatmap(corrmat, square=True,annot=True,fmt='.2f',annot_kws={'size': 10});
sns.distplot(train_df['MEDV'])
plt.scatter(train_df['LSTAT'],train_df['MEDV'])
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.show()
plt.scatter(train_df['RM'],train_df['MEDV'])
plt.xlabel("RM")
plt.ylabel("MEDV")
plt.show()
Y = train_df[['MEDV']].values
train_df.drop(["Id","MEDV"],axis=1,inplace=True)
cols = train_df.columns
X = train_df.values
x_scaled = StandardScaler().fit_transform(X)
train_df = pd.DataFrame(x_scaled,columns=cols)
X = train_df.values
train_df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('MLP', MLPRegressor(max_iter = 500)))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DecisionTree', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
for name, model in models:
    model.fit(X_train,Y_train)
    y = model.predict(X_test)
    print(name + " : " + str(mean_squared_error(Y_test, y,squared=False)))
import tensorflow as tf
from tensorflow import keras
model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[13], activation = 'relu'),
])
model.compile(
  optimizer='sgd',
  loss='mean_squared_error'
)
model.fit(X_train,Y_train, epochs=800)
ypred = model.predict(X_test)
print(mean_squared_error(Y_test, ypred,squared=False))
for degree in range(2,5):
    model = make_pipeline(PolynomialFeatures(degree, interaction_only=False),LinearRegression())
    model.fit(X_train,Y_train)
    y = model.predict(X_test)
    print(" PF: " +str(degree) + " "+ str(mean_squared_error(Y_test, y,squared=False)))
    
test_df = pd.read_csv("../input/boston-housing-dataset/house-price-prediction-with-boston-housing-dataset/test1.csv")

model = KNeighborsRegressor()
model.fit(X_train,Y_train)
test_df["CRIM"] = test_df["CRIM"].replace(np.nan,test_df["CRIM"].median())
test_df["ZN"] = test_df["ZN"].replace(np.nan,test_df["ZN"].median())
test_df["INDUS"] = test_df["INDUS"].replace(np.nan,test_df["INDUS"].mean())
test_df["CHAS"] = test_df["CHAS"].replace(np.nan,test_df["CHAS"].median())
test_df["AGE"] = test_df["AGE"].replace(np.nan,test_df["AGE"].median())
test_df["LSTAT"] = test_df["LSTAT"].replace(np.nan,test_df["LSTAT"].mean())
test_df.drop(["Id"],inplace = True,axis=1)
x = test_df.values
x_scaled = StandardScaler().fit_transform(x)
y = model.predict(x_scaled)
test_df = pd.read_csv("../input/boston-housing-dataset/house-price-prediction-with-boston-housing-dataset/test1.csv")
pred = pd.DataFrame(y)
datasets = pd.concat([test_df["Id"],pred],axis=1)
datasets.columns=["Id","MEDV"]
datasets.to_csv("submi.csv",index=False)
submit = pd.read_csv("./submi.csv")
submit