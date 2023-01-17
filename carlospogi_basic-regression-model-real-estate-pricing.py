import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/lamudi.csv")
df.head() #top 5 rows in the dataset
df.info() #data information
df.describe()
df.hist(bins=50, figsize=(20,15))

plt.show()
corr_matrix = df.corr()
corr_matrix["price"].sort_values(ascending=False)
scatter_matrix(df,figsize=(12,8))


df.plot(kind="scatter", x="price",y="floor_area", alpha=0.1)
df["rooms_per_floor_area"] = df["Bedroom"] / df["floor_area"] # bedroom per floor area 

df["floor_area_per_land_size"] = df["floor_area"] / df["land_size"] #floor area per land size
corr_matrix = df.corr()

corr_matrix["price"].sort_values(ascending=False)
# X = df[['Bedroom','floor_area','land_size']] #select feature

# X = df[['Bedroom','land_size']] #select feature

X = df[['land_size']] #select feature

y = df[['price']].values   #select target var

y = y.reshape(-1,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
# train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
lm = LinearRegression() 

lm.fit(X_train,y_train)
y_train_pred = lm.predict(X_train)
#make prediction using the training set first

y_train_pred = lm.predict(X_train)
sTrain = mean_squared_error(y_train,y_train_pred)

print("Mean Squared error of training set: %2f"%sTrain)
np.sqrt(sTrain)
dt = DecisionTreeRegressor()

dt.fit(X_train,y_train)
dt_y_train_pred = dt.predict(X_train)
dtsTrain = mean_squared_error(y_train,dt_y_train_pred)

print("Mean squared error of testing set: %.2f"%dtsTrain)
np.sqrt(dtsTrain)
rf = RandomForestRegressor() 

rf.fit(X_train,y_train)
rf_y_train_pred = rf.predict(X_train)
rfsTrain = mean_squared_error(y_train,rf_y_train_pred)

print("Mean squared error of testing set: %.2f"%rfsTrain)
np.sqrt(rfsTrain)
svr = SVR(kernel= 'linear')

svr.fit(X_train,y_train)
svr_y_train_pred = svr.predict(X_train)
svrTrain = mean_squared_error(y_train,svr_y_train_pred)

print("Mean squared error of testing set: %.2f"%svrTrain)
np.sqrt(svrTrain)
def display_scores(scores):

    print("Scores:",scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())

scores = cross_val_score(dt, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
scores2 = cross_val_score(lm, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
lm_rmse_scores = np.sqrt(-scores2)

display_scores(lm_rmse_scores)
scores3 = cross_val_score(rf, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
rf_rmse_scores = np.sqrt(-scores3)

display_scores(rf_rmse_scores)
scores4 = cross_val_score(svr, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
svr_rmse_scores = np.sqrt(-scores4)

display_scores(svr_rmse_scores)