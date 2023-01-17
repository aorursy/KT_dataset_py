import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = train.dtypes[train.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
train[skewed_feats] = np.log1p(train[skewed_feats])
test[skewed_feats] = np.log1p(test[skewed_feats])

# Handle categorical data
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Normalize
train_x = train.drop("SalePrice", axis=1)
train_x = train_x.drop("Id", axis=1)
train_y = train["SalePrice"]
train = (train_x - train_x.mean()) / (train_x.max() - train_x.min())
train["SalePrice"] = train_y
test_id = test["Id"]
test = (test - train_x.mean()) / (train_x.max() - train_x.min())
test["Id"] = test_id

#filling NA's with the mean of the column:
train = train.fillna(train.mean())
test = test.fillna(train.mean())
train_x = train.drop("SalePrice", axis=1)
train_y = train["SalePrice"]
X_tr, X_val, y_tr, y_val = train_test_split(train_x, train_y, test_size=0.25)
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model, x, y):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
model_ridge = RidgeCV(alphas = alphas)
cv_ridge = rmse_cv(model_ridge, train_x, train_y)
print(cv_ridge.mean())
alphas = [1, 0.1, 0.001, 0.0005]
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train_x, train_y)
cv_lasso = rmse_cv(model_lasso, train_x, train_y)
print(cv_lasso.mean())
coef = pd.Series(model_lasso.coef_, index = train_x.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
preds = model_lasso.predict(test.drop("Id", axis=1))
solution = pd.DataFrame({"id":test.Id, "SalePrice":np.expm1(preds)})
solution.to_csv("lasso_sol.csv", index = False)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

forest_model = RandomForestRegressor()
forest_model.fit(X_tr, y_tr)
pred_val = forest_model.predict(X_val)

print(mean_squared_error(y_val, pred_val))
forest_model = RandomForestRegressor()
forest_model.fit(train_x, train_y)
preds = forest_model.predict(test.drop("Id", axis=1))
solution = pd.DataFrame({"id":test.Id, "SalePrice":np.expm1(preds)})
solution.to_csv("rf_sol.csv", index = False)
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras import backend as K
nn_model = Sequential()
BatchNormalization()
nn_model.add(Dense(1028,input_dim=288,activation='relu'))
BatchNormalization()
Dropout(0.5)
nn_model.add(Dense(100,input_dim=1028,activation='relu'))
BatchNormalization()
Dropout(0.8)
nn_model.add(Dense(50))
BatchNormalization()
nn_model.add(Dense(1))
#sgd = SGD(lr=0.01)
nn_model.compile( optimizer='adam',loss='mse',metrics=['mean_squared_error'])
nn_model.fit(X_tr,y_tr,validation_data=(X_val,y_val),epochs=30,batch_size=100)
print(np.sqrt(nn_model.evaluate(X_val,y_val)))
preds = nn_model.predict(np.array(X_val))
preds = [p[0] for p in preds]
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
print(rmse(preds,y_val))
preds = nn_model.predict(np.array(test.drop("Id", axis=1)))
preds = [p[0] for p in preds]
solution = pd.DataFrame({"id":test.Id, "SalePrice":np.expm1(preds[0])})
solution.to_csv("nn_sol.csv", index = False)