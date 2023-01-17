import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")



print("Dimensions of train: {}".format(train.shape))

print("Dimensions of test: {}".format(test.shape))
test.head()
plt.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

prices['price'].hist()

prices['price'].head()
list(train)
columns = ['SalePrice','MSSubClass', 'Neighborhood', 'OverallQual', 'OverallCond', '1stFlrSF',

       '2ndFlrSF','KitchenQual',

       'SaleType', 'SaleCondition','GrLivArea']



train[columns].head(10)
import matplotlib.pyplot as plt



class_pivot = pd.pivot_table(train[columns],index="MSSubClass",values="SalePrice",aggfunc=[np.mean])

class_pivot.plot.bar()

plt.show()
class_pivot = pd.pivot_table(train[columns],index="1stFlrSF",values="SalePrice",aggfunc=[np.mean])

class_pivot.plot.bar()

plt.show()
class_pivot = pd.pivot_table(train[columns],index=["OverallQual"],values="SalePrice",aggfunc=[np.mean])

class_pivot.plot.bar()

plt.show()
def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df



train = create_dummies(train,"MSSubClass")

test = create_dummies(test,"MSSubClass")

train = create_dummies(train,"Neighborhood")

test = create_dummies(test,"Neighborhood")

train = create_dummies(train,"KitchenQual")

test = create_dummies(test,"KitchenQual")

train = create_dummies(train,"SaleType")

test = create_dummies(test,"SaleType")

train = create_dummies(train,"SaleCondition")

test = create_dummies(test,"SaleCondition")



list(train)
import pandas as pd

import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer

import seaborn as sns
columns = ['OverallQual', 'OverallCond', '1stFlrSF',

       '2ndFlrSF','KitchenQual_Ex','KitchenQual_Fa','KitchenQual_Gd','KitchenQual_TA']



holdout = test[columns] # from now on we will refer to this

               # dataframe as the holdout data

holdout.head()

    

from sklearn.model_selection import train_test_split



all_X = train[columns]

all_y = train['SalePrice']



X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.20,random_state=0)
all_X.head()
all_y.head()
X_train.head(10)
X_test.head(10)
y_train.head(10)
y_test.head(10)
n_folds = 5

from sklearn.metrics import make_scorer

from sklearn.model_selection import KFold

scorer = make_scorer(mean_squared_error,greater_is_better = False)

def rmse_CV_train(model):

    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)

    rmse = np.sqrt(-cross_val_score(model,X_train,y_train,scoring ="neg_mean_squared_error",cv=kf))

    return (rmse)

def rmse_CV_test(model):

    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)

    rmse = np.sqrt(-cross_val_score(model,X_test,y_test,scoring ="neg_mean_squared_error",cv=kf))

    return (rmse)
lr = LinearRegression()

lr.fit(X_train,y_train)

test_pre = lr.predict(X_test)

train_pre = lr.predict(X_train)

print('rmse on train',rmse_CV_train(lr).mean())

print('rmse on test',rmse_CV_test(lr).mean())
#plot between predicted values and residuals

plt.scatter(train_pre, train_pre - y_train, c = "blue",  label = "Training data")

plt.scatter(test_pre,test_pre - y_test, c = "black",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
# Plot predictions - Real values

plt.scatter(train_pre, y_train, c = "blue",  label = "Training data")

plt.scatter(test_pre, y_test, c = "black",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()
columns = ['OverallQual', 'OverallCond', '1stFlrSF',

       '2ndFlrSF','KitchenQual_Ex','KitchenQual_Fa','KitchenQual_Gd','KitchenQual_TA']



holdout = test[columns] # from now on we will refer to this

               # dataframe as the holdout data



all_X = train[columns]

all_y = train['SalePrice']
print(type(all_X))

print(type(all_y))

print(type(holdout))
all_y.head()
holdout.head()
lr = LinearRegression()

lr.fit(all_X,all_y)

holdout_predictions = pd.DataFrame(lr.predict(holdout), columns=['SalePrice'])

holdout_predictions['Id'] = test['Id']
holdout_predictions[['Id', 'SalePrice']].head()
holdout_predictions[['Id', 'SalePrice']][holdout_predictions['SalePrice'] < 0]
holdout_predictions.loc[holdout_predictions['SalePrice'] < 0, 'SalePrice'] = 500

holdout_predictions[['Id', 'SalePrice']][holdout_predictions['SalePrice'] < 0]
holdout_predictions[['Id', 'SalePrice']].to_csv("houseprices_submission.csv", index=False)