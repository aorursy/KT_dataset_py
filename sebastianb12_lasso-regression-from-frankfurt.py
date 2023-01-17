import numpy as np

import pandas as pd

import seaborn as sns



np.random.seed(42)

random_state= 42

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

print(df_train.shape)

print(df_test.shape)
df_train.head()
df_train.tail()
df_train = df_train.set_index("Id")

df_test = df_test.set_index("Id")



y_train = df_train["SalePrice"]

df_train = df_train.drop("SalePrice",axis=1)

df = pd.concat([df_train, df_test], sort = True)

df.describe()
missing = df.shape[0]-df.count()

print(missing[missing>0])
df["MasVnrType"].fillna("None",inplace=True)



cat_columns = ["MSSubClass", "MSZoning", "Alley", "LotShape", "LandContour", "LotConfig", "LandSlope",\

               "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",\

               "Exterior2nd", "MasVnrType", "Foundation", "Heating", "Electrical", "GarageType", "MiscFeature",\

               "SaleType", "SaleCondition"]



df = pd.get_dummies(df, prefix_sep="_",columns=cat_columns)
for s,i in zip(["Grvl","Pave"],[0,1]):

    df["Street"] = df["Street"].replace(s, i)

    

for s,i in zip(["AllPub","NoSewr","NoSeWa","ELO"],[4,3,2,1]):

    df["Utilities"] = df["Utilities"].replace(s, i)   

    

for s,i in zip(["Ex","Gd","TA","Fa","Po"],[5,4,3,2,1]):

    df["ExterQual"] = df["ExterQual"].replace(s, i) 

    df["ExterCond"] = df["ExterCond"].replace(s, i) 

    df["BsmtQual"] = df["BsmtQual"].replace(s, i) 

    df["BsmtCond"] = df["BsmtCond"].replace(s, i) 

    df["HeatingQC"] = df["HeatingQC"].replace(s, i) 

    df["KitchenQual"] = df["KitchenQual"].replace(s, i) 

    df["FireplaceQu"] = df["FireplaceQu"].replace(s, i) 

    df["GarageQual"] = df["GarageQual"].replace(s, i) 

    df["GarageCond"] = df["GarageCond"].replace(s, i)

    df["PoolQC"] = df["PoolQC"].replace(s, i) 

    

df["BsmtQual"].fillna(0, inplace=True)

df["BsmtCond"].fillna(0, inplace=True)

df["FireplaceQu"].fillna(0, inplace=True)

df["GarageQual"].fillna(0, inplace=True)

df["GarageCond"].fillna(0, inplace=True)

df["PoolQC"].fillna(0, inplace=True)





for s,i in zip(["Gd","Av","Mn","No"],[4,3,2,1]):

    df["BsmtExposure"] = df["BsmtExposure"].replace(s, i) 

df["BsmtExposure"].fillna(0, inplace=True)



for s,i in zip(["GLQ","ALQ","BLQ","Rec","LwQ","Unf"],[6,5,4,3,2,1]):

    df["BsmtFinType1"] = df["BsmtFinType1"].replace(s, i) 

    df["BsmtFinType2"] = df["BsmtFinType2"].replace(s, i) 

df["BsmtFinType1"].fillna(0, inplace=True)

df["BsmtFinType2"].fillna(0, inplace=True)



for s,i in zip(["N","Y"],[0,1]):

    df["CentralAir"] = df["CentralAir"].replace(s, i) 

    

for s,i in zip(["Typ","Min1","Min2","Mod","Maj1","Maj2","Sev","Sal"],[8,7,6,5,4,3,2,1]):

    df["Functional"] = df["Functional"].replace(s, i) 



for s,i in zip(["Fin","RFn","Unf"],[3,2,1]):

    df["GarageFinish"] = df["GarageFinish"].replace(s, i) 

df["GarageFinish"].fillna(0, inplace=True)



for s,i in zip(["Y","P","N"],[3,2,1]):

    df["PavedDrive"] = df["PavedDrive"].replace(s, i) 



for s,i in zip(["GdPrv","MnPrv","GdWo","MnWw"],[4,3,2,1]):    

    df["Fence"] = df["Fence"].replace(s, i) 

df["Fence"].fillna(0, inplace=True)

df["MasVnrArea"].fillna(0, inplace=True)

df = df.fillna(df.mean())
df['Total_SF']= df["2ndFlrSF"] +df["1stFlrSF"] +df["TotalBsmtSF"]



df['Total_No_Bathrooms'] = (df["FullBath"] + (0.5 * df["HalfBath"]) +

                               df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"]))



df['Total_Porch_SF'] = (df["OpenPorchSF"] + df["3SsnPorch"] +

                              df["EnclosedPorch"] + df["ScreenPorch"] +

                              df["WoodDeckSF"])





X_train = df.loc[:1460,:]

X_test = df.loc[1461:,:]

sns.distplot(y_train)
from sklearn import metrics

import matplotlib.pyplot as plt 

def evaluation(y,pred):

    print('MAE:', metrics.mean_absolute_error(y, pred))

    print('MSE:', metrics.mean_squared_error(y, pred))

    print('RMSE:', np.sqrt(metrics.mean_squared_error(y, pred)))

    print('RMSLE:', np.sqrt(metrics.mean_squared_log_error(y, pred)))



    plt.figure(figsize=(8,8))

    plt.scatter(y,pred)

    plt.xlabel('Y Test')

    plt.ylabel('Predicted Y')

    plt.show()
from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import PowerTransformer, RobustScaler

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

from math import sqrt



pipe = Pipeline([('scl', RobustScaler()), ('clf', Lasso(random_state = random_state, max_iter=1e7))])

param_range = np.arange(0.0002,0.0008,0.0001)

param_grid = [{'clf__alpha':param_range}]



#Cross Validation

gs = GridSearchCV(estimator = pipe, param_grid = param_grid, scoring = 'neg_mean_squared_error', cv = 5, n_jobs=-1)

gs = gs.fit(X_train, np.log1p(y_train))

print(np.sqrt(-gs.best_score_))

print(gs.best_params_)



y_pred = gs.predict(X_train)



evaluation(y=y_train,pred=np.expm1(y_pred))



#Nested Cross Validation

# gs = GridSearchCV(estimator = pipe, param_grid = param_grid, scoring = 'neg_mean_squared_error', cv = 10, n_jobs=-1)

# scores = cross_val_score(gs, X_train, np.log1p(y_train), scoring='neg_mean_squared_error', cv = 10)

# print('N-CV: ' + str(np.sqrt(-np.mean(scores))) + ' ,std:'+   str(np.sqrt(np.std(scores))))
residual = np.log1p(y_train) - y_pred

z_score = (residual - residual.mean()) / residual.std()

z_score = z_score.abs()

outliers = z_score > 3

outliers[outliers]
plt.figure(figsize=(6, 6))

plt.scatter(np.log1p(y_train), y_pred)

plt.scatter(np.log1p(y_train)[outliers], y_pred[np.array(outliers)])

plt.plot(range(10, 15), range(10, 15), color="red")
X_train = X_train.drop(outliers[outliers].index)

y_train = y_train.drop(outliers[outliers].index)
pipe = Pipeline([('scl', RobustScaler()), ('clf', Lasso(random_state = random_state, max_iter=1e7))])

param_range = np.arange(0.0002,0.0008,0.0001)

param_grid = [{'clf__alpha':param_range}]





#Cross Validation

gs = GridSearchCV(estimator = pipe, param_grid = param_grid, scoring = 'neg_mean_squared_error', cv = 5, n_jobs=-1)

gs = gs.fit(X_train, np.log1p(y_train))

print("Cross Validation Score:")

print('RMSLE:', np.sqrt(-gs.best_score_))

print("Best Alpha")

print(gs.best_params_)



y_pred = gs.predict(X_train)



print("Scores on Training set:")

evaluation(y=y_train,pred=np.expm1(y_pred))

model = gs.best_estimator_

model.fit(X_train, np.log1p(y_train))



submission_predictions = np.expm1(model.predict(X_test))

df_submission = pd.DataFrame(columns=["Id","SalePrice"])

df_submission["Id"] = df_test.index

df_submission["SalePrice"] = submission_predictions

df_submission.to_csv("submission.csv", index=False)