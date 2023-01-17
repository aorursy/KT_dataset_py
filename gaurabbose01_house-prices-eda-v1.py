import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
#importing data
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
quant = df_train[[f for f in df_train.columns if df_train.dtypes[f] != "object"]]

#plot all variables correlation on Seaborn heatmap
plt.figure(figsize=(30, 30))
sns.heatmap(df_train.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', square= "true")
#creating a new df with only variables that are high correlated with SalePrice
df_train.head()
correls = [f for f in quant if quant["SalePrice"].corr(quant[f])>0.5]
correl_df = df_train[correls]
plt.figure(figsize=(15,15))
sns.heatmap(correl_df.corr(),  annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', square= "true")
#sum of all missing values per column
missing = df_train.isnull().sum()
#filter for those greater than zero
missing = missing[missing>0]
missing.sort_values(inplace=True)
missing.plot.bar()
#I love facetgrid for making multiple charts
golden_vars = correl_df

golden_vars = correl_df.drop(["TotalBsmtSF","TotRmsAbvGrd", "GarageArea"], axis = 1)

def many_dist_plot(df):
    f = pd.melt(df, id_vars="SalePrice")
    print(f)
    g = sns.FacetGrid(f, col="variable", height=5, col_wrap=2, sharex = False, sharey = False)
    g = g.map(sns.distplot, "value")
    
many_dist_plot(golden_vars)


#converting qualitative variables into dummy
def extract_qual(df, cols):
    df_qual = df[cols]
    df_dummy = pd.get_dummies(df_qual)
    return df_dummy

train_qual = df_train[["LotShape", "Neighborhood", "Condition1", "HouseStyle", "ExterQual", "SaleType"]]
test_qual = df_test[["LotShape", "Neighborhood", "Condition1", "HouseStyle", "ExterQual", "SaleType"]]

#by first combining both test and train as they may not match in number of features finally (if one variable has missing values)
train_objs_num = len(train_qual)
dataset = pd.concat(objs=[train_qual, test_qual], axis=0)
dataset_preprocessed = pd.get_dummies(dataset)
train_qual = dataset_preprocessed[:train_objs_num]
test_qual = dataset_preprocessed[train_objs_num:]

train_qual
golden_vars
#1/ tack on qual variables to golden_vars
golden_vars_merged = golden_vars.merge(train_qual, left_index = True, right_index=True)

#2/ remove sales price from combined
Xs = golden_vars_merged.drop(['SalePrice'], axis = 1)
Y = df_train["SalePrice"]

#3 apply logs
def take_log(df, feature):
    df[feature] = np.log1p(df[feature].values)
    return(df)
    
take_log(Xs, "GrLivArea")
take_log(Xs, "1stFlrSF")

Y = np.log1p(Y)


Xs
#fitting linear regression

lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, Xs, Y, scoring="neg_mean_squared_error", cv =5)

mean_MSE = np.mean(MSEs)
print(mean_MSE)

model = lin_reg.fit(Xs, Y)
rsq = model.score(Xs, Y)
print(rsq)

#1/ narrow down to golden quant vars and remove Sale Price
golden_vars_test = correl_df.drop(["TotalBsmtSF","TotRmsAbvGrd", "GarageArea"], axis = 1).drop("SalePrice", axis=1).columns
golden_vars_test = df_test[golden_vars_test]
golden_vars_test
#2/ tack on qual variables
df_test_merged = golden_vars_test.merge(test_qual, left_index = True, right_index = True)
Test_Xs = take_log(df_test_merged, "1stFlrSF")
#applying linear regression to test

my_imputer = SimpleImputer()
Test_Xs = my_imputer.fit_transform(Test_Xs)

test_y = np.exp(model.predict(Test_Xs))
output = pd.DataFrame()
output["Id"] = df_test["Id"]
output["SalePrice"] = test_y
output
output.to_csv(r'KGHousePriceChallenge.csv', index = False)

from sklearn.linear_model import Ridge

ridgeReg = Ridge(alpha=0.05, normalize=True)

model_ridge = ridgeReg.fit(Xs, Y)
pred = np.exp(model_ridge.predict(Test_Xs))

MSEs = cross_val_score(model_ridge, Xs, Y, scoring="neg_mean_squared_error", cv =5)

mean_MSE = np.mean(MSEs)
print(mean_MSE)

output_ridge = pd.DataFrame()
output_ridge["Id"] = df_test["Id"]
output_ridge["SalePrice"] = pred
output_ridge.to_csv(r'KGHousePrice_Ridge.csv', index = False)
pred
pd.DataFrame(Test_Xs)
pd.DataFrame(test_y)
model_lasso
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

lassoReg = Lasso(alpha=0.0003, normalize=True)




def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, Xs, Y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(Xs, Y)
coef = pd.Series(model_lasso.coef_, index = Xs.columns)
coef.head()
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")

pred_lasso = np.exp(model_lasso.predict(Test_Xs))
output_lasso = pd.DataFrame()
output_lasso["Id"] = df_test["Id"]
output_lasso["SalePrice"] = pred
output_lasso.to_csv(r'KGHousePrice_Lasso.csv', index = False)


import xgboost as xgb
dtrain = xgb.DMatrix(Xs, label = Y)
dtest = xgb.DMatrix(Test_Xs)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

labels = Xs.columns
temp = pd.DataFrame(Test_Xs, columns = labels)
Test_Xs = temp
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=5, learning_rate=0.1) 
model_xgb.fit(Xs, Y)
xgb_preds = np.exp(model_xgb.predict(pd.DataFrame(Test_Xs)))
MSEs = cross_val_score(model_xgb, Xs, Y, scoring="neg_mean_squared_error", cv =5)

mean_MSE = np.mean(MSEs)
print(mean_MSE)

output_xgb = pd.DataFrame()
output_xgb["Id"] = df_test["Id"]
output_xgb["SalePrice"] = xgb_preds
output_xgb.to_csv(r'KGHousePriceXGB.csv', index = False)
output_xgb
from sklearn.linear_model import ElasticNet

ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)

model_EN = ENreg.fit(Xs,Y)

pred_EN = np.exp(model_EN.predict(Test_Xs))

MSEs = cross_val_score(model_EN, Xs, Y, scoring="neg_mean_squared_error", cv =5)

mean_MSE = np.mean(MSEs)
print(mean_MSE)
output_en = pd.DataFrame()
output_en["Id"] = df_test["Id"]
output_en["SalePrice"] = pred_EN
output_en.to_csv(r'KGHousePriceEN.csv', index = False)
