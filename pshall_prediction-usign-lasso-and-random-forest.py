

import pandas as pd

import numpy as np

from scipy.stats import skew

from sklearn.linear_model import LassoCV

from sklearn.cross_validation import cross_val_score

import xgboost as xgb



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))

    return(rmse)

    

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#all training and testing data concatenated together

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))



count=0;

for k in all_data['MSSubClass']:

    if k==30 or k==45 or k==70 or k==85 or k==180 or k==190:

        all_data.set_value(count,'MSSubClass',10)

    else :

        all_data.set_value(count,'MSSubClass',20)

    count+=1

    print(count)

    if(count>=1460):

        break



all_data['YearBuilt']=2016-all_data['YearBuilt']

all_data['YearRemodAdd']=2016-all_data['YearRemodAdd']

all_data['GarageYrBlt']=2016-all_data['GarageYrBlt']

all_data['YrSold']=2016-all_data['YrSold']

    

train["SalePrice"] = np.log1p(train["SalePrice"])



numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

##print (skewed_feats)

#

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

all_data=all_data.fillna(all_data.median(0,True))

#all_data.to_csv("./processed_data.csv", index=False)







#---------------------data preprocessing end-----------------





X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice

temp=X_train.join(y)



temp.to_csv("./processed_train_f3.csv", index=False)

X_test.to_csv("./processed_test_f3.csv", index=False)





#model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

#model_lasso = LassoCV(alphas = [0.1, 0.001, 0.0005,.0000001]).fit(X_train, y)

#preds = np.expm1(model_lasso.predict(X_test))

#preds["residuals"] = preds["true"] - preds["preds"]

#solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

#solution['SalePrice']+=test['MiscVal']

#solution.to_csv("ridge_sol.csv", index = False)



#--------------------model2:ridge-----------------

#from sklearn.linear_model import Ridge

#ridgereg = Ridge(alpha=.001,normalize=True)

#ridgereg.fit(X_train, y)

#y_pred = np.expm1(ridgereg.predict(X_test))

#

#final_preds = preds

#solution = pd.DataFrame({"id":test.Id, "SalePrice":final_preds})

#solution['SalePrice']+=test['MiscVal']

#solution.to_csv("merge.csv", index = False)







#model_lasso = LassoCV(alphas = [0.1, 0.001, 0.0005,.0000001]).fit(X_train, y)

#rmse_cv(model_lasso).mean()

#coef = pd.Series(model_lasso.coef_, index = X_train.columns)

#print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

#imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])

#imp_coef.plot(kind = "barh")



#preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

#preds["residuals"] = preds["true"] - preds["preds"]

#preds = np.expm1(model_lasso.predict(X_test))

#solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

#solution['SalePrice']+=test['MiscVal']

#solution.to_csv("ridge_ay4.csv", index = False)

#-----------------------------

dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)

xgb_preds = np.expm1(model_xgb.predict(X_test))

model_lasso = LassoCV(alphas = [0.1, 0.001, 0.0005,.0000001]).fit(X_train, y)

lasso_preds = np.expm1(model_lasso.predict(X_test))

predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

preds = 0.7*lasso_preds + 0.3*xgb_preds

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("ridge_sol.csv", index = False)