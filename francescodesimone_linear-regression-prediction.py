# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr

import xgboost as xgb

import pandas as panda

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.cross_validation import cross_val_score



import numpy as numpyObj

import matplotlib

df= pd.read_csv('../input/test.csv')

df.head()
import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr

import xgboost as xgb

import pandas as panda

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.cross_validation import cross_val_score



import numpy as numpyObj

import matplotlib



def rmse_cv(xgBoostmodel):

    rmse= numpyObj.sqrt(-cross_val_score(xgBoostmodel, Xaxis_trainData, yaxis, scoring="mean_squared_error", cv = 5))

    return(rmse)



def skewNumericFields(entireData):

	numerical_features = entireData.dtypes[entireData.dtypes != "object"].index

	#those skweing numerical

	skewed_features = trainData[numerical_features].apply(lambda x: skew(x.dropna())) 

	skewed_features = skewed_features[skewed_features > 0.75]

	skewed_features = skewed_features.index

	entireData[skewed_features] = numpyObj.log1p(entireData[skewed_features])

	return entireData



def nonNumericFields(entireData):

	columns = entireData.columns.values

	nom_numeric_cols = ['MSSubClass']

	dummy_data = []

	for eachCol in columns:

	    #for every non categirical and column 'mssubclass' u create numerical value using the dummy variable like bit code

	    if entireData[eachCol].dtype.name == 'object' or eachCol in nom_numeric_cols:

		    dummy_data.append(panda.get_dummies(entireData[eachCol].values.astype(str), eachCol))

		    dummy_data[-1].index = entireData.index

		    del entireData[eachCol]

	entireData = panda.concat([entireData] + dummy_data, axis=1)

	#for all empty values replace with mean

	entireData = entireData.fillna(entireData.mean())

	return entireData



testData = panda.read_csv("../input/test.csv")

trainData = panda.read_csv("../input/train.csv")

trainData["SalePrice"] = numpyObj.log1p(trainData["SalePrice"])

yaxis = trainData.SalePrice



entireData = panda.concat((trainData.loc[:,'MSSubClass':'SaleCondition'],

                      testData.loc[:,'MSSubClass':'SaleCondition']))

                      

entireData = skewNumericFields(entireData)

entireData = nonNumericFields(entireData)



Xaxis_testData = entireData[trainData.shape[0]:]

Xaxis_trainData = entireData[:trainData.shape[0]]





xgBoostmodel_ridge = Ridge()

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]

            

xgBoostmodel_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(Xaxis_trainData, yaxis)

rmse_cv(xgBoostmodel_lasso).mean()

coefficients = panda.Series(xgBoostmodel_lasso.coef_, index = Xaxis_trainData.columns)

predictions_lasso = numpyObj.expm1(xgBoostmodel_lasso.predict(Xaxis_testData))



#xgboost library

dataToTest = xgb.DMatrix(Xaxis_testData)

allParameters = {"max_depth":5, "eta":0.07}

dataToTrain = xgb.DMatrix(Xaxis_trainData, label = yaxis)



xgBoostmodel = xgb.cv(allParameters, dataToTrain,  num_boost_round=1000, early_stopping_rounds=100)



cv_ridge = panda.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")



plt.xlabel("alpha")





cv_ridge.min()



xgBoostmodel.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

xgBoostmodel_xgb = xgb.XGBRegressor(n_estimators=10000,max_depth=5,min_child_weight=1.5,reg_alpha=0.75,reg_lambda=0.45,learning_rate=0.07,subsample=0.95)



xgBoostmodel_xgb.fit(Xaxis_trainData, yaxis)

predictions_xgboost = numpyObj.expm1(xgBoostmodel_xgb.predict(Xaxis_testData))



predictedValues = panda.DataFrame({"xgb":predictions_xgboost, "lasso":predictions_lasso})

prediction = 0.7*predictions_lasso + 0.3*predictions_xgboost

predictedValues.plot(x = "xgb", y = "lasso", kind = "scatter")



finalvalues = panda.DataFrame({"id":testData.Id, "SalePrice":prediction})

print (finalvalues)

#finalvalues.to_csv("finalSolution.csv", index = False)