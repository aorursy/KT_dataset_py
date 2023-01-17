import numpy as np

import pandas as pd

import matplotlib

%matplotlib inline 

#Enable Iniline Plotting

import matplotlib.pyplot as plt

from scipy.stats import skew
traindata=pd.read_csv('../data/train.csv')

traindata=traindata.loc[:,'MSSubClass':'SalePrice']

traindata.head()
traindata['SalePrice']=np.log1p(traindata['SalePrice'])

#Get non categorical data

numeric_feats = traindata.dtypes[traindata.dtypes != "object"].index

#traindata[numeric_feats] = (traindata[numeric_feats] - traindata[numeric_feats].mean()) / (traindata[numeric_feats].max() - traindata[numeric_feats].min())

skewed_feats = traindata[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

#Apply transformation

traindata[skewed_feats] = np.log1p(traindata[skewed_feats])



#Convert categorical features

traindata=pd.get_dummies(traindata)





#Remove NaN with mean of the values

traindata=traindata.fillna(np.round(traindata.mean()))



#Shuffle the data

traindata = traindata.sample(frac=1).reset_index(drop=True)



#Generate  X and y datasets



#Make 20% of the Data Test Set for evaluating the different models

testsize=int(traindata.shape[0]*0.8)

X=traindata.drop('SalePrice',1)[0:testsize]

y=traindata[0:testsize]

y=y.loc[:,"SalePrice"]

Xtest=traindata.drop('SalePrice',1)[testsize+1:]

ytest=traindata[testsize+1:]

ytest=ytest.loc[:,"SalePrice"]



X=X.sort_index(axis=1)

Xtest=Xtest.sort_index(axis=1)
from sklearn import tree

regressorCART=tree.DecisionTreeRegressor()

regressorCART=regressorCART.fit(X,y)
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":regressorCART.predict(X), "true":y})

preds["error difference"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "error difference",kind = "scatter")
'''

import pydotplus 

dot_data = tree.export_graphviz(regressorCART, out_file=None, 

                         feature_names=list(X.columns.values),  

                         class_names='SalesPrice' , 

                         filled=True, rounded=True,  

                         special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)  

graph.write_pdf("iris.pdf") 

'''
from sklearn.ensemble import RandomForestRegressor

regressorRandomForest=RandomForestRegressor(n_estimators=10)

regressorRandomForest=regressorRandomForest.fit(X,y)
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":regressorRandomForest.predict(X), "true":y})

preds["error difference"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "error difference",kind = "scatter")
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso, LassoLarsCV

from sklearn.model_selection import cross_val_score, cross_val_predict





def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)



alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge=RidgeCV(alphas=alphas)

cv_ridge=cv_ridge.fit(X,y)

rmse_cv(cv_ridge).mean()
regressorRidge = cv_ridge

regressorRidge=regressorRidge.fit(X,y)
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":regressorRidge.predict(X), "true":y})

preds["error difference"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "error difference",kind = "scatter")
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_lasso= LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X, y)

rmse_cv(cv_lasso).mean()
regressorLasso = cv_lasso

regressorLasso=regressorLasso.fit(X,y)
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":cv_lasso.predict(X), "true":y})

preds["error difference"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "error difference",kind = "scatter")
coef = pd.Series(regressorLasso.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
#Test Set Preview

Xtest.head()
predictedCART =regressorCART.predict(Xtest)



matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":predictedCART, "true":ytest})

preds["error difference"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "error difference",kind = "scatter")
predictedRandomForest = regressorRandomForest.predict(Xtest)



matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":predictedRandomForest, "true":ytest})

preds["error difference"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "error difference",kind = "scatter")
#predictedRidge= cross_val_predict(regressorRidge, Xtest, ytest, cv=10)

predictedRidge=regressorRidge.predict(Xtest)



matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":predictedRidge, "true":ytest})

preds["error difference"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "error difference",kind = "scatter")


scoreCART=regressorRidge.score(Xtest,ytest)

scoreRandomForest=regressorRandomForest.score(Xtest,ytest)

scoreRidge=regressorRidge.score(Xtest,ytest)

scoreLasso=regressorLasso.score(Xtest,ytest)

print("CART Accuracy: "+str(scoreCART*100)+"%")

print("Random Forest Accuracy: "+str(scoreRandomForest*100)+"%")

print("Linear Regression(Ridge) Accuracy: "+str(scoreRidge*100)+"%")

print("Linear Regression(Lasso) Accuracy: "+str(scoreLasso*100)+"%")

modelsScore={scoreCART:"CART",scoreRandomForest:"Random Forest",scoreRidge:"Linear Regression",scoreLasso:"Lasso Regression"}

maxScoreModel=modelsScore[max(modelsScore)]

print("Model with Highest Accuracy is: "+maxScoreModel)
predictedRidge=np.expm1(predictedRidge) #Bring Back the non Log Data

ytest=np.expm1(ytest)





#Plot the Graph

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"Predicted":predictedRidge, "Actual":ytest})

preds.plot(x = "Predicted", y = "Actual",kind = "scatter")
predictedRidgeTrain=np.expm1(regressorRidge.predict(X)) #Bring Back the non Log Data

y=np.expm1(y)



#Plot the Graph

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"Predicted":predictedRidgeTrain, "Actual":y})

preds.plot(x = "Predicted", y = "Actual",kind = "scatter")


#Get the complete Data Set Now

Xcomplete=traindata.drop('SalePrice',1)

ycomplete=traindata.loc[:,'SalePrice']

Xcomplete=Xcomplete.sort_index(axis=1)



#Fit the values

regressorLasso.fit(Xcomplete,ycomplete)



#load Test Data

testdata=pd.read_csv('../data/test.csv')

Ids=testdata.loc[:,'Id']

testdata=testdata.drop('Id',1)





#Use the existing skewed index

testdata[skewed_feats] = np.log1p(testdata[skewed_feats])



#Bit value encoding

testdata=pd.get_dummies(testdata)



testdata=testdata.fillna(np.round(Xcomplete.mean()))

#Generate empty datadrame with same feature names

df_copy = pd.DataFrame(columns=Xcomplete.columns)



#Combine dataframes

result=df_copy.append(testdata,ignore_index=True)

result.sort_index(axis=1)

result=result.fillna(0)



#Predict values

predictedValues=regressorLasso.predict(result)

predictedValues=np.expm1(predictedValues)

outputdataFrame=pd.DataFrame({'Id':Ids,'SalePrice':predictedValues})



#Save to csv

outputdataFrame.to_csv('output.csv',index=False)

outputdataFrame.head()