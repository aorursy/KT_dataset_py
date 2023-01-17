import numpy as np

import pandas as pd

import seaborn as sns

from scipy.stats import skew

import matplotlib.pyplot as plt

from math import ceil

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LassoCV

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Lasso



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print("Size of training data:")

print(train.shape)

print ("Size of test data: ")

print(test.shape)
quantTrain = [f for f in train.columns if train.dtypes[f] != 'object']

quantTrain.remove('SalePrice')

quantTrain.remove('Id')

qualTrain = [f for f in train.columns if train.dtypes[f] == 'object']



print("There are "+str(len(qualTrain)) + " Qualitative Attributes and " + str(len(quantTrain)) + " Quantitative Attributes in the training data\n")

print("Qualitative features are: " + str(qualTrain) + "\n")

print("Quantitative features are: " + str(quantTrain))

fig, ax = plt.subplots(ceil(len(qualTrain)/2), 2, figsize = (10, 100))

i=0

j=0

for feature in qualTrain:

    sns.boxplot(x = feature, y = 'SalePrice', data = train, ax = ax[i, j])

    j=j+1

    if(j==2):

        j=0

        i=i+1

fig.set_size_inches(12, 90)

plt.tight_layout(pad=0.3, w_pad=0.7, h_pad=1)
def pairplot(x, y, **kwargs):

    ax = plt.gca()

    ts = pd.DataFrame({'time': x, 'val': y})

    ts = ts.groupby('time').mean()

    ts.plot(ax=ax)

    plt.xticks(rotation=90)

    

f = pd.melt(train, id_vars=['SalePrice'], value_vars=qualTrain)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

g = g.map(pairplot, "value", "SalePrice")
fig, ax = plt.subplots(int(len(quantTrain)/3), 3, sharey=True, figsize=(15, 90))

i=0

j=0

for feature in quantTrain:

    ax[i, j].scatter(train[feature], train.SalePrice)

    ax[i, j].set_title(feature)

    j=j+1

    if(j==3):

        j=0

        i=i+1

plt.tight_layout(pad=0.3, w_pad=0.7, h_pad=1)

missingTrain = train.isnull().sum()

missingTrain = missingTrain[missingTrain > 0]

missingTrain.plot.bar()
correlation = train.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

plt.figure(figsize=(8, 8))

plt.matshow(correlation)
correlationDict = correlation['SalePrice'].to_dict()

del correlationDict['SalePrice'] # Removing SalePrice from analysis

print("Quantitative feature correlation with Sale Price in descending order:")

for element in sorted(correlationDict.items(), key = lambda l: -abs(l[1])):

    print(str(element[0]) + ": \t" + str(element[1]))
del train['BsmtHalfBath']

del test['BsmtHalfBath']

del train['BsmtFinSF2']

del test['BsmtFinSF2']



print("Size of training data:")

print(train.shape)

print ("Size of test data: ")

print(test.shape)
priceHist = pd.DataFrame({"price":train["SalePrice"]})

priceHist.hist()
logTransformed = np.log(train["SalePrice"])

logHist = pd.DataFrame({"log transformed price":logTransformed})

logHist.hist()



train["SalePrice"] = np.log1p(train["SalePrice"])
reqColumns = train.columns.difference(['Id','SalePrice'])



combinedData = pd.concat((train.loc[:,reqColumns], test.loc[:,reqColumns]))



quantData = combinedData.dtypes[combinedData.dtypes != "object"].index



# Calculating Skewness of the overall Quantitative Data

skewedData = train[quantData].apply(lambda l: skew(l.dropna()))

# Selecting Columns with Skewness > 0.80

skewedData = skewedData[skewedData > 0.80]



# Normalizing Skewed Data with Log transformation

combinedData[skewedData.index] = np.log1p(combinedData[skewedData.index]) # log1p to avoid error due to NaN data

combinedData = pd.get_dummies(combinedData)

combinedData = combinedData.fillna(combinedData.mean())

missingData = combinedData.isnull().sum()

missingData = missingData[missingData > 0]

print("Number of Missing values: " +str(len(missingData)))

print("Size of Combined Data: " + str(len(combinedData)))
stdScaler = StandardScaler()



stdScaler.fit(combinedData[quantData])

scaledData = stdScaler.transform(combinedData[quantData])



X = pd.DataFrame()

X['Feature'] = quantData



for i, feature in enumerate(quantData):

    scaledFeature = scaledData[:, i]

    X["Old Variance"] = str(np.var(combinedData[feature]))

    X["New Variance"] = str(np.var(scaledFeature))

X
trainComb = combinedData[:train.shape[0]]



dropFeature = ['RoofMatl_ClyTile', 'Condition2_PosN', 'MSZoning_C (all)', 'BsmtCond_Po','Condition1_RRNe','Condition2_Artery', \

'Condition2_PosA','Condition2_RRAe','Condition2_RRAn','Condition2_RRNn','Electrical_Mix','ExterCond_Po','Exterior1st_AsphShn',\

'Exterior1st_BrkComm','Exterior1st_CBlock','Exterior1st_ImStucc','Exterior1st_Stone','Exterior2nd_CBlock','Exterior2nd_Other',\

'Functional_Sev','GarageCond_Ex','Heating_Floor','MiscFeature_Gar2','MiscFeature_Othr','MiscFeature_TenC',\

'PoolQC_Ex','PoolQC_Fa','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','RoofStyle_Shed','SaleType_Con','Utilities_NoSeWa']



print("Analysing scatter plots of the all the features, I noticed that there were no significant data for the following attributes:")

print(dropFeature)



fig, ax = plt.subplots(ceil(len(dropFeature)/3), 3, figsize=(15, 70))

i=0

j=0

for feature in dropFeature:

    ax[i, j].scatter(trainComb[feature], train.SalePrice, c = "blue", marker = "s")

    ax[i, j].set_title(feature)

    j=j+1

    if(j==3):

        j=0

        i=i+1



plt.tight_layout(pad=0.3, w_pad=0.7, h_pad=1)



print("Hence, dropping the same will help improve the data")

combinedData = combinedData.drop(dropFeature, axis=1)

print("New Size of Combined Data: " + str(len(combinedData)))

print("\nScatter plots for the same:")
trainComb = combinedData[:train.shape[0]]



def RMSError(model):

    rmse= np.sqrt(-cross_val_score(model, trainComb, train.SalePrice, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)



alphaArr = [1e-4, 5e-4, 1e-3, 5e-3]

lasso = [RMSError(Lasso(alpha = alpha)).mean() for alpha in alphaArr]

lasso = pd.Series(lasso, index = alphaArr)



lasso.plot(title = "RMSE vs Alpha - Lasso")

plt.xlabel("alpha")

plt.ylabel("rmse")
trainComb = combinedData[:train.shape[0]]

testComb = combinedData[train.shape[0]:]



features = [f for f in combinedData.columns]



X = pd.DataFrame()

X['Feature'] = features



for alpha in alphaArr:

    lasso = Lasso(alpha=alpha)

    lasso.fit(trainComb, train.SalePrice)

    X["Alpha - " + str(alpha)] = lasso.coef_

coef = pd.Series(lasso.coef_, index = trainComb.columns)



print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

model_lasso = Lasso(alpha=1e-3, max_iter=1000).fit(trainComb, train["SalePrice"])

X
p_pred = np.expm1(lasso.predict(trainComb))

plt.scatter(p_pred, np.expm1(train.SalePrice))

plt.plot([min(p_pred),max(p_pred)], [min(p_pred),max(p_pred)], c="orange")
model_lassoCV = LassoCV(alphas = alphaArr).fit(trainComb, train.SalePrice)

coef = pd.Series(model_lassoCV.coef_, index = trainComb.columns)



print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
p_pred = np.expm1(model_lassoCV.predict(trainComb))

plt.scatter(p_pred, np.expm1(train.SalePrice))

plt.plot([min(p_pred),max(p_pred)], [min(p_pred),max(p_pred)], c="orange")
trainComb = combinedData[:train.shape[0]]

testComb = combinedData[train.shape[0]:]



import xgboost as xgb

dtrain = xgb.DMatrix(trainComb, label = train.SalePrice)

dtest = xgb.DMatrix(testComb)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)



model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)

model_xgb.fit(trainComb, train.SalePrice)
p_pred = np.expm1(model_xgb.predict(trainComb))

plt.scatter(p_pred, np.expm1(train.SalePrice))

plt.plot([min(p_pred),max(p_pred)], [min(p_pred),max(p_pred)], c="orange")
lassoPred = np.expm1(model_lassoCV.predict(testComb))

xgbPred = np.expm1(model_xgb.predict(testComb))

predictions =  0.7*lassoPred + 0.3*xgbPred



predPlot = pd.DataFrame({"XGBoost":xgbPred, "LassoCV":lassoPred})

predPlot.plot(x = "XGBoost", y = "LassoCV", kind = "scatter")
solution = pd.DataFrame(

    {

        "Id" : test.Id,

        "SalePrice" : predictions

    }, 

    columns=['Id', 'SalePrice']

)

print (solution)

solution.to_csv("lasso_solution.csv", index = False)