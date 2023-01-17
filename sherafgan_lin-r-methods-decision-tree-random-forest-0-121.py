%matplotlib inline
import pandas as pd

pd.options.mode.chained_assignment = None

import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')
trainData = pd.read_csv('../input/train.csv')

y = np.array(trainData['SalePrice'])

trainData.drop(['Id', 'SalePrice'], axis=1, inplace=True)

testData = pd.read_csv('../input/test.csv')

testData.drop(['Id'], axis=1, inplace=True)

data = pd.concat([trainData, testData])

print(trainData.shape, testData.shape, data.shape)

data.head()
trainAndTest = pd.concat([trainData, testData])

trainAndTest.shape
dfs = [trainData, trainAndTest]

def addNewFeature(featureName, featureValues):

    for df in dfs:

        if(len(df) == len(trainData)):

            df[featureName] = featureValues[0:len(trainData)]

        else:

            df[featureName] = featureValues

        df[featureName] = df[featureName].astype(int)
#Remodeled after construction

addNewFeature('RemodAC', np.logical_and(trainAndTest.YearRemodAdd - trainAndTest.YearBuilt, 

                                        trainAndTest.YearBuilt))
#Remodeled after construction

addNewFeature('Age', 2008 - trainAndTest.YearBuilt)
#we don't need these features anymore

trainAndTest = trainAndTest.drop(['YearRemodAdd',  'YearBuilt'], axis=1)
from scipy.stats import skew



#log transform the target:

y = np.log1p(y)



#log transform skewed numeric features:

numeric_feats = trainAndTest.dtypes[trainAndTest.dtypes != "object"].index



skewed_feats = trainData[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



trainAndTest[skewed_feats] = np.log1p(trainAndTest[skewed_feats])
catFswithNaNs = []

for clmn in trainAndTest.loc[:, trainAndTest.dtypes == object]:

    nans =  trainAndTest[clmn].isnull().sum().sum()

    if(nans != 0):

        catFswithNaNs.append(clmn)

        print(clmn + ' NaNs: ', trainAndTest[clmn].isnull().sum().sum())



print('Columns with NaNs: ', len(catFswithNaNs))
catFsWithNaNCategory = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 

                        'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 

                        'PoolQC', 'Fence', 'MiscFeature']

for clmn in catFsWithNaNCategory:

    trainAndTest.loc[trainAndTest[clmn].isnull(), clmn] = 'No' + clmn
catFswithNaNs = [clmn for clmn in catFswithNaNs if clmn not in catFsWithNaNCategory]

for clmn in catFswithNaNs:

    mcvOfClmn = trainAndTest[clmn].value_counts().idxmax()

    trainAndTest.loc[trainAndTest[clmn].isnull(), clmn] = mcvOfClmn
trainAndTest = pd.get_dummies(trainAndTest)
a = trainAndTest.loc[np.logical_not(trainAndTest["LotFrontage"].isnull()), "LotArea"]

b = trainAndTest.loc[np.logical_not(trainAndTest["LotFrontage"].isnull()), "LotFrontage"]

# plt.scatter(x, y)

t = (a <= 25000) & (b <= 150)

p = np.polyfit(a[t], b[t], 1)

trainAndTest.loc[trainAndTest['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p,

                                    trainAndTest.loc[trainAndTest['LotFrontage'].isnull(), 'LotArea'])
# trainAndTest = trainAndTest.fillna(trainAndTest.mean())



from sklearn.preprocessing import Imputer



toImpute = list(trainAndTest.columns)



imp = Imputer(missing_values = 'NaN', strategy='mean', axis=0)

tmp = imp.fit_transform(trainAndTest[toImpute])



i = 0

for clmn in toImpute:

    trainAndTest[clmn] = tmp[:, i]

    i+=1
X = trainAndTest[:trainData.shape[0]]

X_test = trainAndTest[trainData.shape[0]:]
from sklearn.linear_model import Ridge, Lasso

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 20))

    return(rmse)
from sklearn.linear_model import LinearRegression



model_linearReg= LinearRegression(n_jobs=-1)

scores = rmse_cv(model_linearReg)

print(scores)

print('Scores Mean:', np.mean(scores))
model_ridge = Ridge()

def ridge_model_evaluation(alphas):

    cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

    cv_ridge = pd.Series(cv_ridge, index = alphas)

    cv_ridge.plot(title = "Ridge")

    plt.xlabel("alpha")

    plt.ylabel("rmse")

    plt.show()
# alphas = list(np.linspace(0.05, 15))

# alphas = list(np.linspace(5, 10))

# alphas = list(np.linspace(7, 8))

alphas = list(np.linspace(7.4, 7.8, 10))

ridge_model_evaluation(alphas)
model_ridge = Ridge(alpha=7.65)

model_ridge.fit(X, y)

print('CV-20: ', np.mean(rmse_cv(model_ridge)))
model_lasso = Lasso()

def lasso_model_evaluation(alphas):

    cv_ridge = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]

    cv_ridge = pd.Series(cv_ridge, index = alphas)

    cv_ridge.plot(title = "Lasso")

    plt.xlabel("alpha")

    plt.ylabel("rmse")

    plt.show()
# alphas = list(np.linspace(0.0001, 0.002))

alphas = list(np.arange(0.0001, 0.001, 0.0001))

lasso_model_evaluation(alphas)
model_lasso = Lasso(alpha=0.0004)

model_lasso.fit(X, y)

print('CV-20: ', np.mean(rmse_cv(model_lasso)))
from sklearn.linear_model import ElasticNet



model_elastic_net = ElasticNet()

def ElasticNet_model_evaluation(alphas):

    cv_ridge = [rmse_cv(ElasticNet(alpha = alpha)).mean() for alpha in alphas]

    cv_ridge = pd.Series(cv_ridge, index = alphas)

    cv_ridge.plot(title = "ElasticNet")

    plt.xlabel("alpha")

    plt.ylabel("rmse")

    plt.show()
alphas = list(np.linspace(0.001, 0.01))

ElasticNet_model_evaluation(alphas)
model_elastic_net = ElasticNet(alpha=0.001)

model_elastic_net.fit(X, y)

print('CV-20: ', np.mean(rmse_cv(model_elastic_net)))
from sklearn.tree import DecisionTreeRegressor



model_tree = DecisionTreeRegressor()



def DTR_model_evaluation(max_depths):

    cv_ridge = [rmse_cv(DecisionTreeRegressor(max_depth = md)).mean() for md in max_depths]

    cv_ridge = pd.Series(cv_ridge, index = max_depths)

    cv_ridge.plot(title = "DecisionTreeRegressor")

    plt.xlabel("max_depth")

    plt.ylabel("rmse")

    plt.show()
max_depths = list(np.arange(1, 11, 1))

DTR_model_evaluation(max_depths)
model_tree = DecisionTreeRegressor(max_depth=7)

model_tree.fit(X, y)

print('CV-20: ', np.mean(rmse_cv(model_tree)))
from sklearn.ensemble import RandomForestRegressor



model_forest = RandomForestRegressor()



def RFR_model_evaluation(estimatorsL):

    cv_ridge = [rmse_cv(RandomForestRegressor(n_estimators = estimators, n_jobs=-1)).mean() for estimators in estimatorsL]

    cv_ridge = pd.Series(cv_ridge, index = estimatorsL)

    cv_ridge.plot(title = "RandomForestRegressor")

    plt.xlabel("n_estimator")

    plt.ylabel("rmse")

    plt.show()
estimatorsL = list(np.arange(70, 150, 20))

RFR_model_evaluation(estimatorsL)
model_forest = RandomForestRegressor(n_estimators=170, n_jobs=-1)

model_forest.fit(X, y)

print('CV-20: ', np.mean(rmse_cv(model_forest)))
from sklearn.kernel_ridge import KernelRidge



def kernelRidge_model_evaluation(alphas):

    cv_ridge = [rmse_cv(KernelRidge(alpha=alpha)).mean() for alpha in alphas]

    cv_ridge = pd.Series(cv_ridge, index = alphas)

    cv_ridge.plot(title = "KernelRidge")

    plt.xlabel("alpha")

    plt.ylabel("rmse")

    plt.show()
alphas = list(np.linspace(1, 15, 10))

# alphas = list(np.linspace(5, 10, 10))

# alphas = list(np.linspace(7, 8, 10))

kernelRidge_model_evaluation(alphas)
model_kernelRidge = KernelRidge(alpha=7.6)

model_kernelRidge.fit(X, y)

print('CV-20: ', np.mean(rmse_cv(model_kernelRidge)))
# save to file to make a submission

p = np.expm1(model_lasso.predict(X_test))

solution = pd.DataFrame({"id":np.arange(1461, 2920, 1), "SalePrice":p}, columns=['id', 'SalePrice'])

solution.to_csv("submission_1_model_lasso.csv", index = False)