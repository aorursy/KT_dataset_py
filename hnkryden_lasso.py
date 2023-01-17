import numpy as np

import pandas as pd 

from sklearn import preprocessing

from scipy import signal

from sklearn.cross_validation import cross_val_score

from sklearn.metrics import fbeta_score, make_scorer

from sklearn import linear_model

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import math



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

y=train['SalePrice']

yLog=np.log(y)

train=train.drop('SalePrice',1)

train=train.drop('Id',1)

idTest=test['Id']

test=test.drop('Id',1)

nrofRows,nrofCols=train.shape

print('Loaded data')
outEnc=[]

#def preProcessV2(train,test):

dimTransform=['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',

              'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',

              'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical',

              'KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC',

              'Fence','MiscFeature','SaleType','SaleCondition','ExterQual','ExterCond']

#Input features to take the logarithm of

logTrain=['LotArea','GarageArea','OpenPorchSF','EnclosedPorch']



x=np.empty(shape=[train.shape[0],0])

xTest=np.empty(shape=[test.shape[0],0])



for feature in train.columns:

    featureData=train[feature]

    featureDataTest=test[feature]

    if(feature in dimTransform):

        print("Transforming ",feature)

    

        enc = preprocessing.OneHotEncoder(sparse=False)

        le = preprocessing.LabelEncoder()    



        featureData[pd.isnull(featureData)]='NaN'

        featureDataTest[pd.isnull(featureDataTest)]='NaN'  

        

        le.fit(featureData.append(featureDataTest))    

        

        outLE=le.transform(featureData)

        outLEtest=le.transform(featureDataTest)

        

        enc.fit(outLE.reshape(-1,1))

        outEnc=enc.transform(outLE.reshape(-1,1))

        

        outEncTest=enc.transform(outLEtest.reshape(-1,1))

        x=np.append(x,outEnc,axis=1)

        xTest=np.append(xTest,outEncTest,axis=1)

        

    else:

        print("Number feature = ",feature)

        imp=preprocessing.Imputer(missing_values='NaN', strategy='most_frequent')

        imp.fit(featureData.reshape(-1,1))        

        outImp=imp.transform(featureData.reshape(-1,1))

        outImpTest=imp.transform(featureDataTest.reshape(-1,1))

        if(feature in logTrain):

            outImp=np.log(1+outImp)

            outImpTest=np.log(1+outImpTest)

        x=np.append(x,outImp,axis=1)

        xTest=np.append(xTest,outImpTest,axis=1)   

        

# Custom features



trainCustom=1*np.array((train['YearBuilt']-train['YearRemodAdd'])==0)

x=np.append(x,trainCustom.reshape(-1,1),axis=1)



testCustom=1*np.array((test['YearBuilt']-test['YearRemodAdd'])==0)

xTest=np.append(xTest,testCustom.reshape(-1,1),axis=1)



print("Training data rows %d " % train.shape[1])

print("Processed data rows %d " % x.shape[1])
from sklearn.model_selection import GridSearchCV

#regr = linear_model.Ridge(normalize=False,alpha=alpha)

regr = linear_model.Lasso(max_iter=1000000)

#regr = linear_model.ElasticNet(l1_ratio=0.8)





parameters = {'alpha':[1e-5,1e-4,2.5e-4,4e-4,5e-4,7e-4,1e-3,4,6,7,10,20]}

grid = GridSearchCV(regr, parameters,scoring="neg_mean_squared_error")

grid.fit(x,yLog)





testScore=grid.cv_results_['mean_test_score']



testScoreSqrt = (np.sqrt(-testScore))

idx=np.argmin(testScoreSqrt)

print(testScoreSqrt[idx]," at ", grid.param_grid['alpha'][idx])



plt.figure(figsize=(10,6))

plt.plot(np.log(grid.param_grid['alpha']),testScoreSqrt)

plt.show()
def r2logscore(ground_truth,prediction):

    score = np.sqrt(np.sum((np.log(1+ground_truth)-np.log(1+prediction))**2)/len(prediction))

    return score

    

regr=[]



alpha=0.0004

regr = linear_model.Lasso(alpha = alpha,max_iter=100000)

regr.fit(x,yLog)



cross_val=cross_val_score(regr,x,yLog,cv=3,scoring="neg_mean_squared_error")

cross_val_sqrt=np.sqrt(-cross_val)



print("Cross validation score %.6f" % np.mean(cross_val_sqrt))


#Creating test data output

yTest = regr.predict(xTest)

print("Id,SalePrice")

for i in range(len(idTest)):

    print('%d,%f' % (idTest[i],math.exp(yTest[i])))