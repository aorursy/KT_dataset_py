print('########Loading Libraries########')
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn import cross_validation, tree, linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost
import warnings

warnings.filterwarnings("ignore")

print('########Methods#########')
def readData(FileLocation,FileType):
    if FileType=='csv':
        data = pd.read_csv(FileLocation)
    
    print("Number of Columns - " + str(len(data.columns)))
    print("DataTypes - ")
    print(data.dtypes.unique())
    print(data.dtypes)
    print('checking null - ')
    print(data.isnull().any().sum(), ' / ', len(data.columns))
    print(data.isnull().any(axis=1).sum(), ' / ', len(data))
    print(data.head())
    return data
    

def DataCleaning(data, columnsToBeDropped):
    if (columnsToBeDropped):
        data = data.drop(columnsToBeDropped,axis=1)
    data.dropna(thresh=0.8*len(data), axis=1)
    data.dropna(thresh=0.8*len(data))
    data.isnull().any()
    return data
    
   
def scatterPlot(data,ColumnsToPlot,hueColumn):
    with sns.plotting_context("notebook",font_scale=2.5):
        g = sns.pairplot(data[ColumnsToPlot], hue=hueColumn, palette='tab20',size=6)
        g.set(xticklabels=[]);

def PlotDataCorrelation(data,vs):
    print(data.corr().abs().unstack().sort_values()[vs])
    

def featureRankingMatrix(data,x,y):
    ranks = {}
    
    colnames = data.columns
    def ranking(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x,2), ranks)
        return dict(zip(names, ranks))

    rlasso = RandomizedLasso(alpha=0.04)
    rlasso.fit(x, y)
    ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)
    lr = LinearRegression(normalize=True)
    lr.fit(x,y)
    rfe = RFE(lr, n_features_to_select=1, verbose =3 )
    rfe.fit(x,y)
    ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)

    
    lr = LinearRegression(normalize=True)
    lr.fit(x,y)
    ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)


    ridge = Ridge(alpha = 7)
    ridge.fit(x,y)
    ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)


    lasso = Lasso(alpha=.05)
    lasso.fit(x,y)
    ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)

    rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
    rf.fit(x,y)
    ranks["RF"] = ranking(rf.feature_importances_, colnames);

    r = {}
    for name in colnames:
        r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
    methods = sorted(ranks.keys())
    ranks["Mean"] = r
    meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])
    meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
    sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", size=14, aspect=1.9, palette='coolwarm')

def TrainLinearRegressionModel(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.2)
    print ('Linear Regression with OLS')
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    predictions=regr.predict(x_test)
    print('regr score: %.2f' % regr.score(x_test,y_test))
    print("RMSE: %.2f" % math.sqrt(np.mean((regr.predict(x_test) - y_test) ** 2)))
    print("Variance Score: %.2f" % explained_variance_score(predictions,y_test))
    print ('Linear Regression with SGD')
    sgd = linear_model.SGDRegressor()
    sgd.fit(x_train, y_train)
    predictions=sgd.predict(x_test)
    print('sgd score: %.2f' % sgd.score(x_test,y_test))
    print("RMSE: %.2f" % math.sqrt(np.mean((sgd.predict(x_test) - y_test) ** 2)))
    print("Variance Score: %.2f" % explained_variance_score(predictions,y_test))
    print("Linear Regression with XGBoost")
    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)
    xgb.fit(x_train,y_train)
    predictions = xgb.predict(x_test)
    print('xgboost score: %.2f' % sgd.score(x_test,y_test))
    print("RMSE: %.2f" % math.sqrt(np.mean((sgd.predict(x_test) - y_test) ** 2)))
    print("Variance Score: %.2f" % explained_variance_score(predictions,y_test))

def TrainLogisticRegressionModel(x,y):
    logisticRegr = LogisticRegression()
    print("========logisticRegr========")
    x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.2, random_state=0)
    logisticRegr.fit(x_train, y_train)    
    predictions=logisticRegr.predict(x_test)
    print('regr score: %.2f' % logisticRegr.score(x_test,y_test))
    print("RMSE: %.2f" % math.sqrt(np.mean((logisticRegr.predict(x_test) - y_test) ** 2)))
    print("Variance Score: %.2f" % explained_variance_score(predictions,y_test))
      
    print("========SVC========")
    classifier = SVC(kernel='rbf',random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.2)
    classifier.fit(x_train, y_train)    
    predictions=classifier.predict(x_test)
    print('regr score: %.2f' % classifier.score(x_test,y_test))
    print("RMSE: %.2f" % math.sqrt(np.mean((classifier.predict(x_test) - y_test) ** 2)))
    print("Variance Score: %.2f" % explained_variance_score(predictions,y_test))

Male=[]
Female=[]
def GenderToBinaryDataMale(data):
    for row in data["Gender"]:
        if(row == 'Male'):
            Male.append(1)
        else:
            Male.append(0)
    return Male

def GenderToBinaryDataFemale(data):
    for row in data["Gender"]:
        if(row == 'Female'):
            Female.append(1)
        else:
            Female.append(0)
    return Female

print("========Loading Data========")
FileLocation="../input/Social_Network_Ads.csv"
FileType="csv"
data=readData(FileLocation,FileType)
print("========Cleaning Data========")
columnsToBeDropped=['User ID']
data=DataCleaning(data, columnsToBeDropped)
data['Male']=GenderToBinaryDataMale(data)
data['Female']=GenderToBinaryDataFemale(data)
ColumnsToPlot=['Male','Female','Age','EstimatedSalary','Purchased']
hueColumn='Purchased'
print("========Scatter Plot========")
scatterPlot(data,ColumnsToPlot,hueColumn)
print("========Data Correlation========")
PlotDataCorrelation(data,'Purchased')
print("========Feature Ranking matrix========")
y = data.Purchased.values
data = data.drop(['Purchased'], axis=1)
data = data.drop(['Gender'], axis=1)
x = data.as_matrix()
featureRankingMatrix(data,x,y)
print("========Training Model========")
TrainLogisticRegressionModel(data,y)