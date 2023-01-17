# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn import linear_model

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import KFold

import math

from sklearn.svm import SVR

import lightgbm

from xgboost.sklearn import XGBRegressor

from mlxtend.regressor import StackingCVRegressor

from sklearn.pipeline import make_pipeline

from lightgbm import LGBMRegressor

from sklearn import neighbors

from sklearn.linear_model import LogisticRegression 

from sklearn import preprocessing

from sklearn import utils

from sklearn.ensemble import AdaBoostRegressor





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Créer un Pandas DataFrame à partir des fichiers CSV:

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



#enregistrer la colonne  'Id'

train_ID = train['Id']

test_ID = test['Id']



#puis la supprimer puisqu"elle inutile dans le processus de la prediction 

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



#mettre train et test dans une seule dataset

data = train.append(test,sort=False)

#taille des datasets:

print(train.shape)

print(test.shape)# test ne contient pas la colonne 'SalePrice'

print(data.shape)



data.head()
#quelques statisques sur la 'target variable' 'SalePrice'

data.SalePrice.describe() #on constate que le prix de vente moyen est ~ 180921$
#distribution de la variable 'SalePrice' 

plt.hist(train['SalePrice'], color= 'r')

plt.title('Distribution de sales price des maisons', fontsize = 24)

plt.ylabel('observation', fontsize = 20)

plt.xlabel('sales price', fontsize = 20)



plt.show()
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#prendre juste les valeurs numériques

numeric_features = data.select_dtypes(include = [np.number])

#features with the most correlation with the predictor variable

corr = numeric_features.corr()

print(corr['SalePrice'].sort_values(ascending = False)[:5], '\n')

print(corr['SalePrice'].sort_values(ascending = False)[-5:])
#log transforming transformer sale price en une distribution gaussienne

target = np.log(data.SalePrice)

#definir

plt.scatter(x=data['GarageArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

f1=plt.show()

f1

#definir

plt.scatter(x=data['OverallQual'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('OverallQual')

f2=plt.show()

f2

#definir

plt.scatter(x=data['GrLivArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('GrLivArea')

f3=plt.show()

f3

#definir

plt.scatter(x=data['GarageCars'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('GarageCars')

f4=plt.show()

f4
#vérifier l'existence des valeurs nulles:

data[data.isnull().any(axis=1)]
#maintenant on va afficher le nombre des valeurs manquantes pour chaque collone 'feature':

a=data.isnull().sum().sort_values(ascending=False)[:25]

b=a/2919*100

nulls = pd.DataFrame({'Nb_val_nulles': a,'pourcentage': b})

nulls.index.name = 'Features'

nulls
#les colonnes'features' qui ont plus que 1000 valeur manquantes doivent etre supprimées:

data = data.dropna(axis=1, how='any', thresh = 1000)

data.shape
#rempacer les valeurs manquantes par la veleur mean:

data = data.fillna(data.mean())

#train = train.select_dtypes(include= [np.number]).interpolate().dropna()

#vérifier l'existence de valeurs nulles

data[data.isnull().any(axis=1)]
#remplacer les valeurs de type object par des integers

data = pd.get_dummies(data)

data.shape
#Verifying missing values

data[data.isnull().any(axis=1)]

#sum(data.isnull().sum() != 0)
#Supprimer les variables corrélées  entre eux: car ils donnent tous les memes informations:

covariance = data.corr()

allFeatures = [i for i in covariance]

setOfDroppedFeatures = set() 

for i in range(len(allFeatures)) :

    for j in range(i+1,len(allFeatures)): 

        feature1=allFeatures[i]

        feature2=allFeatures[j]

        if abs(covariance[feature1][feature2]) > 0.8: #If the correlation between the features is > 0.8

            setOfDroppedFeatures.add(feature1)

data2 = data.drop(setOfDroppedFeatures, axis=1)

data2.shape
#supprimer les features qui ont la minimale correlation avec notre target variable'SalePrice':

nonCorrelated = [column for column in data2 if abs(data2[column].corr(data2["SalePrice"])) < 0.045]

data2 = data2.drop(nonCorrelated, axis=1)

data2.shape
from sklearn.decomposition import PCA

from sklearn.datasets import load_boston

from sklearn.preprocessing import StandardScaler

boston = load_boston()

X = boston["data"]

scaler = StandardScaler()

x_scaled= scaler.fit_transform(X)

n_col=data.shape[1]

pca=PCA(n_components = n_col)

data3=pca.fit_transform(data)

#test_com=pca.fit_transform(data)

data3.shape

#test_com.shape
#on separe les datastes (Because removing outliers ⇔ removing rows, and we don't want to remove rows from test set)

newTrain = data.iloc[:1460]

newTest = data.iloc[1460:]

#Aprés reduction PCA

"""newTrain3 = data3.iloc[:1460]

newTest3 = data3.iloc[1460:]"""

#Aprés reduction COR

newTrain2 = data2.iloc[:1460]

newTest2 = data2.iloc[1460:]
#Second, definir une fonction quie retourne les valeurs des outliers via la methode  percentile()



def outliers_iqr(ys):

    quartile_1, quartile_3 = np.percentile(ys, [25, 75]) #Get 1st and 3rd quartiles (25% -> 75% of data will be kept)

    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * 1.5) #Get lower bound

    upper_bound = quartile_3 + (iqr * 1.5) #Get upper bound

    return np.where((ys > upper_bound) | (ys < lower_bound)) #Get outlier values



#Third, supprimer les ouliers juste de train dataset 



trainWithoutOutliers = newTrain #We can't change train while running through it

for column in newTrain:

    outlierValuesList = np.ndarray.tolist(outliers_iqr(newTrain[column])[0]) #outliers_iqr() returns an array

    trainWithoutOutliers = newTrain.drop(outlierValuesList) #Drop outlier rows

    

trainWithoutOutliers = newTrain

print(outlierValuesList)

print(trainWithoutOutliers.shape)



#Aprés reduction

trainWithoutOutliers2 = newTrain2

for column in newTrain2:

    outlierValuesList2 = np.ndarray.tolist(outliers_iqr(newTrain2[column])[0]) #outliers_iqr() returns an array

    trainWithoutOutliers2 = newTrain2.drop(outlierValuesList2) #Drop outlier rows

    

trainWithoutOutliers2 = newTrain2

print(outlierValuesList2)

print(trainWithoutOutliers2.shape)
X = trainWithoutOutliers.drop("SalePrice", axis=1) #supprimer la colonne SalePrice 

Y = np.log1p(newTrain["SalePrice"])



#Apré reduction PCA

X2 = trainWithoutOutliers2.drop("SalePrice", axis=1) #supprimer la colonne SalePrice 

Y2 = np.log1p(newTrain2["SalePrice"])
#splitting the data into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 42, test_size = .33)

#Aprés reduction

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, random_state = 42, test_size = .33)
#ça va me servire pour faire la prediction

newTest = newTest.drop("SalePrice", axis=1) 

#apres reduction

newTest2 = newTest2.drop("SalePrice", axis=1) 
rmse_val = [] #to store rmse values for different k

for K in range(20):

    K = K+1

    model = neighbors.KNeighborsRegressor(n_neighbors = K)



    model.fit(X_train, y_train) 

    

    predictions=model.predict(X_test) #make prediction on test set

    error = mean_squared_error(y_test, predictions) #calculate rmse

    rmse_val.append(error) #store rmse values

    print('RMSE value for k= ' , K , 'is:', error)
# on choisi K=5 puisque cette valeur donne RMSE minimal

#créer le modèle

knn = neighbors.KNeighborsRegressor(n_neighbors = 5)

#fitting linear regression on the data

knn_fit = knn.fit(X_train, y_train)
#R square value

print('R square is: {}'.format(knn_fit.score(X, Y)))

#predicting on the test set

predictionknn = knn_fit.predict(X_test)

#evaluating the model on mean square error

knnRMS=mean_squared_error(y_test, predictionknn)

print('RMSE is {}'.format(knnRMS))
rmse_val = [] #to store rmse values for different k

for K in range(20):

    K = K+1

    model = neighbors.KNeighborsRegressor(n_neighbors = K)



    model.fit(X_train2, y_train2) 

    

    predictions=model.predict(X_test2) #make prediction on test set

    error = mean_squared_error(y_test2, predictions) #calculate rmse

    rmse_val.append(error) #store rmse values

    print('RMSE value for k= ' , K , 'is:', error)

    

# on choisi K=5 puisque cette valeur donne RMSE minimal

#créer le modèle

knn = neighbors.KNeighborsRegressor(n_neighbors = 5)

#fitting linear regression on the data

knn_fit2 = knn.fit(X_train2, y_train2)



#R square value

print('R square is: {}'.format(knn_fit2.score(X2, Y2)))

#predicting on the test set

predictionknn2 = knn_fit2.predict(X_test2)

#evaluating the model on mean square error

knnRMS2=mean_squared_error(y_test2, predictionknn2)

print('RMSE is {}'.format(knnRMS2))
lab_enc = preprocessing.LabelEncoder()

y_train_encoded = lab_enc.fit_transform(y_train)

print(y_train_encoded)

print(utils.multiclass.type_of_target(y_train))

print(utils.multiclass.type_of_target(y_train.astype('int')))

print(utils.multiclass.type_of_target(y_train_encoded))
#créer le modèle

logir = LogisticRegression(random_state = 0)

#fitting linear regression on the data

logir_fit = logir.fit(X_train, y_train_encoded)
#R square value

#print('R square is: {}'.format(logir_fit.score(X_test, y_test)))

#predicting on the test set

predictionlogir = logir_fit.predict(X_test)

#evaluating the model on mean square error

logirRMS= mean_squared_error(y_test, predictionlogir)

print('RMSE is {}'.format(logirRMS))
lab_enc = preprocessing.LabelEncoder()

y_train_encoded2 = lab_enc.fit_transform(y_train2)

print(y_train_encoded2)

print(utils.multiclass.type_of_target(y_train2))

print(utils.multiclass.type_of_target(y_train2.astype('int')))

print(utils.multiclass.type_of_target(y_train_encoded2))



#créer le modèle

logir = LogisticRegression(random_state = 0)

#fitting linear regression on the data

logir_fit2 = logir.fit(X_train2, y_train_encoded2)



#R square value

#print('R square is: {}'.format(logir_fit.score(X_test, y_test)))

#predicting on the test set

predictionlogir2 = logir.predict(X_test2)

logirRMS2= mean_squared_error(y_test2, predictionlogir2)

print('RMSE is {}'.format(logirRMS2))
#créer le modèle

lr = linear_model.LinearRegression()

#fitting linear regression on the data

lr_fit = lr.fit(X_train, y_train)
#R square value

print('R square is: {}'.format(lr_fit.score(X, Y)))

#predicting on the test set

predictionlr = lr_fit.predict(X_test)

#evaluating the model on mean square error

lrRMSE=mean_squared_error(y_test, predictionlr)

print('RMSE is {}'.format(lrRMSE))
pred1=np.expm1(lr_fit.predict(newTest))

sub1 = pd.DataFrame() #Create a new DataFrame for submission

sub1['Id'] = test_ID

sub1['SalePrice'] = pred1

sub1.to_csv("submissionlin.csv", index=False) #Convert DataFrame to .csv file

sub1
#créer le modèle

lr = linear_model.LinearRegression()

#fitting linear regression on the data

lr_fit2 = lr.fit(X_train2, y_train2)



#R square value

print('R square is: {}'.format(lr_fit2.score(X2, Y2)))

#predicting on the test set

predictionlr2 = lr_fit2.predict(X_test2)

#evaluating the model on mean square error

lrRMSE2=mean_squared_error(y_test2, predictionlr2)

print('RMSE is {}'.format(lrRMSE2))



"""pred1=np.expm1(lr_fit.predict(newTest2))

sub1 = pd.DataFrame() #Create a new DataFrame for submission

sub1['Id'] = test_ID

sub1['SalePrice'] = pred1

sub1.to_csv("submissionlin.csv", index=False) #Convert DataFrame to .csv file

sub1"""
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

alphas_Rd = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphasL = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
#créer modèle

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphasL, random_state=42, cv=kfolds))
#fitting

lasso_fit=lasso.fit(X_train, y_train)
#prediction on train set

y_train_predict = lasso_fit.predict(X_train)

#prediction on test set

y_test_predict = lasso_fit.predict(X_test)

#MSE trainset

lasso_train = mean_squared_error(y_train, y_train_predict)

print('Mean square error on the Train set is: {}'.format(lasso_train))

#MSE test set

lassoRMS = mean_squared_error(y_test, y_test_predict)

print('Mean square error on the Test set is: {}'.format(lassoRMS))
pred5=np.expm1(lasso_fit.predict(newTest))

sub5 = pd.DataFrame() #Create a new DataFrame for submission

sub5['Id'] = test_ID

sub5['SalePrice'] = pred5

sub5.to_csv("submissionlassofin1.csv", index=False) #Convert DataFrame to .csv file

sub5


#fitting

lasso_fit2=lasso.fit(X_train2, y_train2)

y_train_predict2 = lasso_fit2.predict(X_train2)

#prediction on test set

y_test_predict2 = lasso_fit2.predict(X_test2)

#MSE trainset

lasso_train2 = mean_squared_error(y_train2, y_train_predict2)

print('Mean square error on the Train set is: {}'.format(lasso_train2))

#MSE test set

lassoRMS2 = mean_squared_error(y_test2, y_test_predict2)

print('Mean square error on the Test set is: {}'.format(lassoRMS2))

"""pred5=np.expm1(lasso_fit.predict(newTest))

sub5 = pd.DataFrame() #Create a new DataFrame for submission

sub5['Id'] = test_ID

sub5['SalePrice'] = pred4

sub5.to_csv("submissionlassofin1.csv", index=False) #Convert DataFrame to .csv file

sub5"""
#cére modele

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_Rd, cv=kfolds))
#fiting

ridge_fit=ridge.fit(X_train, y_train)
#prediction on train set

y_train_predict = ridge_fit.predict(X_train)

#prediction on test set

y_test_predict = ridge_fit.predict(X_test)

#MSE trainset

ridge_train = mean_squared_error(y_train, y_train_predict)

print('Mean square error on the Train set is: {}'.format(ridge_train))

#MSE test set

ridgeRMS = mean_squared_error(y_test, y_test_predict)

print('Mean square error on the Test set is: {}'.format(ridgeRMS))
#fiting

ridge_fit2=ridge.fit(X_train2, y_train2)

#prediction on train set

y_train_predict2 = ridge_fit2.predict(X_train2)

#prediction on test set

y_test_predict2 = ridge_fit2.predict(X_test2)

#MSE trainset

ridge_train2 = mean_squared_error(y_train2, y_train_predict2)

print('Mean square error on the Train set is: {}'.format(ridge_train))

#MSE test set

ridgeRMS2 = mean_squared_error(y_test2, y_test_predict2)

print('Mean square error on the Test set is: {}'.format(ridgeRMS2))
#créer modèle

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
#fiting

svr_fit=svr.fit(X_train, y_train)
#prediction on train set

y_train_predict = svr_fit.predict(X_train)

#prediction on test set

y_test_predict = svr_fit.predict(X_test)

#MSE trainset

svr_train = mean_squared_error(y_train, y_train_predict)

print('Mean square error on the Train set is: {}'.format(svr_train))

#MSE test set

svrRMS = mean_squared_error(y_test, y_test_predict)

print('Mean square error on the Test set is: {}'.format(svrRMS))
#fiting

svr_fit2=svr.fit(X_train2, y_train2)

#prediction on train set

y_train_predict2 = svr_fit2.predict(X_train2)

#prediction on test set

y_test_predict2 = svr_fit2.predict(X_test2)

#MSE trainset

svr_train2 = mean_squared_error(y_train2, y_train_predict2)

print('Mean square error on the Train set is: {}'.format(svr_train2))

#MSE test set

svrRMS2 = mean_squared_error(y_test2, y_test_predict2)

print('Mean square error on the Test set is: {}'.format(svrRMS2))
#Gradient boosting regressor model

gbr = GradientBoostingRegressor(n_estimators= 1000, max_depth= 2, learning_rate= .01)

#fiting

gbr_fit=gbr.fit(X_train, y_train)
#prediction on train set

y_train_predict = gbr_fit.predict(X_train)

#prediction on test set

y_test_predict = gbr_fit.predict(X_test)

#MSE trainset

est_train = mean_squared_error(y_train, y_train_predict)

print('Mean square error on the Train set is: {}'.format(est_train))

#MSE test set

grboRMS = mean_squared_error(y_test, y_test_predict)

print('Mean square error on the Test set is: {}'.format(grboRMS))
pred4=np.expm1(gbr_fit.predict(newTest))

sub4 = pd.DataFrame() #Create a new DataFrame for submission

sub4['Id'] = test_ID

sub4['SalePrice'] = pred4

sub4.to_csv("submissiongbr.csv", index=False) #Convert DataFrame to .csv file

sub4
#fiting

gbr_fit2=gbr.fit(X_train2, y_train2)

#prediction on train set

y_train_predict2 = gbr_fit2.predict(X_train2)

#prediction on test set

y_test_predict2 = gbr_fit2.predict(X_test2)

#MSE trainset

est_train2 = mean_squared_error(y_train2, y_train_predict2)

print('Mean square error on the Train set is: {}'.format(est_train2))

#MSE test set

grboRMS2 = mean_squared_error(y_test, y_test_predict)

print('Mean square error on the Test set is: {}'.format(grboRMS2))

"""pred4=np.expm1(gbr_fit.predict(newTest))

sub4 = pd.DataFrame() #Create a new DataFrame for submission

sub4['Id'] = test_ID

sub4['SalePrice'] = pred4

sub4.to_csv("submissiongbr.csv", index=False) #Convert DataFrame to .csv file

sub4"""
# créer modéle

xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
# fiting

xgboost_fit=xgboost.fit(X_train, y_train)
#prediction on train set

y_train_predict = xgboost_fit.predict(X_train)

#prediction on test set

y_test_predict = xgboost_fit.predict(X_test)

#MSE trainset

xgboost_train = mean_squared_error(y_train, y_train_predict)

print('Mean square error on the Train set is: {}'.format(xgboost_train))

#MSE test set

xgboostRMS = mean_squared_error(y_test, y_test_predict)

print('Mean square error on the Test set is: {}'.format(xgboostRMS))
pred4=np.expm1(xgboost_fit.predict(newTest))

sub4 = pd.DataFrame() #Create a new DataFrame for submission

sub4['Id'] = test_ID

sub4['SalePrice'] = pred4

sub4.to_csv("submissionxgb2.csv", index=False) #Convert DataFrame to .csv file

sub4
# fiting

xgboost_fit2=xgboost.fit(X_train2, y_train2)

#prediction on train set

y_train_predict2 = xgboost_fit2.predict(X_train2)

#prediction on test set

y_test_predict2 = xgboost_fit2.predict(X_test2)

#MSE trainset

xgboost_train2 = mean_squared_error(y_train2, y_train_predict2)

print('Mean square error on the Train set is: {}'.format(xgboost_train2))

#MSE test set

xgboostRMS2 = mean_squared_error(y_test2, y_test_predict2)

print('Mean square error on the Test set is: {}'.format(xgboostRMS2))

"""pred4=np.expm1(xgboost_fit.predict(newTest))

sub4 = pd.DataFrame() #Create a new DataFrame for submission

sub4['Id'] = test_ID

sub4['SalePrice'] = pred4

sub4.to_csv("submissionxgb2.csv", index=False) #Convert DataFrame to .csv file

sub4"""
lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )
# fiting

lightgbm_fit=lightgbm.fit(X_train, y_train)
#prediction on train set

y_train_predict = lightgbm_fit.predict(X_train)

#prediction on test set

y_test_predict = lightgbm_fit.predict(X_test)

#MSE trainset

lightgbm_train = mean_squared_error(y_train, y_train_predict)

print('Mean square error on the Train set is: {}'.format(lightgbm_train))

#MSE test set

lightgbmRMS = mean_squared_error(y_test, y_test_predict)

print('Mean square error on the Test set is: {}'.format(lightgbmRMS))
pred3=np.expm1(lightgbm_fit.predict(newTest))

sub3 = pd.DataFrame() #Create a new DataFrame for submission

sub3['Id'] = test_ID

sub3['SalePrice'] = pred3

sub3.to_csv("submissionlight4.csv", index=False) #Convert DataFrame to .csv file

sub3
# fiting

lightgbm_fit2=lightgbm.fit(X_train2, y_train2)

#prediction on train set

y_train_predict2 = lightgbm_fit2.predict(X_train2)

#prediction on test set

y_test_predict2 = lightgbm_fit2.predict(X_test2)

#MSE trainset

lightgbm_train2 = mean_squared_error(y_train2, y_train_predict2)

print('Mean square error on the Train set is: {}'.format(lightgbm_train2))

#MSE test set

lightgbmRMS2 = mean_squared_error(y_test2, y_test_predict2)

print('Mean square error on the Test set is: {}'.format(lightgbmRMS2))

"""pred3=np.expm1(lightgbm_fit.predict(newTest))

sub3 = pd.DataFrame() #Create a new DataFrame for submission

sub3['Id'] = test_ID

sub3['SalePrice'] = pred3

sub3.to_csv("submissionlight4.csv", index=False) #Convert DataFrame to .csv file

sub3"""
#hyperparameter tuning

#ada=AdaBoostRegressor()

#search_grid={'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1],'random_state':[1]}

#search=GridSearchCV(estimator=ada,param_grid=search_grid,scoring='neg_mean_squared_error',n_jobs=1,cv=5)

#search.fit()
#créer le modèle

adab=AdaBoostRegressor(n_estimators=500,learning_rate=0.001,random_state=1)

score=np.mean(cross_val_score(adab,X_train, y_train,scoring='neg_mean_squared_error',cv=7,n_jobs=1))

score
# fiting

adab_fit=adab.fit(X, Y)
#prediction on train set

y_train_predict = adab_fit.predict(X_train)

#prediction on test set

y_test_predict = adab_fit.predict(X_test)

#MSE trainset

adab_train = mean_squared_error(y_train, y_train_predict)

print('Mean square error on the Train set is: {}'.format(adab_train))

#MSE test set

adabRMS = mean_squared_error(y_test, y_test_predict)

print('Mean square error on the Test set is: {}'.format(adabRMS))
"""pred4=np.expm1(xgboost_fit.predict(newTest))

sub4 = pd.DataFrame() #Create a new DataFrame for submission

sub4['Id'] = test_ID

sub4['SalePrice'] = pred4

sub4.to_csv("submissionadab.csv", index=False) #Convert DataFrame to .csv file

sub4"""
#créer le modèle

adab=AdaBoostRegressor(n_estimators=500,learning_rate=0.001,random_state=1)

score2=np.mean(cross_val_score(adab,X_train2, y_train2,scoring='neg_mean_squared_error',cv=7,n_jobs=1))

score2



# fiting

adab_fit2=adab.fit(X2, Y2)



#prediction on train set

y_train_predict2 = adab_fit2.predict(X_train2)

#prediction on test set

y_test_predict2 = adab_fit2.predict(X_test2)

#MSE trainset

adab_train2 = mean_squared_error(y_train2, y_train_predict2)

print('Mean square error on the Train set is: {}'.format(adab_train2))

#MSE test set

adabRMS2 = mean_squared_error(y_test2, y_test_predict2)

print('Mean square error on the Test set is: {}'.format(adabRMS2))



"""pred4=np.expm1(xgboost_fit.predict(newTest))

sub4 = pd.DataFrame() #Create a new DataFrame for submission

sub4['Id'] = test_ID

sub4['SalePrice'] = pred4

sub4.to_csv("submissionadab.csv", index=False) #Convert DataFrame to .csv file

sub4"""
from sklearn.ensemble import RandomForestRegressor

#Créer model

forest = RandomForestRegressor(n_estimators=100)

#fitting

forest.fit(X, Y)

y_test_predict = forest.predict(X_test)
#MSE test set

forestRMS = mean_squared_error(y_test, y_test_predict)

print('Mean square error on the Test set is: {}'.format(forestRMS))
pred4=np.expm1(forest.predict(newTest))

sub4 = pd.DataFrame() #Create a new DataFrame for submission

sub4['Id'] = test_ID

sub4['SalePrice'] = pred4

sub4.to_csv("submissionforestfin.csv", index=False) #Convert DataFrame to .csv file

sub4
from sklearn.ensemble import RandomForestRegressor

#Créer model

forest = RandomForestRegressor(n_estimators=100)

#fitting

forest.fit(X2, Y2)

y_test_predict2 = forest.predict(X_test2)



#MSE test set

forestRMS2 = mean_squared_error(y_test2, y_test_predict2)

print('Mean square error on the Test set is: {}'.format(forestRMS2))



"""pred4=np.expm1(forest.predict(newTest))

sub4 = pd.DataFrame() #Create a new DataFrame for submission

sub4['Id'] = test_ID

sub4['SalePrice'] = pred4

sub4.to_csv("submissionforestfin.csv", index=False) #Convert DataFrame to .csv file

sub4"""
recap = pd.DataFrame({'RMSE': pd.Series([knnRMS,logirRMS,lrRMSE,lassoRMS,ridgeRMS,svrRMS,grboRMS,xgboostRMS,lightgbmRMS, adabRMS, forestRMS], index = ['knn','logiReg','linReg','Lasso','Ridge','SVR','gradboost','Xgboost','lgbm', 'adaboost', 'rdForest']), 'RMSE2': pd.Series([knnRMS2,logirRMS2,lrRMSE2,lassoRMS2,ridgeRMS2,svrRMS2,grboRMS2,xgboostRMS2,lightgbmRMS2, adabRMS2, forestRMS2], index = ['knn','logiReg','linReg','Lasso','Ridge','SVR','gradboost','Xgboost','lgbm', 'adaboost', 'rdForest'])})
recap