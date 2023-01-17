import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing

from IPython.display import display



# Load data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# First look at the data

print(train.shape)

display(train.head())



#NaN

train.fillna(value=-999,inplace=True)

test.fillna(value=-999,inplace=True)



print('Type numeric:')

list_feature_Nan = []

for i in train.select_dtypes(exclude=['object']).columns:

    if (train[i] == -999).astype(int).sum() > 0:

        print("Feature: ", i, "has", round(((train[i] == -999).astype(int).sum()/1460)*100), "% of NaN")

        list_feature_Nan.append(i)

        

print('Type object:') 



for i in train.select_dtypes(include=['object']).columns:

    if (train[i] == -999).astype(int).sum() > 0:

        print("Feature: ", i, "has", round(((train[i] == -999).astype(int).sum()/1460)*100), "% of NaN")     
#Replace numeric feature by mean

train_replace_mean = train

test_replace_mean = test



for i in list_feature_Nan:

    train_replace_mean[i].replace(-999,train[i].mean(),inplace=True)

    test_replace_mean[i].replace(-999,train[i].mean(),inplace=True)
#Label Encoder

le = preprocessing.LabelEncoder()

train_str = train.select_dtypes(include=['object'])

test_str = test.select_dtypes(include=['object']) 

display(train_str.head())



print(train_str.columns.values)



train.drop(train_str.columns.values,axis=1,inplace=True)

test.drop(train_str.columns.values,axis=1,inplace=True)
train_str_dum = pd.get_dummies(train_str)

test_str_dum = pd.get_dummies(test_str)



columns_dum = list(set(train_str_dum) & set(test_str_dum))



train_str_dum = train_str_dum[columns_dum]

test_str_dum = test_str_dum[columns_dum]



#New train and New test

train_flo = train.select_dtypes(exclude=['object'])

test_flo = test.select_dtypes(exclude=['object']) 



new_train = pd.merge(train_flo,train_str_dum,left_index=True,right_index=True)

new_test = pd.merge(test_flo,test_str_dum,left_index=True,right_index=True)



display(new_train.head())

print(new_train.columns.values)
from sklearn.metrics import mean_squared_error

from sklearn.cross_validation import KFold

from sklearn.ensemble import RandomForestRegressor



train_clf = new_train.drop('SalePrice',axis=1)

train_clf.drop('Id',axis=1,inplace=True)



train_clf2 = train_clf.drop(['LotFrontage','MasVnrArea','GarageYrBlt'],axis=1)

train_clf2 = pd.merge(train_clf2,train_replace_mean[['LotFrontage','MasVnrArea','GarageYrBlt']],left_index=True,right_index=True)

train_clf2['tot_sf'] = train_clf2['TotalBsmtSF'] + train_clf2['GrLivArea']

train_clf2['ratio_fl'] = train_clf2['2ndFlrSF'] / train_clf2['1stFlrSF'] 



train_clf3 = train_clf2.copy()

train_clf3['garage_ex'] = (train_clf3['GarageQual_Gd'] + train_clf3['GarageQual_TA'] + train_clf3['GarageQual_Fa'] + train_clf3['GarageQual_Po']) * (train_clf3['GarageCond_Ex'])



index = pd.DataFrame(test.Id,columns = ['Id'])

test_clf = new_test.drop('Id',axis=1)

test_clf2 = test_clf.drop(['LotFrontage','MasVnrArea','GarageYrBlt'],axis=1)

test_clf2 = pd.merge(test_clf2,test_replace_mean[['LotFrontage','MasVnrArea','GarageYrBlt']],left_index=True,right_index=True)

test_clf2['tot_sf'] = test_clf2['TotalBsmtSF'] + test_clf2['GrLivArea']

test_clf2['ratio_fl'] = test_clf2['2ndFlrSF']  / test_clf2['1stFlrSF']



test_clf3 = test_clf2.copy()

test_clf3['garage_ex'] = (test_clf3['GarageQual_Gd'] + test_clf3['GarageQual_TA'] + test_clf3['GarageQual_Fa'] + test_clf3['GarageQual_Po']) * (test_clf3['GarageCond_Ex'])



index = pd.DataFrame(test.Id,columns = ['Id'])

y = np.array(pd.DataFrame(np.log(train.SalePrice)))



train_np = np.array(train_clf2)

train_np2 = np.array(train_clf3)





def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())



def score(clf, train_np, random_state, folds):

    kf = KFold(n = 1460 , n_folds=folds, shuffle = True, random_state = random_state)

    for itrain, itest in kf:

        Xtr, Xte = train_np[itrain], train_np[itest]

        ytr, yte = y[itrain], y[itest]

        clf.fit(Xtr, ytr.ravel())

        pred = pd.DataFrame(clf.predict(Xte))

        return rmse(yte, pred)

    return rmse(y, pred)
import xgboost as xgb

from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor

Cs1 = [2, 3, 4, 5, 6, 7, 8, 9, 10] 

#Cs2 = [10,20,30,40,50, 60, 70, 80, 100]

res = []

res2 = []



print(train_clf2.shape)

print(train_clf3.shape)



#for C in Cs1:

#    res.append(score(xgb.XGBRegressor(n_estimators = 6000, seed = 0, learning_rate = 0.01, max_depth = 3, subsample = 0.8, colsample_bytree = 0.8, colsample_bylevel = 0.8 ),train_np,random_state = 0, folds = C))

#    res2.append(score(xgb.XGBRegressor(n_estimators = 6000, seed = 0, learning_rate = 0.01, max_depth = 3, subsample = 0.8, colsample_bytree = 0.8, colsample_bylevel = 0.8 ),train_np2,random_state = 0, folds = C))

#for C in Cs2:    

#    res2.append(score(AdaBoostRegressor(n_estimators = C, random_state = 42, learning_rate = 0.01, base_estimator = xgb.XGBRegressor(max_depth = 8, seed = 0)),train_np,random_state = 0, folds = 10))





#p1, = plt.plot(Cs1, res,'r-o',label="V1")

#p2, = plt.plot(Cs1, res2,'b-o',label="V2")

#plt.legend([p1,p2])

#plt.show()
plt.hist(np.log(train.SalePrice))

plt.show()
index = pd.DataFrame(test.Id,columns = ['Id'])
#import xgboost as xgb



clf = xgb.XGBRegressor(n_estimators = 6000, seed = 0, learning_rate = 0.01, max_depth = 3, subsample = 0.8, colsample_bytree = 0.8, colsample_bylevel = 0.8 )

#clf = AdaBoostRegressor(n_estimators = 30, learning_rate = 0.1, random_state = 42, base_estimator = xgb.XGBRegressor(max_depth = 8, seed = 0))



train_fin = np.array(train_clf3)

test_fin = np.array(test_clf3)



kf1 = KFold(n = 1460 , n_folds=7, random_state=0, shuffle = True)

#kf2 = KFold(n = 1460 , n_folds=10, random_state=42, shuffle = True)

#kf3 = KFold(n = 1460 , n_folds=10, random_state=123, shuffle = True)



i = 0

res1 = []

res2 = []

res3 = []



for itrain, itest in kf1:

    i = i + 1

    Xtr, Xte = train_fin[itrain], train_fin[itest]

    ytr, yte = y[itrain], y[itest]

    clf.fit(Xtr, ytr.ravel())

    if i == 1:

        pred1 = pd.DataFrame(clf.predict(test_fin))

        print("Fold 1 :", rmse(yte, pd.DataFrame(clf.predict(Xte))))

        res1.append(rmse(yte, pd.DataFrame(clf.predict(Xte))))

    if i > 1 :

        pred1 = pred1 + pd.DataFrame(clf.predict(test_fin))

        print("Fold ",i, " :", rmse(yte, pd.DataFrame(clf.predict(Xte))))

        res1.append(rmse(yte, pd.DataFrame(clf.predict(Xte))))



print(np.mean(res1))



pred_1 = pred1/7 



pred_1.columns = ['SalePrice']

pred_final_1 = pd.DataFrame(np.exp(pred_1), index = new_test.index, columns = ['SalePrice'])

pred_final = pred_final_1



pred_submit = pd.merge(index,pred_final,left_index=True,right_index=True)

pred_submit.head()



pred_submit.to_csv('XGB_CV7.csv',index=False)