# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



%matplotlib inline

# Any results you write to the current directory are saved as output.
df_train =pd.read_csv('../input/train.csv',index_col='Id')

df_test = pd.read_csv('../input/test.csv',index_col='Id')

ID=df_test.index

print(df_train.shape)

print(df_test.shape)
# Outlier detection 

plt.scatter(df_train.GrLivArea, df_train.SalePrice, c = "blue", marker = "s")

plt.title("Looking for outliers")

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.show()



df_train= df_train[df_train.GrLivArea<4000]

print(df_train.shape)
X=df_train.drop(['SalePrice'],axis=1)

Y=df_train.SalePrice

X = pd.concat((X, df_test))

print(Y.shape)
def show_missing_data(X):

    missing_data= (X.isnull().sum()/len(X))*100

    missing_data=missing_data.drop(missing_data[missing_data==0].index,axis=0).sort_values(ascending=False)



    f, ax = plt.subplots(figsize=(10, 8))

    plt.xticks(rotation='90')

    sns.barplot(x=missing_data.index, y=missing_data)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)



show_missing_data(X)
#For the following features, we just replace NA by 'None'

#PoolQC,MiscFeature, Alley, Fence, FireplaceQu,GarageFinish, GarageType, GarageQual, GarageCond, BsmtCond

#BsmtExposure, BsmtQual, BsmtFinType1, BsmtFinType1 , MasVnrType



features_to_fill =['PoolQC','MiscFeature', 'Alley', 'Fence', 'FireplaceQu','MasVnrType','GarageFinish', 'GarageType', 'GarageQual', 'GarageCond', 'BsmtCond','BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']

for k in features_to_fill:

    X[k]=X[k].fillna('None')

    

#LotFrontage : we fill by the median value

X['LotFrontage']=X['LotFrontage'].fillna(np.mean(X['LotFrontage']))

#features we fill with 0

fill_with_zeros=['MasVnrArea','TotalBsmtSF','GarageCars','GarageArea','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','BsmtHalfBath','BsmtFullBath']

for k in fill_with_zeros:

    X[k]=X[k].fillna(0)

    

#Exterior1st,Exterior2nd,KitchenQual,SaleType,Electrical has only 1 missing value, so we just fill the most common value

X['Exterior1st']=X['Exterior1st'].fillna('VinylSd')

X['Exterior2nd']=X['Exterior2nd'].fillna('VinylSd')

X['KitchenQual']=X['KitchenQual'].fillna('TA')

X['Electrical']=X['Electrical'].fillna('SBrkr')

X['SaleType']=X['SaleType'].fillna('WD')





#Functional,MSZoning,electrical : we fill with the most common value

X['Functional']=X['Functional'].fillna('Typ')

X['MSZoning']=X['MSZoning'].fillna('RL')



#Utilities: since 2916 of 2917 are AllPub, let's just drop this feature

X = X.drop(['Utilities'],axis=1)

#in 75% of cases, garageyrblt= yearbuilt, so let's just drop this feature

X=X.drop(['GarageYrBlt'],axis=1)

#First we change some numerical values into categorial values



X = X.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},

                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",

                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}

                      })
#Then we map some categorial values



replacement = {"LandSlope":     {"Gtl": 0, "Mod": 1, "Sev":2},

        "Alley" : {"Grvl" : 1, "Pave" : 2, "None":0},

        "ExterQual": {"Ex": 5 , "Gd": 4 , "TA": 3, "Fa": 2, "Po" : 1},

        "ExterCond":{"Ex": 5 , "Gd": 4 , "TA": 3, "Fa": 2, "Po" : 1},

        "BsmtQual": {"Ex": 5 , "Gd": 4 , "TA": 3, "Fa": 2, "Po" : 1, "None":0},

        "BsmtCond": {"Ex": 5 , "Gd": 4 , "TA": 3, "Fa": 2, "Po" : 1, "None":0},

        "BsmtExposure": {"Gd":4, "Av":3, "Mn":2, "No":1, "None":0},

        "BsmtFinType1": {"GLQ":6,"ALG":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"None":0},

        "BsmtFinType2": {"GLQ":6,"ALG":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"None":0},

        "HeatingQC": {"Ex": 5 , "Gd": 4 , "TA": 3, "Fa": 2, "Po" : 1},

        "CentralAir":{"N":0,"Y":1},

        "KitchenQual":{"Ex": 5 , "Gd": 4 , "TA": 3, "Fa": 2, "Po" : 1},

        "Functional":{"Typ":7, "Min1":6 ,"Min2":5 ,"Mod":4 ,"Maj1":3 ,"Maj2":2, "Sev":1 ,"Sal":0},

        "FireplaceQu":{"Ex": 5 , "Gd": 4 , "TA": 3, "Fa": 2, "Po" : 1, "None":0},

        "GarageFinish": {"Fin":3, "RFn":2, "Unf":1, "None":0},

        "GarageQual": {"Ex": 5 , "Gd": 4 , "TA": 3, "Fa": 2, "Po" : 1, "None":0},

        "GarageCond": {"Ex": 5 , "Gd": 4 , "TA": 3, "Fa": 2, "Po" : 1, "None":0},

        "PavedDrive":{"Y":2,"P":1,"N":0},

        "PoolQC":{"Ex": 5 , "Gd": 4 , "TA": 3, "Fa": 2, "Po" : 1, "None":0},

        "Fence": {"GrPrv":4, "MnPrv":3, "GdWo":2, "MnWw":1 ,"None":0}

       }



X=X.replace(replacement)
X.head()
# Some more feature engineering



X['OverallQualCond']=X.OverallQual*X.OverallCond

X['ExterQualCond']=X.ExterQual*X.ExterCond

X['BsmtQualCond']=X.BsmtQual*X.BsmtCond

X['TotalSF']=X.TotalBsmtSF + X.GrLivArea

X.head()



X = pd.get_dummies(X)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error

from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingRegressor





def score(estimator):

    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(X1)

    rmse=np.sqrt(-cross_val_score(estimator,X1,np.log(Y),scoring="neg_mean_squared_error",cv=kf))

    return(rmse)

    

scale=RobustScaler()

X_scale= scale.fit_transform(X)

X1=X_scale[:1456]

X2=X_scale[1456:]
alpha_list=np.linspace(start=1,stop=50,num=50)    

L=[]



for alpha_val in alpha_list:

    model= Ridge(alpha=alpha_val)

    L.append(score(model).mean())

        

plt.plot(alpha_list,L)

print(min(L))

print(alpha_list[np.argmin(L)])
alpha_list=np.linspace(start=0.0004,stop=0.001,num=50)    

L=[]



for alpha_val in alpha_list:

    model= Lasso(alpha=alpha_val, max_iter=2000 )

    L.append(score(model).mean())

        

plt.plot(alpha_list,L)

print(min(L))

print(alpha_list[np.argmin(L)])

    
learning_rate_list=np.linspace(start=0.05,stop=0.5,num=25)    

L=[]



for learning_rate_val in learning_rate_list:

    model = GradientBoostingRegressor(loss='ls',n_estimators=100,learning_rate=learning_rate_val)

    L.append(score(model).mean())



plt.plot(learning_rate_list,L)

print(min(L))

print(learning_rate_list[np.argmin(L)])
# First stacked regression

X1_train, X1_test, Y_train, Y_test = train_test_split(X1,Y,test_size=0.2)



model1=Ridge(alpha=18)

model2=Lasso(alpha=0.00056, max_iter=2000)

model3=GradientBoostingRegressor(loss='ls',n_estimators=100,learning_rate=0.1)



model1.fit(X1_train,np.log(Y_train))

model2.fit(X1_train,np.log(Y_train))

model3.fit(X1_train,np.log(Y_train))



pred1= model1.predict(X1_test)

pred2= model2.predict(X1_test)

pred3= model3.predict(X1_test)



pred=np.mean((pred1,pred2,pred3),axis=0)

print(np.sqrt(mean_squared_error(np.log(Y_test),pred)))

# Second stacked regression

A, B, Y_A,Y_B = train_test_split(X1,Y,test_size=0.5)

B, C, Y_B,Y_C = train_test_split(B,Y_B,test_size=0.5)



model1=Ridge(alpha=18)

model2=Lasso(alpha=0.00056, max_iter=2000)

model3=GradientBoostingRegressor(loss='ls',n_estimators=100,learning_rate=0.1)



model1.fit(A,np.log(Y_A))

model2.fit(A,np.log(Y_A))

model3.fit(A,np.log(Y_A))



pred1B= model1.predict(B)

pred2B= model2.predict(B)

pred3B= model3.predict(B)



pred1C=model1.predict(C)

pred2C=model2.predict(C)

pred3C=model3.predict(C)
B1_data = {'pred1': pred1B, 'pred2': pred2B, 'pred3' : pred3B}

B1 = pd.DataFrame(data=B1_data)



C1_data = {'pred1': pred1C, 'pred2': pred2C, 'pred3' : pred3C}

C1 = pd.DataFrame(data=C1_data)
meta_model=LinearRegression()

meta_model.fit(B1,np.log(Y_B))

pred= meta_model.predict(C1)

print(np.sqrt(mean_squared_error(np.log(Y_C),pred)))
# Second stacked regression

A, B, Y_A,Y_B = train_test_split(X1,Y,test_size=0.3)

C = X2



model1=Ridge(alpha=18)

model2=Lasso(alpha=0.00056, max_iter=2000)

model3=GradientBoostingRegressor(loss='ls',n_estimators=100,learning_rate=0.1)



model1.fit(A,np.log(Y_A))

model2.fit(A,np.log(Y_A))

model3.fit(A,np.log(Y_A))



pred1B= model1.predict(B)

pred2B= model2.predict(B)

pred3B= model3.predict(B)



pred1C=model1.predict(C)

pred2C=model2.predict(C)

pred3C=model3.predict(C)



B1_data = {'pred1': pred1B, 'pred2': pred2B, 'pred3' : pred3B}

B1 = pd.DataFrame(data=B1_data)



C1_data = {'pred1': pred1C, 'pred2': pred2C, 'pred3' : pred3C}

C1 = pd.DataFrame(data=C1_data)



meta_model=LinearRegression()

meta_model.fit(B1,np.log(Y_B))

pred= meta_model.predict(C1)

pred=np.exp(pred)

pred
test_output = pd.DataFrame({"Id" : ID,"SalePrice": pred})

test_output.set_index("Id", inplace=True)

test_output.to_csv("prediction8.csv")