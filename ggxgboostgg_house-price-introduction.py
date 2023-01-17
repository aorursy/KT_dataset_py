import numpy as np

import seaborn as sb

import matplotlib.pyplot as plt

import pandas as pd 

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error

from math import sqrt

def rmlse(preds,train_data):

    labels = train_data.get_label()

    rmlse = sqrt( mean_squared_error( np.log(labels), np.log(preds) )  ) 

    return 'rmlse',rmlse, False 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

def label_encoding(train: pd.DataFrame, test: pd.DataFrame, col_definition: dict):

        """

        col_definition: encode_col

        """

        n_train = len(train)

        train = pd.concat([train, test], sort=False)

        for f in col_definition['encode_col']:

            try:

                lbl = preprocessing.LabelEncoder()

                train[f] = lbl.fit_transform(list(train[f].values))

            except:

                print(f)

        test = train[n_train:]

        train = train[:n_train]

        return train, test
train = pd.read_csv(os.path.join(dirname, "train.csv"))

test = pd.read_csv(os.path.join(dirname, "test.csv"))

print("Dim_train:",train.shape)

print("Dim_test:",test.shape)
y_train = train[["SalePrice","Id"]].copy().set_index("Id")

train   = train.drop("SalePrice",axis=1).reset_index().set_index("Id")

test    = test.reset_index().set_index("Id")
y_train.describe()
f,ax=plt.subplots(1,2,figsize=(18,8))

sb.distplot(y_train, rug=True,ax=ax[0])

ax[0].set_title('Empirical distribution')

ax[0].set_xlabel('SalePrice')

sb.boxplot(y_train,ax=ax[1])

ax[1].set_title('BoxPlot')

plt.show()

train.info()
train.describe()
pd.DataFrame(train.isnull().sum()/len(train)*100,columns=["% of missing train"]).join(

    pd.DataFrame(test.isnull().sum()/len(test)*100,columns=["% of missing test"])).sort_values(by="% of missing train", ascending=False)
f, ax = plt.subplots(figsize=(9, 9))

sb.heatmap( train.corr(), 

            vmax=.8, square=True)
test = test.loc[:,train.isnull().sum()/len(train)*100<30]

train = train.loc[:,train.isnull().sum()/len(train)*100<30]
non_categorical_missing_train = train.loc[:,(train.dtypes!="object") & (train.isnull().sum()/len(train)*100!=0)].columns

non_categorical_missing_test  = test.loc[:,(test.dtypes!="object") & (test.isnull().sum()/len(test)*100!=0)].columns



for i in non_categorical_missing_test:

     test[i]=test[i].fillna(train[i].median())



for i in non_categorical_missing_train:

     train[i]=train[i].fillna(train[i].median())

        

train = train.fillna("none")

test  = test.fillna("none")
pd.DataFrame(train.isnull().sum()/len(train)*100,columns=["% of missing train"]).join(

    pd.DataFrame(test.isnull().sum()/len(test)*100,columns=["% of missing test"])).sort_values(by="% of missing train", ascending=False)
cols = ['OverallQual','GrLivArea',

       'GarageCars', 'GarageArea', 'TotalBsmtSF',

       '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',

       'YearBuilt']

sb.pairplot(train[cols].join(y_train), size =5)
categorical_variables =  train.loc[:,(train.dtypes=="object")].columns.tolist()
train, test = label_encoding(train, test, col_definition={'encode_col': categorical_variables})
kf = KFold(n_splits=5,shuffle=True,random_state=123)

drop = categorical_variables

test_pred  = []

train_pred = []



for i,(a,b) in enumerate(kf.split(train,y_train.loc[train.index, ])) :

    Xt = train.iloc[a,:]

    yt = np.log(y_train.iloc[a, ])

   # Xt = Xt.drop(drop,axis=1)

    

    Xv = train.iloc[b,:]

    yv = np.log(y_train.iloc[b, ])

    #Xv = Xv.drop(drop,axis=1)

    

    train_data=lgb.Dataset(Xt, yt, categorical_feature=list(categorical_variables))

    test_data=lgb.Dataset(Xv, yv, reference=train_data, categorical_feature=list(categorical_variables))

    

    print('---------- Training fold Nº {} ----------'.format(i+1))

    

   

    params = {'num_leaves': 40,  'min_data_in_leaf': 45, 'objective': 'regression_l1', 'max_depth': 4, 'learning_rate': 0.05, 'boosting': 'gbdt',

         'random_state': 123, 'metric': ['rmse'], 'verbosity': -1,'type':'gamma'}

    

    model1 = lgb.train(params,train_data,valid_sets=[train_data, test_data],verbose_eval=10,num_boost_round=3000 , early_stopping_rounds=200)

                       #, feval=rmlse)



    

    train_pred.append(pd.Series(model1.predict(Xv),

                                index=Xv.index, name="pred"+ str(i)))

    test_pred.append(pd.Series(model1.predict(test),

                                index=test.index, name="fold_" + str(i)  ))

      

test_pred = np.exp(pd.concat(test_pred, axis=1).mean(axis=1))

train_pred = pd.concat(train_pred, axis=1)
print(f'CV: {np.sqrt(mean_squared_error(np.log(y_train), train_pred.mean(axis=1) ))}')

np.log(y_train).describe()
train_pred.mean(axis=1).describe()
test_pred = pd.DataFrame(test_pred.rename("SalePrice"))

test_pred.to_csv("lightgbm_baseline_func:l1.csv", header=True)