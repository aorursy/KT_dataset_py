# Python 3 environment 



#Choosing a algorithm for prediction depends on various factors

#In the case of zillow home price prediction there are two good algorithms which does the job.

#1. Multiple linear regression - gives approximate value of home price.

#2. XGboost - this algorithm is used if we need more accurate prediction.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

P16_df = pd.read_csv("../input/properties_2016.csv")

train16_df = pd.read_csv("../input/train_2016_v2.csv")



#merging

train_df = pd.merge(train16_df, P16_df, on='parcelid', how='left')

train_df = pd.DataFrame(train_df)



#delete all the features whose null percentage is above 99% as shown in EDA_zillow

train_df.drop(['hashottuborspa', 'propertycountylandusecode', 'propertyzoningdesc', 'taxdelinquencyflag','decktypeid','finishedsquarefeet6','typeconstructiontypeid','architecturalstyletypeid','fireplaceflag','yardbuildingsqft26','storytypeid','basementsqft','finishedsquarefeet13','buildingclasstypeid'],axis = 1,inplace=True)

train_df.head()

#We can replace null values by mean value of each column or zero.

#mean_values = train_df.mean(axis=0)

train_df_new = train_df.fillna(0)

train_df_new.head()



 
#Selected features-7 

#No important catogorical features to convert to numeric(taxdeliquencyflag is nullfied)

cols = ['taxamount','calculatedfinishedsquarefeet','garagecarcnt','bathroomcnt','yearbuilt','structuretaxvaluedollarcnt','landtaxvaluedollarcnt']


y =train_df_new['logerror']

X = pd.DataFrame(train_df_new[cols])

#Data is divided into training data and test data.

#There are multiple ways one can split data to get accuracy in prediction

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

X_train.head()
#still in works



import xgboost as xgb

dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns)

dtest = xgb.DMatrix(X_test,y_test,feature_names=X_test.columns)

xgb_params = {

    'eta': 1,

    'max_depth': 1,

    'objective': 'reg:linear',

    'silent': 1

}

bst = xgb.train(xgb_params,dtrain,5)

#model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

feature_imp = bst.get_fscore()

pred = bst.predict(dtest)



#features['pars'] = feature_imp.keys()

#features['importance'] = feature_imp.values()

print(feature_imp.values())

print(feature_imp.keys())

#features.sort_values(by=['importance'],ascending=False,inplace=True)

#fig,ax= plt.subplots()

#fig.set_size_inches(20,10)

#plt.xticks(rotation=90)

#sn.barplot(data=features.head(15),x="importance",y="features",ax=ax,orient="h",color="#34495e")
