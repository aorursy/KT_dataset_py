# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd,numpy as np

import matplotlib.pyplot as plt    # For plotting

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
ef = pd.read_csv('../input/nyc-tax-lot-zoning/pluto.csv')
ef = ef.iloc[:,:10]

ef.head()
df = pd.read_csv('../input/nyc-tax-regression/Train.csv')
new_df = pd.merge(df, ef,  how='inner', left_on=['Block','Lot'], right_on = ['block','lot'],left_index = True,copy = False)

new_df.shape
new_df = new_df[np.isfinite(new_df['block'])]
new_df2 = new_df.drop_duplicates(subset = ['PropertyID'])

new_df2.shape
import pandas_profiling as pp

pp.ProfileReport(new_df2)
new_df2 = new_df2.drop(['Address','State','TotalNoOfUnits','DateOfEvaluation','PropertyID'],axis = 1)

new_df2 = new_df2.drop(['Block','Lot','address','block','lot','bldgclass'],axis = 1)

new_df2.head()
new_df2.iloc[:,[0,2,4,5,9,12,13,17]] = new_df2.iloc[:,[0,2,4,5,9,12,13,17]].astype('category')

new_df2.dtypes
def cut_levels(x, threshold, new_value):

    x = x.copy()

    value_counts = x.value_counts()

    labels = value_counts.index[value_counts < threshold]

    x[np.in1d(x, labels)] = new_value

    return x



# def cut_levels(x, threshold, new_value):

#     value_counts = x.value_counts()

#     labels = value_counts.index[value_counts < threshold]

#     x[np.in1d(x, labels)] = new_value



new_df2['zonedist1'] = cut_levels(new_df2['zonedist1'], 21, 'others')

new_df2['BldgClassCategory'] = cut_levels(new_df2['BldgClassCategory'], 14, 'others')

new_df2['Surroundings'] = cut_levels(new_df2['Surroundings'], 30, 'others')

new_df2['BldgClass_AtEvaluationTime'] = cut_levels(new_df2['BldgClass_AtEvaluationTime'], 20, 'others')



#new_df2['Surroundings'].value_counts()
new_df2['Age'] = 2014-new_df2['YearOfConstruction']

new_df2 = new_df2.drop(['YearOfConstruction'],axis = 1)



new_df2['Age'] = new_df2['Age'].astype('int')

new_df2.dtypes
new_df3 = new_df2[~new_df2.index.duplicated()]

new_df3['builtfar'] = new_df3['builtfar'].fillna(method = 'ffill')
X = new_df3.copy().drop("PropertyEvaluationvalue",axis=1) 

y = new_df3["PropertyEvaluationvalue"]



## Split the data into trainx, testx, trainy, testy with test_size = 0.20 using sklearn

trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.20)



## Print the shape of X_train, X_test, y_train, y_test

print(trainx.shape)

print(testx.shape)

print(trainy.shape)

print(testy.shape)
trainx.head()
# !pip3 install --upgrade git+https://github.com/scikit-learn-contrib/categorical-encoding



from category_encoders import *

enc = TargetEncoder(cols=['Surroundings', 'BldgClassCategory','Borough',

                          'NoOfResidentialUnits','NoOfCommercialUnits','TaxClass_AtEvaluationTime','council','landuse',

                          'BldgClass_AtEvaluationTime','zonedist1','schooldist','ZipCode']).fit(trainx,trainy)

# transform the datasets

trainx2 = enc.transform(trainx.reset_index(drop=True),trainy)

testx2 = enc.transform(testx.reset_index(drop = True))

trainx2['assessland'] = trainx2['assessland'].replace(0,trainx2['assessland'].mean())

testx2['assessland'] = testx2['assessland'].replace(0,testx2['assessland'].mean())

trainx2['LandAreaInSqFt'] = np.log(trainx2['LandAreaInSqFt'])

trainx2['GrossAreaInSqFt'] = np.log(trainx2['GrossAreaInSqFt'])

trainx2['assessland'] = np.log(trainx2['assessland'])

#trainx2['builtfar'] = np.log(trainx2['builtfar'])



testx2['LandAreaInSqFt'] = np.log(testx2['LandAreaInSqFt'])

testx2['GrossAreaInSqFt'] = np.log(testx2['GrossAreaInSqFt'])

testx2['assessland'] = np.log(testx2['assessland'])

#testx2['builtfar'] = np.log(testx2['builtfar'])
trainx2 = trainx2.reset_index()

testx2 = testx2.reset_index()
from sklearn.preprocessing import StandardScaler



## Scale the numeric attributes

scaler = StandardScaler()

scaler.fit(trainx2)



trainx2 = scaler.transform(trainx2)

testx2 = scaler.transform(testx2)

#tf1[['GrossAreaInSqFt','LandAreaInSqFt','Age']] = scaler.transform(tf1[['GrossAreaInSqFt','LandAreaInSqFt','Age']])

trainx2 = pd.DataFrame(trainx2)

testx2 = pd.DataFrame(testx2)



trainx2.head()
# from sklearn.decomposition import PCA



# pca_model = PCA(n_components=6)

# pca_model.fit(trainx2)

# trainx2 = pca_model.transform(trainx2)

# testx2 = pca_model.transform(testx2)
from sklearn.neighbors import KNeighborsRegressor

KNN = KNeighborsRegressor(n_neighbors=15,metric='manhattan',leaf_size=40,p=3,weights='distance',algorithm = 'kd_tree')

%time KNN.fit(trainx2,trainy)
from sklearn.metrics import mean_absolute_error,mean_squared_error

import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# import math

# %time KNN_train_pred = KNN.predict(trainx2)

# print(mean_absolute_percentage_error(KNN_train_pred,trainy))



%time KNN_test_pred = KNN.predict(testx2)

print(mean_absolute_percentage_error(KNN_test_pred,testy))
from sklearn.model_selection import GridSearchCV



grid_params = {

    'n_neighbors':[7,9,11,13,15,17],

    'weights' : ['uniform','distance'],

    'metric' : ['euclidean','manhattan']

}



gs = GridSearchCV(KNeighborsRegressor(),

                  grid_params,

                  verbose = 1,

                  cv = 3,

                  n_jobs = -1

                 )

%time gs_results = gs.fit(trainx2,trainy)
gs_results.best_estimator_
df_sub = pd.DataFrame(knn_subpred2)

df_sub.to_csv('Sub4_226.csv')