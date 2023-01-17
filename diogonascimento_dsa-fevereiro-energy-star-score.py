from pandas import read_csv

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt



df=read_csv('../input/dataset_treino.csv')



def validar(dt):

    features = dt.columns.tolist()

    X=dt[features]

    for feature in features:

        print('------------------------------------')

        print(feature)

        print('Not Available - {}'.format(dt[feature].astype(str).str.count('Not Available').sum()))

        print('N/A - {}'.format(dt[feature].isna().sum()))

        print('Tipo - {}'.format(dt[feature].dtypes))

        

        

cols=['Property Id','Order','Property Name','Parent Property Id','Parent Property Name','BBL - 10 digits','NYC Borough, Block and Lot (BBL) self-reported',

     'NYC Building Identification Number (BIN)','Address 1 (self-reported)','Address 2','Postal Code','Street Number','Street Name',

     'Borough','Primary Property Type - Self Selected','List of All Property Use Types at Property','Largest Property Use Type',

     '2nd Largest Property Use Type','2nd Largest Property Use - Gross Floor Area (ft²)','3rd Largest Property Use Type','Metered Areas (Energy)'

      ,'Fuel Oil #1 Use (kBtu)','Fuel Oil #2 Use (kBtu)','Fuel Oil #4 Use (kBtu)','Fuel Oil #5 & 6 Use (kBtu)',

     'Diesel #2 Use (kBtu)','District Steam Use (kBtu)','Release Date','Water Required?','DOF Benchmarking Submission Status','Latitude',

      'Longitude','Community Board','Council District','Census Tract','NTA']



df2=df.drop(columns=cols)

cols=['3rd Largest Property Use Type - Gross Floor Area (ft²)','Water Use (All Water Sources) (kgal)','Water Intensity (All Water Sources) (gal/ft²)']



df2=df2.drop(columns=cols)

df2=df2.drop(columns='Metered Areas  (Water)')

cols=['Weather Normalized Site EUI (kBtu/ft²)','Weather Normalized Site Electricity Intensity (kWh/ft²)','Weather Normalized Site Natural Gas Intensity (therms/ft²)'

     ,'Weather Normalized Source EUI (kBtu/ft²)','Weather Normalized Site Natural Gas Use (therms)','Weather Normalized Site Electricity (kWh)']



df2=df2.drop(columns=cols)

cols=['Natural Gas Use (kBtu)']

df2=df2.drop(columns=cols)

df2["Electricity Use - Grid Purchase (kBtu)"] = np.where(df2["Electricity Use - Grid Purchase (kBtu)"] == 'Not Available', 0, df2["Electricity Use - Grid Purchase (kBtu)"])

df2["Electricity Use - Grid Purchase (kBtu)"] = np.where(df2["Electricity Use - Grid Purchase (kBtu)"].astype(float) == 0,round(df2["Electricity Use - Grid Purchase (kBtu)"].astype(float).mean(),2), df2["Electricity Use - Grid Purchase (kBtu)"])



df2["Total GHG Emissions (Metric Tons CO2e)"] = np.where(df2["Total GHG Emissions (Metric Tons CO2e)"] == 'Not Available', 0, df2["Total GHG Emissions (Metric Tons CO2e)"])

df2["Total GHG Emissions (Metric Tons CO2e)"] = np.where(df2["Total GHG Emissions (Metric Tons CO2e)"].astype(float) == 0,round(df2["Total GHG Emissions (Metric Tons CO2e)"].astype(float).mean(),2), df2["Total GHG Emissions (Metric Tons CO2e)"])



df2["Direct GHG Emissions (Metric Tons CO2e)"] = np.where(df2["Direct GHG Emissions (Metric Tons CO2e)"] == 'Not Available', 0, df2["Direct GHG Emissions (Metric Tons CO2e)"])

df2["Direct GHG Emissions (Metric Tons CO2e)"] = np.where(df2["Direct GHG Emissions (Metric Tons CO2e)"].astype(float) == 0,round(df2["Direct GHG Emissions (Metric Tons CO2e)"].astype(float).mean(),2), df2["Direct GHG Emissions (Metric Tons CO2e)"])



df2["Indirect GHG Emissions (Metric Tons CO2e)"] = np.where(df2["Indirect GHG Emissions (Metric Tons CO2e)"] == 'Not Available', 0, df2["Indirect GHG Emissions (Metric Tons CO2e)"])

df2["Indirect GHG Emissions (Metric Tons CO2e)"] = np.where(df2["Indirect GHG Emissions (Metric Tons CO2e)"].astype(float) == 0,round(df2["Indirect GHG Emissions (Metric Tons CO2e)"].astype(float).mean(),2), df2["Indirect GHG Emissions (Metric Tons CO2e)"])



df2.dropna(axis=1)

#cols=['DOF Gross Floor Area']

#df2=df2.drop(columns=cols)

cols=['ENERGY STAR Score']

Y=df2['ENERGY STAR Score']

cols=['Electricity Use - Grid Purchase (kBtu)','Total GHG Emissions (Metric Tons CO2e)','Direct GHG Emissions (Metric Tons CO2e)','Indirect GHG Emissions (Metric Tons CO2e)']

df2.drop(columns=cols)

df2=df2.drop(columns='DOF Gross Floor Area')

df2['Electricity Use - Grid Purchase (kBtu)']=df2['Electricity Use - Grid Purchase (kBtu)'].astype(float)

df2['Total GHG Emissions (Metric Tons CO2e)']=df2['Total GHG Emissions (Metric Tons CO2e)'].astype(float)

df2['Direct GHG Emissions (Metric Tons CO2e)']=df2['Direct GHG Emissions (Metric Tons CO2e)'].astype(float)

df2['Indirect GHG Emissions (Metric Tons CO2e)']=df2['Indirect GHG Emissions (Metric Tons CO2e)'].astype(float)

#df2=df2.drop(columns='ENERGY STAR Score')

X=df2

validar(dt=X)
#from dominance_analysis import Dominance

#dominance_regression=Dominance(data=df2.astype(float),target='ENERGY STAR Score',objective=1)
from sklearn import metrics

import xgboost as xgb





#print(metrics.mean_absolute_error(Y,y_1))

from sklearn.model_selection import train_test_split

data_dmatrix = xgb.DMatrix(data=X,label=Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test)

#rmse = np.sqrt(mean_squared_error(y_test, preds))

mae = metrics.mean_absolute_error(y_test, preds)

print("MAE: %f" % (mae))

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 5, 'alpha': 10}

xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

import matplotlib.pyplot as plt



xgb.plot_importance(xg_reg)

plt.rcParams['figure.figsize'] = [5, 5]

plt.show()
from sklearn import preprocessing

Z=pd.concat([df2['Source EUI (kBtu/ft²)'], df2['Site EUI (kBtu/ft²)'],df2['ENERGY STAR Score']], axis=1, sort=False)

Z.head()
import matplotlib.pyplot as plt

#sns.scatterplot(x=Z['Largest Property Use Type - Gross Floor Area (ft²)'], y=Z['ENERGY STAR Score'])

#sns.scatterplot(x=Z['Year Built'], y=Z['ENERGY STAR Score'])

#sns.scatterplot(x=Z['Number of Buildings - Self-reported'], y=Z['ENERGY STAR Score'])

sns.scatterplot(x=Z['Site EUI (kBtu/ft²)'], y=Z['ENERGY STAR Score'])

#sns.scatterplot(x=Z['Occupancy'], y=Z['ENERGY STAR Score'])

#sns.scatterplot(x=Z['ENERGY STAR Score'], y=Z['Electricity Use - Grid Purchase (kBtu)'])

#sns.scatterplot(x=Z['Property GFA - Self-Reported (ft²)'], y=Z['ENERGY STAR Score'])

sns.scatterplot(x=df2['Largest Property Use Type - Gross Floor Area (ft²)'], y=df2['ENERGY STAR Score'])
sns.scatterplot(x=df2['Year Built'], y=df2['ENERGY STAR Score'])
sns.scatterplot(x=df2['Number of Buildings - Self-reported'], y=df2['ENERGY STAR Score'])
sns.scatterplot(x=Z['Source EUI (kBtu/ft²)'], y=Z['ENERGY STAR Score'])
X2=pd.concat([Z['Source EUI (kBtu/ft²)'], Z['Site EUI (kBtu/ft²)']], axis=1, sort=False)

#XX2=pd.DataFrame()

#XX2['Site EUI (kBtu/ft²)']=Z['Site EUI (kBtu/ft²)']

X2.head()

#validar(dt=XX2)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor



regr_1 = DecisionTreeRegressor(max_depth=64)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=64),n_estimators=10, random_state=9)

regr_1.fit(X2, Y)

regr_2.fit(X2, Y)

# Predict

y_1 = regr_1.predict(X2)

y_2 = regr_2.predict(X2)

print(metrics.mean_absolute_error(Y,y_1))

print(metrics.mean_absolute_error(Y,y_2))
import xgboost as xgb

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X2, Y, test_size=0.33, random_state=42)

X_train.head()

xgdmat=xgb.DMatrix(X_train,y_train)

our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':8,'min_child_weight':1}

final_gb=xgb.train(our_params,xgdmat)

tesdmat=xgb.DMatrix(X_test)

y_pred=final_gb.predict(tesdmat)

print(y_pred)

from sklearn.metrics import mean_squared_error

import math

testScore=metrics.mean_absolute_error(y_test.values,y_pred)

print('-------------')

print(testScore)
dfteste=read_csv('../input/dataset_teste.csv')

dfteste.info()
dfteste
newDf=pd.DataFrame()

newDf['Source EUI (kBtu/ft²)']=dfteste['Source EUI (kBtu/ft²)']

newDf['Site EUI (kBtu/ft²)']=dfteste['Site EUI (kBtu/ft²)']

validar(dt=newDf)

newDf.head()
pred=regr_2.predict(newDf)
newDf['pred']=pred.astype(int)
newDf['Property Id']=dfteste['Property Id']
postar=pd.DataFrame()

postar['Property Id']=newDf['Property Id']

postar['score']=newDf['pred']

postar.head()
postar.to_csv('submission.csv',index=False)