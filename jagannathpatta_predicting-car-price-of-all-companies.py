import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
files = ['audi','bmw','ford','hyundi','merc','skoda','toyota','vauxhall','vw']  # file names
dataframes = {}
for f in files:

  dataframes[f+'_data'] = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/'+f+'.csv')
  dataframes[f+'_data']['company'] = [f]*dataframes[f+'_data'].shape[0]
  dataframes[f+'_data'] = dataframes[f+'_data'].reindex(columns = sorted(dataframes[f+'_data'].columns))
  print( f+'_data :' , dataframes[f+'_data'].info() )
dataframes['hyundi_data'].rename(columns={"tax(£)": "tax"} , inplace = True)
dataframes['hyundi_data'].info()
Carsdata = pd.DataFrame()
Carsdata = pd.concat(dataframes.values() )
Carsdata.sample(10)
Carsdata.info()
Carsdata.year.loc[17726]= 2006
from pandas_profiling import ProfileReport
profile = ProfileReport(Carsdata , title ='Pandas Profiling')  # report for data with outliers.
# code snippit for removing outliers.
df = Carsdata
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print('Q1 :', Q1)
print('Q3 :', Q3)
print('IQR ', IQR)
dff = df[~( (df < (Q1 - 1.5*IQR))|(df > (Q3 + 1.5*IQR)) ).any(axis=1)]
dff.shape
profile2 = ProfileReport(dff , title ='Pandas Profiling Report of Cars dataset')   # report for data without outliers.
profile2
fig ,  ax = plt.subplots(1,2,figsize=(15, 5) )
ax[0].scatter( Carsdata.price ,  Carsdata.mileage)
ax[1].scatter( dff.price ,  dff.mileage)

plt.title(' Without Outliers Price VS Mileage ')
plt.xlabel('price')
plt.ylabel('Mileage')
Carsdata.boxplot(['mileage' ],by= ['engineSize'] , figsize=(18 ,5))
Carsdata.boxplot(['price' ],by= ['engineSize'] , figsize=(18 ,5))
Carsdata[Carsdata.engineSize == 1.9]
Carsdata[Carsdata.engineSize == 4.0].fuelType.value_counts()
Carsdata[Carsdata.engineSize == 4.0].transmission.value_counts()
Carsdata[Carsdata.engineSize == 5.2]
Carsdata.boxplot(['price','mileage'] , by = ['transmission'] , figsize=(15 , 4) )
dff.boxplot( ['price','mileage'] , by = ['transmission'] , figsize=(15 , 4) )
Carsdata.boxplot(['price','mileage'] , by= ['fuelType'] , figsize=(15 , 4) )
dff.boxplot(['price','mileage'], by= ['fuelType'] ,  figsize=(15 , 4) )
Carsdata.boxplot(['price']  , by= ['company'], figsize=(15 , 4) )
dff.boxplot(['price']  , by= ['company'], figsize=(15 , 4) )
dff.boxplot(['mpg'] , by= ['transmission'])
dff.boxplot(['mpg'] , by= ['fuelType'])
dff.boxplot(['mpg'] , by= ['company'] , figsize=(10 ,5))
dff.company.value_counts()
Carsdata.company.value_counts()
Carsdata.info()
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error
from math import sqrt
feature_data = Carsdata.drop(columns=['price','model'])      # data with outliers.
target_data = Carsdata.price
###############################################################
feature_data1 = dff.drop(columns=['price','model'])           # data without outliers
target_data1 = dff.price
cat_data = feature_data.select_dtypes(include=['object'])
print(cat_data.columns)
num_data = feature_data.select_dtypes(include=['int','float'])
print(num_data.columns)
oe = OrdinalEncoder()
oe.fit(cat_data)
ss = StandardScaler()
ss.fit(num_data)
cat = pd.DataFrame(data = oe.transform(cat_data) , columns=cat_data.columns)
num = pd.DataFrame(data = ss.transform(num_data), columns= num_data.columns)
cat_pipeline = make_pipeline(OrdinalEncoder())
num_pipeline = make_pipeline(StandardScaler())
preprocessor = make_column_transformer(
              (cat_pipeline,cat_data.columns),
              (num_pipeline,num_data.columns)
)

from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(feature_data, target_data)        # data with outliers
##########################################################################
trainX1, testX1, trainY1, testY1 = train_test_split(feature_data1, target_data1)     # data without outliers
from sklearn.linear_model import LinearRegression
pipeline = make_pipeline(preprocessor, LinearRegression())
pipeline.fit( trainX , trainY)
print("Training Score : ",pipeline.score(trainX, trainY))
print('Testing Score : ',pipeline.score(testX, testY))
print('Root mean square error :',  sqrt(mean_squared_error(testY, pipeline.predict(testX))))
pipeline.fit( trainX1 , trainY1)
print("Training Score : ", pipeline.score( trainX1, trainY1 ))
print('Testing Score : ', pipeline.score( testX1, testY1 ))
print('Root mean square error :', sqrt(mean_squared_error(testY1 , pipeline.predict( testX1 ))))
from sklearn.ensemble import RandomForestRegressor
rf_pipeline = make_pipeline( preprocessor , RandomForestRegressor( n_estimators= 100 ))
rf_pipeline.fit( trainX , trainY)
print("Training Score : ",rf_pipeline.score(trainX, trainY))
print('Testing Score : ',rf_pipeline.score(testX, testY))
print('Root mean square error :',  sqrt(mean_squared_error(testY, rf_pipeline.predict(testX))))
rf_pipeline.fit( trainX1 , trainY1)
print("Training Score : ", rf_pipeline.score( trainX1, trainY1 ))
print('Testing Score : ', rf_pipeline.score( testX1, testY1 ))
print('Root mean square error :', sqrt(mean_squared_error(testY1 , rf_pipeline.predict( testX1 ))))
from sklearn.neighbors import KNeighborsRegressor
kn_pipeline = make_pipeline( preprocessor , KNeighborsRegressor(n_neighbors=4))
kn_pipeline.fit( trainX , trainY)
print("Training Score : ",kn_pipeline.score(trainX, trainY))
print('Testing Score : ',kn_pipeline.score(testX, testY))
print('Root mean square error :',  sqrt(mean_squared_error(testY, kn_pipeline.predict(testX))))
kn_pipeline.fit( trainX1 , trainY1)
print("Training Score : ", kn_pipeline.score( trainX1, trainY1 ))
print('Testing Score : ', kn_pipeline.score( testX1, testY1 ))
print('Root mean square error :', sqrt(mean_squared_error(testY1 , kn_pipeline.predict( testX1 ))))
from sklearn.model_selection import GridSearchCV
gs_pipeline = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=100))
params = {'randomforestregressor__n_estimators':[100,200,250]}
gs = GridSearchCV(gs_pipeline, param_grid=params, cv=5, n_jobs=4)
gs.fit( trainX , trainY)
print("Training Score : ",gs.score(trainX, trainY))
print('Testing Score : ',gs.score(testX, testY))
print('Root mean square error :',  sqrt(mean_squared_error(testY, gs.predict(testX))))
print('******************************')
print('Best params :',gs.best_params_)
print('Best Score :', gs.best_score_ )
gs.fit( trainX1 , trainY1)
print("Training Score : ", gs.score( trainX1, trainY1 ))
print('Testing Score : ', gs.score( testX1, testY1 ))
print('Root mean square error :', sqrt(mean_squared_error(testY1 , gs.predict( testX1 ))))
print('******************************')
print('Best params :', gs.best_params_)
print('Best Score :', gs.best_score_ )
result = pd.DataFrame()
result['actual'] = testY1
result['prediction'] = gs.predict( testX1)
result.sample(10)