#import libraries 

import tensorflow
from tensorflow import keras
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


import os
print(os.listdir("../input"))
#training set consists house id 1- 1460
data= pd.read_csv('../input/train.csv',index_col=0)
#testing set consists house id 1461 - 2919
testdata=pd.read_csv('../input/test.csv', index_col=0)
#insert saleprice column in testdata just so the they have the same column
testdata["SalePrice"]=0

#concate the dataframes together so I can clean the data together. They will be seperated later
total=pd.concat([data,testdata])
print(total.info())

#Fill NAN with 0's
total=total.apply(lambda x: x.fillna(0))
print(total.info())

# conver object to dummy variables
columns=total.columns[data.dtypes == 'object']
total=pd.get_dummies(total, columns =columns)
print(total.head())

#n=now split the data again
data = total.iloc[0:1460,:]
testdata= total.iloc[1460:, :]

print(data.head())
print(testdata.head())

print(data.info())
#Check Correlation
Correlation =data.corr()
pd.DataFrame(Correlation)
correlation_Y=pd.DataFrame(Correlation["SalePrice"])
correlation_Y.sort_values(by= 'SalePrice', ascending=False)


#Correlation to be removed
#to_remove= correlation_Y.head(50).index.tolist()
#print(to_remove)
#Drop columns with no correlation 
data=data.drop(["MSZoning_0","Utilities_0","Exterior1st_0","Exterior2nd_0","KitchenQual_0","Functional_0","SaleType_0"], axis=1)
testdata=testdata.drop(["MSZoning_0","Utilities_0","Exterior1st_0","Exterior2nd_0","KitchenQual_0","Functional_0","SaleType_0"], axis=1)

#Drop columns with low correlation
#data= data.drop(['BsmtFinType2_GLQ', 'RoofMatl_Metal', 'RoofStyle_Mansard', 'GarageCond_Gd', 'Foundation_Wood', 'Condition2_RRAe', 'BldgType_TwnhsE', 'LotConfig_Corner', 'Electrical_0', 'Condition1_RRNe', 'MiscFeature_Gar2', 'RoofMatl_Tar&Grv', 'Fence_GdPrv', 'Condition1_RRAn', 'LotConfig_FR2', 'RoofMatl_ClyTile', 'BsmtFinSF2', 'ExterCond_Ex', 'PoolQC_Gd', 'Foundation_Stone', 'Utilities_NoSeWa', 'Utilities_AllPub', 'SaleType_ConLI', 'RoofMatl_Roll', 'Condition2_RRAn', 'Neighborhood_SawyerW', 'SaleCondition_Alloca', 'Functional_Mod', 'PoolQC_Fa', 'RoofStyle_Flat', 'Exterior2nd_Stone', 'BsmtHalfBath', 'Functional_Sev', 'Exterior1st_Plywood', 'LotConfig_FR3', 'Neighborhood_Blmngtn', 'SaleType_CWD', 'RoofMatl_Membran', 'Heating_GasW', 'Neighborhood_Blueste', 'RoofStyle_Shed', 'MiscVal', 'MiscFeature_TenC', 'Condition1_RRNn', 'Neighborhood_NWAmes', 'GarageType_2Types', 'Exterior2nd_AsphShn', 'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior1st_CBlock'], axis=1)
#testdata= testdata.drop(['BsmtFinType2_GLQ', 'RoofMatl_Metal', 'RoofStyle_Mansard', 'GarageCond_Gd', 'Foundation_Wood', 'Condition2_RRAe', 'BldgType_TwnhsE', 'LotConfig_Corner', 'Electrical_0', 'Condition1_RRNe', 'MiscFeature_Gar2', 'RoofMatl_Tar&Grv', 'Fence_GdPrv', 'Condition1_RRAn', 'LotConfig_FR2', 'RoofMatl_ClyTile', 'BsmtFinSF2', 'ExterCond_Ex', 'PoolQC_Gd', 'Foundation_Stone', 'Utilities_NoSeWa', 'Utilities_AllPub', 'SaleType_ConLI', 'RoofMatl_Roll', 'Condition2_RRAn', 'Neighborhood_SawyerW', 'SaleCondition_Alloca', 'Functional_Mod', 'PoolQC_Fa', 'RoofStyle_Flat', 'Exterior2nd_Stone', 'BsmtHalfBath', 'Functional_Sev', 'Exterior1st_Plywood', 'LotConfig_FR3', 'Neighborhood_Blmngtn', 'SaleType_CWD', 'RoofMatl_Membran', 'Heating_GasW', 'Neighborhood_Blueste', 'RoofStyle_Shed', 'MiscVal', 'MiscFeature_TenC', 'Condition1_RRNn', 'Neighborhood_NWAmes', 'GarageType_2Types', 'Exterior2nd_AsphShn', 'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior1st_CBlock'], axis=1)

print(data.info())
print(testdata.info())
# Now remove SalePrice column from test set
testdata.drop(["SalePrice"],axis=1)
#split out validation data set
Y=data[['SalePrice']]
data.drop(["SalePrice"], axis=1)
X=data
print(testdata)
print(X)
print(Y)
#Define root mean squared error function as rmse is not available in keras' built in loss function
from keras import backend as K
from math import sqrt
from keras.callbacks import EarlyStopping

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

#early stopping
early_stopping= EarlyStopping(patience=3)

#standardize the data
scaler=StandardScaler().fit(X)
rescaledX=scaler.transform(X)


#create model
model = Sequential()
model.add(Dense(330, input_dim=305,init="normal", activation='relu'))
model.add(Dense(75, init="normal", activation='relu')) 
model.add(Dense(32, init="normal", activation='relu'))
model.add(Dense(4, init="normal", activation='relu'))  
model.add(Dense(1, init="normal"))

model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

model.fit(rescaledX,Y, nb_epoch=150, batch_size=5, callbacks= [early_stopping])
scores=model.evaluate(rescaledX,Y, batch_size=5)



#standardize the test data
rescaledtestdata=scaler.transform(testdata)
print(rescaledtestdata)
predictions =model.predict(rescaledtestdata)

testdata["SalePrice"]=predictions
output=testdata[["SalePrice"]]
print(output.head(20))

output.to_csv("prediction_output.csv")