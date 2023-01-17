import pandas as pd

import numpy as np

df = pd.read_csv("../input/train.csv",index_col="Id")
df.head()#Description of the data
from sklearn.ensemble import RandomForestRegressor

#Read csv into DataFrame

df_train = pd.read_csv("../input/train.csv",index_col="Id")

df_test = pd.read_csv("../input/test.csv",index_col="Id")

target = df_train["SalePrice"]#target variable

df_train = df_train.drop("SalePrice",axis=1)

df_train["training_set"] = True

df_test["training_set"] = False
#Create df_full to concat df_train and df_test

df_full = pd.concat([df_train,df_test])

df_full = df_full.interpolate()

df_full = pd.get_dummies(df_full)

df_full.head()
#Seperate df_full into df_train and df_test(By calling different variables)

df_train = df_full[df_full["training_set"] == True]

df_train = df_train.drop("training_set",axis=1)#Deleting 'trainiing_set' variable

df_test = df_full[df_full["training_set"] == False]

df_test = df_test.drop("training_set",axis=1)#Deleting

#Apply RandomForest Functions and seed

rf = RandomForestRegressor(n_estimators=100,n_jobs=-1)

rf.fit(df_train,target)

preds = rf.predict(df_test)
submission = pd.DataFrame({

    "Id": df_test.index,

    "SalePrice":preds

})

submission.to_csv('submission.csv', index=False)