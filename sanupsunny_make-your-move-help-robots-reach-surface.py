import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Loading important libraries

import pandas as pd

import numpy as np

pd.options.display.max_rows=20

pd.options.display.max_columns=20

import time
#Reading CSV Files

start=time.time()

X_train  = pd.read_csv("../input/X_train.csv")

X_test  = pd.read_csv("../input/X_test.csv")

y_train  = pd.read_csv("../input/y_train.csv")

end=time.time()

print("All csv in repsository loaded within {:0.2f} secs".format(end-start))
#Various way to get information from csv file

#a)X_train dataset

print ("Rows     : " ,X_train.shape[0])

print ("Columns  : " ,X_train.shape[1])

print ("\nFeatures : \n" ,X_train.columns.tolist())

print ("\nMissing values :  ", X_train.isnull().sum().values.sum())

print ("\nUnique values :  \n",X_train.nunique())
#You can also store above information in dataframe.Let try this for test dataset

X_test_df = pd.DataFrame({

    "Rows": X_test.shape[0],

    "Columns": X_test.shape[1],

    "Features":X_test.columns.tolist(),

    "Missing Values":X_test.isnull().sum().values.sum(),

    "Unique values":X_test.nunique()

})

X_test_df
#If difficulty in understanding the data or visually confusing try other way 

X_test_df.T

#First dataframe seems better for me .Compared to this but you can use any one.
#What our target column-Surface contains

y_train['surface'].value_counts()
y_train['surface'].value_counts().reset_index().rename(columns={'index': 'target'})
#Joining training set

combined_train=pd.merge(X_train,y_train,how='left',on='series_id')

combined_train.head()

print ("X_train :{} ,Y_train {} ,Combined_train {}".format(X_train.shape,y_train.shape,combined_train.shape))

combined_train.to_csv("combined_train.csv",index=False)

print("Csv file saved")
#Drop unnecessary columns

combined_train.drop(columns=["row_id","measurement_number","group_id"], inplace=True)

combined_train.sample(2)
#excluding target variable and saving variable X

X=combined_train[combined_train.columns.difference(['surface'])]

y = combined_train.surface



#Feature selection

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier



embedded_rf_selector=SelectFromModel(RandomForestClassifier(n_estimators=20),threshold="1.25 *median")

embedded_rf_selector.fit(X,y)

embedded_rf_support=embedded_rf_selector.get_support()

embedded_rf_feature=X.loc[:,embedded_rf_support].columns.tolist()

print(str(len(embedded_rf_feature)),'selected features')

print("Feature include",embedded_rf_feature)



#Considering the selected feature

#manually selected feature(aka Domain knowledge)

#embedded_rf_feature.append('series_id') #-Incase algorithm doesnot select series_id as feature add it manually

embedded_rf_feature

y = combined_train.surface

combined_train_features = embedded_rf_feature

X = combined_train[combined_train_features]

X.columns.tolist()
y.value_counts().reset_index().rename(columns={'index':'Surface','surface' :'Total Count'})
#Splitting our train and test data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)



#Getting to know shape of train_test split

print("X_train {} ,\n X_train {}".format((X_train.shape),(X_train.columns.tolist())))

print("X_test {} ,\n X_train {}".format((X_test.shape),(X_test.columns.tolist())))

print("y_train {} ".format((y_train.shape)))

print("y_test {} ".format((y_test.shape)))
#Randomly selecting ML algorithm for quick test

from sklearn.ensemble import RandomForestClassifier

randomForest=RandomForestClassifier(n_estimators=100)

randomForest.fit(X_train,y_train)

Y_prediction=randomForest.predict(X_test)

randomForest.score(X_train,y_train)

acc_randomForest=round(randomForest.score(X_train,y_train)*100,2)

print("Accuracy as per RandomForestClassifier algorithm:",round(acc_randomForest,2,),"%")



#Diff algo trial

from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

Y_pred_knn=knn.predict(X_test)

acc_knn=round(knn.score(X_train,y_train)*100,2)

print("Accuracy as per KNeighborsClassifier algorithm:",round(acc_knn,2,),"%")
from lightgbm import LGBMClassifier

lgbmclassifier=LGBMClassifier(n_estimators=500)

lgbmclassifier.fit(X_train,y_train)

Y_pred_lgbm=lgbmclassifier.predict(X_test)

acc_lgbm=round(lgbmclassifier.score(X_train,y_train)*100,2)

print("Accuracy as per LGBMClassifier algorithm:",round(acc_lgbm,2,),"%")





print("Making predictions for the following 5 top surfaces:")

print(X_test['series_id'].head())

print("The predictions are")

print(lgbmclassifier.predict(X_test.head()))
submission = pd.DataFrame({'series_id':X_test['series_id'],'surface':Y_pred_lgbm})

submission=submission.sort_values('series_id', ascending=True).drop_duplicates(['series_id'],keep='first').reset_index(drop=True)

submission.to_csv("submission.csv",index=False)

print("CSV saved")
#Check result

print("Shape of submitted file :",submission.shape)

submission.head()