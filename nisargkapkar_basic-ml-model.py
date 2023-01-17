#Import necessary libraries/functions



import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor 

from sklearn.metrics import mean_absolute_error
#Load the data



#read train.csv and store it in 'X'

X=pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col='Id')

#read test.csv and store it in 'X_test'

X_test=pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')
#Examine the data



print(X.shape)

print(X.columns)

print(X.head(n=5))
#Seperate the target variable 



#drop rows with missing target values from 'X'

X.dropna(axis=0,subset=['SalePrice'],inplace=True)



#Store target in 'y' and drop the target column from 'X'

y=X.SalePrice

X.drop(['SalePrice'],axis=1,inplace=True)

print(y)
#Extract numerical and categorical columns



#print categorical column labels with cardinality

for i in X.columns:

    if X[i].dtype=="object":

        print(i,X[i].nunique(),sep=' ')
#Extract numerical and categorical columns



#Divide columns in 3 parts: categorical_columns, numerical_columns and columns_to_drop

categorical_columns=[]

numerical_columns=[]

columns_to_drop=[]



for i in X.columns:

    if X[i].nunique()<15 and X[i].dtype=="object":

        categorical_columns.append(i)

    elif X[i].nunique()>=15 and X[i].dtype=="object":

        columns_to_drop.append(i)

    elif X[i].dtype in ["int64","float64"]:

        numerical_columns.append(i)

        

print(categorical_columns)

print(columns_to_drop)

print(numerical_columns)
#Extract numerical and categorical columns 



#drop 'columns_to_drop' from 'X' and 'X_test'

X=X.drop(columns_to_drop,axis=1)

X_test=X_test.drop(columns_to_drop,axis=1)
#Impute missing data 



#optional

#print column labels with number of missing cells in that corresponding column



#for X dataset

missing_columns=X.isnull().sum()

print("X dataset")

print(missing_columns[missing_columns>0])



print()



#for X_test

missing_columns_test=X_test.isnull().sum()

print("For X_test set")

print(missing_columns_test[missing_columns_test>0])
#Impute missing data



#impute numerical_columns

numerical_imputer=SimpleImputer()



#for X

for i in numerical_columns:

    current_column=np.array(X[i]).reshape(-1,1)

    updated_column=numerical_imputer.fit_transform(current_column)

    X[i]=updated_column



#for X_test

for i in numerical_columns:

    current_column=np.array(X_test[i]).reshape(-1,1)

    updated_column=numerical_imputer.fit_transform(current_column)

    X_test[i]=updated_column
#Impute missing data



#impute categorical_columns

categorical_imputer=SimpleImputer(strategy="most_frequent")



#for X

for i in categorical_columns:

    current_column=np.array(X[i]).reshape(-1,1)

    updated_column=categorical_imputer.fit_transform(current_column)

    X[i]=updated_column



#for X_test

for i in categorical_columns:

    current_column=np.array(X_test[i]).reshape(-1,1)

    updated_column=categorical_imputer.fit_transform(current_column)

    X_test[i]=updated_column
#Impute missing data 



#optional

#print column labels with number of missing cells in that corresponding column



#for X dataset

missing_columns=X.isnull().sum()

print("X dataset")

print(missing_columns[missing_columns>0])



print()



#for X_test

missing_columns_test=X_test.isnull().sum()

print("For X_test set")

print(missing_columns_test[missing_columns_test>0])



#after imputation, there would be no columns with missing data
#Encode categorical columns 



#STEPS:

#get one-hot encoded columns for X and X_test (using fit_transform/transform)

#give names to one-hot encoded columns (using get_feature_names)

#drop categorical columns from X and X_test (using drop)

#oh encoding removes index, add back index (using .index)

#add one-hot encoded columns to X and X_test (using pd.concat)



ohencoder=OneHotEncoder(handle_unknown='ignore',sparse=False)



#for X

ohe_X=pd.DataFrame(ohencoder.fit_transform(X[categorical_columns]))

ohe_X.columns=ohencoder.get_feature_names(categorical_columns)

X.drop(categorical_columns,axis=1,inplace=True)

ohe_X.index=X.index

X=pd.concat([X,ohe_X],axis=1)



#for X_test

ohe_X_test=pd.DataFrame(ohencoder.transform(X_test[categorical_columns]))

ohe_X_test.columns=ohencoder.get_feature_names(categorical_columns)

X_test.drop(categorical_columns,axis=1,inplace=True)

ohe_X_test.index=X_test.index

X_test=pd.concat([X_test,ohe_X_test],axis=1)
#Split Dataset in train and validation set



X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.8,test_size=0.2)
#Define the model



model=RandomForestRegressor()
#Fit the model



model.fit(X_train,y_train)
#Predict and Evaluate on validation set



preds_valid=model.predict(X_valid)

score_valid=mean_absolute_error(y_valid,preds_valid)

print("MAE: ",score_valid)
#Generate Test Prediction



preds_test=model.predict(X_test)

submission=pd.DataFrame({'Id':X_test.index,'SalePrice':preds_test})

submission.to_csv('submission.csv',index=False)