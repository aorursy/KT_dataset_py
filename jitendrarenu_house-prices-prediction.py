#axis = 0 or rows or index will take one row at a time and like that all the rows 

#so the below will take the entire SalePrice column and drop all the rows that are not available with the help of dropna function 

#print (X_train_full.dropna(axis=0,subset=['SalePrice']))
#saving output 

#output = pd.DataFrame({'Id':X_test.index,'SalePrice':pred_test})



#output.to_csv('submission.csv',index=False)
#gradient boosting 

import pandas as pd 



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor



#loading the data

X = pd.read_csv('../input/train.csv',index_col='Id')

X_test = pd.read_csv('../input/test.csv',index_col='Id')



#removing all the rows where the predictor is null 

X.dropna(axis='rows',subset=['SalePrice'],inplace=True)



#seperating predictor from input data 

y = X['SalePrice']



#removing output predictor from input data 

X.drop('SalePrice',axis='columns',inplace=True)



# we need to define all our input data parameters as either int,boolean,float values 

# making sure all the data in the expected shape 



#numerical columns 

numerical_col = [col for col in X.columns if X[col].dtype in ['int64','float64']]



#categorical columns 

#we use cordinality to low value to keep the dataset simple 

#cordinality measures how many unique values are there in each column

#we will use the object caterogical type and cordinality to filter our values in the categorical column 

categorical_col = [col for col in X.columns if X[col].dtype == 'object' and X[col].nunique() <10]



#combining all the numerical columns and categorical columns 

cols = numerical_col + categorical_col 



# making a copy to make sure that we keep the original dataframe untouched 

modified_X = X[cols].copy()

modified_X_test = X_test[cols].copy()



#one hot encoding the variables using pandas dummies method 

X_final = pd.get_dummies(modified_X) 

X_test_final = pd.get_dummies(modified_X_test)



#if we encounter any new type of categories in test data we will face errors in the model process 

# to make sure we will make sure that only the columns matching the training data will be there in the test data

# we will make use of align function in the pandas to align both train and test dataframes 



X_final,X_test_final = X_final.align(X_test_final,join ='left',axis='columns')





#dividing data into train and validation sets 

#keeping random state to zero to make sure results are reproducible 

X_train,X_valid,y_train,y_valid = train_test_split(X_final,y,train_size=0.8,test_size=0.2,random_state=0)





#defining the model 

model = XGBRegressor(random_state=0)



model.fit(X_train,y_train)



preds_valid = model.predict(X_valid)



print ('default XGB:',mean_absolute_error(preds_valid,y_valid))
#XGB with tuning default parameters 

model_2 = XGBRegressor(n_estimators=500,learning_rate=0.05,n_jobs=3,verbose=False,random_state=0)



model_2.fit(X_train,y_train,early_stopping_rounds=8,eval_set=[(X_valid,y_valid)])





preds_valid_2 = model_2.predict(X_valid)



print ('XGB tuned:',mean_absolute_error(preds_valid_2,y_valid))
#final model for submission



final_model = XGBRegressor(n_estimators=500,learning_rate=0.05,random_state=0,verbsoe=False,n_jobs=3)



final_model.fit(X_train,y_train,early_stopping_rounds=8,eval_set=[(X_valid,y_valid)])



preds_test = final_model.predict(X_test_final)

#submitting output 

output = pd.DataFrame({'Id':X_test_final.index,'SalePrice':preds_test})



output.to_csv('submission.csv',index=False)