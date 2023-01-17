# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#loading data



training_data_path = '/kaggle/input/titanic/train.csv'

trainingData = pd.read_csv(training_data_path)



trainingData.head()
trainingData.shape
# Preprocessing 



#Remove rows with missing targets

trainingData.dropna(axis = 0, subset = ['Survived'], inplace = True)



#trainingData.dropna(axis = 0, subset = ['Embarked'], inplace = True)

#Remove rows with missing values

#trainingData.dropna(axis=0,inplace = True)



#Seperate Target from predictors

X = trainingData.drop('Survived',axis=1)

y = trainingData.Survived

#Preprocessing (Extracintg numerical varibles and low cardinaliy categorical variables)



col_description = X.dtypes == 'object'



categorical_cols = []

numerical_cols = []



for i in range(len(col_description)):

    if col_description.iloc[i] and trainingData[col_description.index[i]].nunique() <= 5:

        categorical_cols = categorical_cols + [col_description.index[i]]

    if (not(col_description.iloc[i])):

        numerical_cols = numerical_cols + [col_description.index[i]]

print(categorical_cols,numerical_cols)



#Retaining numerical cols and low cardinality categoical cols (for pipelining)



X = X[categorical_cols+numerical_cols]





#we need to do this with the test set also
from sklearn.model_selection import train_test_split





X_train,X_val,y_train,y_val= train_test_split(X,y)
### One hot Encoding for categorical Variable



# from sklearn.preprocessing import OneHotEncoder





# OH_Encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)



# OH_col_trains = pd.DataFrame(OH_Encoder.fit_transform(X_train[categorical_cols]))

# OH_col_valids = pd.DataFrame(OH_Encoder.transform(X_val[categorical_cols]))



# OH_col_trains.index = X_train.index

# OH_col_valids.index = X_val.index



# X_train = X_train.drop(categorical_cols,axis=1)

# X_val = X_val.drop(categorical_cols,axis=1)



# X_train = pd.concat([X_train,OH_col_trains],axis = 1 )

# X_val = pd.concat([X_val,OH_col_valids], axis = 1)
# X_train.head()
#imputation



# from sklearn.impute import SimpleImputer



# my_imputer = SimpleImputer()



# imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

# imputed_X_val = pd.DataFrame(my_imputer.transform(X_val))



# #imputation removed column names put them back

# imputed_X_train.columns = X_train.columns

# imputed_X_val.columns = X_val.columns
# ### Using Decision Tree Regressor



# from sklearn.tree import DecisionTreeRegressor

# from sklearn.metrics import mean_absolute_error



# model_DecisionTreeRegressor_default = DecisionTreeRegressor()

# model_DecisionTreeRegressor_default.fit(imputed_X_train,y_train)



# predicted_survived = model_DecisionTreeRegressor_default.predict(imputed_X_val)

# predicted_survived = predicted_survived.round()

# mean_absolute_error(y_val,predicted_survived)
# ### Using Random Forest Regressor



# from sklearn.ensemble import RandomForestRegressor

# from sklearn.metrics import mean_absolute_error



# model_RandomForestRegressor_default = RandomForestRegressor()

# model_RandomForestRegressor_default.fit(imputed_X_train,y_train)



# predicted_survived = model_RandomForestRegressor_default.predict(imputed_X_val)

# predicted_survived = predicted_survived.round()

# mean_absolute_error(predicted_survived,y_val)
#Using Pipline



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor



#Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy = 'constant')



#Preprocessing for categorical data

categorical_transformer = Pipeline(steps = [

    ('imputer', SimpleImputer(strategy = 'most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))

])



#Bundle Preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

transformers = [

    ('num',numerical_transformer, numerical_cols),

    ('cat', categorical_transformer, categorical_cols)

])





#Defining Model

model = RandomForestRegressor()



#Creating and Evaluating Pipeline



#Bundle Preprocessing and model

my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),

                             ('model', model)])



#Preprocessing of training data and fit

my_pipeline.fit(X_train, y_train)



#Preprocessing of validation data and prediction

preds = my_pipeline.predict(X_val)

preds = preds.round()



#Evalutaion of the model

score = mean_absolute_error(y_val,preds)

print(score)
# #using XGBoost Regressor

# from xgboost import XGBRegressor

# from sklearn.metrics import mean_absolute_error



# my_model = XGBRegressor()



# my_model.fit(imputed_X_train,y_train)



# preds = my_model.predict(inputed_X_val)

# #preds = preds.round()



# print(mean_absolute_error(preds,y_val))

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.shape



X_test
X_test = test_data[categorical_cols+numerical_cols]



final_prediction = my_pipeline.predict(X_test)



final_prediction = final_prediction.round()
final_prediction = pd.DataFrame(final_prediction)

final_prediction.columns = ['Survived']



final_prediction.index = X_test.PassengerId



final_prediction
gender_submission = final_prediction.to_csv('Titanic_prediction',index = True)