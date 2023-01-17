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
import numpy as np

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesClassifier

from sklearn.impute import SimpleImputer



#Read the input data

train = pd.read_csv("../input/train.csv")



#Label for prediction to be used in Training Data

train_Y = train.SalePrice



#Drop colum of label from the dataframe

train = train.drop(['SalePrice','Id'], axis = 1)



print(len(train))

print(len(train_Y))
#Separate numeric attributes from the training data



numeric_cols = list(train._get_numeric_data().columns)

print(numeric_cols)
#Create a numeric_df from the training data of just the numeric columns



numeric_df = train[numeric_cols]

numeric_df.head()
#Separate categorical attributes from training data



categorical_cols = list(set(train.columns) - set(numeric_cols))

print(categorical_cols)
#Create separate dataframe of just categorical attributes



categorical_df = train[categorical_cols]

categorical_df.head()
#Apply One Hot Encoding on categorical columns



one_hot_encoded_df = pd.get_dummies(categorical_df)

one_hot_encoded_df.head()
#Create Training set by combining the numeric df and one_hot_encoded_df



train_df = pd.concat([numeric_df, one_hot_encoded_df], axis = 1)

train_df.head()
#Create a list of all the columns in the Final training dataset



cols_list = list(train_df.columns)

print(cols_list)
#Handling Missing values using SimpleImputer



myImputer = SimpleImputer()



train_df = myImputer.fit_transform(train_df)
#Using ExtraTreesClassifier to extract important features from the training dataset



#Building a forest to determine feature importances



forest = ExtraTreesClassifier(n_estimators=200, random_state=0)



forest.fit(train_df, train_Y)
#Extract the important features from the training dataset 



importances = list(np.argsort(forest.feature_importances_))



#Extract the top 30 important features from total columns list



indices = importances[-80:]



important_features = []

for i in indices:

    important_features.append(cols_list[i])

    

print(important_features)

#Create the final training data set



train_df = pd.concat([numeric_df, one_hot_encoded_df], axis = 1)



train_X = train_df[important_features]



train_X = myImputer.fit_transform(train_X)
#Fit the training data 

model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.005)



model.fit(train_X,train_Y)
#Predicting on test data



test = pd.read_csv("../input/test.csv")



test_numeric = list(test._get_numeric_data().columns)



test_categorical = list(set(test.columns)-set(test_numeric))



numeric_test_df = test[test_numeric]



categorical_test_df = test[test_categorical]



one_hot_encoded_test_df = pd.get_dummies(categorical_test_df)



test_final = pd.concat([numeric_test_df, one_hot_encoded_test_df], axis=1)



test_X = test_final[important_features]



test_X = myImputer.fit_transform(test_X)



predicted_prices = model.predict(test_X)



print(predicted_prices)
my_submission = pd.DataFrame({"Id": test.Id,"SalePrice":predicted_prices})



my_submission.to_csv("submission.csv", index=False)