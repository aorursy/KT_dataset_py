import pandas as pd

import numpy as np



test_data = pd.read_csv("test.csv").drop('id',axis=1)

train_data = pd.read_csv("train.csv").drop('id',axis=1)



# Describe the shape of the train dataset

print(train_data.shape)

print(test_data.shape)



# Check for null values

print(train_data.isnull().any().any())

print(test_data.isnull().any().any())



# Perform data processing

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X = train_data.drop('species',axis=1)

y = le.fit_transform(train_data['species'])

target_name = le.inverse_transform(y)



# Train Test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)



# Min Max Scaler

from sklearn.preprocessing import MinMaxScaler

minmaxscale = MinMaxScaler()

scaler = minmaxscale.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)



# fit a logistic regression model

from sklearn.linear_model import LogisticRegression

LR_model = LogisticRegression(C=1000, multi_class='multinomial',solver='lbfgs')

LR_model.fit(X_train_scaled,y_train)

print(LR_model.score(X_train_scaled,y_train))

print(LR_model.score(X_test_scaled,y_test))



# Predict using the model for the sample_submission.csv dataset

test_data_scaled = scaler.transform(test_data)

y_pred = LR_model.predict_proba(test_data_scaled)

y_pred.shape



test_data2 = pd.read_csv("test.csv")



submission =  pd.DataFrame(data=y_pred,columns=list(le.classes_))

submission.insert(0, 'id', test_data2.id)

submission.reset_index()

submission.to_csv('submission.csv', index=False)