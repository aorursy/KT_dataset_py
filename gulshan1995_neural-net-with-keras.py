import os, sys, re

from keras.models import Sequential

from keras.layers import Dense

import time

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



print (time.time())

dataset = pd.read_csv("../input/loan.csv")
dataset = dataset.fillna(0) ## filling missing values with zeros
dataset['application_type'] = dataset['application_type'].astype('category').cat.codes

dataset['addr_state'] = dataset['addr_state'].astype('category').cat.codes

dataset['earliest_cr_line'] = pd.to_datetime(dataset['earliest_cr_line'])

dataset['earliest_cr_line'] = (dataset['earliest_cr_line']-dataset['earliest_cr_line'].min())/np.timedelta64(1,'D')

dataset['emp_length'] = dataset['emp_length'].astype('category').cat.codes

dataset['grade'] = dataset['grade'].astype('category').cat.codes

dataset['home_ownership'] = dataset['home_ownership'].astype('category').cat.codes

dataset['initial_list_status'] = dataset['initial_list_status'].astype('category').cat.codes

dataset['issue_d'] = pd.to_datetime(dataset['issue_d'])

dataset['issue_d'] = (dataset['issue_d']-dataset['issue_d'].min())/np.timedelta64(1,'D')

dataset['last_credit_pull_d'] = pd.to_datetime(dataset['last_credit_pull_d'])

dataset['last_credit_pull_d'] = (dataset['last_credit_pull_d']-dataset['last_credit_pull_d'].min())/np.timedelta64(1,'D')

dataset['last_pymnt_d'] = pd.to_datetime(dataset['last_pymnt_d'])

dataset['last_pymnt_d'] = (dataset['last_pymnt_d']-dataset['last_pymnt_d'].min())/np.timedelta64(1,'D')

dataset['loan_status'] = dataset['loan_status'].astype('category').cat.codes

dataset['next_pymnt_d'] = pd.to_datetime(dataset['next_pymnt_d'])

dataset['next_pymnt_d'] = (dataset['next_pymnt_d']-dataset['next_pymnt_d'].min())/np.timedelta64(1,'D')

dataset['purpose'] = dataset['purpose'].astype('category').cat.codes

dataset['pymnt_plan'] = dataset['pymnt_plan'].astype('category').cat.codes

dataset['sub_grade'] = dataset['sub_grade'].astype('category').cat.codes

dataset['term'] = dataset['term'].astype('category').cat.codes

dataset['verification_status'] = dataset['verification_status'].astype('category').cat.codes

dataset['verification_status_joint'] = dataset['verification_status_joint'].astype('category').cat.codes

non_numerics = [x for x in dataset.columns\

if not (dataset[x].dtype == np.float64 or dataset[x].dtype == np.int8 or dataset[x].dtype == np.int64)]

df = dataset

df = df.drop(non_numerics,1)
def LoanResult(status):

    if (status == 5) or (status == 1) or (status == 7):

        return 1

    else:

        return 0



df['loan_status'] = df['loan_status'].apply(LoanResult)
train, test = train_test_split(df, test_size = 0.25)



##running complete data set will take a lot of time, hence reduced the data set

X_train = train.drop('loan_status',1).values[0:50000, :]

Y_train = train['loan_status'].values[0:50000]



X_test = test.drop('loan_status',1).values[0:1000, :]

Y_test = test['loan_status'].values[0:1000]



X_pred = test.drop('loan_status',1).values[1001:2000, :]
seed = 8 

np.random.seed(seed)

# Create the model 

model = Sequential()



# Define the three layered model

model.add(Dense(110, input_dim = 126, kernel_initializer = "uniform", activation = "relu"))

model.add(Dense(110, kernel_initializer = "uniform", activation = "relu"))

model.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))

#

# Compile the model

model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])

#

# Fit the model

model.fit(X_train, Y_train, epochs= 220, batch_size=200)
performance = model.evaluate(X_test, Y_test)

print("%s: %.2f%%" % (model.metrics_names[1], performance[1]*100))

#
# Predict using the trained model

prediction = model.predict(X_pred)

rounded_predictions = [np.round(x) for x in prediction]

print(rounded_predictions)
