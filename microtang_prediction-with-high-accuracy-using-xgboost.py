#import the library

import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix

from termcolor import colored

import warnings

warnings.filterwarnings('ignore')
# Importing the dataset

df = pd.read_csv('../input/HR_comma_sep.csv')

df.head()
#data process

X = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary']]

y = df['left']

# Encoding the categorical data

labelencoder_X_1 = LabelEncoder()

X['sales'] = labelencoder_X_1.fit_transform(X['sales'])

labelencoder_X_2 = LabelEncoder()

X['salary'] = labelencoder_X_2.fit_transform(X['salary'])

# Splitting the dataset into the Training set and Validation set

Xt, Xv, yt, yv = train_test_split(X, y, test_size = 0.25, random_state = 0)

dt = xgb.DMatrix(Xt.as_matrix(),label=yt.as_matrix())

dv = xgb.DMatrix(Xv.as_matrix(),label=yv.as_matrix())
#Build the model

params = {

    "eta": 0.2,

    "max_depth": 4,

    "objective": "binary:logistic",

    "silent": 1,

    "base_score": np.mean(yt),

    'n_estimators': 1000,

    "eval_metric": "logloss"

}

model = xgb.train(params, dt, 3000, [(dt, "train"),(dv, "valid")], verbose_eval=200)
#Prediction on validation set

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(yv, (y_pred>0.5))

print(colored('The Confusion Matrix is: ', 'red'),'\n', cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

print(colored('The Accuracy on Test Set is: ', 'blue'), colored(predict_accuracy_on_test_set, 'blue'))
# Input the data you want to predict

print("please input the folowing information of this employee:satisfaction_level")

satisfaction_level = input("satisfaction_level:")

print("please input the folowing information of this employee:last_evaluation")

last_evaluation = input("last_evaluation:")

print("please input the folowing information of this employee:number_project")

number_project = input("number_project:")

print("please input the folowing information of this employee:average_montly_hours")

average_montly_hours = input("average_montly_hours:")

print("please input the folowing information of this employee:time_spend_company")

time_spend_company = input("time_spend_company:")

print("please input the folowing information of this employee:Work_accident")

Work_accident = input("Work_accident:")

print("please input the folowing information of this employee:promotion_last_5years")

promotion_last_5years = input("promotion_last_5years:")

print("please input the folowing information of this employee:sales")

sales = input("sales:")

print("please input the folowing information of this employee:salary")

salary = input("salary:")

#  Encoding categorical data

sales_input = labelencoder_X_1.transform(np.array([[sales]]))

salary_input = labelencoder_X_2.transform(np.array([[salary]]))



# Make prediction

new_prediction = model.predict(xgb.DMatrix([[float(satisfaction_level), float(last_evaluation), int(number_project), int(average_montly_hours), int(time_spend_company), int(Work_accident), int(promotion_last_5years), sales_input, salary_input]]))

if(new_prediction > 0.5):

    print(colored("This employee will leave!", 'red'))

else:

    print(colored("This employee will not leave!", 'green'))