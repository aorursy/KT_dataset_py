import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
dataset = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")
dataset.head()
dataset.shape
dataset.dtypes
dataset.isnull().sum()
dataset["Location"].value_counts().plot(kind='bar', figsize=(10,10))
for col in dataset.columns:

    if (dataset[col].isnull().sum()) > 20000:

        print("'"+col+"',")
work_data = dataset.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm'], axis = 1)
work_data.shape
work_data.dtypes
work_data_object = work_data.select_dtypes(include="object")

work_data_number = work_data.select_dtypes(exclude="object")
print(work_data_object.shape)

print(work_data_number.shape)
for col in work_data_number.columns:

    mean = work_data_number[col].mean()

    work_data_number[col].fillna(mean, inplace = True)
for col in work_data_object.columns:

    mode = work_data_object[col].mode()[0]

    work_data_object[col].fillna(mode, inplace = True)
work_data_number.isnull().sum().plot()
work_data_object.isnull().sum()
label = LabelEncoder()

work_data_object = work_data_object.astype(str).apply(label.fit_transform)

# for col in work_data_object.columns:

#     work_data_object[col] = label.fit_transform(work_data_object[col].astype(str))
work_data_object.isnull().sum()
work_data = pd.concat([work_data_object, work_data_number], axis = 1)
work_data.shape
work_data.head()
work_data["WindGustDir"].unique()
corr_value = work_data.corr()["RainTomorrow"].values
for index, value in enumerate(corr_value):

    value = value * 100

    if(value > 20 or value < -20):

        print(index+1, value)

print("last index value :", index+1)
work_data.drop(work_data.columns[[1,2,3,4,5,7,8,9,12,13,18,19]], axis= 1, inplace = True)
X = work_data.drop(["RainTomorrow"], axis = 1)

Y = work_data[["RainTomorrow"]]
X.head()
Y.head()
Y.shape
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 5)
xtest.shape
xtest.shape
logistic_model = LogisticRegression()

logistic_model.fit(xtrain, ytrain)
y_predict_logistic = logistic_model.predict(xtest)
accuracy_score(y_predict_logistic, ytest)
confusion_matrix(y_predict_logistic, ytest)
logistic_model.score(xtrain, ytrain)
logistic_model.score(xtest, ytest)