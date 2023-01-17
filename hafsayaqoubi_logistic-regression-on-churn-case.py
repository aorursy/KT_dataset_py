#import datasets

import pandas as pd

import numpy as np

internet_data = pd.read_csv("../input/internetdata/internet_data.csv")

customer_data = pd.read_csv("../input/customerdata/customer_data.csv")

churn_data = pd.read_csv("../input/churndata/churn_data.csv")
# fusion des 3 datasets en un seul, en fonction de la colonne CustomerID

# use intersection of keys from both frames

df_1 = pd.merge(internet_data, customer_data, how='inner', on='customerID')



# dataframe final

dataset = pd.merge(df_1, churn_data, how='inner', on='customerID')
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
#Replacing NAN values in totalcharges

dataset['TotalCharges'] = dataset['TotalCharges'].replace(' ', np.nan)

dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'])
dataset.isnull().sum()
#TotalCharges comprend des missing values

from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='NaN', strategy='median', axis=0)

imr = imr.fit(dataset[['TotalCharges']])

dataset['TotalCharges'] = imr.transform(dataset[['TotalCharges']]).ravel()

dataset.isnull().sum()
dataset[numerical_features].describe()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(14, 4))

dataset[dataset.Churn == "No"][numerical_features].hist(bins=30, color="green", alpha=0.5, ax=ax)

dataset[dataset.Churn == "Yes"][numerical_features].hist(bins=30, color="red", alpha=0.5, ax=ax)
fig, ax = plt.subplots(1, 2, figsize=(14, 4))

dataset[dataset.Churn == "No"]['Contract'].value_counts().plot('bar', ax=ax[0],

                                       color = ['red', 'green', 'orange']).set_title('not churned')

dataset[dataset.Churn == "Yes"]['Contract'].value_counts().plot('bar', ax=ax[1], 

                                        color = ['red', 'green', 'orange']).set_title('churned')
from sklearn.preprocessing import LabelEncoder

lblE = LabelEncoder()

labeled_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

multi_cols=["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 

            "TechSupport", "StreamingTV",            "StreamingMovies", "Contract", "PaymentMethod"]

# Encode labels in labeled_cols columns. 

dataset[labeled_cols]= dataset[labeled_cols].apply(lambda col:lblE.fit_transform(col))

dataset[multi_cols]= dataset[multi_cols].apply(lambda col:lblE.fit_transform(col))

pd.get_dummies(dataset,columns=multi_cols, drop_first=True)
dataset.drop(['customerID'], axis=1, inplace=True)

#no need anymore for customerID
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])
from sklearn.model_selection import KFold

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression



X = dataset.iloc[:, :-1] # =>dataframe

y = dataset.iloc[:, 19] #=>series

kfold = model_selection.KFold(n_splits=10, random_state=100)

model_kfold = LogisticRegression()

results_kfold = model_selection.cross_val_score(model_kfold, X, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))



from sklearn.model_selection import cross_validate

from sklearn.metrics import recall_score

scoring = ['precision_macro', 'recall_macro']

scores = cross_validate(model_kfold, X, y, scoring=scoring,cv=kfold)

print("Precision: %.2f%%" % (scores['test_precision_macro'].mean()*100))

print("Recall: %.2f%%" % (scores['test_recall_macro'].mean()*100))