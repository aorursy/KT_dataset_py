import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
pd.set_option('display.max_columns', None)

data.head()
data.shape
data.describe()
#Distribution of the target variable

sns.set_style('darkgrid')

plt.figure(figsize = (6,5))

g = sns.countplot(x = 'Churn', data = data)

i=0

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2., height + 0.1,

    '{}'.format(height),ha="center")

    i += 1

display()
#Visualizing Binary columns

cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','PaperlessBilling']

total = len(data['Churn'])

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(19,10), dpi= 60)

axes = axes.flatten()

for i, ax in zip(cols, axes):

    g = sns.countplot(x = i, data = data, ax = ax, hue = 'Churn')

    g.set_ylabel('Percentage')

    for p in g.patches:

      height = p.get_height()

      g.text(p.get_x()+p.get_width()/2., height + 0.1,

      '{:1.2f}'.format(height/total),ha="center")

display()
#Replacing spaces with null values in total charges column

data['TotalCharges'] = data["TotalCharges"].replace(" ",np.nan)



#Dropping null values from total charges column which contain .15% missing data 

data = data[data["TotalCharges"].notnull()]

data = data.reset_index()[data.columns]



#convert to float type

data["TotalCharges"] = data["TotalCharges"].astype(float)



#replace 'No internet service' to No for the following columns

replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

                'TechSupport','StreamingTV', 'StreamingMovies']

for i in replace_cols : 

    data[i]  = data[i].replace({'No internet service' : 'No'})



#replace 'No internet service' to No for MultipleLines

    data['MultipleLines']  = data['MultipleLines'].replace({'No phone service' : 'No'})

    

#replace values

data["SeniorCitizen"] = data["SeniorCitizen"].replace({1:"Yes",0:"No"})
#Visualizing Tenure

plt.figure(figsize = (18,4))

sns.countplot(x = 'tenure',data = data, hue = 'Churn')

plt.show()
from sklearn.preprocessing import LabelEncoder



#Make dummy variables for catigorical variables with >2 levels

dummy_columns = ["MultipleLines","InternetService","OnlineSecurity",

                 "OnlineBackup","DeviceProtection","TechSupport",

                 "StreamingTV","StreamingMovies","Contract",

                 "PaymentMethod"]



df = pd.get_dummies(data, columns = dummy_columns)



#Encode catigorical variables with 2 levels

enc = LabelEncoder()

encode_columns = ["Churn","PaperlessBilling","PhoneService",

                  "gender","Partner","Dependents","SeniorCitizen"]



for col in encode_columns:

    df[col] = enc.fit_transform(df[col])

    

#Remove customer ID column

del df["customerID"]





#Make TotalCharges column numeric, empty strings are zeros

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors = 'coerce').fillna(0)
from sklearn.model_selection import train_test_split



#Split data into x and y

y = df[["Churn"]]

x = df.drop("Churn", axis=1)



#Create test and training sets

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= .2, random_state= 1)
from xgboost import XGBClassifier

from xgboost import plot_importance

from sklearn.metrics import accuracy_score



#Build XGBoost model

model = XGBClassifier()

model.fit(x_train, y_train.values.ravel())





#Predictions for test data

y_pred = model.predict(x_test)

predictions = [round(value) for value in y_pred]



#Accuracy

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



#Feature importance

fig, ax = plt.subplots(figsize=(10, 8))

plot_importance(model, ax = ax)

plt.show()