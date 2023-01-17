import pandas as pd

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rc

import seaborn as sns

import pylab as pl

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn import linear_model

import warnings
#Disabling warnings

warnings.simplefilter("ignore")
#importing data

data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv", na_values=' ')
#Shape and columns

print(data.shape)

print(data.columns)
#Peek at data

data.head(10)
#Checking for missing data

data.isna().sum()
#Replacing missing total charges with 0 and type conversion

data['TotalCharges'] = data['TotalCharges'].fillna(0)

pd.to_numeric(data['TotalCharges'])

data['TotalCharges'][:5]
#Checking proportions of male & female customers

pl.figure(figsize =(10,3))

data.groupby('gender').customerID.count().plot('barh')

pl.ylabel('Gender', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Proportion of male & female customers', fontsize=12)

plt.show()



plt.pie(data["gender"].value_counts().values, labels=["Males","Females"], autopct="%1.0f%%", wedgeprops={"linewidth":2,"edgecolor":"white"})

plt.title("Proportion of male & female customers")

plt.show()
#Plotting the proportions of senior, married, and single customers



#Setting up figure

pl.figure(figsize =(10,10))

rc('font', weight='bold')



#Male's Data

#Senior Males

mc=data[np.logical_and.reduce([data['gender']=='Male', data['Partner']=='No'])].groupby('SeniorCitizen').customerID.count()

#Married Males

mm=data[data['gender']=='Male'].groupby('Partner').customerID.count()

#Single Males

sm=data[np.logical_and.reduce([data['gender']=='Male', data['SeniorCitizen']==0, data['Partner']=='No'])].groupby('customerID').customerID.count()



#Female's Data

#Senior Females

fc=data[np.logical_and.reduce([data['gender']=='Female', data['Partner']=='No'])].groupby('SeniorCitizen').customerID.count()

#Married Females

mf=data[data['gender']=='Female'].groupby('Partner').customerID.count()

#Single Females

sf=data[np.logical_and.reduce([data['gender']=='Female', data['SeniorCitizen']==0, data['Partner']=='No'])].groupby('customerID').customerID.count()





#Bars for plotting

mfbar1 = [mc[1],fc[1]]

mfbar2 = [mm[1],mf[1]]

mfbar3 = [sm.count(),sf.count()]

 

# The position of the bars on the x-axis

r = [0,1]

 

# Names of group and bar width

names = ['Male','Female']

barWidth = 0.5

 

# Creating bars

plt.bar(r, mfbar1, edgecolor='white', width=barWidth, label='Senior Citizen')

plt.bar(r, mfbar2, bottom=mfbar1, edgecolor='white', width=barWidth, label='Married')

plt.bar(r, mfbar3, bottom=mfbar2, edgecolor='white', width=barWidth, label='Single')



 

# Custom X axis

plt.xticks(r, names, fontweight='bold')

plt.xlabel("Gender", fontsize=12)

plt.ylabel("Total Count", fontsize=12)

plt.title("Proportions of senior, married, and single Customers", fontsize=12)

plt.legend(loc='best')

 

# Show graphic

plt.show()
#Plotting total usage/non of phone service

pl.figure(figsize =(10,3))

data.groupby(['PhoneService']).customerID.count().plot(kind='barh')

pl.ylabel('Phone Service', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Phone Service Usage', fontsize=15)

plt.show()
#Plotting usage/non of multiplelines

pl.figure(figsize =(10,3))

data.groupby(['MultipleLines']).customerID.count().plot(kind='barh')

pl.ylabel('Multiple Lines', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Multiple Lines Usage', fontsize=15)

plt.show()
#Plotting usage/non of internet service

pl.figure(figsize =(10,3))

data.groupby(['InternetService']).customerID.count().plot(kind='barh')

pl.ylabel('Internet Service', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Internet Service Usage', fontsize=15)

plt.show()
#Plotting usage/non of online security

pl.figure(figsize =(10,3))

data.groupby(['OnlineSecurity']).customerID.count().plot(kind='barh')

pl.ylabel('Online Security', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Online Security', fontsize=15)

plt.show()
#Plotting usage/non of online backup

pl.figure(figsize =(10,3))

data.groupby(['OnlineBackup']).customerID.count().plot(kind='barh')

pl.ylabel('Online Backup', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Online Backup', fontsize=15)

plt.show()
#Plotting usage/non of device protection

pl.figure(figsize =(10,3))

data.groupby(['DeviceProtection']).customerID.count().plot(kind='barh')

pl.ylabel('Device Protection', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Device Protection', fontsize=15)

plt.show()
#Plotting usage/non of tech support

pl.figure(figsize =(10,3))

data.groupby(['TechSupport']).customerID.count().plot(kind='barh')

pl.ylabel('Tech Support', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Tech Support', fontsize=15)

plt.show()
#Plotting usage/non of tv streaming

pl.figure(figsize =(10,3))

data.groupby(['StreamingTV']).customerID.count().plot(kind='barh')

pl.ylabel('Streaming TV', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Streaming TV', fontsize=15)

plt.show()
#Plotting usage/non of streaming movies

pl.figure(figsize =(10,3))

data.groupby(['StreamingMovies']).customerID.count().plot(kind='barh')

pl.ylabel('Streaming Movies', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Streaming Movies', fontsize=15)

plt.show()
#Plotting contracts

pl.figure(figsize =(10,3))

data.groupby(['Contract']).customerID.count().plot(kind='barh')

pl.ylabel('Contract', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Contract', fontsize=15)

plt.show()
#Plotting billing

pl.figure(figsize =(10,3))

data.groupby(['PaperlessBilling']).customerID.count().plot(kind='barh')

pl.ylabel('Paperless Billing', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Paperless Billing', fontsize=15)

plt.show()
#Plotting usage of different payment methods

pl.figure(figsize =(10,3))

data.groupby(['PaymentMethod']).customerID.count().plot(kind='barh')

pl.ylabel('Payment Methods', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Payment Methods', fontsize=15)

plt.show()
#Checking proportions of churned and retained customers

pl.figure(figsize =(10,3))

data.groupby('Churn').customerID.count().plot('barh')

pl.ylabel('Current Status of Customer{Churned:Yes, Retained:No}', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Proportion of Churned & Retained customers', fontsize=12)

plt.show()



plt.pie(data["Churn"].value_counts().values, labels=["Retained","Churned"], autopct="%1.0f%%", wedgeprops={"linewidth":2,"edgecolor":"white"})

plt.title("Proportion of Churned & Retained customers")

plt.show()
#Data Transformations

encode = LabelEncoder()

encode.fit(['Male','Female'])

data['gender'] = encode.transform(data['gender'])



encode.fit(['No','Yes'])

data['Partner'] = encode.transform(data['Partner'])



encode.fit(['No','Yes'])

data['Dependents'] = encode.transform(data['Dependents'])



encode.fit(['No','Yes'])

data['PhoneService'] = encode.transform(data['PhoneService'])



encode.fit(['No','No phone service','Yes'])

data['MultipleLines'] = encode.transform(data['MultipleLines'])



encode.fit(['DSL','Fiber optic','No'])

data['InternetService'] = encode.transform(data['InternetService'])



encode.fit(['No','No internet service','Yes'])

data['OnlineSecurity'] = encode.transform(data['OnlineSecurity'])



encode.fit(['No','No internet service','Yes'])

data['OnlineBackup'] = encode.transform(data['OnlineBackup'])



encode.fit(['No','No internet service','Yes'])

data['DeviceProtection'] = encode.transform(data['DeviceProtection'])



encode.fit(['No','No internet service','Yes'])

data['TechSupport'] = encode.transform(data['TechSupport'])



encode.fit(['No','No internet service','Yes'])

data['StreamingTV'] = encode.transform(data['StreamingTV'])



encode.fit(['No','No internet service','Yes'])

data['StreamingMovies'] = encode.transform(data['StreamingMovies'])



encode.fit(['Month-to-month','One year','Two year'])

data['Contract'] = encode.transform(data['Contract'])



encode.fit(['No','Yes'])

data['PaperlessBilling'] = encode.transform(data['PaperlessBilling'])



encode.fit(['Bank transfer (automatic)','Credit card (automatic)','Electronic check', 'Mailed check'])

data['PaymentMethod'] = encode.transform(data['PaymentMethod'])



encode.fit(['No','Yes'])

data['Churn'] = encode.transform(data['Churn'])
#Correlation matrix & Heatmap

pl.figure(figsize =(15,15))

corrmat = data.corr()

sns.heatmap(corrmat, annot=True, fmt='.1f', vmin=0, vmax=1, square=True);
#dropping uncorrelated columns with target column:Churn

data=data.drop(columns=['customerID','gender', 'PhoneService', 'MultipleLines', 'InternetService', 'StreamingTV', 'StreamingMovies'])

data.head(5)
#Labels and featureSet columns

columns = data.columns.tolist()

columns = [c for c in columns if c not in ['Churn']]

target = 'Churn'



X = data[columns]

y = data[target]
#Splitting data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)



print("Training FeatureSet:", X_train.shape)

print("Training Labels:", y_train.shape)

print("Testing FeatureSet:", X_test.shape)

print("Testing Labels:", y_test.shape)
#Initializing the model with some parameters.

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)

#Fitting the model to the data.

model.fit(X_train, y_train)

#Generating predictions for the test set.

predictions = model.predict(X_test)

#Computing the Model Accuracy

print("Random Forrest Accuracy:",round(metrics.accuracy_score(y_test, predictions),2)*100)

#Computing the error.

print("Mean Absoulte Error:", round(mean_absolute_error(predictions, y_test),2)*100)

#Computing classification Report

print("Classification Report:\n", classification_report(y_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y_test, predictions),

    index = [['actual', 'actual'], ['0','1']],

    columns = [['predicted', 'predicted'], ['0', '1']])

print(df)
#Initializing the model with some parameters.

model = SVC(gamma='auto')

#Fitting the model to the data.

model.fit(X_train, y_train)

#Generating predictions for the test set.

predictions = model.predict(X_test)

#Computing the Model Accuracy

print("SVM Accuracy:",round(metrics.accuracy_score(y_test, predictions),2)*100)

#Computing the error.

print("Mean Absoulte Error:", round(mean_absolute_error(predictions, y_test),2)*100)

#Computing classification Report

print("Classification Report:\n", classification_report(y_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y_test, predictions),

    index = [['actual', 'actual'], ['0','1']],

    columns = [['predicted', 'predicted'], ['0', '1']])

print(df)
#Initializing the model with some parameters.

model = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

#Fitting the model to the data.

model.fit(X_train, y_train)

#Generating predictions for the test set.

predictions = model.predict(X_test)

#Computing the Model Accuracy

print("SGD Accuracy:",round(metrics.accuracy_score(y_test, predictions),2)*100)

#Computing the error.

print("Mean Absoulte Error:", round(mean_absolute_error(predictions, y_test),2)*100)

#Computing classification Report

print("Classification Report:\n", classification_report(y_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y_test, predictions),

    index = [['actual', 'actual'], ['0','1']],

    columns = [['predicted', 'predicted'], ['0', '1']])

print(df)