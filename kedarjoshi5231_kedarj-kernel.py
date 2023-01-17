import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

# Loading the CSV with pandas

data = pd.read_csv('../input/predict-the-churn-for-customer-dataset/Train File.csv')

data.head()
#Find blanks in data

data.info()
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors = 'coerce')

data.loc[data['TotalCharges'].isna()==True]
data[data['TotalCharges'].isna()==True] = 0

data['OnlineBackup'].unique()
data['gender'].replace(['Male','Female'],[0,1],inplace=True)

data['Partner'].replace(['Yes','No'],[1,0],inplace=True)

data['Dependents'].replace(['Yes','No'],[1,0],inplace=True)

data['PhoneService'].replace(['Yes','No'],[1,0],inplace=True)

data['MultipleLines'].replace(['No phone service','No', 'Yes'],[0,0,1],inplace=True)

data['InternetService'].replace(['No','DSL','Fiber optic'],[0,1,2],inplace=True)

data['OnlineSecurity'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

data['OnlineBackup'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

data['DeviceProtection'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

data['TechSupport'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

data['StreamingTV'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

data['StreamingMovies'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

data['Contract'].replace(['Month-to-month', 'One year', 'Two year'],[0,1,2],inplace=True)

data['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)

data['PaymentMethod'].replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)

data['Churn'].replace(['Yes','No'],[1,0],inplace=True)



data.pop('customerID')

data.info()
corr = data.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})

heat_map=plt.gcf()

heat_map.set_size_inches(20,15)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.show()
data.pop('TotalCharges')
from sklearn.model_selection import train_test_split 

train, test = train_test_split(data, test_size = 0.25)



train_y = train['Churn']

test_y = test['Churn']



train_x = train

train_x.pop('Churn')

test_x = test

test_x.pop('Churn')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report



logisticRegr = LogisticRegression()

logisticRegr.fit(X=train_x, y=train_y)



test_y_pred = logisticRegr.predict(test_x)

confusion_matrix = confusion_matrix(test_y, test_y_pred)

print('Intercept: ' + str(logisticRegr.intercept_))

print('Regression: ' + str(logisticRegr.coef_))

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr.score(test_x, test_y)))

print(classification_report(test_y, test_y_pred))



confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'Churn'), ('No churn', 'Churn'))

heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)

plt.ylabel('True label', fontsize = 14)

plt.xlabel('Predicted label', fontsize = 14)
data['Churn'].value_counts()
from sklearn.utils import resample



data_majority = data[data['Churn']==0]

data_minority = data[data['Churn']==1]



data_minority_upsampled = resample(data_minority,

replace=True,

n_samples=5174, #same number of samples as majority classe

random_state=1) #set the seed for random resampling

# Combine resampled results

data_upsampled = pd.concat([data_majority, data_minority_upsampled])



data_upsampled['Churn'].value_counts()