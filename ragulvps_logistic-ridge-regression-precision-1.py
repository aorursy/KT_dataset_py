import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import RidgeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.metrics import recall_score,precision_score,f1_score,classification_report



print(" Import Complete")

data =pd.read_csv("../input/logistic-regression-heart-disease-prediction/framingham_heart_disease.csv")

data.head()
data.describe()
data.isnull().sum().sort_values(ascending=False)
data.shape
sns.heatmap(data.isnull())
sns.violinplot(x=data['diabetes'],y=data['glucose'],hue=data['male'])

plt.xlabel('Diabetes')

plt.ylabel('Glucose')

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(x=data['age'],y=data['glucose'],hue=data['diabetes'])
new = data[['diabetes','glucose']].groupby(by=['diabetes']).mean()

new.reset_index(inplace=True) 

new['glucose'][0]

data['glucose'][data['diabetes']==1]=[data['glucose'][data['diabetes']==1].fillna(new['glucose'][1])]

data['glucose'][data['diabetes']==0]=[data['glucose'][data['diabetes']==0].fillna(new['glucose'][0])]



#for i in data['diabetes'].unique())]]
data['glucose'].isnull().sum()
data_cleaned=data.dropna(axis=0)

print('Shaoe cleaned data:',data_cleaned.shape)

data_cleaned.head()
num_col=['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke',

         'prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']

x=data_cleaned[num_col]

y=data_cleaned['TenYearCHD']



x.shape ,y.shape
scaler = StandardScaler()

x_scaled=scaler.fit_transform(x)

x_scaled
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,random_state = 26,test_size=.15)



print('Shape of X_train:',x_train.shape)

print('Shape of y_train:',y_train.shape)

print('Shape of X_test:',x_test.shape)

print('Shape of y_test:',y_test.shape)
logreg = LogisticRegression().fit(x_train,y_train)

y_pred=logreg.predict(x_test)

#y_pred
from sklearn.metrics import recall_score,precision_score,f1_score,classification_report

print('Accuracy Score :',accuracy_score(y_test,y_pred))

print('F1 Score:',f1_score(y_test,y_pred))

print('Recall Score:',recall_score(y_test,y_pred))

print('Precision Score',precision_score(y_test,y_pred))

print('Confusion Matrix:',confusion_matrix(y_test,y_pred))

#print('Classification Report:',classification_report(y_test,y_pred))
RC = RidgeClassifier().fit(x_train,y_train)

y_pred_RC = RC.predict(x_test)

#y_pred_RC
from sklearn.metrics import precision_score,f1_score,classification_report

print('Accuracy Score By lasso:',accuracy_score(y_test,y_pred_RC))

print('F1 Score:',f1_score(y_test,y_pred_RC))

print('Recall Score:',recall_score(y_test,y_pred_RC))

print('Precision Score',precision_score(y_test,y_pred_RC))

#print('Classification Report:',classification_report(y_test,y_pred_RC ))
print('Confusion Matrix:', confusion_matrix(y_test,y_pred_RC))