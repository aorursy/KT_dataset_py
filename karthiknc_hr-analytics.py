import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.preprocessing import LabelEncoder,StandardScaler
data=pd.read_csv("../input/hr-analytics-case-study/general_data.csv")
data.head()
data.info()
data.isnull().sum()
data.shape
data.describe().T
cols=data.columns
#checking for any special chars in data

for i in cols:

    if np.dtype(data[i]).name=="object":

        print(i,":",sum(data[i]=="?"))
sns.pairplot(data)
sns.countplot(x="Attrition",data=data)
sns.countplot(x="Attrition",hue="Gender",data=data)
#changing the categorical values to numberical values for modelling

le=LabelEncoder()
data['BusinessTravel'] = le.fit_transform(data['BusinessTravel'])

data['Department'] = le.fit_transform(data['Department'])

data['EducationField'] = le.fit_transform(data['EducationField'])

data['Gender'] = le.fit_transform(data['Gender'])

data['JobRole'] = le.fit_transform(data['JobRole'])

data['MaritalStatus'] = le.fit_transform(data['MaritalStatus'])

data['Over18'] = le.fit_transform(data['Over18'])

#Target Variable

data['Attrition']=le.fit_transform(data['Attrition'])
#dropping the unnecessary cols

data.drop(['EmployeeCount','EmployeeID','StandardHours'],axis=1, inplace = True)
plt.figure(figsize=(12,7))

sns.boxplot(y='Age',x='Attrition',data=data)
data.isnull().sum()
#filling the null values with 0

data.fillna(0,inplace=True)
data.isnull().sum()
#modelling 

x=data.drop('Attrition',axis=1)

y=data['Attrition']
#x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
logrmodel=LogisticRegression()
logrmodel.fit(x_train,y_train)
pred=logrmodel.predict(x_test)
print(classification_report(y_test,pred))
#print confusion matrix

confusion_matrix(y_test,pred)
#printing Accuracy Score

acc=metrics.accuracy_score(y_test,pred)

print(acc)