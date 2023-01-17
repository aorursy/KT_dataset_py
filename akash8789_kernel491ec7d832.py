import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
data.head()
X=data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y=data['Outcome']
X.head()
y.head()
X[X.isnull()].count()
y[y.isnull()].count()
df=data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age']]
l=[]

for i in y:

    if i==0:

        l.append('Non-Diabetic')

    else:

        l.append('Diabetic')
df['Outcome']=l
df.head()
X.plot(kind="box",figsize=(16,6),subplots=True)

plt.show()
sns.distplot(df.Age,label='Outcome')

plt.show()
X.BMI.describe()
X.BMI.quantile(0.95)
X.BloodPressure.describe()
X.BloodPressure.quantile(0.95)
X.Insulin.describe() #Outlier---------------
X[X.Insulin.values>200]
fig,ax=plt.subplots(figsize=(10,7))

sns.heatmap(data.corr(),annot=True,cmap="Reds",ax=ax,square=True)
sns.pairplot(df,hue='Outcome')
X.columns
from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler()
Scaler.fit(X.BloodPressure.values.reshape(-1,1))
Scaler.data_max_,Scaler.data_min_
X['BloodPressure']=Scaler.fit_transform(X.BloodPressure.values.reshape(-1,1))
Scaler.fit(X.Insulin.values.reshape(-1,1))
X['Insulin']=Scaler.fit_transform(X.Insulin.values.reshape(-1,1))
Scaler.fit(X.DiabetesPedigreeFunction.values.reshape(-1,1))
X['DiabetesPedigreeFunction']=Scaler.fit_transform(X.DiabetesPedigreeFunction.values.reshape(-1,1))
Scaler.fit(X.BMI.values.reshape(-1,1))
X['BMI']=Scaler.fit_transform(X.BMI.values.reshape(-1,1))
Scaler.fit(X.Glucose.values.reshape(-1,1))
X['Glucose']=Scaler.fit_transform(X.Glucose.values.reshape(-1,1))
X.head()
df_x=X[['Glucose','BloodPressure','Insulin','BMI']]
df_x.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df_x,y,train_size=0.75,random_state=100)
y_train.shape,x_train.shape,x_test.shape
x_train.reset_index(drop=True,inplace=True)

x_test.reset_index(drop=True,inplace=True)

y_train.reset_index(drop=True,inplace=True)

y_test.reset_index(drop=True,inplace=True)
from sklearn.neighbors import  KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
knn.score(x_test,y_test)