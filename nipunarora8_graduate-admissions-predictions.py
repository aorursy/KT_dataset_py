import numpy as np

import pandas as pd
df=pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
df.info()
df.columns
y=df['Chance of Admit ']

X=df.drop(['Serial No.','Chance of Admit '],axis=1)
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

sns.heatmap(X.corr(),cbar=True,annot=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X=scaler.fit_transform(X)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)

print(lr.score(X_test,y_test))
pred=lr.predict(X_test)
plt.scatter(y_test,pred)

plt.plot(y_test,y_test,color='blue')

plt.show()
plt.scatter(y_test,pred)

plt.plot(pred,pred,color='red')

plt.show()