
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv('../input/financial-distress/Financial Distress.csv')
df.head(5)
df.shape
df.info()
df.describe()
## Financial Distrees according to working Hour plot

df=df.rename(columns={'Financial Distress':'Financial_Distress'})
df.head(5)
plt.scatter(df['Time'],df['Financial_Distress'],color='red')
plt.xlabel('Time')
plt.ylabel('Financial Distress')
## Split the data
x=df[['Time']]
y=df['Financial_Distress']

xtrain,xtest,ytest,ytrain= train_test_split(x,y,train_size=.3)
lg=LinearRegression()
lg
#Train The Model
lg.fit(df[['Time']],df['Financial_Distress'])
#Accuracy 

lg.score(df[['Time']],df['Financial_Distress']) ##Tooo Toooo bad
##Predict
xinput=input("What is Your Working Time")
array=np.array(xinput)

array2=array.astype(np.float)

value=[[array2]]

result=lg.predict(value)

predict=np.array(value)

predict=predict.item()

print('Your Distress Percantage is', predict)


