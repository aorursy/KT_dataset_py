import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('../input/focusing-on-mobile-app-or-website/Ecommerce Customers')

df.head(5)
df.describe().round(2)
df=df.rename( columns={'Time on Website':'Time_on_Web'})

df
df=df.rename(columns={'Yearly Amount Spent':'Yearly_Amount_Spent'})
df
df
df=df.rename(columns={'Time on App':'Time_on_App'})
## Customer buy products by App

plt.scatter(df['Time_on_App'],df['Yearly_Amount_Spent'],marker='*',color='red')
plt.xlabel("Time_on_App")
plt.ylabel("Yearly Spent")
plt.title("Customers Spent by App")


x=df[["Time_on_App"]]
y=df['Yearly_Amount_Spent']
xtest,xtrain,ytest,ytrain=train_test_split(x,y,test_size=.3)
lgg=LinearRegression()
lgg.fit(xtrain,ytrain)
lgg.score(xtest,ytest)
##Predict

n=input('How much time you spent in app to buy products from E-Commerce Sites (monthly-hour)  ')
array=np.array(n)
array2=array.astype(np.float)

value=[[array2]]
result=lgg.predict(value).round(2)

yearly_spent=np.array(result)
yearly_spent=yearly_spent.item()

print(" Your Predicted Yearly Spent in Ecommerce Site",yearly_spent,'$')