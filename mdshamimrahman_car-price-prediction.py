
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')
df.head(5)
df.info

df.shape
#plot the fig 

plt.title('Car Price According to Engine Size ')
plt.scatter(df['enginesize'],df['price'],color='red')
plt.xlabel('Engine Size')
plt.ylabel('Price')
##Create Dimension

x=df[['enginesize']]
y=df['price']
#Split train and test dataset

xtest,xtrain,ytest,ytrain=train_test_split(x,y,test_size=.4)
#Create Object

lg=LinearRegression()
lg
lg.fit(xtrain,ytrain)
lg.score(xtest,ytest)
##Boom got nice Accuracy 76%
#Prediction
n=input("Please Tell the Ingine Size")
array=np.array(n)
array2=array.astype(np.float)

value=[[array2]]

car_price=lg.predict(value)

result=np.array(car_price)

result=result.item()

print('The Predicted price is', result)

## Now check if the model work properly or not

lg.coef_
lg.intercept_
##Y=mx+c
168.17010495*55-8159.774620506905

#Boom Model work Perfectly