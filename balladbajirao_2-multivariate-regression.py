import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

print("All packages installed successfully!")
url = 'https://raw.githubusercontent.com/MainakRepositor/Datasets-/master/House%20rent.csv'

df = pd.read_csv(url,error_bad_lines=False)

df.head(10)
df.shape
print("Are there null values ?",df.isnull().values.any())
df.isnull().sum()
median = df.Bedrooms.median()

median
df.Bedrooms = df.Bedrooms.fillna(int(median))

print("Dataframe after replacing the null values :\n")

df
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

reg = LinearRegression()
reg.fit(df[['Area','Bedrooms','Age']],df.Rent)
reg.coef_
reg.intercept_
reg.predict([[920,3,10]])
y = 34.66745591*920+2440.61024394*3+-354.75983519*10+-24001.729279468844

y
r = 100 - (y-reg.predict([[920,3,10]]))

print("Accuracy of model = ",int(r),"%")
x = df.iloc[:,0].values.reshape(-1,1)

x1 = df.iloc[:,1].values.reshape(-1,1)

x2 = df.iloc[:,2].values.reshape(-1,1)

y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/4,random_state=0)

x1_train,x1_test,y_train,y_test = train_test_split(x1,y,test_size=1/4,random_state=0)

x2_train,x2_test,y_train,y_test = train_test_split(x2,y,test_size=1/4,random_state=0)
reg.fit(x_train,y_train)

reg.fit(x1_train,y_train)

reg.fit(x2_train,y_train)
y_pred = reg.predict(x_test)

y_pred = reg.predict(x1_test)

y_pred = reg.predict(x2_test)
plt.figure(figsize=(25,20))

plt.subplot(3,2,1)

plt.scatter(x_train,y_train,c='k')

plt.plot(x_train,reg.predict(x_train),color='blue')

plt.xlabel('Area of house',size=18)

plt.ylabel('Rent of house',size=18)

plt.title('House Rent Prediction (Training set) \n',size=24)



plt.subplot(3,2,2)

plt.scatter(x_test,y_test,c='k')

plt.plot(x_train,reg.predict(x_train),color='blue')

plt.xlabel('Area of house',size=18)

plt.ylabel('Rent of house',size=18)

plt.title('House Rent Prediction (Test set) \n',size=24)



plt.subplot(3,2,3)

plt.scatter(x1_train,y_train,c='k')

plt.plot(x1_train,reg.predict(x1_train),color='green')

plt.xlabel('Bedrooms',size=18)

plt.ylabel('Rent of house',size=18)



plt.subplot(3,2,4)

plt.scatter(x1_test,y_test,c='k')

plt.plot(x1_train,reg.predict(x1_train),color='green')

plt.xlabel('Bedrooms',size=18)

plt.ylabel('Rent of house',size=18)



plt.subplot(3,2,5)

plt.scatter(x2_train,y_train,c='k')

plt.plot(x2_train,reg.predict(x2_train),color='red')

plt.xlabel('Age of house',size=18)

plt.ylabel('Rent of house',size=18)



plt.subplot(3,2,6)

plt.scatter(x2_test,y_test,c='k')

plt.plot(x2_train,reg.predict(x2_train),color='red')

plt.xlabel('Age of house',size=18)

plt.ylabel('Rent of house',size=18)



plt.show()