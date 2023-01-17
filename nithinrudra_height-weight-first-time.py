import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_squared_error

from mpl_toolkits.mplot3d import Axes3D
!ls ../input/500-person-gender-height-weight-bodymassindex
df=pd.read_csv("../input/500-person-gender-height-weight-bodymassindex/500_Person_Gender_Height_Weight_Index.csv")

df.head()
X=df.iloc[:,1:-1].values

Y=df.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
reg=LinearRegression()

reg.fit(x_train,y_train)
print("-"*5,"Model for age-height","-"*5)

print("y =",reg.coef_[0],"x1+",reg.coef_[1],"x2+",reg.intercept_)
#Training Data Scatter plot

fig=plt.figure()

ax=Axes3D(fig)

ax.scatter(x_train[0:,0],x_train[:,1],y_train,c="red")

ax.set_xlabel("Height")

ax.set_ylabel("Weight")

ax.set_zlabel("Index")
y_pred=reg.predict(x_test)
r2score=r2_score(y_test,y_pred)

mse=mean_squared_error(y_test,y_pred)

rmse=np.sqrt(mse)
print("So we can say",round((r2score*100),4),"% is variability in our model from actual data")

print("Method used to ch")
print("In this notebook I have used Root Mean Squared error to find the accuracy so the value of RMSE is",rmse)
x1=[i for i in range (1,len(y_pred)+1)]

plt.scatter(x1,y_test-y_pred,c="green")

plt.plot(x1,y_pred*[0],c="red")
