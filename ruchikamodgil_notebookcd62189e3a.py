import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
df.head()

X=df.iloc[:,0]
Y=df.iloc[:,1]

plt.scatter(X,Y)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,Y,train_size=0.7,random_state=100)
train_x = train_x[:,np.newaxis]
test_x = test_x[:,np.newaxis]
def linearreg(X,Y):
    n=len(X)
    mx=np.mean(X)
    my=np.mean(Y)
    SS_xx=0
    SS_xy=0
    for i in range(n):
        SS_xx=SS_xx+(X[i]-mx)**2
        SS_xy=SS_xy+(X[i]-mx)*(Y[i]-my)
    m=SS_xy/SS_xx
    c=my-m*mx
    return m,c
m,c=linearreg(X,Y)
print(m,c)
y_pred=(m*X+c)
plt.scatter(X,Y)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.plot(X,y_pred,label="regression line",color="red")
plt.show()
from sklearn.metrics import mean_squared_error,r2_score
print(mean_squared_error(Y,y_pred))
print(r2_score(Y,y_pred))
z = [i for i in range (1,len(Y)+1,1)]
plt.plot(z,Y,color='r',linestyle='-')
plt.plot(z,y_pred,color='g',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Comparison')
plt.show()
from sklearn import linear_model
regr=linear_model.LinearRegression()
regr.fit(train_x,train_y)
print(regr.coef_)
print(regr.intercept_)
plt.scatter(X,Y)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.plot(X,X*regr.intercept_+regr.coef_,'-r')
plt.show()
print(mean_squared_error(Y,X*regr.intercept_+regr.coef_))
print(r2_score(Y,X*regr.intercept_+regr.coef_))
z = [i for i in range (1,len(Y)+1,1)]
plt.plot(z,Y,color='r',linestyle='-')
plt.plot(z,X*regr.intercept_+regr.coef_,color='g',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Comparison')
plt.show()
def gradientdescent(X,Y):
    m=0
    c=0
    n=float(len(X))
    L=.001   #Learning Rate
    epochs=1000
    for i in range(epochs):
        y_pred=m*X+c
        Dm=(-2/n)*(np.sum(X*(Y-y_pred)))
        Dc=(-2/n)*(np.sum(Y-y_pred))
        m=m-L*Dm
        c=c-L*Dc
        
    
        plt.plot(X,y_pred,label="regression line")
        
    return m,c
plt.scatter(X,Y)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
gradientdescent(X,Y)
plt.show()
m,c=gradientdescent(X,Y)
print(m,c)
y_pred=(m*X+c)
plt.scatter(X,Y)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.plot(X,y_pred,label="regression line",color="red")
plt.show()
c = [i for i in range (1,len(Y)+1,1)]
plt.plot(c,Y,color='r',linestyle='-')
plt.plot(c,y_pred,color='g',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Comparison')
plt.show()
print(mean_squared_error(Y,y_pred))
print(r2_score(Y,y_pred))
