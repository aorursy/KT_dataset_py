import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split 
from sklearn import metrics


df=pd.read_csv('../input/Salary_Data.csv')
df.head(2)


X=df.iloc[:,0].values
X=X.reshape(-1,1)


y=df.iloc[:,1].values
#y=np.swapaxes(y,0,1)
#y
y=y.reshape(-1,1)

##Splitting data into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_test
##fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(X_train,y_train)
#predict test results
y_predict=model.predict(X_test)
model.intercept_
model.coef_
plt.scatter(x=X_train,y=y_train,color='red')
plt.plot(X_train,model.predict(X_train))
plt.title('Salary vs Experience(Training set)')
plt.ylabel=('Salary')
plt.xlabel=('Experience')
plt.show()
plt.scatter(x=X_test,y=y_test,color='red')
plt.plot(X_train,model.predict(X_train))
plt.title('Salary vs Experience(Test set)')
plt.ylabel=('Salary')
plt.xlabel=('Experience')
plt.show()
plt.scatter(x=X_test,y=y_test,color='red')
plt.plot(X_test,model.predict(X_test))
plt.title('Salary vs Experience(predicted set)')
plt.ylabel=('Salary')
plt.xlabel=('Experience')
plt.show()
y_predict.mean()
y_test.mean()
y_train.mean()
metrics.r2_score(y_test,y_predict)
metrics.mean_absolute_error(y_test,y_predict)

