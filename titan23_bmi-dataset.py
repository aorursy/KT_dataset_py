import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
df=pd.read_csv('../input/500_Person_Gender_Height_Weight_Index.csv')

print(df.sample(frac=0.1)) # this will print only 10% of total data ie: 10%  of df
df=pd.get_dummies(df)

print(df)

X=df.iloc[:,[0,1,3,4]].values

Y=df.iloc[:,2].values
X_nu=df[["Height","Weight","Index"]]



X_nu.corr()



X_nu.hist(bins=50)
plt.scatter(X_nu.Index,Y,color="g")

plt.grid()
plt.scatter(X_nu.Weight,Y,color="r")

plt.grid()
plt.scatter(X_nu.Height,Y,color="teal")

plt.grid()
X_train=X[:400]

X_test=X[400:]



Y_train=Y[:400]

Y_test=Y[400:]




from sklearn.linear_model import LinearRegression

teacher=LinearRegression()

learner=teacher.fit(X_train,Y_train)


Yp=learner.predict(X_test)

c=learner.intercept_

m=learner.coef_

print("c is {}  \n m is {}  \n Yp is {}".format(c,m,Yp))



xlist=list(X_train)

ylist=list(Y_train)

yplist=list(Yp)



mytable=pd.DataFrame({"input":xlist,"out":ylist})

print(mytable)

from sklearn.metrics import mean_squared_error,accuracy_score

Error=mean_squared_error(Yp,Y_test)

np.sqrt(Error)


import seaborn as sns

sns.barplot(x=Y_test,y=Yp,data=df)


y_pred_cls=np.zeros_like(Yp)

y_pred_cls[Yp>2.5]=1



y_test_cls=np.zeros_like(Yp)

y_test_cls[Y_test>2.5]=1
print(accuracy_score(y_test_cls,y_pred_cls))
