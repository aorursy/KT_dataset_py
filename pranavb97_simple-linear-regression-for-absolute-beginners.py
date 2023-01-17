import numpy as nm

import matplotlib.pyplot as mpl

import pandas as pd
data_set=pd.read_csv('../input/salary/Salary.csv')
print("Data size: ",data_set.shape)

data_set.info()
x=data_set.iloc[:,:-1].values

y=data_set.iloc[:,1].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
mpl.scatter(x_train,y_train,color="brown")

mpl.title("Salary Vs Experience(Training dataset)")

mpl.xlabel("Years of Experience")

mpl.ylabel("Salary")

mpl.show()
from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)

x_pred=reg.predict(x_train)
error = pd.DataFrame({"Actual": y_test,

                      "Predictions": y_pred,

                      "Difference": nm.abs(y_test-y_pred)})

print(error)
mpl.scatter(x_train,y_train,color="purple")

mpl.plot(x_train,x_pred,color="black")

mpl.title("Salary Vs Experience(Training dataset)")

mpl.xlabel("Years of Experience")

mpl.ylabel("Salary")

mpl.show()
mpl.scatter(x_test,y_test,color="red")

mpl.plot(x_train,x_pred,color="black")

mpl.title("Salary Vs Experience(Test dataset)")

mpl.xlabel("Years of Experience")

mpl.ylabel("Salary")

mpl.show()
pred_sal=reg.predict([[9.2]])

print("Predicted salary: ",pred_sal)
print("Training accuracy: ",(reg.score(x_train,y_train))*100)

print("Test accuracy: ",(reg.score(x_test,y_test))*100)