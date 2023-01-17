#Machine Learning 

import pandas as pd

df = pd.read_csv('../input/years-of-experience-and-salary-dataset/Salary_Data.csv')

df
df.shape
df.size
import matplotlib.pyplot as plt

plt.scatter(df['YearsExperience'], df['Salary'])

plt.show()
x = df.iloc[:,[0]].values

x
y = df.iloc[:,1].values

y
# sklearn -package  model_selection - library    train_test_split - module

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x , y ,test_size = 0.3, random_state = 0)
x_train.shape
x_test.shape
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)

y_predict
y_test
df1 = pd.DataFrame({'Actual':y_test, 'Predicted': y_predict,'Difference in %': ((y_predict-y_test)/(y_test)*100)})

df1
import matplotlib.pyplot as plt

plt.scatter(x_train,y_train, c='r')

plt.plot(x_train,model.predict(x_train))
import matplotlib.pyplot as plt

plt.scatter(x_test,y_test, c='r')

plt.plot(x_test,model.predict(x_test))
df2 = df1

#df2.figure(figsize=(20,10))

df2.plot(kind='bar')

plt.show()
#y =mx+c

c = model.intercept_

c
m = model.coef_

m
y_manual = m*3+c

y_manual
y_pred_unique = model.predict([[3]])

y_pred_unique