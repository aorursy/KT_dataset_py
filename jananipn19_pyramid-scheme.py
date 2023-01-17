#importing pandas library
import pandas as pd
ds = pd.read_csv("../input/pyramid_scheme.csv")
print(ds)
#dropping the columns: Cost price and Sales_Commission
t = ['cost_price','sales_commission']
ds.drop(t,inplace = True, axis= 1)
print(ds)

x = ds.iloc[:,:3]
y = ds['profit']

#Splitting train_data and test_data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#developing linear model for train_data
from sklearn import linear_model
mdl = linear_model.LinearRegression()
mdl.fit(x_train,y_train) 
y_pred = mdl.predict(x_test)


#Checking the accuracy of the model
from sklearn.metrics import r2_score,mean_squared_error
scr = r2_score(y_pred,y_test)
mse = mean_squared_error(y_pred,y_test)
print("Mean_Squared_Error: ")
print(mse)
print("Variance: ")
print(scr)
import matplotlib.pyplot as plt
plt.plot(y_test, y_pred, color='blue', linewidth=3)