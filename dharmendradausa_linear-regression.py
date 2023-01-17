import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
toys = pd.read_excel('../input/toys-sales-data/toys.xlsx')
toys.head(5)
toys.columns=['Month','Sales','Price','adexp','proexp']
X = toys[['Price','adexp','proexp']] #independent variable or input
y = toys['Sales'] #dependent variable or target
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=0) #split the data on test size=0.2
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train) #training the model
reg.score(X_test,y_test)#score on the test data
reg.coef_
reg.intercept_
y_prediction = reg.predict(X_test)# predict the sales on the test data
y_prediction
from sklearn.metrics import r2_score      #accuracy of the model
r2_score(y_test,y_prediction)
data_for_prediction = pd.read_excel('../input/data-prediction/toys2.xlsx')#importing new dataset for getting predictions
toys_pred = reg.predict(data_for_prediction)#predictions based on the new dataset
toys_pred
a=toys['Sales']
df = pd.DataFrame(a)
df1=pd.DataFrame(toys_pred)
df1.columns=['Predicted_Sales']
compare = pd.concat([df,df1],axis=1)
compare#sales is the column of acutal Sales, the model accuracy is 0.6202129505262453 since less observations are there, 
#accuracy can be increased with more number of observations.
