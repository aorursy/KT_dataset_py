# Importing all libraries required in this notebook

import pandas as pd

import numpy as np  

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Reading data from remote link

url = "http://bit.ly/w-data"

df = pd.read_csv(url)

df.head()
df.shape
df.isnull().sum()
df.Hours.mean()
df.describe()
# Plotting the distribution of scores

df.plot(x='Hours', y='Scores', style='gx')  

plt.title('Hours vs Percentage')  

plt.xlabel('Hours Studied')  

plt.ylabel('Percentage')  

plt.show() 
from pandas_profiling import ProfileReport
profile = ProfileReport(df, title="EDA Report")

profile
profile.to_file("Report.html")
X = df.iloc[:, :-1].values  #Independent variable

y = df.iloc[:, 1].values  #Target variable
X.shape
y.shape
from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                            test_size=0.2, random_state=0) 
print(X_test)

print(y_test)
from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()  

regressor.fit(X_train, y_train) 



print("Training complete.")
# Plotting the regression line

line = regressor.coef_*X+regressor.intercept_

print("Regression Equation =", regressor.coef_,"X","+",regressor.intercept_)

# Plotting for the test data

plt.scatter(X,y)

plt.plot(X,line)
# Testing data

pred = regressor.predict(X_test) # Predicting the scores

pred
# Comparing Actual vs Predicted

df2 = pd.DataFrame({'Actual': y_test, 'Predicted': pred})  

df2
# We can also test with our own data

hours = [[9.25]]

own_pred = regressor.predict(hours)

print("No of Hours = {}".format(hours))



if own_pred>=100:

    print("You are Perfect! You have scored 100%")

else:

    print("Predicted Percentage Score = {}".format(own_pred[0]))
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.metrics import explained_variance_score



mse = mean_absolute_error(y_test,pred) 

mae = mean_absolute_error(y_test,pred)

r2 = r2_score(y_test,pred)

evs = explained_variance_score(y_test,pred)



print(mse)

print(mae)

print(r2)

print(evs)
import statsmodels.api as sm

X_train2 = sm.add_constant(X_train)

model = sm.OLS(y_train,X_train2)

result = model.fit()

result.params



result.summary()
X_test2 = sm.add_constant(X_test)
pred2 = result.predict(X_test2) 
Residual = pred2 - y_test
sns.distplot(Residual)
df.plot(x='Hours', y='Scores', style='go')
import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(Residual)
df.plot(x='Hours', y='Scores', style='o')
#Saving model to disk

import pickle

model = pickle.dump(regressor,open('model.pkl','wb'))



# loading model



model = pickle.load(open('model.pkl','rb'))

print(model.predict([[9.25]]))