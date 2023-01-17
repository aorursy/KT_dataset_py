import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
seed = 0
np.random.seed(seed)
from sklearn.datasets import load_boston
# Load the Boston Housing dataset from sklearn
boston = load_boston()
bos = pd.DataFrame(boston.data)
# give our dataframe the appropriate feature names
bos.columns = boston.feature_names
# Add the target variable to the dataframe
bos['Price'] = boston.target
# For student reference, the descriptions of the features in the Boston housing data set
# are listed below
print(boston.DESCR)
bos.head()
bos.columns
bos.describe()
# Select target (y) and features (X)
X = bos.iloc[:,:-1]
y = bos.iloc[:,-1]

# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=seed, shuffle=True)
bos.plot(x='CRIM', y='Price', style='o',c='black')
bos.plot(x='NOX', y='Price', style='o',c='coral')
plt.title('CRIM vs Price')
plt.xlabel('crime')
plt.ylabel('Price')
plt.show()
bos.plot(x='NOX', y='Price', style='o',c='coral')
plt.title('NOX vs Price')
plt.xlabel('NOX')
plt.ylabel('Price')
plt.show()
plt.scatter(bos.TAX, bos.Price, color='r')
plt.scatter(bos.AGE, bos.Price, color='g')
plt.xlabel('Tax')
plt.ylabel('AGE')
plt.show()
bos.columns

plt.boxplot(bos.CRIM,showmeans=True)
# correlation
corr = bos.corr().round(2)
corr.style.background_gradient(cmap='coolwarm')
from sklearn.linear_model import LinearRegression
slm = LinearRegression()
slm.fit(x_train, y_train)
coeff_df = pd.DataFrame(slm.coef_, X.columns, columns=['Coefficient'])
coeff_df

importance = slm.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
y_pred = slm.predict(x_test)
actual_vs_predict = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
actual_vs_predict.head()
def metrics(m,X,y):

    yhat = m.predict(X)

    SS_Residual = sum((y-yhat)**2)

    SS_Total = sum((y-np.mean(y))**2)

    r_squared = round(1 - (float(SS_Residual))/SS_Total,4)

    adj_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1),4)

    return r_squared,adj_r_squared
print('R squared Value:', metrics(slm,x_test,y_test)[0])
print('Adjusted R sqaure Value:', metrics(slm,x_test,y_test)[1])
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#Iteration 2
# remove TAX,INDUS,AGE,DIS based on correlation and coefficient value combinedly
columns=['CRIM', 'ZN', 'CHAS','DIS', 'NOX', 'RM', 'RAD','PTRATIO', 'B', 'LSTAT']
X = bos[columns]
y = bos.iloc[:,-1]

# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=seed, shuffle=True)

from sklearn.linear_model import LinearRegression
slm = LinearRegression()
slm.fit(x_train, y_train)
coeff_df = pd.DataFrame(slm.coef_, X.columns, columns=['Coefficient'])
coeff_df

y_pred = slm.predict(x_test)
actual_vs_predict = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
actual_vs_predict.head()
def metrics(m,X,y):

    yhat = m.predict(X)
    #print(yhat)
    SS_Residual = sum((y-yhat)**2)

    SS_Total = sum((y-np.mean(y))**2)

    r_squared = round(1 - (float(SS_Residual))/SS_Total,4)

    adj_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1),4)

    return r_squared,adj_r_squared

print('R squared Value:', metrics(slm,x_test,y_test)[0])
print('Adjusted R sqaure Value:', metrics(slm,x_test,y_test)[1])
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#iteration3
# adding AGE based on correlation and coefficient value combinedly
columns=['CRIM', 'CHAS','DIS', 'NOX', 'RM','B','INDUS', 'RAD','PTRATIO', 'LSTAT']
X = bos[columns]
y = bos.iloc[:,-1]

# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=seed, shuffle=True)

from sklearn.linear_model import LinearRegression
slm = LinearRegression()
slm.fit(x_train, y_train)
coeff_df = pd.DataFrame(slm.coef_, X.columns, columns=['Coefficient'])
coeff_df
y_pred = slm.predict(x_test)
actual_vs_predict = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
actual_vs_predict.head()

def metrics(m,X,y):

    yhat = m.predict(X)
    #print(yhat)
    SS_Residual = sum((y-yhat)**2)

    SS_Total = sum((y-np.mean(y))**2)

    r_squared = round(1 - (float(SS_Residual))/SS_Total,4)

    adj_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1),4)

    return r_squared,adj_r_squared

print('R squared Value:', metrics(slm,x_test,y_test)[0])
print('Adjusted R sqaure Value:', metrics(slm,x_test,y_test)[1])
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
bos.hist()
## using log transform to normalise data
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)
bos_new=pd.DataFrame(transformer.transform(bos))
bos_new.columns=bos.columns
X_new = bos_new.iloc[:,:-1]
# y_new = bos_new.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(X_new,y,test_size=0.3, random_state=seed, shuffle=True)


from sklearn.linear_model import LinearRegression
slm = LinearRegression()
slm.fit(x_train, y_train)
coeff_df = pd.DataFrame(slm.coef_, X_new.columns, columns=['Coefficient'])
coeff_df
y_pred = slm.predict(x_test)
actual_vs_predict = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

def metrics(m,X,y):

    yhat = m.predict(X)
    #print(yhat)
    SS_Residual = sum((y-yhat)**2)

    SS_Total = sum((y-np.mean(y))**2)

    r_squared = round(1 - (float(SS_Residual))/SS_Total,4)

    adj_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1),4)

    return r_squared,adj_r_squared
print('co efficients are:',coeff_df)
print('R squared Value:', metrics(slm,x_test,y_test)[0])
print('Adjusted R sqaure Value:', metrics(slm,x_test,y_test)[1])
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


actual_vs_predict.head()
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(x_train,
          y_train) 
# Evaluate the output

print(ridge.intercept_)
print(ridge.coef_)
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(x_train,
          y_train) 
# Evaluate the output

print(lasso.intercept_)
print(lasso.coef_)
bos_new.head()
#Iteration 1
plt.scatter(actual_vs_predict.Actual, actual_vs_predict.Predicted, color='r')
plt.xlabel('Actual')
plt.ylabel('Predict')
#Iteration 4 after Log transform
plt.scatter(actual_vs_predict.Actual, actual_vs_predict.Predicted, color='r')
plt.xlabel('Actual')
plt.ylabel('Predict')

