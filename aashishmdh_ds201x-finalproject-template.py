import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
score = pd.read_csv('../input/student_scores.csv')
score.shape
score.head()
score.describe()
score.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show() 
X = score.iloc[:, :-1].values  
y = score.iloc[:, 1].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
model = LinearRegression()  
model.fit(X_train, y_train) 
print(model.intercept_)  
print(model.coef_)  
# y_pred contains all the predicted values for the X_test data
y_pred = model.predict(X_test)  
# Creating a dataframe to show how the prediction and actual data looks like
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
dataset = pd.read_csv("../input/petrol_consumption.csv")
dataset.head()
dataset.describe()
dataset.plot(kind='box',layout=(2,2), sharex=False, sharey=False, figsize=(10,8))
plt.show()
dataset.hist(figsize=(12,10))
plt.show()
X = dataset[['Petrol_tax', 'Average_income', 'Paved_Highways',  
       'Population_Driver_licence(%)']]
y = dataset['Petrol_Consumption']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df
y_pred = regressor.predict(X_test) 
df_regressor = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_regressor)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
model_gnb = GaussianNB()  
model_gnb.fit(X_train, y_train)
y_pred = model_gnb.predict(X_test) 
df_gnb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_gnb)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
model_dtc = DecisionTreeClassifier() 
model_dtc.fit(X_train, y_train)
y_pred = model_dtc.predict(X_test) 
df_dtc = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_dtc)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
model_sgd = SGDClassifier()
model_sgd.fit(X_train, y_train)
y_pred = model_sgd.predict(X_test) 
df_sgd = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_sgd)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
y_pred = model_knn.predict(X_test) 
df_knn = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_knn)
model_svm = SVC(kernel="linear", C=0.025)
model_svm.fit(X_train, y_train)
y_pred = model_svm.predict(X_test) 
df_svm = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_svm)