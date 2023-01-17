import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('../input/Admission_Predict.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values
#Removing Serial
X = X[:, 1:]
#Split the Dataset to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#Fitting LinearRegression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predicting with Test Set
y_pred = regressor.predict(X_test)
print(y_pred.reshape(-1,1))
#Building The Optimal Model With BackWard Elimination, Eliminating Columns With P-Value > 0.05
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((400,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 2, 3, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 2, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Predicting with the optimal Model
X_trainOPT, X_testOPT, y_trainOPT, y_testOPT = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
regressorOPT = LinearRegression()
regressorOPT.fit(X_trainOPT, y_trainOPT)
#Final Predictions
y_predOPT = regressorOPT.predict(X_testOPT)
print(y_predOPT.reshape(-1,1))
#Plotting Each Independet Variable with the Dependent One
#GRE
X_GRE = X_testOPT[:, 1]
X_GRE = X_GRE.reshape(-1,1)
regressorGRE = LinearRegression()
regressorGRE.fit(X_GRE, y_testOPT)


plt.scatter(X_GRE, y_testOPT, color = 'red')
plt.plot(X_GRE, regressorGRE.predict(X_GRE), color = 'blue')
plt.title('GRE vs Chance of Admission')
plt.xlabel('GRE')
plt.ylabel('Chance of Admission')
plt.show()
#TOEFL SCORE
X_TOEFL = X_testOPT[:, 2]
X_TOEFL = X_TOEFL.reshape(-1,1)
regressorTOEFL = LinearRegression()
regressorTOEFL.fit(X_TOEFL, y_testOPT)


plt.scatter(X_TOEFL, y_testOPT, color = 'red')
plt.plot(X_TOEFL, regressorTOEFL.predict(X_TOEFL), color = 'blue')
plt.title('TOEFL Score vs Chance of Admission')
plt.xlabel('TOEFL Score')
plt.ylabel('Chance of Admission')
plt.show()
#LOR
X_LOR = X_testOPT[:, 3]
X_LOR = X_LOR.reshape(-1,1)
regressorLOR = LinearRegression()
regressorLOR.fit(X_LOR, y_testOPT)


plt.scatter(X_LOR, y_testOPT, color = 'red')
plt.plot(X_LOR, regressorLOR.predict(X_LOR), color = 'blue')
plt.title('LOR vs Chance of Admission')
plt.xlabel('LOR')
plt.ylabel('Chance of Admission')
plt.show()
#CGPA
X_CGPA = X_testOPT[:, 4]
X_CGPA = X_CGPA.reshape(-1,1)
regressorCGPA = LinearRegression()
regressorCGPA.fit(X_CGPA, y_testOPT)


plt.scatter(X_CGPA, y_testOPT, color = 'red')
plt.plot(X_CGPA, regressorCGPA.predict(X_CGPA), color = 'blue')
plt.title('CGPA vs Chance of Admission')
plt.xlabel('CGPA')
plt.ylabel('Chance of Admission')
plt.show()
#Research as RS
X_RS = X_testOPT[:, 5]
X_RS = X_RS.reshape(-1,1)
regressorRS = LinearRegression()
regressorRS.fit(X_RS, y_testOPT)


plt.scatter(X_RS, y_testOPT, color = 'red')
plt.plot(X_RS, regressorRS.predict(X_RS), color = 'blue')
plt.title('RS vs Chance of Admission')
plt.xlabel('RS')
plt.ylabel('Chance of Admission')
plt.show()