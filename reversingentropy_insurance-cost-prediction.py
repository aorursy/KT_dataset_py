import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

ins = pd.read_csv('/kaggle/input/insurance/insurance.csv')
print(ins.columns)
print(ins.head())
print(ins.shape)
ins.info()
print("Checking for number of Null or NaN values : ")
print(ins.isna().sum())
##No Null Values.
ins = pd.get_dummies(ins)
#One-Hot Encoding, categories converted.

ins.columns = ['age', 'bmi', 'children', 'charges', 'female', 'male',
       'non-smoker', 'smoker', 'northeast', 'northwest',
       'southeast', 'southwest']
ins.head()
corr = ins.corr().round(2)
corr.style.background_gradient(cmap='coolwarm')
 
sns.distplot(ins['charges'],bins=200)
#charges not normally distributed, skewed to the left
zoomin = sns.distplot(ins['charges'],bins=50)

zoomin.set_xlim(0, 20000)
plt.show()
#Most of the patients are charges ~1250 to 15000


sns.pairplot(ins[['age', 'bmi','non-smoker','smoker','charges']])


ins['non-smoker'].sum()

ins['smoker'].sum()

ins.describe()
#separate X and Y target dataset.
Y = ins['charges']
X = ins[['age','bmi','non-smoker','smoker']]

#split dataset into train and test sets with 80/20 split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ",Y_train.shape)
print("Shape of y_test",Y_test.shape)


#data normalisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#make linear regression model
ins_mod = linear_model.LinearRegression()
#train the model with X and Y training data sets
ins_mod.fit(X_train, Y_train)

#predict on Y for values in X test dataset
Y_pred = ins_mod.predict(X_test)

print('Coefficients: \n', ins_mod.coef_)
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
print('Variance score: %.2f' % r2_score(Y_test, Y_pred))
ins_mod.intercept_

plt.scatter(Y_test, Y_pred)
plt.show()
coefficients = ins_mod.coef_
bias = ins_mod.intercept_
print("Y =", bias, "+", coefficients[0], "* AGE +", coefficients[1], "* BMI +",coefficients[2], "* NON-SMOKER", coefficients[3], "* SMOKER")
#Make a list of features to predict the charges.
newList = [[26, 27, 1,0 ], [18, 30, 0, 1]]
n = 1
for x in newList:
  predict_charges = bias + (coefficients[0] * x[0]) + (coefficients[1] * x[1]) + (coefficients[2] * x[2])+(coefficients[3] * x[3])
  print("Predicted charges %d =  %d" % (n, predict_charges))
  n+=1
  print("\n")
