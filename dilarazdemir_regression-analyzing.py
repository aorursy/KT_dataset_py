# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1 = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")

df2 = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
df1.info()
df1.head()
df1.isnull().sum()
df2.info()
df2.head()
df2.isnull().sum()
df1.corr()
sns.heatmap(df1.corr(), annot = True, fmt = ".2f")

plt.show()
pelvic_incidence = np.array(df1.loc[:,'pelvic_incidence']).reshape(-1,1)

sacral_slope = np.array(df1.loc[:,'sacral_slope']).reshape(-1,1)

plt.figure(figsize=(10,10))

plt.scatter(pelvic_incidence,sacral_slope)

plt.xlabel("Pelvic Incidence")

plt.ylabel("Sacral Slope")

plt.show()
# LinearRegression

from sklearn.linear_model import LinearRegression

reg = LinearRegression()



# Predict space

predict_space = np.linspace(min(pelvic_incidence), max(pelvic_incidence)).reshape(-1,1)

# Fit

reg.fit(pelvic_incidence,sacral_slope)

# Predict

predicted = reg.predict(predict_space)

# R^2 

print('R^2 score: ',reg.score(pelvic_incidence, sacral_slope))

# Plot regression line and scatter

plt.figure(figsize=(10,10))

plt.plot(predict_space, predicted, color='red', linewidth=2)

plt.scatter(pelvic_incidence,sacral_slope)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
x = (df1.iloc[:,[0,2]]).values # [pelvic_incidence,lumbar_lordosis_angle]

y = df1.sacral_slope.values.reshape(-1,1)


multiple_linear_regression = LinearRegression()

multiple_linear_regression.fit(x,y)





print("b0: ",multiple_linear_regression.intercept_)

print("b1,b2:",multiple_linear_regression.coef_)



multiple_linear_regression.predict(np.array([[63.0278175 , 39.60911701],[40.47523153, 39.60911701]]))

# first values of [[pelvic_incidence,lumbar_lordosis_angle],[sacral_slope,lumbar_lordosis_angle]]
x = np.array(df1.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(df1.loc[:,'sacral_slope']).reshape(-1,1)





from sklearn.linear_model import LinearRegression



lr = LinearRegression()



lr.fit(x,y)

# predict

y_head = lr.predict(x)



plt.plot(x,y_head,color="red",label = "linear")



# polynomial regression = y = b0 + b1*x + b2*x^2 + b3*x^3 + ... + bn*x^n



from sklearn.preprocessing import PolynomialFeatures



polynomial_regression = PolynomialFeatures(degree = 3)



x_polynomial = polynomial_regression.fit_transform(x)

# fit

linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y) 



# visualize



y_head2 = linear_regression2.predict(x_polynomial)





plt.plot(x,y_head2,color = "green", label = "poly")

plt.legend()

plt.show()
# normalization

x = (x - np.min(x))/(np.max(x)-np.min(x))

y = (y - np.min(y))/(np.max(y)-np.min(y))
x = np.array(df1.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(df1.loc[:,'sacral_slope']).reshape(-1,1)





from sklearn.tree import DecisionTreeRegressor # random state = 0

tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)





tree_reg.predict([[5.5]])



x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)



y_head = tree_reg.predict(x_)

#%% visualize

plt.figure(figsize = (10,10))

plt.scatter(x,y,color = "red")

plt.plot(x_,y_head,color="green")

plt.xlabel("Pelvic Incidence")

plt.ylabel("Sacral Slope")

plt.show()
x = np.array(df1.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(df1.loc[:,'sacral_slope'])





from sklearn.ensemble import RandomForestRegressor





rf = RandomForestRegressor(n_estimators = 40,random_state = 42)



rf.fit(x,y)





x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = rf.predict(x_)



#visualize

plt.figure(figsize = (10,10))

plt.scatter(x,y,color="blue")

plt.plot(x_,y_head,color = "red")

plt.xlabel("Pelvic Incidence")

plt.ylabel("Sacral Slope")

plt.show()
x = np.array(df1.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(df1.loc[:,'sacral_slope'])



from sklearn.ensemble import RandomForestRegressor





rf = RandomForestRegressor(n_estimators = 100,random_state = 42)



rf.fit(x,y)



y_head = rf.predict(x)





from sklearn.metrics import r2_score



print("r_score", r2_score(y,y_head))
x = np.array(df1.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(df1.loc[:,'sacral_slope'])

plt.figure(figsize=(10,10))

plt.scatter(pelvic_incidence,sacral_slope)

plt.xlabel("Pelvic Incidence")

plt.ylabel("Sacral Slope")



from sklearn.linear_model import LinearRegression



#linear regression model

linear_reg = LinearRegression()





linear_reg.fit(x,y)



y_head = linear_reg.predict(x)

plt.plot(x, y_head , color = "red")





#%% 



from sklearn.metrics import r2_score



print("r_square score: ", r2_score(y, y_head))