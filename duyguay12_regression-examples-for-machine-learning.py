# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the data

dframe = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
dframe.head()
# Check whether there are empty rows or not.

dframe.info()
dframe.describe()
# Correlation 

# We excluded "Serial No" with data.iloc[:,1:]) )



dframe.iloc[:,1:].corr()
# Correlation map

f, axx = plt.subplots(figsize=(10,10))

sns.heatmap(dframe.iloc[:,1:].corr(), linewidths=0.5, cmap="Blues", annot=True,fmt=".1f", ax=axx)

plt.show()
# Drop the duplicated values of the Chance of Admit.

df= dframe.drop_duplicates(subset=["Chance of Admit "])

df.info()
df= df.drop_duplicates(subset="CGPA")

df= df.drop_duplicates(subset="GRE Score")

df= df.drop_duplicates(subset="TOEFL Score")

df.info()
df.describe()
# Correlation 

# We excluded "Serial No" with data.iloc[:,1:]) )



df.iloc[:,1:].corr()
# Correlation map

f, axx = plt.subplots(figsize=(10,10))

sns.heatmap(df.iloc[:,1:].corr(), linewidths=0.5, cmap="Blues", annot=True,fmt=".2f", ax=axx)

plt.show()
df.columns
# Mean value of "Chance of Admit " is 0.677368.

# Output is on above; df.describe()



# Create a new column for High and Low.



df["Admit Level"] = ["Low" if each < 0.677368 else "High" for each in df["Chance of Admit "]]

df.head()
df.info()
# Vizualization

# CGPA, GRE Score and TOEFL Scores / Chance of Admit



import plotly.graph_objs as go



trace1 = go.Scatter(

                        x = df["Chance of Admit "],

                        y = df.CGPA,

                        mode = "markers",

                        name = "CGPA",

                        marker = dict(color="rgba(255, 100, 128, 0.8)"),

                        text = df["Admit Level"]

                        )

trace2 = go.Scatter(

                        x = df["Chance of Admit "],

                        y = df["GRE Score"],

                        mode = "markers",

                        name = "GRE Score",

                        marker = dict(color="rgba(80, 80, 80, 0.8)"),

                        text = df["Admit Level"]

                        )

trace3 = go.Scatter(

                        x = df["Chance of Admit "],

                        y = df["TOEFL Score"],

                        mode = "markers",

                        name = "TOEFL Score",

                        marker = dict(color="rgba(0, 128, 255, 0.8)"),

                        text = df["Admit Level"]

                        )

data = [trace1, trace2, trace3]

layout = dict(title="CGPA, GRE Score and TOEFL Scores v Chance of Admit",

             xaxis=dict(title="Chance of Admit", ticklen=5, zeroline=False),

             yaxis=dict(title="Values", ticklen=5, zeroline=False)

             )

fig = dict(data=data, layout=layout)

iplot(fig)
# Sklearn library

from sklearn.linear_model import LinearRegression



linear_reg = LinearRegression()
print(df.CGPA.values.shape)

print(df["Chance of Admit "].values.shape)



# Reshape

x = df.CGPA.values.reshape(-1,1)

y = df["Chance of Admit "].values.reshape(-1,1)

print("After resphape:\nX:", x.shape)

print("Y:", y.shape)
linear_reg.fit(x,y)
# Formula

# y = b0 + b1*x



b0 = linear_reg.intercept_

print("b0:", b0) # the spot where the linear line cuts the y-axis



b1 = linear_reg.coef_

print("b1:", b1) # slope



print("Linear Regression Formula:", "y = {0} + {1}*x".format(b0,b1))
x[0:5]
# CGPA-9.65 = Chance of Admit -0.92

df[df.CGPA == 9.65].loc[:,"Chance of Admit "]
linear_reg.predict([[9.8]])
print(min(x), max(x))
# CGPA values that will be predicted.



# Chance of Admit (predicted values)

y_head = linear_reg.predict(x)



plt.figure(figsize=(10,10))

plt.scatter(x,y, alpha=0.7)  # Real values (blue)

plt.plot(x,y_head, color="red") # Predicted values for numpay array (arr).

plt.show()
# Same shapes

print(y.shape, y_head.shape)
# R Square Library

from sklearn.metrics import r2_score

# y: Chance of Admit values

# y_head: predicted Chance of Admit values with LR

print("r_square score: ", r2_score(y, y_head))
# Sklearn library

# we already imported -- > from sklearn.linear_model import LinearRegression
# Define and reshape the variables



x1 = df.loc[:, ["CGPA", "GRE Score", "TOEFL Score"]]

y1 = df["Chance of Admit "].values.reshape(-1,1)
# Creat the model and fit the x&y values.

multiple_linear_regression = LinearRegression()

multiple_linear_regression.fit(x1,y1)
# Formula

# y = b0 + b1*x1 + b2*x2 + ... bn*xn

b0 = multiple_linear_regression.intercept_

b1,b2,b3 = zip(*multiple_linear_regression.coef_) 

print("b1:", b1, "b2:", b2, "b3:", b3)

print("b0:", multiple_linear_regression.intercept_)

print("b1, b2:", multiple_linear_regression.coef_)

print("Multiple Linear Regression Formula:", "y = {0} + {1}*x1 + {2}*x2 + {3}*x3".format(b0,b1,b2,b3))
print("CGPA:", min(x1["CGPA"]),"-", max(x1["CGPA"]))

print("GRE Score:", min(x1["GRE Score"]),"-", max(x1["GRE Score"]))

print("TOEFL Score:", min(x1["TOEFL Score"]), "-", max(x1["TOEFL Score"]))

plt.figure(figsize=(10,5))

plt.scatter(df["Chance of Admit "], df.CGPA, color="blue", label="CGPA")

plt.scatter(df["Chance of Admit "], df["GRE Score"], color="green", label="GRE Score")

plt.scatter(df["Chance of Admit "], df["TOEFL Score"], color="orange", label="TOEFL Score")

plt.legend()

plt.show()
# 1st: CGPA: 6.8 - 9.92

# 2nd: GRE Score: 290 - 340

# 3rd: TOEFL Score: 92 - 120

# Prediction: Chance of Admit



print("Values= np.array( [[6,280,90]])) Prediction =",

      multiple_linear_regression.predict(np.array( [[6,280,90]])))



print("Values= np.array( [[8,300,100]])) Prediction =",

      multiple_linear_regression.predict(np.array( [[8,300,100]])))



print("Values= np.array( [[10,350,130]])) Prediction =",

      multiple_linear_regression.predict(np.array( [[10,350,130]])))
x1.head()
y1_head = multiple_linear_regression.predict(x1)

y1_head[:5]
plt.figure(figsize=(10,20))



plt.scatter(y, x1.iloc[:,0], color="blue", alpha=0.7) # CGPA

plt.scatter(y1_head, x1.iloc[:,0], color="black", alpha=0.7)



plt.scatter(y, x1.iloc[:,1], color="green", alpha=0.7) # GRE Score

plt.scatter(y1_head, x1.iloc[:,1], color="black", alpha=0.7)



plt.scatter(y, x1.iloc[:,2],color="orange", alpha=0.7) # TOEFL  Score

plt.scatter(y1_head, x1.iloc[:,2], color="black", alpha=0.7)

plt.show()
# R Square Library



# Imported on previous sections

# from sklearn.metrics import r2_score



# y: Chance of Admit values

# y1_head: predicted Chance of Admit values with MLR

print("r_square score: ", r2_score(y,y1_head))
# Sklearn library 

from sklearn.preprocessing import PolynomialFeatures



# We have chose the second degree equation with (degree=2)

polynomial_regression = PolynomialFeatures(degree=2)

# y = b0 + b1*x + b2*x^2

x = df["TOEFL Score"].values.reshape(-1,1)

# y = df["Chance of Admit "].values.reshape(-1,1)

x_ploynominal = polynomial_regression.fit_transform(x)



linear_regression_poly = LinearRegression()

linear_regression_poly.fit(x_ploynominal, y)
# Linear Regression (LR) section: x = df.CGPA.values.reshape(-1,1)

# Linear Regression (LR) section: y = df["Chance of Admit "].values.reshape(-1,1)

print("x:\n", x[:5], "\ny:\n",y[:5])
# Predicted values

y_head_poly = linear_regression_poly.predict(x_ploynominal)

y_head_poly[:5]
plt.figure(figsize=(10,10))

plt.scatter(x, y, color="blue", alpha=0.7) # CGPA

plt.scatter(x, y_head_poly, label="poly (degree=2)", color="black") # predicted Chance of Admit

plt.xlabel("TOEFL Score")

plt.ylabel("chance")

plt.legend()

plt.show()
# y = b0 + b1*x + b2*x^2 + ..... b10*x^10

polynomial_regression7 = PolynomialFeatures(degree=7)



# x = df.CGPA.values.reshape(-1,1)

x_ploynominal_7 = polynomial_regression7.fit_transform(x)



linear_regression_poly_7 = LinearRegression()

linear_regression_poly_7.fit(x_ploynominal_7, y)



# Predicted values

y_head_poly_7 = linear_regression_poly_7.predict(x_ploynominal_7)
# y = b0 + b1*x + b2*x^2 + ..... b30*x^30

polynomial_regression30 = PolynomialFeatures(degree=30)



# x = df.CGPA.values.reshape(-1,1)

x_ploynominal_30 = polynomial_regression30.fit_transform(x)



linear_regression_poly_30 = LinearRegression()

linear_regression_poly_30.fit(x_ploynominal_30, y)



# Predicted values

y_head_poly_30 = linear_regression_poly_30.predict(x_ploynominal_30)
plt.figure(figsize=(12,12))

plt.scatter(x, y, color="blue", alpha=0.7) # TOEFL Score

plt.scatter(x, y_head_poly, label="poly (degree=2)", color="black", alpha="0.7") # predicted Chance of Admit

plt.scatter(x, y_head_poly_7, label="poly (degree=7)", color="red", alpha="0.7") # predicted Chance of Admit

plt.scatter(x, y_head_poly_30, label="poly (degree=30)", color="green", alpha="0.7") # predicted Chance of Admit

plt.xlabel("TOEFL Score")

plt.ylabel("chance")

plt.legend()

plt.show()
# R Square Library



# Imported on previous sections

# from sklearn.metrics import r2_score



print("r_square score for degree=2: ", r2_score(y, y_head_poly))

print("r_square score for degree=7: ", r2_score(y, y_head_poly_7))

print("r_square score for degree=30: ", r2_score(y, y_head_poly_30))

df.head()
# Decision Tree Library

from sklearn.tree import DecisionTreeRegressor



x = df["TOEFL Score"].values.reshape(-1,1)

y = df["Chance of Admit "].values.reshape(-1,1)



tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)
plt.scatter(df["TOEFL Score"] , df["Chance of Admit "],alpha=0.8)

plt.xlabel("TOEFL Score")

plt.ylabel("Chance of Admit")

plt.show()
y_head_dtr = tree_reg.predict(x)
plt.scatter(x, y, color="blue", alpha = 0.7)

plt.scatter(x, y_head_dtr, color="black", alpha = 0.4)

plt.xlabel("TOEFL Score")

plt.ylabel("Chance of Admit")

plt.show()
# Let's make a new array in the range of TOEFL Score values increased by 0.01

x001= np.arange(min(x), max(x), 0.01).reshape(-1,1) # (start, end, increase value)

y_head001dtr = tree_reg.predict(x001)
len(np.unique(y_head001dtr))



# 19 unique values for all values
plt.figure(figsize=(20,10))

plt.scatter(x,y, color="blue", s=100, label="real TOEFL Score") # real y (Chance of Admit) values

plt.scatter(x001,y_head001dtr, color="red", alpha = 0.7, label="predicted TOEFL Score") # to see the predicted values one by one

plt.plot(x001,y_head001dtr, color="black", alpha = 0.7)  # to see the average values for each leaf.

plt.legend()

plt.show()
# Same shapes, y and y_head_dtr

print(y.shape, y_head_dtr.shape, y_head001dtr.shape)
from sklearn.metrics import r2_score



print("r_score: ", r2_score(y,y_head_dtr))
from sklearn.model_selection import cross_val_score

#cross_val_score(tree_reg, boston.data, boston.target, cv=10)

print(tree_reg.score(x001, y_head001dtr))

print(tree_reg.score(x, y))
from sklearn.metrics import r2_score

print("r_score: ", r2_score(y,y_head_dtr))



from sklearn.model_selection import cross_val_score

print(tree_reg.score(x001, y_head001dtr))

print(tree_reg.score(x, y))
plt.scatter(x, y, color="blue", alpha = 0.7)

plt.xlabel("TOEFL Score")

plt.ylabel("Chance of Admit")

plt.show()
x = df["TOEFL Score"].values.reshape(-1,1)

y = df["Chance of Admit "].values.reshape(-1,1)



print(min(x), max(x))

print(min(y), max(y))
# Random Forest Regression Library



from sklearn.ensemble import RandomForestRegressor

 

random_forest_reg = RandomForestRegressor(n_estimators = 100, random_state = 42)

# n_estimators = 100 --> Tree number

# random_state = 42  --> Sample number

random_forest_reg.fit(x,y)



print(random_forest_reg.predict([[98]]))
# New prediction examples with (Start, End, Increase)

x001 = np.arange(min(x), max(x), 0.01).reshape(-1,1)

y_head001rf = random_forest_reg.predict(x001)



print(min(x001), max(x001))

print(min(y_head001rf), max(y_head001rf))
len(np.unique(y_head001rf))





# 46 unique values for all values
plt.figure(figsize=(20,10))

plt.scatter(x,y, color="blue", label="real TOEFL Score")

plt.scatter(x001,y_head001rf, color="red", label="predicted TOEFL Score")

plt.plot(x001,y_head001rf, color="black")

plt.legend()

plt.xlabel("TOEFL Score")

plt.ylabel("Chance of Admit")

plt.show()
from sklearn.model_selection import cross_val_score



print(tree_reg.score(x001, y_head001rf))

print(tree_reg.score(x, y))
from sklearn.metrics import r2_score



y_headrf = random_forest_reg.predict(x)

print("r_score: ", r2_score(y,y_headrf))



from sklearn.model_selection import cross_val_score

print(tree_reg.score(x001, y_head001rf))

print(tree_reg.score(x, y))