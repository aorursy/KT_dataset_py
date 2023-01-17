import matplotlib.pyplot as plt

%matplotlib notebook

import numpy as np

import pandas as pd

import statsmodels.formula.api as sm

import seaborn as sns
# Our model is in the data variable.

data = pd.read_csv("../input/Admission_Predict.csv")
data.head() # Informations about the content of the dataset.
data.info() # We can see that no "null" or "NaN" variable in our dataset.
# We can see the mathematical informations as mean,std etc..

data.describe()
# How many people did and did not a research.

# 1 = Yes , 0 = No

data["Research"].value_counts(dropna=False)
# Information about the "TOEFL Score" column. 

sns.countplot(data["TOEFL Score"] , order = data["TOEFL Score"].value_counts().index)

plt.show()
plt.figure(figsize=(13,6))

sns.heatmap(data.corr() , annot=True)



plt.tight_layout()

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,7,1)

sns.boxplot(x="Research",y="GRE Score", data=data,palette="PRGn")

plt.title("GRE Score")



plt.subplot(1,7,2)

sns.boxplot(x="Research" , y="TOEFL Score",data=data,palette="PRGn")

plt.title("TOEFL Score")



plt.subplot(1,7,3)

sns.boxplot(x="Research" , y="University Rating",data=data,palette="PRGn")

plt.title("University Rating")



plt.subplot(1,7,4)

sns.boxplot(x="Research" , y="SOP",data=data,palette="PRGn")

plt.title("SOP")



plt.subplot(1,7,5)

sns.boxplot(x="Research" , y="LOR ",data=data,palette="PRGn")

plt.title("LOR")



plt.subplot(1,7,6)

sns.boxplot(x="Research" , y="CGPA",data=data,palette="PRGn")

plt.title("CGPA")



plt.subplot(1,7,7)

sns.boxplot(x="Research" , y="Chance of Admit ",data=data,palette="PRGn")

plt.title("Chance Of Admit")



plt.tight_layout()

plt.show()
corr_m = data.corr()

corr_m["Research"].sort_values(ascending=False)
sns.pairplot(data , x_vars=["GRE Score" , "TOEFL Score" , "University Rating" , "SOP","LOR ","CGPA","Research"] , y_vars="Chance of Admit ")



plt.tight_layout()

plt.show()
corr_m = data.corr()

corr_m["Chance of Admit "].sort_values(ascending=False)
# data.drop(["Serial No."],axis=1,inplace=True)  # Drop the "serial no" column.

y = data["Chance of Admit "].values 

x = data.drop(["Chance of Admit "] , axis=1)
r = sm.OLS(endog = y , exog = x).fit() # OLS kullanarak "p value" gibi değerlere bakabiliriz.

r.summary()



# P>|t| is important. It should be <0.05 .If it is <5 , the column is effects our dataset logically.

# We should drop the "SOP" column and calculate again.
x.drop(["SOP"],axis=1,inplace=True)

r = sm.OLS(endog = y , exog = x).fit() # OLS kullanarak "p value" gibi değerlere bakabiliriz.

r.summary()

#We dropped the "OLS" column and as you can see , all the values close to 0. So there is no problem out there.
#Fitting Polynomial Regression to dataset

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 9) # We created a polynomial function has degree 9.

x_poly = poly_reg.fit_transform(x) # We convert out data(x) to the polynomial and keep it in the x_poly.

lin_reg = LinearRegression() # If we want to calculate polynomial , we should create Linear first than convert.

lin_reg.fit(x_poly,y) # We converted out linear regression using x_poly and y.

y_head = lin_reg.predict(x_poly) # We made a guess for all in the x_poly and keep it in the y_yead.



from sklearn.metrics import r2_score # Score

print(r2_score(y,y_head)) # Calculate r2 score.