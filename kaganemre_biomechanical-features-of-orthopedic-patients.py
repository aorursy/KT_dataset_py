#We will work on the biomechanical-features-of-orthopedic-patients dataset.
#After examining our dataset and making various visualizations. We will try to do some predictions
#We will use linear and polynomial regressions

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #data visualization
import csv
from matplotlib import pyplot,pylab

#Let's start with the dataset review
df=pd.read_csv("/kaggle/input/column-2c-wekacsv/column_2C_weka.csv") #read dataset with pandas
df.head(n=10) #Let's look at the top 10 lines of our dataset.
df.tail(10)
df.columns  #We see terms on the anatomy of the pelvis in the columns
df.dtypes  #just one float dtype
df.shape
df.isnull().sum()  #We don't have NaN values.
df.describe() #Some statistical values
df.rename(columns={"class":"anatomy"},inplace=True) #more understandable 

df.columns
class_1=df.groupby(["anatomy"]) 

class_1["anatomy"].value_counts()["Normal"]
class_1["anatomy"].value_counts()["Abnormal"]
x=len(class_1.get_group("Normal"))
y=len(class_1.get_group("Abnormal"))
#We have 210 abnormal and 100 normal people in our dataset.

width=0.25

anatomy_types=["Normal","Abnormal"]
Counts=[x,y]

plt.style.use("fivethirtyeight")

plt.bar(anatomy_types,Counts,width=width,color=["k","r"])

plt.xlabel1=("Anatomic Types")
plt.ylabel1=("Counts")

plt.title=("Anatomic Type Counts")

plt.legend(loc=(0.8,0.9))
plt.grid()
plt.tight_layout()
plt.show()                                 
#Let's see the relation between pelvic_incidence and sacral_slope

x_1=df["pelvic_incidence"]
y_1=df["sacral_slope"]
plt.style.use("seaborn")

plt.scatter(x_1,y_1,edgecolor="black",linewidth=1,alpha=0.75)

#x axis=pelvic_incidence
#y axis=sacral_slope

plt.tight_layout()

plt.show()

#Our distribution seems to be suitable for linear regression. Let's make some predictions using linear regression

# Basic: y=b0+x*b1

# b0=constant,b1=coefficant,x=independent variable, y=dependent variable

# We will use sklearn

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
x_1.shape,y_1.shape
x_2=df["pelvic_incidence"].values.reshape(-1,1)
y_2=df["sacral_slope"].values.reshape(-1,1)

x_2.shape,y_2.shape
linear_reg.fit(x_2,y_2)
b0=linear_reg.intercept_
b1=linear_reg.coef_

b0,b1
y_head=linear_reg.predict(x_2)
plt.scatter(x_1,y_1,color="red")
plt.plot(x_2,y_head)



#x axis=pelvic_incidence
#y axis=sacral_slope

plt.suptitle("Relation Between Pelvic_İncidance and Sacral_Slope ")


plt.show()
linear_reg.predict([[45],[55],[23]]) #sacral_slope predictions by changing pelvic_incidances values
from sklearn.metrics import r2_score
r2_score(y_2,y_head) #Accuracy
#Let's try with polynomial regression
from sklearn.preprocessing import PolynomialFeatures
x_3=df["pelvic_incidence"].values.reshape(-1,1)
y_3=df["sacral_slope"].values.reshape(-1,1)
polynomial_reg=PolynomialFeatures(degree=2) 
x_polynomial=polynomial_reg.fit_transform(x_3)
x_polynomial
linear_reg2=LinearRegression()
linear_reg2.fit(x_polynomial,y_3)
linear_reg2.predict(polynomial_reg.fit_transform([[45],[55],[23]])) #sacral_slope predictions by changing pelvic_incidances values
 
y_head2=linear_reg2.predict(x_polynomial)
plt.plot(x_3,y_head2,color="green") 
plt.scatter(x_3,y_3)
r2_score(y_3,y_head2) #accuracy
