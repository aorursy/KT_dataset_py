#Import all the necessary libraries
import numpy as np # numerical python
import pandas as pd # data structures and tool
import matplotlib.pyplot as plt
import seaborn as sns

#Import dataset to be used
df = pd.read_csv("../input/usa-cers-dataset/USA_cars_datasets.csv")
df.head()
df.dtypes
#All the datatypes are perfectly correct


#Lets delete all the columns which will be of no use to us.Eg- Vin, lot, unnmaed:0
df= df.drop(["vin","lot","Unnamed: 0"], axis= 1)

#Statistical summary to know our data
df.describe()
#Now let us start by analysing the behavior of Price
sns.distplot(df["price"])

#Now, Lets look our dependent variable nature with a few independent variables
y= df["price"]
x= df["year"]
plt.scatter(x,y)
plt.show()
#price vs brand - Which are the most expensive Cars in US?
df_1= df.groupby(["brand"]).price.mean()
df_2= df_1.sort_values(ascending= True)
df_2.plot(kind= "barh")


#Now lets explore which car is the most popular car in US?
df_3= df.groupby("brand").brand.count()
df_4= df_3.sort_values(ascending= False)
df_4

#Relationship with Categorical Variables
#Color of Ford vs Price

