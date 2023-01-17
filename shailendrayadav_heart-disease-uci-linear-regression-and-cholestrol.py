#importing the data in DataFrame df

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn import linear_model  #for linear regression

df=pd.read_csv("../input/heart.csv")

df

df.isnull().sum()
#Above result shows that there is no missing data.

#Taking first 10 rows for analysis

df.head(10)

# lets rename columns for better readability

df=df.rename(columns ={"chol":"Cholestrol","ca":"cardiac_arrest","thalach":"tahacemia_count"})



df
df
#grouping the dataframe with sex as male and female

grp=df.groupby("sex")

grp.max() #getting max conditions of male and female patients
#getting  splitted dataframes based on sex in groupbyDataframe object

male=grp.get_group(0) #0 means sex == 0,female

female=grp.get_group(1) #1 means sex == 1,male

male

# we will analyse further based on male and female patients
female
#plotting based on age groups of patients as male,female and both

df.sort_values(by="age",ascending=True)

male.sort_values(by="age",ascending=True)

female.sort_values(by="age",ascending=True)
df
grp.max()
#comparision of High cholestrol levels in male and female groups.

grp.max().plot(x="age",y="Cholestrol",kind="bar",rot=0)

grp.max().plot(x="age",y="cardiac_arrest",kind="bar",rot=0)

plt.show()
#Graph shows how men have cholestrol levels in their late years.

male.tail(10).plot(x="age",y="Cholestrol",kind="bar")

plt.show()
#Graph shows how women have cholestrol levels in their late years.

female.tail(10).plot(x="age",y="Cholestrol",kind="bar")

plt.show()
df.tail(10).plot(x="age",y="Cholestrol",kind="bar")

plt.show()
#let's see a scatter to analyse the Trend of Cholestrol levels in Patients.

#All patients

df.plot.scatter("age","Cholestrol",color="r")

plt.xlabel("Age")

plt.ylabel("Cholestrol")

#Male, sex = 1

male.plot.scatter("age","Cholestrol",color="blue")

#Female, sex= 0

female.plot.scatter("age","Cholestrol",color="g")
df.plot.scatter("age","Cholestrol",color="r")

plt.xlabel("Age")

plt.ylabel("Cholestrol")
#learning to plot data with other variable as well.

df.plot.scatter("age","tahacemia_count",color="y")

plt.xlabel("Age")

plt.ylabel("tahacemia_count")
#male and female patient

male.age.count(),female.age.count()
#pie chart  to visualise the patients a

#Pie chart learning

patients =[male.age.count(),female.age.count()]

patients

everyone =["Male","Female"]

plt.pie(patients,labels=everyone,radius=2,autopct="%0.2f%%",shadow=True,explode=[0.1,0.1],startangle =45)

plt.axis("equal")

plt.show()





male.head()
male.Cholestrol.max(),male.Cholestrol.max(),male.Cholestrol.max()
#patinets with heart disease disease.

patients =[male.cardiac_arrest.max(),female.cardiac_arrest.max()]

patients

heart_fail =["Male","Female"]

plt.pie(patients,labels=heart_fail,radius=2,autopct="%0.2f%%",shadow=True,explode=[0.1,0.1],startangle =45)

plt.axis("equal")

plt.show()
#patinets with heart disease disease.

patients =[male.cardiac_arrest.max(),male.Cholestrol.max(),male.trestbps.max()]

plt.figure(1)

plt.title("Male")

heart_fail =["heart attack","cholestrol","trestbps"]

plt.pie(patients,labels=heart_fail,radius=2,autopct="%0.2f%%",shadow=True,explode=[0.1,0.1,0.1],startangle =45)

plt.axis("equal")

plt.show()

patients =[female.cardiac_arrest.max(),female.Cholestrol.max(),female.trestbps.max()]

plt.figure(2)

plt.title("Female")

heart_fail =["heart attack","cholestrol","trestbps"]

plt.pie(patients,labels=heart_fail,radius=2,autopct="%0.2f%%",shadow=True,explode=[0.1,0.1,0.1],startangle =45)

plt.axis("equal")

plt.show()
##Average age of males and females

female.age.mean(),male.age.mean()

df["age"]
reg=linear_model.LinearRegression()
reg.fit(df[["age"]],df.Cholestrol)
#y= m*x + b

# m is Coefficient so below is m

reg.coef_
#b -> intercept

reg.intercept_
#lets predict the cholestrol level at the age of 80

result=reg.predict([[80]])

#y= m*x+b

cholestrol=1.2194412*80+179.9674706591244

result,cholestrol
#improving my result with more variables in consideration

mreg=linear_model.LinearRegression()

mreg.fit(df[["age","cardiac_arrest","trestbps"]],df.Cholestrol)
mreg.coef_
mreg.intercept_
Cholestrol_level=mreg.predict([[45,1,130]])

Cholestrol_level