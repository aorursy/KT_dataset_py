import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv("/kaggle/input/life-expectancy-who/Life Expectancy Data.csv")

df.head()
df.shape
df.Country.unique()
df.Country.value_counts()
df.info() #checking data types
df.isnull().sum() #checking null values in our dataset
df.Alcohol.isnull().sum()
df.Alcohol.value_counts() 
#from the above data, we can conclude most countries have an alcohol consumption around 0.01

#so we will fill the nan columns with 0.01
df.Alcohol.fillna(0.01,inplace=True)
df.Alcohol.isnull().sum()#in the alcohol column we have removed the nan values
df["Hepatitis B"].value_counts()
df["Hepatitis B"].fillna(99.0,inplace=True)
a=df.Population.mean()
df["Population"].fillna(a,inplace=True)
b=df.GDP.median()
df.GDP.fillna(b,inplace=True)
df.dropna(inplace=True)

#dropping the remaining na values
df.isnull().sum()
df.head(1)
# Developing vs Developed counts

#plt.subplots(figsize=(12,6))

df['Status'].value_counts().plot.bar(width=0.9,color="red",alpha=0.75)
df.columns
#year vs Life Expectancy

plt.subplots(figsize=(8,4))

sns.lineplot("Year","Life expectancy ",data=df,marker="o")

plt.title('Life Expectancy by Year')

plt.show()

# year vs alcohol_consumption

plt.subplots(figsize=(8,4))

sns.set_palette("rocket")

sns.lineplot("Year","Alcohol",data=df,marker="o")

plt.title('Alcohol Consumption by Year')

plt.show()
# year vs GDP_growth

plt.subplots(figsize=(8,4))

sns.set_palette("summer")

sns.lineplot("Year","GDP",data=df,marker="o")

plt.title('GDP growth with Year')

plt.show()
# year vs population_growth

plt.subplots(figsize=(8,4))

sns.set_palette("Greys")

sns.lineplot("Year","Population",data=df,marker="o")

plt.title('Population growth with Year')

plt.show()
#country

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

le.fit(df.Country.drop_duplicates())

df.Country=le.transform(df.Country)
#status

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

le.fit(df.Status.drop_duplicates())

df.Status=le.transform(df.Status)
df.info()
#dividing our data

y=df["Life expectancy "]

x=df.drop("Life expectancy ",axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33)
from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(X_train,y_train)
reg.score(X_train,y_train)
#our_model_score


#applying ridge regression

from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

ls=Ridge()

params={"alpha":[0.001,0.002,1,10,20,20,40,50,100]}

lreg=GridSearchCV(ls,params,cv=3)

lreg.fit(X_train,y_train)



y_train_pred = lreg.predict(X_train)

y_test_pred = lreg.predict(X_test)



print(lreg.score(X_test,y_test))
#playing with grid_search_cv params

print("Best model score:",lreg.best_score_)

print("Best value of alpha:",lreg.best_params_)