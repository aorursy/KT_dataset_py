#Importing libraries

#Loading and Manipulating Data

import pandas as pd 

import numpy as np 



#Visualizations

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns 

#Loading Data

dext=pd.read_csv("../input/external_data.csv",)

dev=pd.read_csv("../input/evolution content consumption.csv",index_col=["date"],parse_dates=["date"])

dus=pd.read_csv("../input/users.csv",index_col=0)

dpeb=pd.read_csv("../input/pebmed_content.csv",parse_dates=["access_timestamp"])
#Basic info

dpeb.info()

#First 15 lines of the dataset

dpeb.head(5)

#how many countries are considered in this dataset?

dpeb.country.nunique()

#Countplot of the 5 countries with the most number of content

plt.figure(figsize=(12,4))

plt.title("Top 5 countries")

sns.countplot(x="country",data=dpeb,#selecting top 5 countries#

              order=dpeb.country.value_counts().iloc[:5].index)

#Considering brazilian entries

d_BR=dpeb.loc[dpeb.country=="BR"]



#Top 5 cities with most entries

plt.figure(figsize=(12,4))

plt.title("Top 5 cities")

sns.countplot(x="city", data=d_BR,order=d_BR.city.value_counts().iloc[:5].index)



#Countplot of content

plt.figure(figsize=(12,4))

plt.title("Contents")

sns.countplot(x="content", data=d_BR)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='x-large'  

)

d_BR.category.unique()

plt.figure(figsize=(12,4))

plt.title("Categories")

sns.countplot(x="category", data=d_BR)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='x-large'  

)

d_BR.category.unique()

plt.figure(figsize=(12,4))

plt.title("SubCategories")

sns.countplot(x="subcategory", data=d_BR)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='x-large'  

)
#Basic info

dev.info()

#First 5 lines

dev.head()
#We just want brazilian entries

ev_BR=dev.loc[dev.country=="BR"]

#Countplot of aboutcovid

plt.figure(figsize=(12,4))

plt.title("Covid related content")

sns.countplot(x="about covid?", data=ev_BR)
#Selecting number of access column

s=ev_BR["total access"]

#Suming all access by date to create a time series

sev_BR=s.groupby("date").sum()

sev_BR
#Ploting the time series

plt.figure(figsize=(12,4))

plt.title("Acessos")

sns.lineplot(data=sev_BR)
#This Time Series relates date and Number of user_ids that logged that day

s_BR=d_BR.groupby(d_BR["access_timestamp"].dt.date).count()["user_id"]

s_BR
#Plot of the Time Series

plt.figure(figsize=(12,4))

plt.title("Consultas COVID")

sns.lineplot(data=s_BR)
dext
#Library to handle datetime objects

import datetime

 # The format

format_str = '%d/%m/%Y' 

#Formating column "id_solution" to datetime format

dext.id_solution=dext.id_solution.map(lambda p: datetime.datetime.strptime(p[-10:], format_str))

#Time Series that relates date to Total Number of cases in that day

sext_BR=dext.groupby(dext["id_solution"]).sum()["casosNovos"]







#Even with a error message the series is well defined

sext_BR

#Ploting Series of new COVID cases

plt.figure(figsize=(12,4))

plt.title("Casos de Covid")

sns.lineplot(data=sext_BR)
#Creating a Dataframe to analyse the two series together.

dt =pd.concat([s_BR, sext_BR], axis=1, keys=['Consultas', 'Casos'])

dt

#Observe that this DataFrame contains nullvalues.
#Ploting both Series

import matplotlib.pyplot as plt

ax = dt.plot(figsize=(12, 4), fontsize=14)

plt.show()
plt.figure(figsize=(12,4))

plt.title("CasosxConsultas")

sns.scatterplot(x="Casos",y="Consultas",data=dt)
#Spearman correlation matrix

corr_p = dt[['Casos', 'Consultas']].corr(method='spearman')

print(corr_p)