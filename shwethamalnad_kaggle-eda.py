import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_2017 = pd.read_csv("/kaggle/input/kaggle-survey/multipleChoiceResponses_2017.csv",encoding='ISO-8859-1')

data_2018 = pd.read_csv("/kaggle/input/kaggle-survey/multipleChoiceResponses_2018.csv")

data_2019 = pd.read_csv("/kaggle/input/kaggle-survey/multiple_choice_responses_2019.csv")
data_2019.shape,data_2018.shape,data_2017.shape
data_2017.drop(0,inplace=True)

data_2018.drop(0,inplace=True)

data_2019.drop(0,inplace=True)
data_year = data_2017.shape[0],data_2018.shape[0],data_2019.shape[0]

year=["2017","2018","2019"]
plt.plot(year,data_year)
data_2017["GenderSelect"].value_counts(),data_2018["Q1"].value_counts(),data_2019["Q2"].value_counts()
gender_per_2017=((data_2017["GenderSelect"].value_counts())/data_2017.shape[0])*100

gender_per_2018=((data_2018["Q1"].value_counts())/data_2018.shape[0])*100

gender_per_2019=((data_2019["Q2"].value_counts())/data_2019.shape[0])*100

gender_per_2017,gender_per_2018,gender_per_2019
fig =plt.figure(figsize=(20,10))



ax1 = plt.subplot2grid((1,3),(0,0))

plt.title("2017 Data",weight="bold",size=15)

sns.countplot(data_2017["GenderSelect"],order=data_2017["GenderSelect"].value_counts().index)

plt.xticks(weight='bold',rotation=45)



ax1 = plt.subplot2grid((1,3),(0,1))

plt.title("2018 Data",weight="bold",size=15)

sns.countplot(data_2018["Q1"],order=data_2018["Q1"].value_counts().index)

plt.xticks(weight='bold',rotation=45)



ax1 = plt.subplot2grid((1,3),(0,2))

plt.title("2019 Data",weight="bold",size=15)

sns.countplot(data_2019["Q2"],order=data_2019["Q2"].value_counts().index)

plt.xticks(weight='bold',rotation=45)

plt.show()



degree_per_2018=((data_2018["Q4"].value_counts())/data_2018.shape[0])*100

degree_per_2019=((data_2019["Q4"].value_counts())/data_2019.shape[0])*100

degree_per_2018,degree_per_2019
fig = plt.figure(figsize=(20,10))

ax1 = plt.subplot2grid((1,2),(0,0))

plt.title("Higest Level of Education",weight='bold',size=27)

sns.countplot(y=data_2019["Q4"],order=data_2019['Q4'].value_counts().index)

plt.xlabel("2019 Data",weight='bold',size=25)

plt.yticks(rotation=45,weight='bold',size=12)



ax1 = plt.subplot2grid((1,2),(0,1))

sns.countplot(y=data_2018["Q4"],order=data_2018['Q4'].value_counts().index)

plt.xlabel("2018 Data",weight="bold",size=25)

plt.yticks(rotation=45,weight='bold',size=12)

plt.subplots_adjust(top=0.85)

plt.show()
fig = plt.figure(figsize=(20,10))

ax1=plt.subplot2grid((1,4),(0,1))

plt.title("2019_data",weight='bold')

sns.countplot(y=data_2019["Q5"],order=data_2019["Q5"].value_counts().index)





ax1=plt.subplot2grid((1,4),(0,2))

plt.title("2018_data",weight='bold')

sns.countplot(y=data_2018["Q6"],order=data_2018["Q6"].value_counts().index)



ax1=plt.subplot2grid((1,4),(0,3))

plt.title("2017_data",weight='bold')

sns.countplot(y=data_2017["CurrentJobTitleSelect"],order=data_2017["CurrentJobTitleSelect"].value_counts().index)





plt.show()