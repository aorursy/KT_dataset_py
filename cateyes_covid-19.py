# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data1=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

data5=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data7=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
data5
data5.describe()
plt.figure(figsize=(25,16))



plt.subplot(2,2,1)   



plt.plot(data5.ObservationDate,data5.Confirmed, color='b')



plt.xlabel('Confirmed')



plt.title('Confirmed of Covid-19')





plt.subplot(2,2,2)  



plt.plot(data5.ObservationDate,data5.Deaths,color="r")



plt.xlabel('Deaths')



plt.title('Deaths of Covid-19')





plt.subplot(2,2,3)  



plt.plot(data5.ObservationDate,data5.Recovered,color="g")



plt.xlabel('Recovered')



plt.title('Recovered of Covid-19')





plt.show()
data51=data5['Country/Region'].value_counts()



data51=pd.DataFrame(data51).reset_index()



data511=data51.head(15)



data511
plt.figure(figsize=(12,6))



plt.bar(data511['index'], data511['Country/Region'],color='orange')



plt.xticks(rotation=90)



plt.xlabel('Country/Region')



plt.ylabel('Count')



plt.title("Count of Covid-19")



plt.show()
data11=data1.describe()



data11
plt.figure(figsize=(14,8))



plt.subplot(2,2,1)



data1.boxplot(column='1/22/20')



plt.subplot(2,2,2)



data1.boxplot(column='9/12/20')



plt.show()
data7.head()
data71=data7.gender.value_counts()



data71=pd.DataFrame(data71).reset_index()



data71
male = data7[data7.gender == "male"]



female = data7[data7.gender == "female"]



plt.hist(male.gender,bins= 1,color='g')



plt.hist(female.gender,bins= 1,color='y')



plt.xlabel("Gender")



plt.ylabel("Count")



plt.title("Rate of Male and Female")



plt.show()
data72=data7.age.value_counts()

data72=pd.DataFrame(data72).reset_index()

data72
plt.figure(figsize=(12,8))



plt.bar(data72['index'], data72['age'],color='purple')



plt.xticks(rotation=90)



plt.xlabel('Age')



plt.title("Age and Covid-19")



plt.show()
data73=pd.melt(frame=data7,id_vars='gender',value_vars=['age'],ignore_index=True)

data74=data73.dropna(inplace=True)

data74=data73.dropna()

data74
data74.dtypes
male = data74[data74.gender == "male"]

female=data74[data74.gender=='female']

male , female
data75=male.value_counts()

data75=pd.DataFrame(data75).reset_index()

data76=female.value_counts()

data76=pd.DataFrame(data76).reset_index()

data75
plt.figure(figsize=(14,8))





plt.subplot(2,2,1)



plt.hist(data75.value,color='blue',hatch="o")



plt.xlabel("Age")



plt.ylabel("Confirmed")







plt.subplot(2,2,2)



plt.hist(data76.value,color='yellow',hatch='o')



plt.xlabel("Age")



plt.ylabel("Confirmed")





plt.show()