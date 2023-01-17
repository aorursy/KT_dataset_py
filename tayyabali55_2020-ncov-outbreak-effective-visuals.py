# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data Visulizations

import seaborn as sns

import dateutil.parser





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
file = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")

file = file.drop(['Unnamed: 0'], axis = 1) 

file.info()
round(file.describe())
# first few record of the dataset

file.head(10)
#file['Date last updated']=pd.to_datetime(file['Date last updated']).apply(lambda x: x.date())

uniq_dates = list(file['Date last updated'].unique())

uniq_dates

confirmed=[]

suspected=[]

recovered=[]

deaths=[]

for x in uniq_dates:

    confirmed.append(file[file['Date last updated']==x].Confirmed.sum())

    suspected.append(file[file['Date last updated']==x].Suspected.sum())

    recovered.append(file[file['Date last updated']==x].Recovered.sum())

    deaths.append(file[file['Date last updated']==x].Deaths.sum())





line_plot= pd.DataFrame()

line_plot['Date']=uniq_dates

line_plot['Confirmed']=confirmed

line_plot['Suspected']=suspected

line_plot['Recovered']=recovered

line_plot['Deaths']=deaths

line_plot.head()


line_plot = line_plot.set_index('Date')

plt.figure(figsize=(20,15))

sns.lineplot(data=line_plot)

plt.xticks(rotation=15)

plt.title('Number of Coronavirus Cases Over Time', size=20)

plt.xlabel('Time', size=20)

plt.ylabel('Number of Cases', size=20)

plt.show()
plt.figure(figsize=(20,6))

sns.pairplot(file, size=3.5);
plt.figure(figsize=(20,6))

sns.pairplot(file,hue='Country' ,size=3.5);
plt.figure(figsize=(20,6))

sns.pairplot(file,hue='Province/State' ,size=3.5);
data= pd.DataFrame(file.groupby(['Country'])['Confirmed','Suspected','Recovered','Deaths'].agg('sum')).reset_index()

data.head(19)
data= pd.DataFrame(file.groupby(['Country'])['Confirmed','Suspected','Recovered','Deaths'].agg('sum')).reset_index()



data.sort_values(by=['Confirmed'], inplace=True,ascending=False)



plt.figure(figsize=(12,6))



#  title

plt.title("Number of Patients Confirmed Infected by Corona Virus, by Country")



# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country

sns.barplot(y=data['Country'],x=data['Confirmed'],orient='h')



# Add label for vertical axis

plt.ylabel("Number of Confirmed Patients")
data.sort_values(by=['Suspected'], inplace=True,ascending=False)



plt.figure(figsize=(12,6))



#  title

plt.title("Number of Patients Suspected Infected by Corona Virus, by Country")



# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country

sns.barplot(y=data['Country'],x=data['Suspected'],orient='h')



# Add label for vertical axis

plt.ylabel("Number of Suspected Patients")
data.sort_values(by=['Recovered'], inplace=True,ascending=False)



plt.figure(figsize=(12,6))



#  title

plt.title("Number of Patients Recovered from by Corona Virus, by Country")



# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country

sns.barplot(y=data['Country'],x=data['Recovered'],orient='h')



# Add label for vertical axis

plt.ylabel("Number of Recovered Patients")
data.sort_values(by=['Deaths'], inplace=True,ascending=False)



plt.figure(figsize=(12,6))



#  title

plt.title("Number of Patients Died by Corona Virus, by Country")



# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country

sns.barplot(y=data['Country'],x=data['Deaths'],orient='h')



# Add label for vertical axis

plt.ylabel("Number of Deaths")
china= file[file['Country'] == 'Mainland China']

china_data= pd.DataFrame(china.groupby(['Province/State'])['Confirmed','Suspected','Recovered','Deaths'].agg('sum')).reset_index()

china_data.head(35)
china_data.sort_values(by=['Confirmed'], inplace=True,ascending=False)



plt.figure(figsize=(25,10))



#  title

plt.title("Number of Patients Confirmed Infected by Corona Virus, by States")



# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country

sns.barplot(x=china_data['Province/State'],y=china_data['Confirmed'],orient='v')





# Add label for vertical axis

plt.ylabel("Number of Confirmed Patients")
china_data.sort_values(by=['Deaths'], inplace=True,ascending=False)



plt.figure(figsize=(25,10))



#  title

plt.title("Number of Patients Died by Corona Virus, by States")



# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country

sns.barplot(x=china_data['Province/State'],y=china_data['Deaths'],orient='v')





# Add label for vertical axis

plt.ylabel("Number of Deaths")