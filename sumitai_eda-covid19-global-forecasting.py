# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import seaborn as sns 

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_csv = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

#

train_csv.head()
# combine Country region and Province state: and avoide NAN values 

train_csv['combine state'] = train_csv['Country/Region'].fillna('') + str(': ') +train_csv['Province/State'].fillna('')

train_csv.info()
train_csv.describe()
from pandas_profiling import ProfileReport

train_profile = ProfileReport(train_csv, title='Pandas Profiling Report', html={'style':{'full_width':True}})

train_profile
# CREATE SCATTER PLOT 

sns.set()

sns.pairplot(train_csv)
# we dont need the whole dataframe to analyze

train_csv = train_csv.set_index('Date')

col = ['combine state',

       'ConfirmedCases', 'Fatalities']

df = train_csv[col]

'''df = df.groupby('Country/Region')

#df = df.sort_values(by = 'Fatalities', ascending = False )

df'''

df.head()
# extract the unique names from the dataframe 

country = df['combine state'].unique()

for i in range(len(country)):

    df1 = df[df['combine state'].str.match(country[i])]

    if len(df1)>1:

        if max(df1['Fatalities'].values) > 100 :

            plt.figure(figsize = [20,5])

            title = str(country[i])

            plt.title(title, fontsize=12)

            #plt.subplot(2,1,1)

            #plt.plot(df1.index, df1['Fatalities'], '-o')

            sns.barplot(x=df1.index,y=df1['Fatalities'])

            sns.pointplot(x=df1.index,y=df1['Fatalities'],color='Black')

            plt.tight_layout()

            title = str(country[i])



            plt.xlabel('Time', fontsize=12)

            plt.ylabel('Number of People', fontsize=12)

            sns.barplot(x=df1.index,y=df1['ConfirmedCases'])

            sns.pointplot(x=df1.index,y=df1['ConfirmedCases'],color='Black')

            #plt.subplot(2,2,1)

            #plt.plot(df1.index, df1['ConfirmedCases'], '-*')



            plt.legend(fontsize=12)



            plt.xticks(rotation = 50)

            plt.show()
Country = []

Fatalities = []

affected = []




for i in range(len(country)):

    df1 = df[df['combine state'].str.match(country[i])]

    if len(df1)>1:

        if max(df1['Fatalities'].values) > 0:

            Country.append(country[i])

            Fatalities.append(max(df1['Fatalities'].values))

            affected.append(max(df1['ConfirmedCases'].values))

            #print( 'Name of the country: ', country[i], ', Total number of deaths: ', max(df1['Fatalities'].values), ', Total affected people: ', max(df1['ConfirmedCases'].values)  )

final = pd.DataFrame((np.array(Country), np.array(Fatalities), np.array(affected)))

final = pd.DataFrame.transpose(final)

columns = ['country/province', 'Fatalities', 'Number of Confirmed cases']

final.columns = columns

final
plt.figure(figsize = [30,10])

final = final.sort_values(by='Fatalities', ascending=False)

final10 = final.iloc[:int(len(final)*0.15), :]

#plt.bar(final10['country/province'],final10['Fatalities'])

sns.barplot(x=final10['country/province'],y=final10['Fatalities'])

sns.pointplot(x=final10['country/province'],y=final10['Fatalities'],color='Black')

plt.xticks(rotation = 45)

plt.title("Top 15% country with maximum number of faltalities reported", fontsize=25)

plt.xlabel('Country', fontsize=20)

plt.ylabel('Number of People', fontsize=20)



plt.figure(figsize = [30,10])



final = final.sort_values(by='Number of Confirmed cases', ascending=False)

final1 = final.iloc[:int(len(final)*0.15), :]

#plt.bar(final1['country/province'],final1['Number of Confirmed cases'] )

sns.barplot(x=final1['country/province'],y=final1['Number of Confirmed cases'])

sns.pointplot(x=final1['country/province'],y=final1['Number of Confirmed cases'],color='Black')

plt.title("Top 15% country with maximum number of Confirmed cases reported", fontsize=25)

plt.xlabel('Country', fontsize=20)

plt.ylabel('Number of People', fontsize=20)

plt.xticks(rotation = 45)

plt.show()
