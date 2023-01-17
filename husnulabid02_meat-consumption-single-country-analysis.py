# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
rawData = pd.read_csv('/kaggle/input/meatconsumption/meat_consumption_worldwide.csv', sep = ",")

df = rawData



rawData.head()
df.info()
print("Types of meat : ", df['SUBJECT'].unique())

print("\nCountries: \t",df['LOCATION'].unique())
df = df.loc[df['LOCATION'] == 'NZL']  #CHANGE HERE TO WATCH OTHER COUNTRY



#Selecting only row with kg per capita

dfKgCap = df.loc[df['MEASURE'] == 'KG_CAP']

dfKgCap.drop(['LOCATION', 'MEASURE'], axis=1, inplace=True)



#Pivoting

dfKgCapPivot = dfKgCap.pivot(index='TIME', columns='SUBJECT', values='Value')



dfKgCapPivot.head()
ax = dfKgCapPivot.plot(figsize=(10,5))

ax.set_ylabel('Meat Consumption : KG_CAP')

ax.legend(bbox_to_anchor=(1, 1))

ax.set_title('In New Zeland different types of meat consumption kg/capita')
dfKgCapFrom90To17 = dfKgCap.loc[dfKgCap['TIME'] <= 2017]

dfKgCapFrom90To17 = dfKgCapFrom90To17.groupby(by = ['TIME']).sum()

ax2 = dfKgCapFrom90To17.plot(figsize=(10,5))

ax2.set_ylabel('Meat Consumption : KG_CAP')

ax2.set_title('In New Zeland overall total meat consumption kg/capita by year (1990-2017) [28 years]')


dfKgCapFrom91To17 = dfKgCap.loc[(dfKgCap['TIME'] >= 1991) & (dfKgCap['TIME'] <= 2017)]

dfKgCapAvg = dfKgCapFrom91To17.groupby(by = ['SUBJECT']).Value.mean()



dfKgCapAvg
dfKgCapFrom91To17 = dfKgCap.loc[(dfKgCap['TIME'] >= 1991) & (dfKgCap['TIME'] <= 2017)]

dfKgCapFrom91To17 = dfKgCapFrom91To17.groupby(by = ['SUBJECT']).Value.sum()

ax3 = dfKgCapFrom91To17.plot.bar(figsize=(10,5))

ax3.set_ylabel('Meat Consumption : KG_CAP')

ax3.set_xlabel('Meat Type')

ax3.set_title('In New Zeland different types of meat consumption kg/capita (1991-2017) [27 years]')
dfThTon = df.loc[df['MEASURE'] == 'THND_TONNE']

dfThTon.drop(['LOCATION', 'MEASURE'], axis=1, inplace=True)



#Pivoting

dfThTonPivot = dfThTon.pivot(index='TIME', columns='SUBJECT', values='Value')



dfThTonPivot.head()
ax = dfThTonPivot.plot(figsize=(10,5))

ax.set_ylabel('Meat Consumption : Thousand Tonne')

ax.legend(bbox_to_anchor=(1, 1))

ax.set_title('In New Zeland different types of meat consumption in Thousand Tonnes')
dfThTonAll = dfThTon.groupby(by = ['TIME']).sum()

ax2 = dfThTonAll.plot(figsize=(10,5))

ax2.set_ylabel('Meat Consumption : Thousand Tonne')

ax2.set_title('In New Zeland overall meat consumption in Thousand Tonnes by year (1990-2026) [36 years]')
dfThTonFrom91To17 = dfThTon.loc[(dfThTon['TIME'] >= 1991) & (dfThTon['TIME'] <= 2017)]

dfThTonAvg = dfThTonFrom91To17.groupby(by = ['SUBJECT']).Value.mean() #per year



dfThTonAvg
dfThTonFrom91To17 = dfThTon.loc[(dfThTon['TIME'] >= 1991) & (dfThTon['TIME'] <= 2017)]

dfThTonTotal = dfThTonFrom91To17.groupby(by = ['SUBJECT']).Value.sum()

ax3 = dfThTonTotal.plot.bar(figsize=(10,5))

ax3.set_ylabel('Meat Consumption : Thousand tonnes')

ax3.set_xlabel('Meat Type')

ax3.set_title('In New Zeland total different types of meat consumption in Thousand tonnes (1991-2017) [27 years]')
dfThTonTotal = dfThTon.groupby(by = ['SUBJECT']).Value.sum()

ax4 = dfThTonTotal.plot.pie(title = 'Portions of different types of meat consumption in New Zeland')

ax4.set_ylabel('')
import matplotlib.pyplot as plt



dfThTonFrom91To95 = dfThTon.loc[(dfThTon['TIME'] >= 1991) & (dfThTon['TIME'] <= 1995)]

dfThTonFrom06To10 = dfThTon.loc[(dfThTon['TIME'] >= 2006) & (dfThTon['TIME'] <= 2010)]

dfThTonFrom11To15 = dfThTon.loc[(dfThTon['TIME'] >= 2011) & (dfThTon['TIME'] <= 2015)]

dfThTonFrom21To25 = dfThTon.loc[(dfThTon['TIME'] >= 2021) & (dfThTon['TIME'] <= 2025)]



dfThTonFrom91To95 = dfThTonFrom91To95.groupby(by = ['SUBJECT']).Value.sum()

dfThTonFrom06To10 = dfThTonFrom06To10.groupby(by = ['SUBJECT']).Value.sum()

dfThTonFrom11To15 = dfThTonFrom11To15.groupby(by = ['SUBJECT']).Value.sum()

dfThTonFrom21To25 = dfThTonFrom21To25.groupby(by = ['SUBJECT']).Value.sum()





fig = plt.figure (figsize=(18,7))

fig.suptitle('Change of meat consumption over 35 years (New Zeland)', size = 22)





ax5 = plt.subplot(1, 4, 1)

ax5.set_title('From 1991 to 1995')

dfThTonFrom91To95.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax6 = plt.subplot(1, 4, 2)

ax6.set_title('From 2006 to 2010')

dfThTonFrom06To10.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax7 = plt.subplot(1, 4, 3)

ax7.set_title('From 2011 to 2015')

dfThTonFrom11To15.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax8 = plt.subplot(1, 4, 4)

ax8.set_title('From 2021 to 2025')

dfThTonFrom21To25.plot.pie(autopct='%1.0f%%')

plt.ylabel("")
