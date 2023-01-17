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
print('Country that has highest & lowest consumption of meat (kg/capita) by considering from 2000 to 2017\n')



dfKgCap = df.loc[(df['MEASURE'] == 'KG_CAP') & (df['TIME'] >= 2000) & (df['TIME'] <= 2017)] #as before missing data



print('Country that has the highest Consumption : ', dfKgCap.groupby(by = ['LOCATION']).Value.sum().idxmax())

print('Country that has the lowest Consumption : ', dfKgCap.groupby(by = ['LOCATION']).Value.sum().idxmin())



print('\nHigest Consumption of meat on : ', dfKgCap.groupby(by = ['TIME']).Value.sum().idxmax())

print('Lowest Consumption of meat on : ', dfKgCap.groupby(by = ['TIME']).Value.sum().idxmin())
print('Country that has highest consumption of different types of meat (kg/capita) by considering from 2000 to 2017\n')



dfKgCapBeef = dfKgCap.loc[(dfKgCap['SUBJECT'] == 'BEEF')]

print('Highest consumption of Beef : ',dfKgCapBeef.groupby(by = ['LOCATION']).Value.sum().idxmax())

#print('Country that has the lowest Consumption of Beef : ',dfKgCapBeef.groupby(by = ['LOCATION']).Value.sum().idxmin())



dfKgCapPoul = dfKgCap.loc[(dfKgCap['SUBJECT'] == 'POULTRY')]

print('Highest consumption of Poultry : ',dfKgCapPoul.groupby(by = ['LOCATION']).Value.sum().idxmax())

                           

dfKgCapSheep = dfKgCap.loc[(dfKgCap['SUBJECT'] == 'SHEEP')]

print('Highest consumption of Sheep : ',dfKgCapSheep.groupby(by = ['LOCATION']).Value.sum().idxmax())



dfKgCapPig = dfKgCap.loc[(dfKgCap['SUBJECT'] == 'PIG')]

print('Highest consumption of Pig : ',dfKgCapPig.groupby(by = ['LOCATION']).Value.sum().idxmax())
import matplotlib.pyplot as plt



dfThTon = df.loc[(df['MEASURE'] == 'THND_TONNE') & (df['TIME'] >= 2000)] #as before missing data



dfThTonFrom00To05 = dfThTon.loc[(dfThTon['TIME'] >= 2000) & (dfThTon['TIME'] <= 2005)]

dfThTonFrom09To14 = dfThTon.loc[(dfThTon['TIME'] >= 2009) & (dfThTon['TIME'] <= 2014)]

dfThTonFrom15To20 = dfThTon.loc[(dfThTon['TIME'] >= 2015) & (dfThTon['TIME'] <= 2020)]

dfThTonFrom21To26 = dfThTon.loc[(dfThTon['TIME'] >= 2021) & (dfThTon['TIME'] <= 2026)]



dfThTonFrom00To05 = dfThTonFrom00To05.groupby(by = ['SUBJECT']).Value.sum()

dfThTonFrom09To14 = dfThTonFrom09To14.groupby(by = ['SUBJECT']).Value.sum()

dfThTonFrom15To20 = dfThTonFrom15To20.groupby(by = ['SUBJECT']).Value.sum()

dfThTonFrom21To26 = dfThTonFrom21To26.groupby(by = ['SUBJECT']).Value.sum()





fig = plt.figure (figsize=(18,7))

fig.suptitle('Change of meat consumption over 27 years', size = 22)





ax1 = plt.subplot(1, 4, 1)

ax1.set_title('From 2000 to 2005')

dfThTonFrom00To05.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax2 = plt.subplot(1, 4, 2)

ax2.set_title('From 2009 to 2014')

dfThTonFrom09To14.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax3 = plt.subplot(1, 4, 3)

ax3.set_title('From 2015 to 2020')

dfThTonFrom15To20.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax4 = plt.subplot(1, 4, 4)

ax4.set_title('From 2021 to 2026')

dfThTonFrom21To26.plot.pie(autopct='%1.0f%%')

plt.ylabel("")
country = 'NZL' #CHANGE HERE FOR OTHER COUNTRY



dfThTon = df.loc[(df['MEASURE'] == 'THND_TONNE') & (df['TIME'] >= 2000) & (df['LOCATION'] == country) ] 



dfThTonFrom00To05 = dfThTon.loc[(dfThTon['TIME'] >= 2000) & (dfThTon['TIME'] <= 2005)]

dfThTonFrom09To14 = dfThTon.loc[(dfThTon['TIME'] >= 2009) & (dfThTon['TIME'] <= 2014)]

dfThTonFrom15To20 = dfThTon.loc[(dfThTon['TIME'] >= 2015) & (dfThTon['TIME'] <= 2020)]

dfThTonFrom21To26 = dfThTon.loc[(dfThTon['TIME'] >= 2021) & (dfThTon['TIME'] <= 2026)]



dfThTonFrom00To05 = dfThTonFrom00To05.groupby(by = ['SUBJECT']).Value.sum()

dfThTonFrom09To14 = dfThTonFrom09To14.groupby(by = ['SUBJECT']).Value.sum()

dfThTonFrom15To20 = dfThTonFrom15To20.groupby(by = ['SUBJECT']).Value.sum()

dfThTonFrom21To26 = dfThTonFrom21To26.groupby(by = ['SUBJECT']).Value.sum()





fig = plt.figure (figsize=(18,7))

fig.suptitle('Change of meat consumption over 27 years in {}'.format(country), size = 22)



ax1 = plt.subplot(1, 4, 1)

ax1.set_title('From 2000 to 2005')

dfThTonFrom00To05.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax2 = plt.subplot(1, 4, 2)

ax2.set_title('From 2009 to 2014')

dfThTonFrom09To14.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax3 = plt.subplot(1, 4, 3)

ax3.set_title('From 2015 to 2020')

dfThTonFrom15To20.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax4 = plt.subplot(1, 4, 4)

ax4.set_title('From 2021 to 2026')

dfThTonFrom21To26.plot.pie(autopct='%1.0f%%')

plt.ylabel("")