import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/holidays_events.csv")
df.info()
for index, row in df.iterrows():

        

    date = row['date']

    

    elements = date.split("-")

    year = elements[0]



    df.loc[index,'year'] = year
df2 = df.groupby(['year', 'locale']).count().reset_index()
df3 = df2[['year','locale', 'type']]
df3
local = []

national = []

regional = []



for index, row in df3.iterrows():

    

    locale = row['locale']

    type_value = row['type']

    

    if locale == "Local":

        local.append(type_value)

    elif locale == "National":

        national.append(type_value)

    else:

        regional.append(type_value)
x1 = ['2012', '2013', '2014', '2015', '2016', '2017']

x = [1,2,3,4,5,6]

N = 6

ind = np.arange(N)
ax = plt.subplot()

plt.suptitle('Number of holidays per year', fontsize=14, fontweight='bold')



local = ax.bar(ind-0.2, local, width=0.2, color='b', align='center')

national = ax.bar(ind, national,width=0.2, color='g', align='center')

regional = ax.bar(ind+0.2, regional,width=0.2, color='r', align='center')



ax.set_xticklabels(('', '2012', '2013', '2014', '2015', '2016', '2017'))



ax.set_xlabel('Year')

ax.set_ylabel('Number of holidays')



ax.legend((local, national,  regional), ('Local', 'National', 'Regional'), loc = 'upper left', bbox_to_anchor=(1, 0.5))



plt.show()