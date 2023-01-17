import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





#setting seaborn style plots

sns.set()



#importing data and showing sample

year = '2016'

df = pd.read_csv('../input/' + year + '.csv')



#our target

TARGET = df['Happiness Score']



#happiest country

print("The happiest country for the year of " + year + " is: " + df[df['Happiness Rank']==1]['Country'][0])

def ecdf(data):

    n = len(data)

    x = np.sort(data)

    y = np.arange(1,n+1)/n

    return x,y



#plotting ecdf and overlay percentiles

x_ecdf, y_ecdf = ecdf(TARGET)

plt.plot(x_ecdf,y_ecdf,marker='.',linestyle='none')

plt.margins(0.02)



percentiles = np.array([25,50,75])

ptiles = np.percentile(TARGET,percentiles)

plt.plot(ptiles,percentiles/100,marker='D',linestyle='none', color='black')

plt.show()
#find correlation coefficients

dicts = {}

keys = list(df.columns[6:-1]) 

for i in keys:

    ind = i

    x = df[ind]

    y = TARGET

    corr = np.corrcoef(x,y)[0,1]

    dicts[i] = corr

m = max(dicts.items(), key=lambda k: k[1])

print("Variable with maximum correlation to Happiness Score: " + m[0])

print("============================================================================================")

print(dicts)
percentages_sub = df[keys]

percentages = pd.DataFrame()

for i in percentages_sub.columns:

    percentages[i + " influence_pct"] = percentages_sub[i]/TARGET

    

print(percentages.describe())
#plot to show correlation strength

ind = df[m[0]]

plt.plot(ind,TARGET,marker='.',linestyle='none')

plt.show()
#What two combinations will produce a good correlation?

two_cols = df[keys]

new_cols = pd.DataFrame()

for k in two_cols.columns:

    for m in two_cols.columns:

        if k != m:

            new_cols[k+"_and_"+m] = two_cols[k] + two_cols[m]



dicts = {}

keys1 = list(new_cols.columns) 

for i in keys1:

    x = new_cols[i]

    y = TARGET

    corr = np.corrcoef(x,y)[0,1]

    dicts[i] = corr

m = max(dicts.items(), key=lambda k: k[1])

print(m)
ind = new_cols[m[0]]

plt.plot(ind,TARGET,marker='.',linestyle='none')

plt.show()
df.Region = df.Region.astype('category')

sns.swarmplot(x=df.Region, y=TARGET, data=df)

plt.xticks(rotation=90)

plt.show()

print(df[df['Region'] == 'Southeastern Asia'])