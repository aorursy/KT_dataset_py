import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic plotting 

import seaborn as sns

plt.style.use('ggplot')

%matplotlib inline 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# read in the data from the provided csv file

df = pd.read_csv('../input/seaice.csv')



# drop the 'Source Data' column as it obscures more useful columns and doesn't tell us much

df.drop('Source Data', axis = 1, inplace = True)

df['Date'] = pd.to_datetime(df[['Year','Month','Day']])

df.index = df['Date'].values

NHem = df[df['hemisphere'] == 'north']

SHem = df[df['hemisphere'] == 'south']



# Obtain Monthly data

NHem_month, SHem_month = NHem.resample('1M').mean(), SHem.resample('1M').mean()

for monthly,Hem,Hem_name in zip([NHem_month,SHem_month],[NHem,SHem],['North / Arctic','South / Antarctic']):

    monthly['Month'] = monthly['Month'].apply(lambda x:int(x))

    monthly['Year'] = monthly['Year'].apply(lambda x:int(x))
for monthly,Hem_name in zip([NHem_month,SHem_month],['North / Arctic','South / Antarctic']):

    plt.figure(figsize=(10,4))

    month_short = monthly.pivot("Month", "Year", "Extent")

    # plt.xticks(rotation=60)

    plt.title(Hem_name+' | Monthly Mean of Sea Ice Extent [10**6 sq km]',size=10)

    sns.heatmap(month_short, annot=True, fmt="1.1f", linewidths=.5,cmap='jet_r',annot_kws={"size":4})
fig, ((ax1, ax2)) = plt.subplots(1, 2, sharex='row',sharey='row',figsize=(7,2))

sns.regplot(x="Year", y="Extent", data=NHem_month[NHem_month.Month==2],ax=ax1)

ax1.set_title('North/Arctic | February mean',size=10)

sns.regplot(x="Year", y="Extent", data=SHem_month[SHem_month.Month==9],ax=ax2)

ax2.set_title('South/Antarctic | September mean',size=10)



fig2, ((ax3, ax4)) = plt.subplots(1, 2, sharex='row',sharey='row',figsize=(7,2))

sns.regplot(x="Year", y="Extent", data=NHem_month[NHem_month.Month==9],ax=ax3)

ax3.set_title('North/Arctic | September mean',size=10)

sns.regplot(x="Year", y="Extent", data=SHem_month[SHem_month.Month==2],ax=ax4)

ax4.set_title('South/Antarctic | February mean',size=10)