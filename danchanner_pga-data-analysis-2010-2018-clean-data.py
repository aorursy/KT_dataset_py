import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



#seaborn default

sns.set()

#Load the PGA data (2019 or historical)

f = '/kaggle/input/pga_historical_cleansed.csv'

data = pd.read_csv(f)
data.info()
pivot = pd.pivot_table(data = data, index = 'Player Name', columns = 'Variable', values = 'Value')

pivot.head()
#List of key stats under 'Variable'

AAD = 'Average Approach Shot Distance - (AVG)'

FedEx = "FedExCup Season Points - (POINTS)"

DriveD = "Driving Distance - (AVG.)"

GIR = 'Greens in Regulation Percentage - (%)'

PF10 = "Putting from - > 10' - (% MADE)"

P4 = "Putting from 4-8' - (% MADE)"

SS = "Sand Save Percentage - (%)"

UD = "Scrambling - (%)"

#Select driving distance values for each season

dt = data[['Season','Value']][data['Variable']==DriveD].groupby(['Season']).mean()

dt.plot(legend=False,figsize=(5,4),table=False)

plt.title('Average Driving Distance')

plt.show()
data2010 = data[data['Season']==2010]

data2018 = data[data['Season']==2018]



sns.distplot(data2010['Value'][data['Variable']==DriveD],bins=50, hist=False,label='2010')

sns.distplot(data2018['Value'][data['Variable']==DriveD],bins=50, hist=False,label='2018')

plt.title('Driving Distance')

plt.xlabel('Distance')

plt.show()

#Select approach shot distance values for each season

dt = data[['Season','Value']][data['Variable']==AAD].groupby(['Season']).mean()

dt.plot(legend=False,figsize=(5,4),table=False)

plt.title('Average Approach Shot Distance')

plt.show()
sns.distplot(data2010['Value'][data['Variable']==GIR],bins=50, hist=False,label='2010')

sns.distplot(data2018['Value'][data['Variable']==GIR],bins=50, hist=False,label='2018')

plt.title('GIR')

plt.xlabel('GIR %')

plt.show()
sns.distplot(data2010['Value'][data['Variable']==UD],bins=50, hist=False,label='2010')

sns.distplot(data2018['Value'][data['Variable']==UD],bins=50, hist=False,label='2018')



plt.title('Scrambling')

plt.xlabel('%')

plt.show()
sns.distplot(data2010['Value'][data['Variable']==PF10],bins=50, hist=False,label='2010')

sns.distplot(data2018['Value'][data['Variable']==PF10],bins=50, hist=False,label='2018')



plt.title('Putting from outside 10 feet')

plt.xlabel('%')

plt.show()
sns.distplot(data['Value'][data['Variable']==FedEx],bins=50, hist=False)



plt.title('FedEx Points Won')

plt.xlabel('Points')

plt.show()
#remove 0 values and re-set the 2010/18 data

data = data[data['Value']>0]

data2010 = data[data['Season']==2010]

data2018 = data[data['Season']==2018]
#Now convert the data into pivot table format

data2010p=data2010.pivot_table(index = ['Player Name'],columns='Variable',values='Value')

data2018p=data2018.pivot_table(index = ['Player Name'],columns='Variable',values='Value')

data2010p.head()
#2010 figures

data2010p.corr()
#2018 figures

data2018p.corr()
#Driving Distance

x = "Driving Distance - (AVG.)"

y = "FedExCup Season Points - (POINTS)"



sns.regplot(x,y,data2010p,label='2010')

sns.regplot(x,y,data2018p,label='2018')



plt.show()
#Greens in regulation

x = "Driving Distance - (AVG.)"

y = 'Greens in Regulation Percentage - (%)'



sns.regplot(x,y,data2010p,label='2010')

sns.regplot(x,y,data2018p,label='2018')



plt.show()