import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt 

%matplotlib inline 



import seaborn as sns 
#Import Data

raw_data = pd.read_csv('../input/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv', index_col=0, parse_dates=True)

#display(raw_data)

raw_data.describe()
# Histogram plot of features



plt.figure(figsize=(10,6)) 

# Add title 

plt.title("Hs") 

sns.distplot(a=raw_data['Hs'], kde=False) 



plt.figure(figsize=(10,6))

plt.title("Hmaz") 

sns.distplot(a=raw_data['Hmax'], kde=False) 



plt.figure(figsize=(10,6))

plt.title("Tz") 

sns.distplot(a=raw_data['Tz'], kde=False) 



plt.figure(figsize=(10,6))

plt.title("TP") 

sns.distplot(a=raw_data['Tp'], kde=False) 



plt.figure(figsize=(10,6))

plt.title("SST") 

sns.distplot(a=raw_data['SST'], kde=False) 



plt.figure(figsize=(10,6))

plt.title("Peak Direction") 

sns.distplot(a=raw_data['Peak Direction'], kde=False)  

# Remove Outliers

from scipy import stats

data= raw_data[(np.abs(stats.zscore(raw_data)) < 3).all(axis=1)]

data.describe()
#Rename Data

data = data.rename({'Hs': 'Average_Significant_WH', 'Hmax': 'Maximum_WH', 'Tz': 'ZeroUpcrossing_WP', 'Tp': 'PeakEnegery_WP', 'SST': 'Sea_Temp'}, axis=1)  # new method

data.index.names = ['Date']



#data.head()
#Different way to plot the distribution



plt.figure(figsize=(10,6)) 

# Add title 

plt.title("Sea_Temp") 

sns.kdeplot(data=data['Sea_Temp'], label="Sea_Temp", shade=True)





plt.figure(figsize=(10,6))

plt.title("Average_Significant_WH") 

sns.kdeplot(data=data['Average_Significant_WH'], label="Average_Significant_WH", shade=True)





plt.figure(figsize=(10,6))

plt.title("Maximum_WH") 

sns.kdeplot(data=data['Maximum_WH'], label="Maximum_WH", shade=True)





plt.figure(figsize=(10,6))

plt.title("ZeroUpcrossing_WP") 

sns.kdeplot(data=data['ZeroUpcrossing_WP'], label="ZeroUpcrossing_WP", shade=True)





plt.figure(figsize=(10,6))

plt.title("PeakEnegery_WP") 

sns.kdeplot(data=data['PeakEnegery_WP'], label="PeakEnegery_WP", shade=True)





plt.figure(figsize=(10,6))

plt.title("Peak Direction") 

sns.kdeplot(data=data['Peak Direction'], label="Peak Direction", shade=True)
sns.jointplot(x=data['Average_Significant_WH'], y=data['Maximum_WH'], kind="kde", height = 8 , scale = 1) 
# Experimenting with heat plots. 

# Interesting is the Sea_Temp seems to have the same relation to PeakEnergy as it does Peak Direction.



correlation_matrix = data.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,vmax=0.8,square = True, annot = True)

plt.show()
#Pairplot is a nice abstraction of most of the work done so far



sns.pairplot(data)
cols_plot = ['Average_Significant_WH', 'PeakEnegery_WP', 'Sea_Temp', 'Maximum_WH', 'ZeroUpcrossing_WP','Peak Direction',]

axes = data[cols_plot].plot(marker=',', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
# Set the width and height of the figure

plt.figure(figsize=(14,6))

# Add title

plt.title("Sea Temp")

sns.lineplot(x=data.index.month, y="Sea_Temp", data=data)
#Simple max search shows Jan meaning more analysis is needed

print(data.loc[data['Sea_Temp'].idxmax()])

print('\n',data.loc[data['Sea_Temp'].idxmin()])



#Create Summer and Winter Data sets and get averages



Jan = data[(data.index.month.isin([1]))]

Feb = data[(data.index.month.isin([2]))]

July = data[(data.index.month.isin([7]))]

Aug = data[(data.index.month.isin([8]))]



JanAv = Jan['Sea_Temp'].mean()

FebAv = Feb['Sea_Temp'].mean()

JulyAv = July['Sea_Temp'].mean()

AugAv = Aug['Sea_Temp'].mean()



print("\nJan:",JanAv,"\nFeb:",FebAv,"\nJuly:",JulyAv,"\nAug:",AugAv)

plt.figure(figsize=(14,6))

# Add title

plt.title("Av. WH")

sns.lineplot(x=data.index.month, y="Average_Significant_WH", data=data)



plt.figure(figsize=(14,6))

# Add title

plt.title("Max WH")

sns.lineplot(x=data.index.month, y="Maximum_WH", data=data)
#Create Summer and Winter Data sets and get averages



Summer = data[(data.index.month.isin([1,2,12]))]

Winter = data[(data.index.month.isin([6,7,8]))]



SummerMax = Summer['Maximum_WH'].mean()

SummerSig = Summer['Average_Significant_WH'].mean()



WinterMax = Winter['Maximum_WH'].mean()

WinterSig = Winter['Average_Significant_WH'].mean()



print("Summer:",SummerMax,SummerSig,"\nWinter:",WinterMax,WinterSig)
# from pandas.plotting import autocorrelation_plot

# autocorrelation_plot(data['PeakEnegery_WP'])



# sns.kdeplot(data['PeakEnegery_WP'], data['Peak Direction'])