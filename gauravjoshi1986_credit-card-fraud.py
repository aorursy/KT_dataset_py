import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
data = pd.read_csv("../input/creditcard.csv")

data.head()
class_freq = data['Class'].value_counts()



print(class_freq)
# Class data is highly imbalanced

sns.countplot(x='Class', data=data, palette='Set3')
data.describe()
# check for missing data in dataset

data.isnull().sum()
X_data = data.iloc[:,1:29]



# correlation matrix for margin features

corr = X_data.corr()

#corr = data.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 10))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap='viridis', vmax=.3,

            square=True, xticklabels=5, yticklabels=5,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)



#sns.heatmap(corr, vmax=1, square=True, annot=True, cmap='viridis')

plt.title('Correlation between different fearures')
# select on fraud observetions

fraud_points = data.loc[data['Class'] == 1]
# Create bins

step = 3600

bin_range = np.arange(0, 172801, step)



out, bins  = pd.cut(fraud_points['Time'], bins=bin_range, include_lowest=True, right=False, retbins=True)

#out, bins  = pd.cut(data['Time'], bins=bin_range, include_lowest=True, right=False, retbins=True)



#modify the plot to include just the lower closed interval of the range for aesthetic purpose

out.cat.categories = ((bins[:-1]/3600)+1).astype(int)

out.value_counts(sort=False).plot(kind='bar', title= 'Fraud per Hr')
# Create new dataframe with frequence count for Genuine and fraud transactions

out_all, bins  = pd.cut(data['Time'], bins=bin_range, include_lowest=True, right=False, retbins=True)

out_all.cat.categories = ((bins[:-1]/3600)+1).astype(int)



# convert seriese to dataframe and add class attributes

out_df = out_all.to_frame(name=None)

out_df['Class'] = data['Class']



# count class and hr wise frequency

out_grp = out_df.groupby(['Time', 'Class'])['Time'].count().unstack('Class').fillna(0).astype(int)
fig = plt.figure() # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes

ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.



width = 0.4



out_grp.ix[:,0].plot(kind='bar', color='green', ax=ax, width=width, position=1)

out_grp.ix[:,1].plot(kind='bar', color='red', ax=ax2, width=width, position=0)



ax.set_ylabel('Genuine')

ax2.set_ylabel('Fraud')



plt.show()
out_grp.ix[:,0:2].plot(kind='bar',grid=True,subplots=True,sharex=True); 

plt.show()
# percent of fraud and genuine transactions within class per hour

out_grp['col_perc_0'] = 100*out_grp.ix[:,0]/out_grp.ix[:,0].sum()



out_grp['col_perc_1'] = 100*out_grp.ix[:,1]/out_grp.ix[:,1].sum()



print (out_grp.sum())

print(out_grp.head())
# plot group percent of fraud and genuine transactions per hour

out_grp.ix[:,2:4].plot(kind='bar', cmap = 'brg');
# get hr wise percent between fraud and genuine transactions

out_grp['row_perc_0'] = 100*out_grp.ix[:,0]/out_grp.ix[:,0:2].sum(axis=1)



out_grp['row_perc_1'] = 100*out_grp.ix[:,1]/out_grp.ix[:,0:2].sum(axis=1)



print (out_grp.sum())

print(out_grp.head())
# plot group percent of fraud transactions per hour

out_grp.ix[:,5].plot(kind='bar', title= '% Fraud per Hr') 
# let's do some EDA on repationship between time and Amount for tranactions

def label_Hr (row):

    val = int(row['Time']/3600) + 1

    if val > 24:

        val = val - 24

    return val



data['Hr'] = data.apply(lambda row: label_Hr(row), axis=1)
data.head()
data['Hr'].describe()
# major transaction are in 

data_fraud = data[data['Class'] == 1]



sns.regplot('Hr', 'Amount', data=data_fraud, fit_reg=False)



# Create the boxplot

sns.boxplot('Hr', 'Amount', data=data_fraud)
# frequency plot

sns.distplot(data_fraud['Hr'], bins=24)