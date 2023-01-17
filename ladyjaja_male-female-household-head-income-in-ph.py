import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import ttest_ind

import scipy.stats as stats



raw_data = pd.read_csv('../input/Family Income and Expenditure.csv')

print('Shape: ', raw_data.shape)

#print('Columns: ', raw_data.columns)

print('Household Head Sex (HHS): ', raw_data['Household Head Sex'].unique())

print('Number of rows per HHS: ', raw_data.groupby('Household Head Sex')['Household Head Sex'].count())



female = raw_data[raw_data['Household Head Sex'] == 'Female']

female_rowcount = len(female.index)

male = raw_data[raw_data['Household Head Sex'] == 'Male']

male_rowcount = len(male.index)

print('HOUSEHOLD HEAD SEX (HHS)')

print ('Number of rows where HHS is Female: ', female_rowcount, 'Percentage: ', "{:.2%}".format(female_rowcount/(female_rowcount+male_rowcount)))

print ('Number of rows where HHS is Male: ', male_rowcount, 'Percentage: ', "{:.2%}".format(male_rowcount/(female_rowcount+male_rowcount)))

print('TOTAL HOUSEHOLD INCOME (THI)')

print ('Mean THI where HHS is Female: ', "{:,.2f}".format(female['Total Household Income'].mean()))

print ('Mean THI where HHS is Male: ', "{:,.2f}".format(male['Total Household Income'].mean()))

plt.figure(figsize=(10,10))

#plt.ylim(0, 5000000)

plt.xticks(rotation=30)

#plt.axes.ticklabel_format(axis='y', style='plain')

#graph = sns.scatterplot(x=raw_data['Region'], y=raw_data['Total Household Income'], hue=raw_data['Household Head Sex'], hue_order=['Male', 'Female'])

graph = sns.scatterplot(x=raw_data['Household Head Sex'], y=raw_data['Total Household Income'], hue=raw_data['Household Head Sex'], hue_order=['Male', 'Female'])

ylabels = ['{:,}'.format(y) for y in graph.get_yticks()]

graph.set_yticklabels(ylabels)

print('Region: ', raw_data['Household Head Sex'].unique())

print('Number of rows per Region: ', raw_data.groupby('Region')['Region'].count())



femalebyreg = female.groupby('Region')['Household Head Sex'].count().to_frame()

femalebyreg.rename(columns={'Household Head Sex': 'Female HHS Count'}, inplace=True)

#print(femalebyreg)

malebyreg = male.groupby('Region')['Household Head Sex'].count().to_frame()

malebyreg.rename(columns={'Household Head Sex': 'Male HHS Count'}, inplace=True)

#print(malebyreg)

femalebyregincome = female.groupby('Region')['Total Household Income'].mean().to_frame()

femalebyregincome.rename(columns={'Total Household Income': 'Female THI Mean'}, inplace=True)

#print(femalebyregincome)

malebyregincome = male.groupby('Region')['Total Household Income'].mean().to_frame()

malebyregincome.rename(columns={'Total Household Income': 'Male THI Mean'}, inplace=True)

#print(malebyregincome)

hhsbyreg = pd.concat([femalebyreg, malebyreg, femalebyregincome, malebyregincome], axis=1)

#print(hhsbyreg.columns)

hhsbyreg['Diff in THI Mean'] = hhsbyreg['Female THI Mean'] - hhsbyreg['Male THI Mean'] 

print(hhsbyreg[['Female THI Mean', 'Male THI Mean']])
hhsbyreg['Female THI Mean'] - hhsbyreg['Male THI Mean'] 
# Both datasets do not follow normal distribution



#income = male['Total Household Income'].tolist()

income = female['Total Household Income'].tolist()

income.sort()

incomemean = np.mean(income)

incomestd = np.std(income)

print(incomemean, incomestd)

fit = stats.norm.pdf(income, incomemean, incomestd)

plt.plot(income, fit)

plt.show()
t, p = ttest_ind(female['Total Household Income'], male['Total Household Income'])

different = "Reject NULL Hypothesis" if p < 0.01 else "NULL Hypothesis can't be rejected"

print("t test results, p is compared to 0.01")

print ("t = ", t)

print ("p = ", p)

print("Conclusion: ", different)