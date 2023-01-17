import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#import seaborn as sns
from pathlib import Path
plt.style.use('seaborn-whitegrid')
print(plt.style.available) # it's nice to see whats available :)
# path = Path(r'D:\Coding/comptab_2018-01-29 16_00_comma_separated.csv')
path = Path('../input/comptab_2018-01-29 16_00_comma_separated.csv') #  kaggle path
data = pd.read_csv(path)
data.head(5)
zero_trade = data.loc[data['Importer reported quantity']==0].index
data = data.drop(zero_trade)
#data.describe
plt.figure(figsize=(20,10))
plt.subplot(121)
data.boxplot(column='Importer reported quantity', showfliers=False)
plt.title('Amount of imported goods per trade', fontsize=20)

plt.subplot(122)
data.boxplot(column='Exporter reported quantity', showfliers=False)
plt.title('Amount of exported goods per trade', fontsize=20)
class_distribution = data['Class'].value_counts(normalize=True)
plt.figure(figsize=(10,10))
class_distribution.head(5).plot(kind='bar', fontsize=18)
plt.title('5 most traded animal classes', fontsize=20)
plt.xticks(rotation=40)
plt.figure(figsize=(10,10))
plt.figure(figsize=(20,8))
data['Source'].value_counts(normalize=True).plot(kind='bar', fontsize=18)
plt.title('Relative amounts of different sources of traded animal goods', fontsize=20)
plt.xticks(rotation=0) #  this rotates the labels of x-axis by 90°
plt.figure(figsize=(15,10))
plt.subplot(511)
data['Class'].loc[data['Source']=='W'].value_counts(normalize=True).head(5).plot.barh(fontsize=18)
plt.title('Top five traded animal classes caught in the wild', fontsize=20)
plt.subplot(513)
data['Order'].loc[data['Source']=='A'].value_counts(normalize=True).head(5).plot.barh(fontsize=18)
plt.title('Top five traded plant orders propagated artificially', fontsize=20)
plt.subplot(515)
data['Class'].loc[data['Source']=='C'].value_counts(normalize=True).head(5).plot.barh(fontsize=18)
plt.title('Top five traded animal classes that were bred in captivity', fontsize=20)
#  when it comes to captivity and breeding, things get a little more complicated and blurred.

animal_classes = ['Reptilia', 'Aves', 'Actinopteri', 'Mammalia', 'Anthozoa']
stacked_data = pd.DataFrame()
for i in animal_classes:
    stacked_data[i] = data['Term'].loc[data['Class']==i].value_counts()
stacked_data.head(15).plot.barh(figsize=(20,15), fontsize=24, stacked=True)
#plt.xscale('log') #  applying logscale removes Reptilia bar color?!
plt.legend(fontsize=30)
plt.title('Amount of 15 most traded animal goods', fontsize=24)
plt.figure(figsize=(20,10))

plt.subplot(121)
toptrader_imp = data['Importer'].value_counts(normalize=True)
toptrader_imp.head(5).plot(kind='bar', fontsize=18)
plt.title('Top 5 importing countries', fontsize=20)
plt.xticks(rotation=0)

plt.subplot(122)
toptrader_exp = data['Exporter'].value_counts(normalize=True)
toptrader_exp.head(5).plot(kind='bar', fontsize=18)
plt.title('Top 5 exporting countries', fontsize=20)
plt.xticks(rotation=0)
all_appendices = data['App.'].value_counts(normalize=True)
known_amounts = data.loc[data['Importer reported quantity']==data['Exporter reported quantity'], 'App.'].value_counts(normalize=True)
y = all_appendices - known_amounts
y.plot.barh(figsize=(20,5), fontsize=24)
plt.title('Relative difference in known trade amounts compared to sum of all trades', fontsize=20)
