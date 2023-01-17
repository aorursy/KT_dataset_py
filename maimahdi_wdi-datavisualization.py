# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
indicators_df = pd.read_csv('../input/world-development-indicators/Indicators.csv')

indicators_df

indicators_df.head(20)
years = indicators_df.loc[indicators_df['IndicatorCode'] == 'NY.GNP.PCAP.CD',['Year']].Year.unique()

years
richcountry_1960= indicators_df.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 1962").sort_values(by = 'Value')[-10:]

richcountry_2014 = indicators_df.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 2014").sort_values(by= 'Value')[-10:]
plt.figure(figsize=(16,4))

graph_rich = sns.barplot(x = "CountryName", y = "Value", palette = "BuGn", data = richcountry_1960)

plt.xlabel('Country', fontsize = 14)

plt.ylabel('Average Income ($)',  fontsize=14)

plt.title('The 10 Countries with Highest Average Income in 1960', fontsize = 14)
plt.figure(figsize=(16,4))

graph_rich1 = sns.barplot(x = "CountryName", y = "Value", palette = "BuGn", data = richcountry_2014)

plt.xlabel('Country', fontsize = 14)

plt.ylabel('Average Income ($)',  fontsize=14)

plt.title('The 10 Countries with Highest Average Income in 2014', fontsize = 14)
for key, group in richcountry_1960.groupby(['CountryName']):

    for key1, group1 in richcountry_2014.groupby(['CountryName']):

        if key == key1:

            print (key)
lowest_1960 = indicators_df.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 1962").sort_values(by = 'Value', ascending = True)[:10]

lowest_2014 = indicators_df.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 2014").sort_values(by = 'Value', ascending = True)[:10]

plt.figure(figsize=(16,4))

graph_rich = sns.barplot(x = "CountryName", y = "Value", palette = "BuGn", data = lowest_1960)

plt.xlabel('Country', fontsize = 14)

plt.ylabel('Average Income ($)',  fontsize=14)

plt.title('The 10 Countries with  Lowest Average  Income in 1960', fontsize = 14)
plt.figure(figsize=(16,4))

graph_rich = sns.barplot(x = "CountryName", y = "Value", palette = "BuGn", data = lowest_2014)

plt.xlabel('Country', fontsize = 14)

plt.ylabel('Average Income ($)',  fontsize=14)

plt.title('The 10 Countries with  Lowest Average  Income in 2014', fontsize = 14)
year = indicators_df.loc[indicators_df['IndicatorCode'] == 'SP.DYN.CDRT.IN',['Year']].Year.unique()

year
lowest_1960 = indicators_df.query("IndicatorCode == 'SP.DYN.CDRT.IN' & CountryName != list & Year == 1960").sort_values(by = 'Value', ascending = True)[:10]

lowest_2013 = indicators_df.query("IndicatorCode == 'SP.DYN.CDRT.IN' & CountryName != list & Year == 2013").sort_values(by = 'Value', ascending = True)[:10]

plt.figure(figsize=(16,4))

graph_rich = sns.barplot(x = "CountryName", y = "Value", palette = "PuBu", data = lowest_1960)

plt.xlabel('Country', fontsize = 14)

plt.ylabel('Death Average',  fontsize=14)

plt.title('The 10 Countries with  Lowest Average  Death in 1960', fontsize = 14)
plt.figure(figsize=(16,4))

graph_rich = sns.barplot(x = "CountryName", y = "Value", palette = "PuBu", data = lowest_2013)

plt.xlabel('Country', fontsize = 14)

plt.ylabel('Average Death',  fontsize=14)

plt.title('The 10 Countries with  Lowest Average Death in 2013', fontsize = 14)
highest_1960= indicators_df.query("IndicatorCode == 'SP.DYN.CDRT.IN' & CountryName != list & Year == 1960").sort_values(by = 'Value')[-10:]

highest_2013 = indicators_df.query("IndicatorCode == 'SP.DYN.CDRT.IN' & CountryName != list & Year == 2013").sort_values(by= 'Value')[-10:]

plt.figure(figsize=(16,4))

graph_rich = sns.barplot(x = "CountryName", y = "Value", palette = "PuBu", data = highest_1960)

plt.xlabel('Country', fontsize = 14)

plt.ylabel('Death Average',  fontsize=14)

plt.title('The 10 Countries with  Highest Average  Death in 1960', fontsize = 14)
plt.figure(figsize=(16,4))

graph_rich1 = sns.barplot(x = "CountryName", y = "Value", palette = "PuBu", data = highest_2013)

plt.xlabel('Country', fontsize = 14)

plt.ylabel('Death Average',  fontsize=14)

plt.title('The 10 Countries with  Highest Average  Death in 2013', fontsize = 14)