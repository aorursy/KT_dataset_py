# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
chocodata = pd.read_csv('../input/flavors_of_cacao.csv')
chocodata.head()
chocodata.columns
#Changing Column Names
old_colnames = chocodata.columns
new_colnames = ['Company', 'BeanOriginBarName', 'REF', 'ReviewDate', 'Cocoa', 'CompanyLocation', 'Rating', 'BeanType', 'BeanOrigin']
chocodata = chocodata.rename(columns = dict(zip(old_colnames, new_colnames)))
chocodata.head()
#Converting Cocoa column to float
chocodata['Cocoa'] = chocodata['Cocoa'].str.replace('%','').astype(float)
chocodata.head()
chocodata['CompanyLocation'].sort_values().unique()
#Fixing the issues in Company Location Names
chocodata['CompanyLocation'] = chocodata['CompanyLocation'].str.replace('Eucador','Ecuador')\
                               .str.replace('Amsterdam','Netherlands')\
                               .str.replace('Belgium','Germany')\
                               .str.replace('Domincan Republic', 'Dominican Republic')\
                               .str.replace('Niacragua', 'Nicaragua')\
                               .str.replace('U.K.', 'England')\
                               .str.replace('U.S.A.', 'United States of America')                                  
chocodata['CompanyLocation'].sort_values().unique()
#Checking for data issues in Bean Origin
chocodata['BeanOrigin'].sort_values().unique()
#Finding No. of entries for each Bean Origin Location
chocodata['BeanOrigin'].value_counts().head()
#Finding no. of NULL values
chocodata['BeanOrigin'].isnull().value_counts()
#Identifying the record with NULL value in BeanOrigin
chocodata[chocodata['BeanOrigin'].isnull() == True]
#Replacing Bean Origin Value for the record with Bean Origin or Bar Name Column
chocodata['BeanOrigin'] = chocodata['BeanOrigin'].fillna(chocodata['BeanOriginBarName'])
chocodata['BeanOrigin'].isnull().value_counts()
chocodata['BeanOrigin'].sort_values().unique()
chocodata['BeanOrigin'].value_counts().head(10)
#Identifying only those with Comma separated names
chocodata[chocodata['BeanOrigin'].str.contains(',')]['BeanOrigin'].sort_values().unique()
chocodata[chocodata['BeanOrigin'].str.contains('/')]['BeanOrigin'].sort_values().unique()
chocodata[chocodata['BeanOrigin'].str.contains('&')]['BeanOrigin'].sort_values().unique()
chocodata[chocodata['BeanOrigin'].str.contains('\(')]['BeanOrigin'].sort_values().unique()
chocodata[chocodata['BeanOrigin'].str.contains('Ven$|Ven,|Venez,|Venez$')]['BeanOrigin'].sort_values().unique()
## Text preparation (correction) func
def txt_prep(text):
    replacements = [
        ['-', ', '], ['/ ', ', '], ['/', ', '], ['\(', ', '], [' and', ', '], [' &', ', '], ['\)', ''],
        ['Dom Rep|DR|Domin Rep|Dominican Rep,|Domincan Republic', 'Dominican Republic'],
        ['Mad,|Mad$', 'Madagascar, '],
        ['PNG', 'Papua New Guinea, '],
        ['Guat,|Guat$', 'Guatemala, '],
        ['Ven,|Ven$|Venez,|Venez$', 'Venezuela, '],
        ['Ecu,|Ecu$|Ecuad,|Ecuad$', 'Ecuador, '],
        ['Nic,|Nic$', 'Nicaragua, '],
        ['Cost Rica', 'Costa Rica'],
        ['Mex,|Mex$', 'Mexico, '],
        ['Jam,|Jam$', 'Jamaica, '],
        ['Haw,|Haw$', 'Hawaii, '],
        ['Gre,|Gre$', 'Grenada, '],
        ['Tri,|Tri$', 'Trinidad, '],
        ['C Am', 'Central America'],
        ['S America', 'South America'],
        [', $', ''], [',  ', ', '], [', ,', ', '], ['\xa0', ' '],[',\s+', ','],
        [' Bali', ',Bali']
    ]
    for i, j in replacements:
        text = re.sub(i, j, text)
    return text
chocodata['BeanOrigin'].str.replace('.','').apply(txt_prep).unique()
chocodata['BeanOrigin'] = chocodata['BeanOrigin'].str.replace('.','').apply(txt_prep)
chocodata['BeanOrigin'].sort_values().unique()
chocodata.head(10)
#Creating a new column to identify if the Chocolate bar is a pure variant or based on a blend
chocodata['Isblend'] = np.where(chocodata['BeanOrigin'].str.contains(','), 'Blend', 'Pure')
#Verifying if the data is fine in the new column
chocodata[chocodata['BeanOrigin']=='Peru,Ecuador,Venezuela'].head()
#Verifying if the data is fine in the new column
chocodata[chocodata['BeanOrigin']=='Venezuela'].head()
chocodata['Isblend'].value_counts()
chocodata.head()
chocodata.describe().T
chocodata.dtypes
#f, ax = plt.subplots(figsize = (12,4), sharex=True,sharey = True)
chocodata['Rating'].plot(kind = 'hist', figsize = (12,4), bins=10)
chocodata[(chocodata['Rating'] >= 3.0)&(chocodata['Rating'] < 4)]['Rating'].plot(kind = 'hist', figsize = (12,4), bins = 2)
k = chocodata['Isblend'].value_counts()
print(k)
chocodata['Isblend'].value_counts().plot(kind = 'Bar', figsize = (14,4))
plt.xlabel('Type of Bean used', fontsize = 14)
plt.ylabel('No. of Chocolate Bars', fontsize = 14)
plt.show()
data1 = chocodata.groupby(by = "Isblend").Rating.mean()
print(data1)
data1.plot(kind = "bar")
plt.xlabel("Type of Bean (Blend/ Pure)", fontsize = 14)
plt.ylabel("Mean Rating", fontsize = 14)
plt.show()
f, ax = plt.subplots(figsize = [6,16])
sns.boxplot(data = chocodata, x = "Rating", y = "CompanyLocation")
plt.subplots(figsize = (14,4))
sns.barplot(data = chocodata.nlargest(10, "Rating"), x = "BeanOriginBarName", y = "Rating", hue = "Rating")
plt.legend(loc = "upper-left", bbox_to_anchor=(1,1))
plt.show()
#Understanding the variance in the Ratings of the Chocolates for different years
k = sns.FacetGrid(chocodata[(chocodata['ReviewDate']>=2010) & (chocodata['ReviewDate']<=2016)], row = "ReviewDate", aspect = 4)
k = k.map(plt.hist, 'Rating')
data1 = chocodata.groupby(by = ["Rating"]). BeanOriginBarName.nunique()
data1.columns = ["Rating", "NoofVal"]
print(data1)
data1.plot(kind = 'bar', y  = "NoofVal", x="Rating", figsize = (18,6), title = "No. of Chocolate Bars by Rating")
plt.xlabel("Year", fontsize = 14)
plt.ylabel("NoofVal", fontsize = 14)
plt.show()
#Creating a Crosstab from an existing DataFrame
flow = pd.crosstab(
    chocodata['CompanyLocation'],
    chocodata['ReviewDate'],
    chocodata['Rating'], aggfunc='mean'
)
#replace NaN with 0
flow.fillna(value = 0, inplace = True)
#Creating a new column tot which is the sum of all values row level (axis = 1 refers to "by rows")
flow['tot'] = flow.mean(axis=1)
#Sorting the Table by tot
flow = flow.sort_values('tot', ascending=False)
flow.head()
#dropping the column tot so that the Locations will be sorted in ascending order
flow = flow.drop('tot', axis=1)
#plotting a heatmap using the flow dataset just created
fig, ax = plt.subplots(figsize=[10,6])
sns.heatmap(flow.head(20), cmap='RdBu_r', linewidths=.5)
ax.set_title('Goods Flow from Company location, mean rating by years')