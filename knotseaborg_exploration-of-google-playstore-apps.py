import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

raw_data = pd.read_csv("../input/googleplaystore.csv")

raw_data = raw_data[raw_data['Category'] != '1.9'] #Fix Category
raw_data.loc[:, 'Category'] = raw_data['Category'].astype('category')
raw_data.loc[:, 'Reviews'] = raw_data['Reviews'].astype('int64') #Fix Reviews
#Fixing Size
s_filter = raw_data.Size == 'Varies with device'
raw_data.loc[~(s_filter), 'Size'] = raw_data['Size'][~(s_filter)].apply(lambda x: float(x[:-1])*1000000 if x[-1] == 'M' else float(x[:-1])*1000 if x[-1] == 'k'else float(x))
raw_data.loc[s_filter, 'Size'] = raw_data['Size'][~(s_filter)].mean()
raw_data.loc[:, 'Size'] = raw_data['Size'].astype('float64')
#Fixing Installs
raw_data = raw_data[raw_data['Installs'] != '0']
raw_data.loc[:, 'Installs'] = raw_data['Installs'].apply(lambda x: int(''.join(x[:-1].split(','))))
#Fixing Types
raw_data.loc[:, 'Type'] = raw_data['Type'].astype('category')
#Fixing Price
raw_data.loc[:, 'Price'] = raw_data['Price'].apply(lambda x: float(x[1:]) if x[0] == '$' else float(x))
#Fixing Content Rating
raw_data.loc[:, 'Content Rating'] = raw_data['Content Rating'].astype('category')
#Fixing Genres
raw_data = pd.concat([raw_data.drop('Genres', axis=1), raw_data['Genres'].str.get_dummies(sep=';')], axis=1)
#Fixing Last Updated
raw_data.loc[:, 'Last Updated'] = raw_data['Last Updated'].apply(lambda x: datetime.strptime(x, '%B %d, %Y'))
#Dropping Current Ver
raw_data = raw_data.drop('Current Ver', axis = 1)
#Fixing Android Ver {We only need the minimum supported version}
raw_data.loc[:, 'Android Ver'] = raw_data['Android Ver'].apply(lambda x: 4.4 if type(x) == float or x=='Varies with device' else x[:3]).astype('float64')
#renaming columns
clean_data = raw_data
clean_data.head()
#clean_data.Size.value_counts()
import matplotlib.pyplot as plt

#Distribution of categories
category_grouped = clean_data.groupby(['Category']).sum()
category_grouped.sort_values('Installs', ascending = False, axis = 0, inplace= True)

sample = category_grouped['Installs'].head(7)
fig, ax = plt.subplots()
plt.barh(sample.index,width = sample.values)
ax.set_xlabel('No.of Installs')
ax.set_title('Which types of App are the most popular?')
plt.show()
category_grouped = clean_data[['Category','Rating', 'Reviews', 'Size','Installs', 'Price']].groupby('Category').mean()

#Category vs Size
sample = category_grouped['Size'].sort_values(ascending = False).head(10)
fig, ax = plt.subplots()
ax.barh(sample.index, width = sample.values)
ax.set_xlabel('Average Memory Size')
ax.set_title('What is the average Size of different applications?')
plt.show()
#Category vs Reviews
fig, ax = plt.subplots()
sample = category_grouped['Reviews'].sort_values(ascending = False).head(10)
ax.barh(sample.index, width = sample.values)
ax.set_xlabel('Average Number of Reviews')
ax.set_title('On Average, how often users interact with application creators?')
plt.xticks(rotation = 60)
plt.show()
#Category vs Average Installs
fig, ax_l = plt.subplots(1,2, figsize=(10,4))
sample = category_grouped['Installs'].sort_values(ascending = False).head(10)
ax_l[0].barh(sample.index, width = sample.values)
ax_l[0].set_xlabel('Average Installs')
ax_l[0].set_title('What is the average number of Installs for different applications?')

sample = clean_data['Category'].value_counts().head(8)
ax_l[1].axis('equal')
ax_l[1].set_title('Top 8 common applications')
explode = (0, 0.1, 0, 0, 0, 0, 0, 0.1)
ax_l[1].pie(sample, labels = sample.index, explode = explode);
plt.show()
#Correlation analysis
sample = clean_data[['Installs', 'Reviews', 'Size', 'Android Ver']]
fig, ax = plt.subplots()
ax.set_xticklabels(['', 'Installs', 'Reviews', 'Size', 'Android Ver'])
ax.set_yticklabels(['', 'Installs', 'Reviews', 'Size', 'Android Ver'])
print(sample.corr())
ax.matshow(sample.corr())
plt.show()
#Checking the distribution of Installations across minimum supported android version
fig, ax = plt.subplots()
ax.bar(x = clean_data['Android Ver'], height = clean_data['Installs'])
ax.set_title('Installations across Android versions')
ax.set_xlabel('Minimum Android verison')
ax.set_ylabel('Installs')
plt.show()
sample = clean_data[['Category', 'Installs']]
sample.loc[:, 'Installs'] = clean_data['Installs'].apply(lambda x: 0 if x==0 else np.log10(x))
sample = sample.groupby('Category')
count = 0
for name, group in sample:
    if name in ['SOCIAL', 'COMMUNICATION', 'PRODUCTIVITY', 'GAME']:
        fig, ax_l = plt.subplots(1,2, figsize=(15, 5))
        x,y = (count>>1, count%2)
        group.hist(ax = ax_l[0], column='Installs')
        ax_l[0].set_title('Distribution of Installations for '+name)
        ax_l[0].set_ylabel('Frequency')
        ax_l[0].set_xlabel('log10(Installs)')
        group.boxplot(ax = ax_l[1], column='Installs')
        ax_l[1].set_title('Boxplot of Installations for '+name)
        ax_l[1].set_ylabel('log10(Installs)')
        count+=1
#333clean_data[clean_data['Category'] == 'COMMUNICATION'].hist(column = ['Rating', 'Size', 'Reviews', 'Installs'])
plt.show()
sample = clean_data['Content Rating'].value_counts()
#General distribution
fig, ax = plt.subplots()
ax.axis('equal')
ax.pie(sample[:-1], labels = sample.index[:-1])
ax.set_title('Distribution of App among user demographics')
plt.show()
#Distribution of apps by categories
#sample = clean_data[clean_data['Content Rating'] == 'Everyone'].groupby
x,y = (-1,-1)
fig, ax_l = plt.subplots(3,2,figsize = (15,15))
for name, group in clean_data.groupby('Content Rating'):
    y=y+1
    x=x+1
    sample = group['Category'].value_counts().head(10)
    sample = sample[sample > 0]
    ax_l[int(x/2)][int(y%2)].set_title('Apps for '+name)
    ax_l[int(x/2)][int(y%2)].barh(sample.index, width = sample)
    ax_l[int(x/2)][int(y%2)].grid(linestyle='-')
plt.subplots_adjust(wspace=0.5)
plt.show()
#Distribution of apps by categories
#sample = clean_data[clean_data['Content Rating'] == 'Everyone'].groupby
x,y = (-1,-1)
fig, ax_l = plt.subplots(3,2,figsize = (18,22))
for name, group in clean_data.groupby('Content Rating'):
    y=y+1
    x=x+1
    sample = group.groupby('Category').sum()['Installs'].sort_values(ascending=False).head(10)
    sample = sample[sample>0]
    ax_l[int(x/2)][int(y%2)].set_title('Installs of apps for '+name)
    ax_l[int(x/2)][int(y%2)].barh(sample.index, width = sample)
    ax_l[int(x/2)][int(y%2)].grid(linestyle = '-')
plt.subplots_adjust(wspace=0.5)
plt.show()