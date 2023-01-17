import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

choko = pd.read_csv('../input/flavors_of_cacao.csv')
# import data
data = pd.read_csv('../input/flavors_of_cacao.csv')

# explore 
data.head()
# explore data type
data.dtypes
# initial clean
# rename cols for easier manipuation
data.columns = ['company', 'origin_specific', 'REF', 'review_date', 'cocoa_percent', 'company_location', 'rating', 'bean_type', 'origin_broad']

# modify data type
data['cocoa_percent'] = data['cocoa_percent'].str.replace('%','').astype(float)/100

# drop rows with null values
data = data.dropna()

data.head()
data.describe(include='all').T
# Distribution of ratings

fig, ax = plt.subplots(figsize=(16,3))
sns.distplot(data['rating'], ax=ax)
ax.set_title('Figure 1: Distribution of Ratings')
plt.show()
# Distribution of cocoa percentages

fig, ax = plt.subplots(figsize=(16,3))
sns.distplot(data['cocoa_percent'], ax=ax)
ax.set_title('Figure 2: Distribution of Cocoa %')
plt.show()
plt.figure(figsize=(16,5))
plt.grid(linewidth=0.6)
plt.scatter(data['rating'], data['cocoa_percent'])
plt.axhline(data['cocoa_percent'].mean(), color='g', linestyle='-.', linewidth=0.8)
plt.axvline(data['rating'].mean(), color='g', linestyle='-.', linewidth=0.8)
plt.xlabel('Rating')
plt.ylabel('Cocoa %')
plt.title('Figure 3: Rating vs. Cocoa %')
# explore
data['company_location'].value_counts().head(10)
# top 10 manufacturers
top_manus = data['company_location'].value_counts().head(10).to_frame().reset_index()
top_manus.columns = ['country', 'count']

m_usa = data[data['company_location'] == 'U.S.A.']['rating'].mean()
m_fra = data[data['company_location'] == 'France']['rating'].mean()
m_can = data[data['company_location'] == 'Canada']['rating'].mean()
m_unk = data[data['company_location'] == 'U.K.']['rating'].mean()
m_ita = data[data['company_location'] == 'Italy']['rating'].mean()
m_ecu = data[data['company_location'] == 'Ecuador']['rating'].mean()
m_aus = data[data['company_location'] == 'Australia']['rating'].mean()
m_blg = data[data['company_location'] == 'Belgium']['rating'].mean()
m_swi = data[data['company_location'] == 'Switzerland']['rating'].mean()
m_ger = data[data['company_location'] == 'Germany']['rating'].mean()

top_manus['rating'] = [m_usa, m_fra, m_can, m_unk, m_ita, m_ecu, m_aus, m_blg, m_swi, m_ger]

top_manus
# create subplots
fig, ax = plt.subplots(1,2, figsize=(16,3))
fig.suptitle('Figure 4: 10 Largest Manufacturing Countries and Their Ratings', fontsize=14)

# plot top 10 manufacturers count
ax1 = sns.barplot(x=top_manus['country'], y=top_manus['count'], color='MediumTurquoise', ax=ax[0])
ax1.set(xlabel='', ylabel='Number of Products')
xrotate = ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# plot top 10 manufacturers rating
ax2 = sns.barplot(x=top_manus['country'], y=top_manus['rating'], color='PaleTurquoise', ax=ax[1])
ax2.set(xlabel='', ylabel='Rating')
xrotate = ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.axhline(top_manus['rating'].mean(), color='grey', linestyle='-.', linewidth=1.0)
# pull the best-rated chocolates (4 & 5); recall that no chocolates were rated 4.5
best = []

for index, row in data.iterrows():
    if row['rating'] == 4 or row['rating'] == 5:
        best.append(row)

best = pd.DataFrame(best)

# plot
fig, ax = plt.subplots(figsize=(16,3))
sns.countplot(x='company_location', data=best, order=best['company_location'].value_counts().index, color='PowderBlue', ax=ax)
ax.set_title('Figure 5: Countries with Best-Rated Products')
ax.set(xlabel='', ylabel='Number of Products')
plt.show()

print("Number of products with 4.0 rating or higher: {}".format(len(best)))
best['company'].value_counts().head(20)
companies = []

for index, row in best.iterrows():
    if row['company'] == 'Soma':
        companies.append(row)
    if row['company'] == 'Bonnat':
        companies.append(row)
    if row['company'] == 'Amedei':
        companies.append(row)
    if row['company'] == 'Valrhona':
        companies.append(row)
    if row['company'] == 'Idilio (Felchlin)':
        companies.append(row)
    if row['company'] == 'Fresco':
        companies.append(row)
    if row['company'] == 'Pierre Marcolini':
        companies.append(row)
    if row['company'] == 'Patric':
        companies.append(row)
    if row['company'] == 'Domori':
        companies.append(row)
    if row['company'] == 'Michel Cluizel':
        companies.append(row)
    if row['company'] == 'Pralus':
        companies.append(row)
    if row['company'] == 'A. Morin':
        companies.append(row)

companies = pd.DataFrame(companies)
# create subplots
fig, ax = plt.subplots(1,2, figsize=(16,3))
fig.suptitle('Figure 6: Top Companies* and their Locations', fontsize=14)

# plot top 10 manufacturers count
ax1 = sns.countplot(x='company', data=companies, order=companies['company'].value_counts().index, color='MediumTurquoise', ax=ax[0])
ax1.set(xlabel='', ylabel='Number of Products')
xrotate = ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
                  
# plot top 10 manufacturers locations
ax2 = sns.countplot(x='company_location', data=companies, order=companies['company_location'].value_counts().index, color='PaleTurquoise', ax=ax[1])
ax2.set(xlabel='', ylabel='Number of Products')
xrotate = ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

print("*Includes companies with 3 or more products rated 4.0 and up")
# explore bean origins

print(data['origin_broad'].value_counts().head(10))
print('\nWhere a blank origin indicates a blend')
# top 10 bean producers
beans = data['origin_broad'].value_counts().head(10).to_frame().reset_index()
beans.columns = ['country', 'count']

m_ven = data[data['origin_broad'] == 'Venezuela']['rating'].mean()
m_ecu = data[data['origin_broad'] == 'Ecuador']['rating'].mean()
m_per = data[data['origin_broad'] == 'Peru']['rating'].mean()
m_mad = data[data['origin_broad'] == 'Madagascar']['rating'].mean()
m_dom = data[data['origin_broad'] == 'Dominican Republic']['rating'].mean()
m_nic = data[data['origin_broad'] == 'Nicaragua']['rating'].mean()
m_bra = data[data['origin_broad'] == 'Brazil']['rating'].mean()
m_bol = data[data['origin_broad'] == 'Bolivia']['rating'].mean()
m_bel = data[data['origin_broad'] == 'Belize']['rating'].mean()
m_ble = data[data['origin_broad'].str.contains(',')]['rating'].mean()

beans['rating'] = [m_ven, m_ecu, m_per, m_mad, m_dom, m_ble, m_nic, m_bra, m_bol, m_bel]

# create subplots
fig, ax = plt.subplots(1,2, figsize=(16,3))
fig.suptitle('Figure 7: 10 Biggest Bean Producers and Their Ratings', fontsize=14)

# plot top 10 bean producers count
ax1 = sns.barplot(x=beans['country'], y=beans['count'], color='DarkSeaGreen', ax=ax[0])
ax1.set(xlabel='', ylabel='Count')
xrotate = ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# plot top 10 bean producers rating
ax2 = sns.barplot(x=beans['country'], y=beans['rating'], color='LightGreen', ax=ax[1])
ax2.set(xlabel='', ylabel='Rating')
xrotate = ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
#ax2.yaxis.tick_right()
#ax2.yaxis.set_label_position('right')
ax2.axhline(data['rating'].mean(), color='grey', linestyle='-.', linewidth=1.0)


print('* Where a blank origin indicates a blend')
# get top 10 bean producers (by rating)
# recall that the df 'best' includes all products rated 4.0 or higher

beans = best['origin_broad'].value_counts().head(10)
beans = pd.DataFrame(beans).reset_index()

# plot
fig, ax = plt.subplots(figsize=(16,3))
sns.barplot(x='index', y='origin_broad', data=beans, color='PowderBlue', ax=ax)
ax.set_title('Figure 5: Countries with Best-Rated Products')
ax.set(xlabel='', ylabel='Number of Products')
xrotate = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

print('* Where a blank origin indicates a blend')
# side look: types of blends
data[data['origin_broad'].str.len()==1]['origin_specific'].unique()
