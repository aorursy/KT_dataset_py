#Import the required Libraries.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#Read the data in pandas

inp0= pd.read_csv("../input/AttributeDataSet.csv")

inp1= pd.read_csv("../input/DressSales.csv")
inp0.info()
inp0.head()
# Print the information about the attributes of inp0 and inp1.

inp0.info()
inp1.info()
# Column fixing, correcting size abbreviation. count the percentage of each size category in "Size" column.

pd.unique(inp0['Size'])
inp0['Size'].replace(['XL','L','M','S','s','small','free'], ['Extra Large','Large','Medium','Small','Small','Small','Free'], inplace=True)
pd.unique(inp0['Size'])
# Print the value counts of each category in "Size" column.

# What is the value of the lowest percentage, the highest percentage and the percentage of Small size categories in the column named “Size”?



inp0['Size'].value_counts(normalize=True)*100
# Print the null count of each variables of inp0 and inp1.

inp0.isnull().sum()
inp1.isnull().sum()
# Print the data types information of inp1 i.e. "Dress Sales" data.

inp1.info()
# Try to convert the object type into float type of data. YOU GET ERROR MESSAGE.

#inp1['09-12-2013'] = pd.to_numeric(inp1['09-12-2013'])
# Do the required changes in the "Dress Sales" data set to get null values on string values.

inp1.loc[inp1['09-12-2013']== 'Removed',"09-12-2013"] = np.NaN

inp1.loc[inp1['14-09-2013']== 'removed',"14-09-2013"] = np.NaN

inp1.loc[inp1['16-09-2013']== 'removed',"16-09-2013"] = np.NaN

inp1.loc[inp1['18-09-2013']== 'removed',"18-09-2013"] = np.NaN

inp1.loc[inp1['20-09-2013']== 'removed',"20-09-2013"] = np.NaN

inp1.loc[inp1['22-09-2013']== 'Orders',"22-09-2013"] = np.NaN
# Convert the object type columns in "Dress Sales" into float type of data type.

inp1['09-12-2013'] = pd.to_numeric(inp1['09-12-2013'], downcast='float')

inp1['14-09-2013'] = pd.to_numeric(inp1['14-09-2013'], downcast='float')

inp1['16-09-2013'] = pd.to_numeric(inp1['16-09-2013'], downcast='float')

inp1['18-09-2013'] = pd.to_numeric(inp1['18-09-2013'], downcast='float')

inp1['20-09-2013'] = pd.to_numeric(inp1['20-09-2013'], downcast='float')

inp1['22-09-2013'] = pd.to_numeric(inp1['22-09-2013'], downcast='float')
inp1.info()
# Print the null percetange of each column of inp1.

inp1.isnull().sum()/len(inp1.index)*100
# Drop the columns in "Dress Sales" which have more than 40% of missing values.

inp1 = inp1.drop(['26-09-2013','30-09-2013','10-02-2013','10-04-2013','10-08-2013','10-10-2013' ],axis=1)

inp1.info()
inp1.isnull().sum()/len(inp1.index)*100
inp1.head()
# Create the four seasons columns in inp1, according to the above criteria.

inp1['Summer'] = inp1['09-06-2013'] + inp1['10-06-2013'] + inp1['29-08-2013'] + inp1['31-08-2013'] + inp1['09-08-2013']

inp1['Autumn'] = inp1['09-10-2013'] + inp1['14-09-2013'] + inp1['16-09-2013'] + inp1['18-09-2013'] + inp1['20-09-2013'] + inp1['22-09-2013'] + inp1['24-09-2013'] + inp1['28-09-2013']

inp1['Winter'] = inp1['09-12-2013'] + inp1['10-12-2013'] + inp1['09-02-2013']

inp1['Spring'] = inp1['09-04-2013']
# calculate the sum of sales in each seasons in inp1 i.e. "Dress Sales".

inp1[['Summer','Autumn','Winter','Spring']].sum()
# Merge inp0 with inp1 into inp0. this is also called left merge.

inp0 = pd.merge(left=inp0,right=inp1, how='left', left_on='Dress_ID', right_on='Dress_ID')

inp0.head()
# Now Drop the Date columns from inp0 as it is already combined into four seasons.

inp0.drop(inp0.loc[:,'29-08-2013':'10-12-2013'].columns, axis= 1, inplace= True)
# Print the null count of each columns in inp0 dataframe i.e. combined data frame of inp0 and inp1 without date columns.

inp0.isnull().sum()
#inp0.drop(inp0.loc[:,'29-08-2013_x':'Spring_y'].columns, axis= 1, inplace= True)

#inp0.head()
inp0.isnull().sum()
# Deal with the missing values of Type-1 columns: Price, Season, NeckLine, SleeveLength, Winter and Autumn.

inp0['Price'].fillna(inp0['Price'].mode(), inplace=True)

inp0['Season'].fillna(inp0['Season'].mode(), inplace=True)

inp0['NeckLine'].fillna(inp0['NeckLine'].mode(), inplace=True)

inp0['SleeveLength'].fillna(inp0['SleeveLength'].mode(), inplace=True)

inp0['Winter'].fillna(inp0['Winter'].mode(), inplace=True)

inp0['Autumn'].fillna(inp0['Autumn'].mode(), inplace=True)
# Deal with the missing values for Type-2 columns: Material, FabricType, Decoration and Pattern Type.

inp0.dropna(axis=0, subset=['Material','FabricType','Decoration','Pattern Type'], inplace=True)

inp0.isnull().sum()
#correcting the spellings.

inp0['Season'].unique()
inp0['Season'].replace(['Automn','winter'], ['Autumn','Winter'], inplace=True)

inp0['Season'].unique()
inp0['SleeveLength'].unique()
inp0['SleeveLength'].replace(['thressqatar'], ['threequarter'], inplace=True)

inp0['SleeveLength'].unique()
inp0.head()
# Group "Style" categories into "Others" which have less than 50000 sales across all the seasons.

inp0['total'] = inp0['Summer'] + inp0['Autumn'] + inp0['Winter'] + inp0['Spring']

style_group = inp0['total'].groupby(inp0['Style']).sum().reset_index()

res = style_group.loc[style_group['total']<50000]

res
inp0['Style'].replace(['Flare','Novelty','bohemian','party','party','sexy','vintage','work'], 'Others', inplace=True)

inp0['Style'].unique()
# Calculate the percentage of each categories in the "Style" variable.

inp0['Style'].value_counts(normalize=True)*100
# Group "Neckline" categories into "Others" which have less than 50000 sales across all the seasons.

neck_group = inp0['total'].groupby(inp0['NeckLine']).sum().reset_index()

res = neck_group.loc[neck_group['total']<50000]

res
inp0['NeckLine'].replace(['Sweetheart','boart-neck','bowneck','open','peterpan-collor','slash-neck','turndowncollor'], 'Others', inplace=True)

inp0['NeckLine'].unique()
# Group "Sleeve length" categories into "Others" which have less than 50000 sales across all the seasons.

sleeve_group = inp0['total'].groupby(inp0['SleeveLength']).sum().reset_index()

res = sleeve_group.loc[sleeve_group['total']<50000]

res
inp0['SleeveLength'].replace(['butterfly','threequarter'], 'Others', inplace=True)

inp0['SleeveLength'].unique()
# Group "material" categories into "Others" which have less than 25000 sales across all the seasons.

mat_group = inp0['total'].groupby(inp0['Material']).sum().reset_index()

res = mat_group.loc[mat_group['total']<25000]

res
inp0['Material'].replace(['linen','lycra','model','nylon','other','shiffon','spandex'], 'Others', inplace=True)

inp0['Material'].unique()
# Group "fabric type" categories into "Others" which have less than 25000 sales across all the seasons.

fab_group = inp0['total'].groupby(inp0['FabricType']).sum().reset_index()

res = fab_group.loc[fab_group['total']<25000]

res
res['FabricType']
inp0['FabricType'].replace([res['FabricType']], 'Others', inplace=True)

inp0['FabricType'].unique()
# Group "patern type" categories into "Others" which have less than 25000 sales across all the seasons.

pat_group = inp0['total'].groupby(inp0['Pattern Type']).sum().reset_index()

res = pat_group.loc[pat_group['total']<25000]

res
inp0['Pattern Type'].replace([res['Pattern Type']], 'Others', inplace=True)

inp0['Pattern Type'].unique()
# Group "decoration" categories into "Others" which have less than 25000 sales across all the seasons.

dec_group = inp0['total'].groupby(inp0['Decoration']).sum().reset_index()

res = dec_group.loc[dec_group['total']<25000]

res
inp0['Decoration'].replace([res['Decoration']], 'Others', inplace=True)

inp0['Decoration'].unique()
inp0.head()
x = inp0['Autumn'].max() - inp0['Autumn'].quantile(0.75)

x
# Describe the numerical variale: "Autumn".

inp0['Autumn'].describe()
# plot the boxplot of "Autumn" column.

plt.boxplot(inp0['Summer'])

plt.show()
# Find the maximum and 99th percentile of Winter season.

print(inp0['Winter'].max()) 

print(inp0['Winter'].quantile(0.99))



x = inp0['Winter'].max() - inp0['Winter'].quantile(0.99)

x
# Find the maximum and 99th percentile of Summer season.

print(inp0['Summer'].max()) 

print(inp0['Summer'].quantile(0.99))

x = inp0['Summer'].max() - inp0['Summer'].quantile(0.99)

x
# Find the maximum and 99th percentile of Spring season.

print(inp0['Spring'].max()) 

print(inp0['Spring'].quantile(0.99))

x = inp0['Spring'].max() - inp0['Spring'].quantile(0.99)

x
# Find the maximum and 99th percentile of Autumn season.

print(inp0['Autumn'].max()) 

print(inp0['Autumn'].quantile(0.99))



x = inp0['Autumn'].max() - inp0['Autumn'].quantile(0.99)

x
# Find the Mean of Ratings for each Price category.

inp0['Rating'].groupby(inp0['Price']).mean()
# Find the median of Ratings for each Style category.

inp0['Rating'].groupby(inp0['Style']).median()
inp0['Recommendation'].groupby(inp0['Season']).mean()
# Size vs Recommendation.

inp0.groupby('Size')['Recommendation'].sum()
# plot the heat map of Style, price and Recommendation.

res = pd.pivot_table(data=inp0, index='Style', columns='Price', values='Recommendation')

res
sns.heatmap(res, cmap='RdYlGn', annot=True)
# plot the heat map of Season, material and Recommendation.

res = pd.pivot_table(data=inp0, index='Material', columns='Season', values='Recommendation')

plt.figure(figsize=[10,10])

sns.heatmap(res[['Summer','Winter']], cmap='RdYlGn', annot=True)