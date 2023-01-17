import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pylab import rcParams
%matplotlib inline
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
df = pd.read_csv("../input/googleplaystore.csv")
df.head()
print("The shape of the data is ",df.shape)
df.describe().T
plt.figure(figsize=(8,6,))
sns.heatmap(df.isnull(), cbar = False)
total = df.isnull().sum().sort_values(ascending  = False)
percent = (df.isnull().sum()/df.count()).sort_values(ascending = False)
temp = pd.concat([total, percent], axis = 1, keys = ['total','percentage'])
temp.head()
#Dropping observations having missing values in any column
df.dropna(how = 'any', inplace = True)
print("Length of Unique App names = ", len(df['App'].unique()))
print("Legth of the Total App name = ", df.shape[0])
print("Duplicate Apps = ",df.shape[0]- len(df['App'].unique()))
df[df['App'] == 'Coloring book moana']
df.drop_duplicates(subset = 'App', keep = 'first', inplace = True)
temp = df['Category'].value_counts().reset_index() #A temporary dataframe for this plot

plt.figure(figsize=(12,12))
ax = plt.subplot(111)
plt.pie(x = temp['Category'], labels= temp['index'],autopct= '%1.1f%%')
plt.legend()
ax.legend(bbox_to_anchor=(1.4, 1))
plt.show()
plt.figure(figsize=(10,7))
sns.distplot(df['Rating'])
plt.legend(['Rating'])
plt.show()
print("The average rating in the appstore is ",np.average(df['Rating']))
top = np.array(df.Category.value_counts().reset_index()['index'])
print("Most Occuring Categories\n",top[:6])
plt.figure(figsize= (15,10))
plt.suptitle("Ratings of Different Categories",fontsize = 22)

plt.subplot(2,3,1)
sns.kdeplot(df[df['Category'] == 'FAMILY']['Rating'], shade = True)
plt.title('Rating of FAMILY Apps')

plt.subplot(2,3,2)
sns.kdeplot(df[df['Category'] == 'GAME']['Rating'], shade = True)
plt.title('Rating of GAME Apps')


plt.subplot(2,3,3)
sns.kdeplot(df[df['Category'] == 'TOOLS']['Rating'], shade = True)
plt.title('Rating of TOOLS Apps')


plt.subplot(2,3,4)
sns.kdeplot(df[df['Category'] == 'FINANCE']['Rating'], shade = True)
plt.title('Rating of FINANCE Apps')


plt.subplot(2,3,5)
sns.kdeplot(df[df['Category'] == 'LIFESTYLE']['Rating'], shade = True)
plt.title('Rating of LIFESTYLE Apps')


plt.subplot(2,3,6)
sns.kdeplot(df[df['Category'] == 'PRODUCTIVITY']['Rating'], shade = True)
plt.title('Rating of PRODUCTIVITY Apps')

plt.show()
import scipy.stats as stats
htest = stats.f_oneway(df[df['Category'] == 'FAMILY']['Rating'],
              df[df['Category'] == 'GAME']['Rating'],
              df[df['Category'] == 'TOOLS']['Rating'],
              df[df['Category'] == 'FINANCE']['Rating'],
              df[df['Category'] == 'PRODUCTIVITY']['Rating'],
              df[df['Category'] == 'LIFESTYLE']['Rating'],
              )
print("The P value of the test is ",htest[1])
plt.figure(figsize=(18,9))
f = sns.violinplot(x = df['Category'], y = df['Rating'], palette= 'coolwarm')
f.set_xticklabels(f.get_xticklabels(), rotation = 90)
plt.show()
print(df['Reviews'].head())
df['Reviews'] = df['Reviews'].astype(dtype = 'int')
plt.figure(figsize=(15,8))
sns.kdeplot(df['Reviews'], color = 'Green', shade = True)
plt.title('Distribution of Ratings')
print("Number of Apps with more than 1M reviews",df[df['Reviews'] > 1000000].shape[0])
print("\nTop 20 apps with most reviews: \n",df[df['Reviews'] > 1000000].sort_values(by = 'Reviews', ascending = False).head(20)['App'])
print("For all apps")
sns.jointplot(x = 'Reviews', y= 'Rating',data = df[df['Reviews']>100000], color = 'darkorange') 
plt.show()

print("For apps below 1M reviews")
sns.jointplot(x = 'Reviews', y= 'Rating',data = df[df['Reviews']<100000], color = 'darkorange') 
plt.show()
df['Installs'].dtype
df['Installs'].head()
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',',''))
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+',''))
df['Installs'] = df['Installs'].astype(dtype = 'int')
df['Installs'].head()
df['Installs'].unique()
plt.figure(figsize=(12,8))
f = sns.countplot(df['Installs'], palette= "viridis" )
f.set_xticklabels(f.get_xticklabels(), rotation = 30)
plt.show()
sorted_values = sorted(df['Installs'].unique())
df['Installs Classes'] = df['Installs'].replace(sorted_values, range(0,len(sorted_values)))
df['Installs Classes'].head()
plt.figure(figsize=(12,9))
sns.boxplot(y = df['Rating'], x = df['Installs Classes'], palette= 'Blues')
plt.show()
import scipy.stats as sp

plt.figure(figsize=(13,13))
plt.subplot(2,2,1)
f = sns.kdeplot(df[df['Installs Classes'] == 5]['Rating'], shade = True, color = 'purple')
plt.title("Ratings variation for apps above 5 installs")
f.set_xticks([1,2,3,4,5])

plt.subplot(2,2,2)
f = sns.kdeplot(df[df['Installs Classes'] == 6]['Rating'], shade = True, color = 'purple')
plt.title("Ratings variation for apps above 500 installs")
f.set_xticks([1,2,3,4,5])

plt.subplot(2,2,3)
f = sns.kdeplot(df[df['Installs Classes'] == 17]['Rating'], shade = True, color = 'purple')
plt.title("Ratings variation for apps above 500M installs")
f.set_xticks([1,2,3,4,5])

plt.subplot(2,2,4)
f = sns.kdeplot(df[df['Installs Classes'] == 18]['Rating'], shade = True, color = 'purple')
plt.title("Ratings variation for apps above 1B installs")
f.set_xticks([1,2,3,4,5])

plt.show()
print("Variation in Rating of installs above 100 installs ",sp.variation(df[df['Installs Classes'] == 5]['Rating']))
print("Variation in Rating of installs above 500 installs ",sp.variation(df[df['Installs Classes'] == 6]['Rating']))
print("Variation in Rating of installs above 500M installs ",sp.variation(df[df['Installs Classes'] == 17]['Rating']))
print("Variation in Rating of installs above 1B installs ",sp.variation(df[df['Installs Classes'] == 18]['Rating']))

df['Size'].head()
print(df['Size'].unique())
df['Size'] = df['Size'].apply(lambda x: x.replace('M', '*1000'))
df['Size'] = df['Size'].apply(lambda x: x.replace('k', ''))
df['Size'].replace('Varies with device', '-1', inplace = True)
df['Size'] = df['Size'].apply(lambda x: eval(x))
df['Size'] = df['Size'].replace(-1,np.nan) #Changing the values to null then we can fill them with mean value
df['Size'].fillna(np.mean(df['Size']), inplace = True) 
plt.figure(figsize=(18,9))
sns.distplot(df['Size'], color = 'darkred')
plt.xlabel('Size in KBs')
plt.xticks(list(range(0, int(max(df['Size'])), 5000)))
plt.show()
plt.figure(figsize=(10,10))
plt.scatter(x = df['Size'], y = df['Rating'], color = 'orange')
plt.xlabel('Size in KBs')
plt.ylabel('Rating')
plt.title('Rating vs Size')
plt.show()
plt.figure(figsize= (12,12))
sns.regplot(y = df['Size'], x = df['Installs Classes'], color = 'grey')
plt.title('Size vs Installs')
plt.show()
temp = df['Type'].value_counts().reset_index()

# plt.figure(figsize=(9,9))
rcParams['figure.figsize'] = 9,9
plt.pie(x = temp['Type'], labels= temp['index'], autopct= '%1.1f%%', colors = ['lightblue','lightgreen'], 
        shadow= True, explode=(0.25,0), startangle= 90)
plt.show()
df['Price'].unique()
df['Price'] = df['Price'].apply(lambda x: x.replace('$',''))
df['Price'] = df['Price'].astype('float')
plt.figure(figsize=(10,7))
sns.kdeplot(df[df['Type'] == 'Paid']['Price'], color = 'blue', shade = True)
plt.xlabel('Prices of Apps')
plt.title('Pricing Distribution of Paid Apps')
plt.show()
paid_prices = df[df['Type'] == 'Paid']['Price']
sns.jointplot(y = df[df['Type'] == 'Paid']['Rating'], x = df[df['Type'] == 'Paid']['Price'], color= 'teal')
plt.show()
df.loc[df['Price'] == 0,'Price_Class'] = 'Free'
df.loc[(df['Price'] > 0) & (df['Price'] <=1), 'Price_Class'] = 'Cheap'
df.loc[(df['Price'] > 1) & (df['Price'] <=3), 'Price_Class'] = 'Above Cheap'
df.loc[(df['Price'] > 3) & (df['Price'] <=6), 'Price_Class'] = 'Average'
df.loc[(df['Price'] > 6) & (df['Price'] <=16), 'Price_Class'] = 'Above Average'
df.loc[(df['Price'] > 16) & (df['Price'] <=40), 'Price_Class'] = 'Expensive'
df.loc[(df['Price'] > 40), 'Price_Class'] = 'Too Expensive'
temp = df[df['Type'] == 'Paid']['Price_Class'].value_counts().reset_index()

sns.barplot(x = temp['index'], y = temp['Price_Class'], palette= 'autumn')
plt.xlabel('Price Classes')
plt.ylabel('Counts')
plt.show()
df[['Price_Class','Rating','Reviews']].groupby('Price_Class').mean()
plt.figure(figsize=(13,10))
f = sns.violinplot(x = df['Price_Class'], y = df['Rating'], palette= 'Wistia')
f.set_xticklabels(f.get_xticklabels(), fontdict= {'fontsize':13})
f.set_xlabel('Price Class', fontdict= {'fontsize':17})
f.set_ylabel('Rating', fontdict= {'fontsize':17})
f.set_title('Rating vs Price Class',fontdict= {'fontsize':17})
plt.show()
df['Content Rating'].head()
df['Content Rating'].unique()
plt.figure(figsize=(16,7))
plt.suptitle('Content Rating Shares on playstore')
plt.subplot(1,2,1)
sns.countplot(x = df['Content Rating'], palette='summer')

plt.subplot(1,2,2)
temp = df['Content Rating'].value_counts().reset_index()
plt.pie(x = temp['Content Rating'], labels = temp['index'])

plt.show()
sns.boxplot(x = df['Content Rating'], y = df['Rating'], palette= 'hls')
df['Genres'].head()
df['Genres'].unique()
df['Genres'].value_counts()
df['Genres'] = df['Genres'].apply(lambda x: x.split(';')[0])
df['Genres'].unique()
df['Genres'].value_counts()
df['Genres'].replace('Music & Audio','Music', inplace = True)
df['Genres'].value_counts().tail()
temp = df[['Genres','Rating','Reviews']].groupby(by = 'Genres').mean().sort_values(by = 'Rating',ascending = False)
print(temp.head(1))
print(temp.tail(1))
plt.figure(figsize=(14,8))
f = sns.boxplot(x = df['Genres'], y = df['Rating'], palette= 'rainbow')
f.set_xticklabels(f.get_xticklabels(), rotation = 90)
plt.show()
df['Last Updated'].head(10)
from datetime import datetime
df['Last Updated'] = pd.to_datetime(df['Last Updated'])
df['Last Updated'].max()
df['Last Updated TimeDelta'] = df['Last Updated'].max() - df['Last Updated'] 
print(df['Last Updated TimeDelta'][0])
sns.jointplot(df['Last Updated TimeDelta'].dt.days, df['Rating'], COLOR = 'brown')
plt.show()
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot = True, cmap = 'Reds')
plt.show()
sns.regplot(x = df['Reviews'], y = df['Installs'], color = 'green')