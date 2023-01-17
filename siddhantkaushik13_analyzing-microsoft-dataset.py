import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df_msft = pd.read_csv('../input/windows-store/msft.csv')
df_msft.head(10)
df_msft.isnull().any()
df_msft.isnull().sum()
df_msft.isnull()
df_msft.drop([5321],axis = 0, inplace = True)
df_msft.isnull().any().any()
df_msft.shape
df_msft.describe()
df_msft['Price'] = df_msft['Price'].replace('Free', '0')
df_msft['Price'] = df_msft['Price'].apply(lambda x: str(x).replace('₹', '') if '₹' in str(x) else str(x))
df_msft['Price'] = df_msft['Price'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))
df_msft['Price'] = df_msft['Price'].astype(float)
df_msft['Price']
sns.heatmap(df_msft.corr(), annot = True, fmt = '1f', cmap='YlGnBu')
df_msft.hist(figsize=(13,8))
data_agg = df_msft.groupby(['Category'])
rating = data_agg['Rating'].agg(np.mean)
price = data_agg['Price'].agg(np.sum)
num_of_people_rated = data_agg['No of people Rated'].agg(np.mean)
print(rating)
print(price)
plt.plot(price, 'rp')
plt.xticks(rotation = 90)
plt.xlabel('Categories of Books')
plt.ylabel('Price')
plt.show()
plt.plot(rating, 'go')
plt.xticks(rotation=90)
plt.xlabel('Categories of Books')
plt.ylabel('Rating')
plt.show()
plt.plot(num_of_people_rated, 'bo')
plt.xticks(rotation=90)
plt.xlabel('Categories of Books')
plt.ylabel('Number Of People rated')
plt.show()
sns.jointplot(x = 'Rating', y = 'No of people Rated', kind='kde', data=df_msft)
plt.figure(figsize=(15,8))
sns.countplot(x = 'Category', data=df_msft)
plt.xticks(rotation = 20)


