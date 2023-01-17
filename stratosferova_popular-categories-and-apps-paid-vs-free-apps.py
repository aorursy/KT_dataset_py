import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data_googleplaystore = pd.read_csv('../input/googleplaystore.csv')
data_googleplaystore[(data_googleplaystore['App'] == 'Life Made WI-Fi Touchscreen Photo Frame')] #strange row
data_googleplaystore.drop(data_googleplaystore[(data_googleplaystore['App'] == 'Life Made WI-Fi Touchscreen Photo Frame')].index, inplace=True)
data_googleplaystore['Reviews'] = data_googleplaystore['Reviews'].fillna(0).astype(int) #change type of Reviews 
data_googleplaystore.info() #check
data_googleplaystore.sort_values(by='Reviews', ascending=False).head()#duplicates!
data_googleplaystore.drop_duplicates('App', keep='last', inplace=True) #drop_duplicates
data_googleplaystore.sort_values(by='Reviews', ascending=False).head() 
data_googleplaystore['Category'].value_counts()
data_googleplaystore['Category'].value_counts().head(10).plot(kind='bar')
data_googleplaystore.groupby(['Category','Genres'], as_index=False)['Reviews'].sum().sort_values(by='Category', ascending=False)
data_googleplaystore.groupby(['Category'], as_index=False)['Reviews'].sum().sort_values(by='Reviews', ascending=False).head(10)
data_googleplaystore.groupby(['Category'], as_index=False).mean().sort_values(by='Rating', ascending=False).head(10)
data_category = data_googleplaystore.groupby(['Category'], as_index=False).mean()
data_category.head()
plt.figure(figsize = (10,5))
x = data_category.Rating
y = data_category.Reviews
z = data_category.Category
rng = np.random.RandomState(0)
colors = rng.rand(33)
sizes = 500 * rng.rand(33)

plt.scatter(x, y, c=colors, s=sizes, alpha = 0.5, cmap='magma')
plt.xlabel("Rating")
plt.ylabel("Reviews")
for i, j in zip(x, y):
    plt.text(i, j, '%.1f' % i, ha='center', va='bottom')

plt.colorbar()
plt.show()
plt.figure(figsize = (15,7))

def list_for_visio(data_row):
    list_of_smth = []
    for i in data_row:
        list_of_smth.append(i)
    return list_of_smth


categories = list_for_visio(data_category.Category)
ratings = list_for_visio(data_category.Rating)
reviews = list_for_visio(data_category.Reviews)

for j, category in enumerate(categories):
    x = ratings[j]
    y = reviews[j]
    plt.scatter(x, y,  c='green', s=400, marker='H', alpha = 0.3)
    plt.text(x, y, category, fontsize=10, horizontalalignment='center', family='monospace', color='black', rotation=15)

plt.xlabel("Rating")
plt.ylabel("Reviews")
plt.show()
data_googleplaystore.groupby(['Category'], 
                             as_index=False).get_group('DATING').sort_values(by='Reviews', 
                                                                             ascending=False).head()
#the worst category in mean rating and rewiews
data_googleplaystore.groupby(['Category'], 
                             as_index=False).get_group('EVENTS').sort_values(by='Reviews', 
                                                                             ascending=False).head()
#the best category in mean rating and rewiews
data_googleplaystore.groupby(['Category']).size()
data_mm = data_googleplaystore.groupby(['Category']).agg([np.sum, np.mean, np.min, np.max])
data_mm.head(3) #nice, I doesn't help us at all)
data_mm['Reviews'].sort_values(by='sum',
ascending=False).head()
#interesting, but I want to see app+category!
pd.pivot_table(data_googleplaystore[data_googleplaystore['Rating'] > 4.0],
index=['Category', 'App'], aggfunc='mean').sort_values(by='Reviews',
ascending=False).head(10)
pd.pivot_table(data_googleplaystore[data_googleplaystore['Reviews'] > 100000], 
index=['Category', 'App'], 
aggfunc='mean').sort_values(by='Rating', 
ascending=False).head(10)
data_googleplaystore['Type'].value_counts()
data_googleplaystore[data_googleplaystore['Type'] == 'Free'].mean()
data_googleplaystore[data_googleplaystore['Type'] == 'Paid'].mean()
data_free = data_googleplaystore[data_googleplaystore['Type'] == 'Free']
data_paid = data_googleplaystore[data_googleplaystore['Type'] == 'Paid']
data_paid['Installs'].value_counts()
data_free['Installs'].value_counts()
data_paid['Price'].value_counts().head(10)
data_paid.groupby('Price').mean().dropna().sort_values(by='Reviews', ascending=False).head()
data_paid.groupby(['Category',
'App', 'Price'], as_index=False)['Reviews',
'Rating'].mean().sort_values(by='Reviews', ascending=False).head(10)
import seaborn as sns
paid_visio = data_paid.groupby(['Category',
'App', 'Price'], as_index=False)['Reviews',
'Rating'].mean().sort_values(by='Reviews', ascending=False).head(100)


plt.figure(figsize=(8,5))
sns.scatterplot(x=paid_visio.Reviews, y=paid_visio.Rating, hue=paid_visio.Price)
plt.legend(bbox_to_anchor=(1, 1), loc=0, borderaxespad=0.3)
free_visio = data_free.groupby(['Category',
'App', 'Price'], as_index=False)['Reviews',
'Rating'].mean().sort_values(by='Reviews', ascending=False).head(100)

plt.figure(figsize=(8,5))
sns.scatterplot(x=free_visio.Reviews, y=free_visio.Rating)
sns.scatterplot(x=paid_visio.Reviews, y=paid_visio.Rating, hue=paid_visio.Price)
plt.legend(bbox_to_anchor=(1, 1), loc=0, borderaxespad=0.3)
