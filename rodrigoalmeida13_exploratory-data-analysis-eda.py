import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/filpkart-onlineorders/OnlineOrders_of_a_ecommerce_website.csv')

df.head()
# Using datetime

df['crawl_timestamp'] = pd.to_datetime(df['crawl_timestamp'])
# Sorting rows by date

df = df.reindex(df['crawl_timestamp'].sort_values().index)
# Starting from index 0

df.reset_index(drop=True, inplace=True)
df['crawl_timestamp'][8082] - df['crawl_timestamp'][0]
df.shape
df['product_category_tree'][0]
def clear(x):

    """Removes some characters"""

    return x.replace('["', ' ').replace('"]', ' ').split('>>')
df['product_category_tree'] = df['product_category_tree'].apply(clear)
# Done!

df['product_category_tree'][0]
# Number of categories accessed

df['categories_clicked'] = df['product_category_tree'].apply(len)
df.head(8)
df['categories_clicked'].mean()
df['first_click'] = df['product_category_tree'].apply(lambda x: x[0])

df['last_click'] = df['product_category_tree'].apply(lambda x: x[-1])
df.head()
#Discount percentage

df['discount_pct'] = df['discounted_price']/(df['retail_price']/100)
bym = df.groupby(by=df['crawl_timestamp'].dt.month_name())['retail_price'].mean().sort_values()

sns.barplot(bym.index, bym.values, palette=sns.cubehelix_palette(len(bym.values)))

plt.xticks(rotation=60)

plt.title("Average gain in retail prices - per month")

plt.show()
df.groupby(by=df['crawl_timestamp'].dt.month_name())['retail_price'].mean().sort_index().plot(style='-o', color='green')
df.groupby(by=df['crawl_timestamp'].dt.hour)['retail_price'].mean().plot.bar(color='orange', label='retail_price')

df.groupby(by=df['crawl_timestamp'].dt.hour)['discounted_price'].mean().plot.bar(color='blue', label='discounted_price')

df.groupby(by=df['crawl_timestamp'].dt.hour)['retail_price'].mean().plot(color='red', label='')



plt.legend()

plt.title("Average gain in retail prices - per hour")

plt.show()

#Decide for yourself
df.groupby(by=df['crawl_timestamp'].dt.hour)['retail_price'].sum().plot.bar(color='orange', label='retail_price')

df.groupby(by=df['crawl_timestamp'].dt.hour)['discounted_price'].sum().plot.bar(color='blue')

df.groupby(by=df['crawl_timestamp'].dt.hour)['retail_price'].sum().plot(color='red', label='')



plt.legend()

plt.title("Total retail price gain - per hour")

plt.show()
category = df['last_click'].value_counts().nlargest(10).sort_values()

sns.barplot(category.values, category.index, palette=sns.cubehelix_palette(len(category.values)))

plt.show()
product = df['product_name'].value_counts().nlargest(10).sort_values()

sns.barplot(product.values, product.index, palette=sns.cubehelix_palette(len(category.values)))

plt.xticks(rotation=45, ha='right')

plt.show()
byday = df.groupby(by=df['crawl_timestamp'].dt.day_name())['retail_price'].mean().sort_values()

sns.barplot(byday.index, byday.values, palette=sns.cubehelix_palette(len(byday.values)))

plt.xticks(rotation=60)

plt.title("Average gain in retail prices - per day weekly")

plt.show()
fc = data=df['first_click'].value_counts().nlargest(10).sort_values()



sns.barplot(fc.values, fc.index,

            palette=sns.cubehelix_palette(len(fc.values)))



plt.title('First category look')

plt.show()