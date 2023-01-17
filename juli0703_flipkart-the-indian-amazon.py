import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
df = pd.read_csv('../input/flipkart_com-ecommerce_sample.csv')
df.info()
df.head()
plt.figure(figsize=(15,8))
sns.heatmap(df.isnull(),
            cmap='plasma',
            yticklabels=False,
            cbar=False)
plt.title('Missing Data?',fontsize=20)
plt.xticks(fontsize=15)
plt.show()
df.duplicated().value_counts()
#make this column into a datetime type for workability

df['crawl_timestamp'] = pd.to_datetime(df['crawl_timestamp'])
df['crawl_year'] = df['crawl_timestamp'].apply(lambda x: x.year)
df['crawl_month'] = df['crawl_timestamp'].apply(lambda x: x.month)
print(df.product_category_tree[1])
print('\n')

for i in df.product_category_tree[1].split('>>'):
    print(i)
df.product_category_tree[10].split('>>')[1][1:]
#This .apply(lambda) will create a main category column from the first item in the product_category_tree column

df['MainCategory'] = df['product_category_tree'].apply(lambda x: x.split('>>')[0][2:])
#These functions will be .apply() to the df. These functions will draw the second, third and fourth items from the product_category_tree
#try except statements because an index error occurs when there is no second/third/fourth item in the product_category_tree.

def secondary(x):
    try:
        return x.split('>>')[1][1:]
    except IndexError:
        return 'None '
    
def tertiary(x):
    try:
        return x.split('>>')[2][1:]
    except IndexError:
        return 'None '
    
def quaternary(x):
    try:
        return x.split('>>')[3][1:]
    except IndexError:
        return 'None '
df['SecondaryCategory'] = df['product_category_tree'].apply(secondary)
df['TertiaryCategory'] = df['product_category_tree'].apply(tertiary)
df['QuaternaryCategory'] = df['product_category_tree'].apply(quaternary)
plt.figure(figsize=(12,9))
df.groupby('crawl_month')['crawl_month'].count().plot(kind='bar')
plt.title('Sales Count by Month',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Month',fontsize=12)
plt.ylabel('Sales Count',fontsize=12)
plt.show()
print(df.groupby('crawl_month')['crawl_month'].count())
plt.figure(figsize=(10,6))
df.groupby('crawl_year')['crawl_year'].count().plot(kind='bar')
plt.title('Sales Count by Year',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.ylabel('Sales Count',fontsize=12)
plt.show()
print(df.groupby('crawl_year')['crawl_year'].count())
plt.figure(figsize=(12,8))
df['MainCategory'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Main Category',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Main Categories by Sales.\n')
print(df['MainCategory'].value_counts()[:10])
plt.figure(figsize=(12,8))
df['SecondaryCategory'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Secondary Category',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Secondary Categories by Sales.\n')
print(df['SecondaryCategory'].value_counts()[:10])
plt.figure(figsize=(12,8))
df['TertiaryCategory'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Tertiary Category',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Tertiary Categories by Sales.\n')
print(df['TertiaryCategory'].value_counts()[:10])
df[df['TertiaryCategory']=='Western Wear ']['product_name'][60:70]
plt.figure(figsize=(12,8))
df['QuaternaryCategory'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Quaternary Category',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Quaternary Categories by Sales.\n')
print(df['QuaternaryCategory'].value_counts()[:10])
df['retail_price'].max()
df[df['retail_price']==571230.000000]
#discount percent = ((retail - sale) / retail) * 100

df['discount_%'] = round(((df['retail_price'] - df['discounted_price']) / df['retail_price'] * 100),1) 
df[['product_name','retail_price','discounted_price','discount_%']].head()
MainCategoryDiscount = pd.DataFrame(df.groupby('MainCategory').agg({
    'discount_%':[(np.mean)],
    'MainCategory':['count']
}))

SecondaryCategoryDiscount = pd.DataFrame(df.groupby('SecondaryCategory').agg({
    'discount_%':[np.mean],
    'SecondaryCategory':['count']
}))

TertiaryCategoryDiscount = pd.DataFrame(df.groupby('TertiaryCategory').agg({
    'discount_%':[np.mean],
    'TertiaryCategory':['count']
}))

QuaternaryCategoryDiscount = pd.DataFrame(df.groupby('QuaternaryCategory').agg({
    'discount_%':[np.mean],
    'QuaternaryCategory':['count']
}))
MainCategoryDiscount.head()
MainCategoryDiscount.columns = ['_'.join(col) for col in MainCategoryDiscount.columns]
SecondaryCategoryDiscount.columns = ['_'.join(col) for col in SecondaryCategoryDiscount.columns]
TertiaryCategoryDiscount.columns = ['_'.join(col) for col in TertiaryCategoryDiscount.columns]
QuaternaryCategoryDiscount.columns = ['_'.join(col) for col in QuaternaryCategoryDiscount.columns]
MainCategoryDiscount.head()
MainCategoryDiscount = MainCategoryDiscount.sort_values(by=['MainCategory_count'],ascending=False)[:20]
SecondaryCategoryDiscount = SecondaryCategoryDiscount.sort_values(by=['SecondaryCategory_count'],ascending=False)[:20]
TertiaryCategoryDiscount = TertiaryCategoryDiscount.sort_values(by=['TertiaryCategory_count'],ascending=False)[:20]
QuaternaryCategoryDiscount = QuaternaryCategoryDiscount.sort_values(by=['QuaternaryCategory_count'],ascending=False)[:20]
plt.figure(figsize=(12,8))
MainCategoryDiscount['discount_%_mean'].sort_values(ascending=True).plot(kind='barh')
plt.title('Main Category by Discount',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('MainCategory',fontsize=12)
plt.show()
print('Main Category by Discount (Percentage)\n')
print(MainCategoryDiscount['discount_%_mean'].sort_values(ascending=False)[:8])
plt.figure(figsize=(12,8))
SecondaryCategoryDiscount['discount_%_mean'].sort_values(ascending=True).plot(kind='barh')
plt.title('Secondary Category by Discount',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('SecondaryCategory',fontsize=12)
plt.show()
print('Secondary Category by Discount (Percentage)\n')
print(SecondaryCategoryDiscount['discount_%_mean'].sort_values(ascending=False)[:8])
plt.figure(figsize=(12,8))
TertiaryCategoryDiscount['discount_%_mean'].sort_values(ascending=True).plot(kind='barh')
plt.title('Tertiary Category by Discount',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('Tertiary Category',fontsize=12)
plt.show()
print('Tertiary Category by Discount (Percentage)\n')
print(TertiaryCategoryDiscount['discount_%_mean'].sort_values(ascending=False)[:8])
plt.figure(figsize=(12,8))
QuaternaryCategoryDiscount['discount_%_mean'].sort_values(ascending=True).plot(kind='barh')
plt.title('Quaternary Category by Discount',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('QuaternaryCategory',fontsize=12)
plt.show()
print('Quaternary Category by Discount (Percentage)\n')
print(QuaternaryCategoryDiscount['discount_%_mean'].sort_values(ascending=False)[:8])
#MainCategoryDiscount
