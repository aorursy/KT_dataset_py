import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from string import ascii_letters
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.cluster.hierarchy import linkage,dendrogram
train_df = pd.read_csv("../input/mercaritest/train.tsv", delimiter='\t')
test_df = pd.read_csv("../input/mercaritest/test.tsv", delimiter='\t')
train_df.head()
# Checking missing values
print(pd.isnull(train_df).sum())
print("------------")
print(pd.isnull(test_df).sum())
# Fill those products with No Brand with NoBrand
train_df["brand_name"] = train_df["brand_name"].fillna("NoBrand")
test_df["brand_name"] = test_df["brand_name"].fillna("NoBrand")
# Fill those products with no category with No/No/No
train_df["category_name"] = train_df["category_name"].fillna("No/No/No")
test_df["category_name"] = test_df["category_name"].fillna("No/No/No")
def split(txt):
    try :
        return txt.split("/")
    except :
        return ("No Label", "No Label", "No Label")
train_df['general_category']='' 
train_df['subcategory_1'] = '' 
train_df['subcategory_2'] = ''
# zip to make it work faster and so does lambda
train_df['general_category'],train_df['subcategory_1'],train_df['subcategory_2'] = \
zip(*train_df['category_name'].apply(lambda x: split(x)))
train_df.head()
# force Python to display entire number
pd.set_option('float_format', '{:f}'.format)

train_df.describe()
train_df.price.plot.hist(bins=50, figsize=(8,4), edgecolor='white',range=[0,300])
plt.title('Price Distribution')
np.log(train_df['price']+1).plot.hist(bins=50, figsize=(8,4), edgecolor='white')
plt.title('Price Distribution (log price +1 )')
sns.set(rc={'figure.figsize':(11.7,8.27)})

ax = sns.countplot('general_category',data=train_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Count of each general category')
ax = sns.countplot('item_condition_id',data=train_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Count of each item condition')
ax = sns.countplot(x="item_condition_id", hue="general_category", data=train_df, palette="Set3")
ax.set_title('Count of each item condition by general category')
def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]
pd.crosstab(train_df.general_category,train_df.item_condition_id).apply(lambda r: r/r.sum(), axis=1).style.apply(highlight_max,axis=1)
print("There are",train_df['brand_name'].nunique(),"brands in this dataset")
train_df.brand_name.value_counts()[:10]
top10_brands = ['NoBrand','PINK', 'Nike',"Victoria's Secret", 'LuLaRoe','Apple','FOREVER 21','Nintendo','Lululemon','Michael Kors']
# Subset those top 10 brands
df = train_df[train_df.brand_name.isin(top10_brands)]
df.pivot_table(index='brand_name',columns='item_condition_id',aggfunc={'price':'mean'}).style.apply(highlight_max,axis=1)
Apple = df[df['brand_name'] == 'Apple']
Apple[Apple['item_condition_id'] == 1].head(5)
Apple[Apple['price'] > 100].head(5)
ax = sns.boxplot(x="general_category", y="item_condition_id", data=train_df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Condition distribution by general category')

plt.tight_layout()
plt.show()
ax = sns.boxplot(x="general_category", y="price", data=train_df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_yscale('log')
ax.set_title('Price distribution by general category')

plt.tight_layout()
plt.show()
ax = sns.countplot(x="general_category", hue="shipping", data=train_df, palette="Set3")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Count of shipping by general category')
ax = sns.boxplot(x="shipping", y="price", data=train_df)
ax.set_yscale('log')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Price distribution by shipping')
ax = sns.violinplot(x='general_category', y='price', hue='shipping', data=train_df, palette="Pastel1",legend_out=False)
plt.legend(loc='lower left')
ax.set_yscale('log')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Price distribution by general category and shipping')
train_df.groupby(['subcategory_1'])["price"].mean().sort_values(ascending=False)[:10]
print("There are",train_df['subcategory_1'].nunique(),"subcategory 1 in this dataset")
train_df.groupby(['subcategory_2'])["price"].mean().sort_values(ascending=False)[:10]
print("There are",train_df['subcategory_2'].nunique(),"subcategory 2 in this dataset")
train_df.groupby(['brand_name'])["price"].mean().sort_values(ascending=False)[:10]
print("There are",train_df['brand_name'].nunique(),"brands in this dataset")
train_df.head()
train_df.item_description = train_df.item_description.fillna('Empty')
train_df['log_price'] = np.log(train_df['price']+1)

train_df['des_len'] = train_df.item_description.apply(lambda x : len(x))
df = train_df.groupby(['des_len'])['log_price'].mean().reset_index()

plt.plot('des_len','log_price', data=df, marker='o', color='mediumvioletred')
plt.show()
train_df['name_len'] = train_df.name.apply(lambda x : len(x))
df = train_df.groupby(['name_len'])['log_price'].mean().reset_index()

plt.plot('name_len','log_price', data=df, marker='o', color='mediumvioletred')
plt.show()
from keras.preprocessing.text import Tokenizer
text = np.hstack([train_df.item_description.str.lower(), 
                      train_df.name.str.lower()])
tok_raw = Tokenizer()
tok_raw.fit_on_texts(text)
train_df["seq_item_description"] = tok_raw.texts_to_sequences(train_df.item_description.str.lower())
train_df["seq_name"] = tok_raw.texts_to_sequences(train_df.name.str.lower())
train_df['desc_point'] = train_df.seq_item_description.apply(lambda x : np.linalg.norm(x))
train_df['name_point'] = train_df.seq_name.apply(lambda x : np.linalg.norm(x))
fig = plt.figure()
ax = plt.gca()
ax.scatter(train_df['desc_point'] ,train_df['price'] , c='blue', alpha=0.05)
ax.set_yscale('log')
fig = plt.figure()
ax = plt.gca()
ax.scatter(train_df['name_point'] ,train_df['price'] , c='blue', alpha=0.05)
ax.set_yscale('log')
train_df.head()
tr = train_df.drop(['train_id','brand_name','category_name','item_description','name','price','shipping'
                    ,'general_category','subcategory_1','subcategory_2','seq_item_description','seq_name'],axis=1)
model = KMeans(n_clusters = 12)
scaler = StandardScaler()
model.fit(tr)
labels = model.predict(tr)
cluster = make_pipeline(scaler,model)
train_df['cluster']=labels
clusters = pd.get_dummies(train_df['cluster'],prefix='Cluster',drop_first=False)

train_test = pd.concat([train_df,clusters],axis=1).drop('cluster',axis=1)
conditions  = pd.get_dummies(train_df['item_condition_id'],prefix='Condition',drop_first=False)

train_df = pd.concat([train_df,conditions],axis=1)
train_df.head()
general_category  = pd.get_dummies(train_df['general_category'],drop_first=True)

train_df = pd.concat([train_test,general_category],axis=1).drop('general_category',axis=1)
train_df = train_df.drop(['train_id','log_price'],axis=1)
corr = train_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True)
