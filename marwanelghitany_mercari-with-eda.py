import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



import gc

import scipy

from wordcloud import WordCloud



## Text Cleaning

import string

from nltk.stem import SnowballStemmer, WordNetLemmatizer

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS



from collections import Counter, OrderedDict

## Text Embedding

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import LabelBinarizer





# dimensionality reduction

from sklearn.decomposition import PCA, TruncatedSVD



## Modelling

from sklearn.linear_model import Ridge



plt.style.use('ggplot')

%matplotlib inline
PATH = "../input/"
df_train = pd.read_csv(f'{PATH}train.tsv', sep='\t')

df_test = pd.read_csv(f'{PATH}test_stg2.tsv', sep='\t')
print(f"Train consists of {df_train.shape[0]:,} observations\nTest consists of {df_test.shape[0]:,} observations")
df_train['price'].describe()
df_train['price'].isnull().sum()
print("There Exist {} items sold for Free!".format(df_train[df_train['price'] == 0].shape[0]))
fig, ax = plt.subplots(2,figsize=(11,7))



sns.distplot( df_train[df_train['price'] <  df_train['price'].quantile(0.99)]['price'], ax=ax[0], 

             label='Price', color="green")



sns.distplot( np.log1p(df_train[df_train['price'] <  df_train['price'].quantile(0.99)]['price']), ax=ax[1],

             label="log Price", color="blue")



ax[0].legend(loc=0)

ax[1].legend(loc=0)

ax[0].set_title("")

ax[1].set_title("")

ax[0].axis("off")

ax[1].axis("off")

plt.show()
nrow_train = df_train.shape[0]

y_train = np.log1p(df_train['price'])
null_df = pd.DataFrame(df_train.dtypes).T.rename(index={0: 'dtype'})

null_df = null_df.append(pd.DataFrame(df_train.isnull().sum()).T.rename(index={0: 'count'}))

null_df = null_df.append(pd.DataFrame(df_train.isnull().sum() /df_train.shape[0] * 100 ).T.rename(index={0: '%'}))

null_df
df_train['brand_name'] = df_train['brand_name'].fillna("Other")

df_train['item_description'] = df_train['item_description'].fillna("No description yet")

df_train['item_description'] = df_train['item_description'].astype(str)

df_train['shipping'] = df_train['shipping'].astype('category')

df_train['item_condition_id'] = df_train['item_condition_id'].astype('category')



df_test['brand_name'] = df_test['brand_name'].fillna("Other")

df_test['item_description'] = df_test['item_description'].fillna("No description yet")

df_test['item_description'] = df_test['item_description'].astype(str)

df_test['shipping'] = df_test['shipping'].astype('category')

df_test['item_condition_id'] = df_test['item_condition_id'].astype('category')



df_train['logprice'] = y_train



df_train.duplicated().sum()
plt.figure(figsize=(11,7))



sns.kdeplot(df_train[df_train['shipping'] == 1].loc[:, 'logprice'], shade=True, color="g", bw=.09, label="Seller")



sns.kdeplot(df_train[df_train['shipping'] == 0].loc[:, 'logprice'], shade=True, color="r", bw=.09, label="Buyer")



plt.show()
plt.figure(figsize=(15,7))

ax = sns.boxplot(x='shipping', y='logprice', data=df_train)

ax.set_xticklabels(["buyer", "seller"])

plt.ylabel("Log (price)")

plt.xlabel("Item Condition")



plt.title("Shipping and price interaction")

plt.show()
brand_cat = df_train.groupby(by='brand_name').agg({'price':np.median})

brand_cat = pd.DataFrame(brand_cat).rename(columns={1:'price'}).reset_index()

brand_cat = brand_cat.sort_values('price', ascending=False)
plt.figure(figsize=(15,7))

ax = sns.scatterplot(x='price', y='brand_name', data=brand_cat[:25], color='r')

ax.invert_yaxis()

plt.title("Most 25 Expensive Brands")

plt.show()
len(brand_cat)
d = dict()

for brand, pr in zip(brand_cat.brand_name[:200], brand_cat.price[:200]):

    d[brand] = pr
cloud = WordCloud(width=1440, height=1080, background_color='lightgrey', colormap='cividis', random_state=42).generate_from_frequencies(frequencies=d)

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')

plt.show()
conditions = df_train['item_condition_id'].value_counts()
plt.figure(figsize=(15,7))

ax = sns.countplot(y = df_train['item_condition_id'])

ax.set_xticklabels([int(x/1000) for x in ax.get_xticks()])

plt.ylabel("Item Condition")

plt.xlabel("Freq (000s)")

plt.title("Frequency of Each item Condition")

plt.show()
plt.figure(figsize=(15,7))

sns.boxplot(x='item_condition_id', y='logprice', data=df_train)

plt.ylabel("log(price)")

plt.xlabel("Item Condition")

plt.title("Item Condition Iteraction with price")

plt.show()
df_train.category_name.head()
def split_categ(c):

        try:

            c1, c2, c3 = c.split("/")

            return c1, c2, c3

        except:

            return ("No label","No label","No label")
df_train['cat1'], df_train['cat2'], df_train['cat3']= zip(*df_train.category_name.apply(split_categ))

df_test['cat1'], df_test['cat2'], df_test['cat3']= zip(*df_test.category_name.apply(split_categ))
plt.figure(figsize=(15,7))

ax = sns.countplot(df_train.cat1)

ax.set_yticklabels([int(y/1000) for y in ax.get_yticks()])

plt.ylabel("Freq (000s)")

plt.xlabel("Main Category classes")

plt.title("Most Category popular")

plt.show()
plt.figure(figsize=(15,7))

sns.boxplot(x='cat1', y='logprice', data=df_train)

plt.ylabel("log(price)")

plt.xlabel("Item Category")

plt.title("Item Category Iteraction with price")

plt.show()
cat2S = df_train.groupby('cat2')['cat2'].count()

cat2S = pd.DataFrame(cat2S).rename(columns={'cat2': 'freq'}).reset_index()

cat2S = cat2S.sort_values(by='freq', ascending=False)
len(cat2S)
plt.figure(figsize=(15,7))

ax = sns.scatterplot(y='cat2', x='freq', data=cat2S.head(25))

ax.invert_yaxis()

ax.set_xticklabels([int(x/1000) for x in ax.get_xticks()])

plt.xlabel("Freq (000s)")

plt.ylabel("Sub Category classes")

plt.title("Most 25 Sub Category popular")

plt.show()
cat3S = df_train.groupby('cat3')['cat3'].count()

cat3S = pd.DataFrame(cat3S).rename(columns={'cat3': 'freq'}).reset_index()

cat3S = cat3S.sort_values(by='freq', ascending=False)
plt.figure(figsize=(15,7))

ax = sns.scatterplot(y='cat3', x='freq', data=cat3S.head(25))

ax.invert_yaxis()

ax.set_xticklabels([int(x/1000) for x in ax.get_xticks()])

plt.xlabel("Freq (000s)")

plt.ylabel("Sub Category classes")

plt.title("Most 25 Sub-Sub Category popular")

plt.show()
cloud = WordCloud(width=1440, height=1080, background_color='lightgrey', colormap='cividis', random_state=4).generate(" ".join(df_train['cat3'].astype(str)[:200]))



plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')

plt.show()
vectorizer = TfidfVectorizer(max_df=0.9, max_features=1_000, stop_words = "english")
X_desc_train = vectorizer.fit_transform(df_train['item_description'])

X_desc_test  = vectorizer.transform(df_test['item_description'])
print("We have {} keywords from our Description Corpus".format(len(vectorizer.get_feature_names())))
print("Tf-IDF Matrix shape {}, the Original matrix shape {:,}".format(X_desc_train.get_shape(), df_train.shape[0]))
NUM_TOP_BRANDS = 2500

top_brands = df_train['brand_name'].value_counts().index[:NUM_TOP_BRANDS]

df_train.loc[~df_train['brand_name'].isin(top_brands), 'brand_name'] = "Other"



brand_lv = LabelBinarizer(sparse_output=True)

X_brand_train  = brand_lv.fit_transform(df_train['brand_name'])

X_brand_test   = brand_lv.transform(df_test['brand_name'])
X_brand_train.get_shape() , X_brand_test.get_shape() 
MIN_NAME_DF = 10

name_cv = CountVectorizer(min_df=MIN_NAME_DF)

X_name_train  = name_cv.fit_transform(df_train['name']) 

X_name_test   = name_cv.transform(df_test['name']) 
X_name_train.shape, X_name_test.shape
cat1_cv = CountVectorizer()

X_cat1_train  = cat1_cv.fit_transform(df_train['cat1'])

X_cat1_test   = cat1_cv.transform(df_test['cat1'])
X_cat1_train.shape, X_cat1_test.shape
cat2_cv = CountVectorizer()

X_cat2_train  = cat2_cv.fit_transform(df_train['cat2'])

X_cat2_test   = cat2_cv.transform(df_test['cat2'])
X_cat2_train.shape, X_cat2_test.shape
X_dummies_train = scipy.sparse.csr_matrix( pd.get_dummies(df_train[['shipping', 'item_condition_id']], sparse=True).values)

X_dummies_test  = scipy.sparse.csr_matrix( pd.get_dummies(df_test[['shipping', 'item_condition_id']], sparse=True).values)
X_dummies_train.shape, X_dummies_test.shape
X_train = scipy.sparse.hstack((

                        X_dummies_train,

                        X_desc_train,

                        X_name_train,

                        X_brand_train,

                        X_cat1_train,

                        X_cat2_train

                        )).tocsr()



X_test = scipy.sparse.hstack((

                        X_dummies_test,

                        X_desc_test,

                        X_name_test,

                        X_brand_test,

                        X_cat1_test,

                        X_cat2_test

                        )).tocsr()
X_train.shape, X_test.shape
model = Ridge(solver='lsqr', fit_intercept=False)

model.fit(X_train, y_train)
preds = model.predict(X_test)
df_test['price'] = np.expm1(preds)
df_test[['test_id', 'price']].to_csv('ridge_submission_v2.csv', index=False)