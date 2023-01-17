# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import scipy
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import gc
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Install 7zip and unzip all files
!apt install --assume-yes p7zip-full
!7z x ../input/mercari-price-suggestion-challenge/train.tsv.7z -y
!7z x ../input/mercari-price-suggestion-challenge/test.tsv.7z -y
!7z x ../input/mercari-price-suggestion-challenge/test_stg2.tsv.zip -y
!7z x ../input/mercari-price-suggestion-challenge/sample_submission.csv.7z -y
!7z x ../input/mercari-price-suggestion-challenge/sample_submission_stg2.csv.zip -y
df_train = pd.read_csv('train.tsv',sep="\t", index_col=0)
df_train.head(10)
print(df_train.name.isna().sum())
print(df_train.name.isnull().sum())
na_catg = (df_train.category_name.isna().sum()/len(df_train))*100

print('Only ',round(na_catg, 3), "% off all category_name data is missing")

df_train['category_name'] = df_train['category_name'].fillna('missing')
df_train[df_train['category_name'] == 'Electronics/Computers & Tablets/Components & Parts'].shape
def split_category_name(df):
    # function from @Chris Defreitas
    category_split = df['category_name'].str.split(r'/', n=2, expand=True)
    for i in [0,1,2]:
        df['category_' + str(i)] = category_split[i]
    return df
df_train = split_category_name(df_train)
df_train.head()
electronics = df_train[df_train['category_0'] == 'Electronics']
electronics.shape
electronics.describe()
a_brand = (electronics.brand_name.isna().sum()/len(electronics))*100

print(round(a_brand, 3), "% off all brand_name data is missing")
# get the electronic brands
_brands = electronics['brand_name'].value_counts().index.to_list()

# remove brand with only 1 character
_brands.remove('M')

def check_brand_name(name, brand):
    match = [b for b in _brands if b.lower() in name.lower()]
    if match:
        return match[0]
    return brand
df_brand_null = electronics[electronics['brand_name'].isna() == True]
for i in df_brand_null.index:
    
    name = electronics.loc[i, 'name']
    
    brand = electronics.loc[i, 'brand_name']
    
    # First we try with name columns value
    res = check_brand_name(name, brand)
    
    if pd.isna(res):
        description = electronics.loc[i, 'item_description']
        res = check_brand_name(description, brand)
        
    electronics.at[i, 'brand_name'] = res

na_brand = (electronics.brand_name.isna().sum()/len(electronics))*100

print(round(na_brand, 3), "% off all brand_name data is missing")

electronics['brand_name'].fillna('missing', inplace=True)
# plot the distribution

plt.subplot(1, 2, 1)

(electronics['price']).plot.hist(bins=50, figsize=(12, 6), edgecolor = 'white', range = [0, 200], color='g')

plt.xlabel('price', fontsize=12)
plt.title('Electronics Prices Distribution', fontsize=12)

plt.subplot(1, 2, 2)

np.log(electronics['price']+1).plot.hist(bins=50, figsize=(12,6), edgecolor='white', color='g')

plt.xlabel('log(price+1)', fontsize=12)
plt.title('Electronics Prices Distribution', fontsize=12)
plt.show()

for column in ['category_name', 'brand_name']:
    electronics[column] = electronics[column].astype('category')

cv = CountVectorizer()
X_name = cv.fit_transform(electronics["name"])

cv2 = CountVectorizer()
X_category = cv2.fit_transform(electronics["category_name"])

count_descp = TfidfVectorizer(max_features=50000,
                              ngram_range=(1, 3),
                              stop_words="english")

# X_descp = count_descp.fit_transform(electronics["item_description"])

vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(electronics["brand_name"])

# Dummy Encoders
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(electronics[[
    "item_condition_id", "shipping"]], sparse=True).values)

X_dummies = X_dummies.astype(float)

X = scipy.sparse.hstack((X_dummies, X_brand, X_category, X_name)).tocsr()
y = np.log1p(electronics['price']).values
del df_train, electronics
gc.collect()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

models = {'linear_regression': LinearRegression(),
          'random_forest': RandomForestRegressor(),
          'Ridge': Ridge()}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    error = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(name, ': ', 'MSE: ', error, 'R2: ', r2)