# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df=pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
df.head()
test_list = ["Men", "Men's", "Man","Man's"]
test = ["F" if "Women" in item else "M" for item in df['title_orig']]
test2 = ["F" if "Women's" in item else "M" for item in df['title_orig']]


df['Gender'] = ['F' if 'F' in (test[i] or test2[i]) else 'M' for i,x in enumerate(test)]
df.drop(columns=['title','currency_buyer','merchant_id','merchant_has_profile_picture','merchant_profile_picture','product_url','product_picture','product_id'], inplace=True)
df.drop(columns='crawl_month',inplace=True)
print(df['urgency_text'].value_counts())
print(df['urgency_text'].isna().sum())
df['urgency_text'].replace(np.nan,'N',inplace=True)
df['urgency_text'].replace('Quantité limitée !', 'Y',inplace=True)
df['urgency_text'].replace('Réduction sur les achats en gros', 'Y', inplace=True)
df.drop(columns='has_urgency_banner',inplace=True)
df['origin_country'].value_counts()
df.replace(['US','VE','SG','GB','AT'],'Other',inplace=True)
df['origin_country'].value_counts()
df['inventory_total'] = ["Full" if ele == 50 else "Not Full" for ele in df['inventory_total']]
df.inventory_total.value_counts()
df[['merchant_title','merchant_name']][:15]
counts = pd.DataFrame(df['merchant_title'].value_counts())
df['repeat'] = ['Y' if counts.loc[ele][0] > 1 else "N" for ele in df['merchant_title']]
df.drop(columns=['merchant_title','merchant_name','merchant_info_subtitle'],inplace=True)
df.drop(columns='theme',inplace=True)
df.drop(columns='product_color',inplace=True)

items = ['Shirt','Dress','Shorts','Pants','Skirt','Sweater']
clothes_test = [np.nan]*1573
for item in items:
    for ind,ele in enumerate(df['title_orig']):
        if clothes_test[ind] is np.nan and item in ele:
            clothes_test[ind]=item
clothes = pd.DataFrame(clothes_test)
print(clothes.value_counts())
print(clothes.value_counts().sum())
df['Clothing'] = clothes
df[df['Clothing'].isna()][['title_orig','tags']]
items_round2 = ['beachwear', 'beach wear','swimsuit','romper','jumpsuit','t-shirts','blouse']
for item in items_round2:
    for ind,ele in enumerate(df['tags']):
        if clothes_test[ind] is np.nan and item in ele:
            clothes_test[ind]=item
clothes = pd.DataFrame(clothes_test)
df['Clothing'] = clothes
df['Clothing'].value_counts(dropna=False)
df[df['Clothing'].isna()][['title_orig','tags']]
items_round3 = ['bikini', 'Bikini','T-shirt','Shorts','Vest','Tank','tank']
for item in items_round3:
    for ind,ele in enumerate(df['tags']):
        if clothes_test[ind] is np.nan and item in ele:
            clothes_test[ind]=item
clothes = pd.DataFrame(clothes_test)
df['Clothing'] = clothes
df['Clothing'].value_counts(dropna=False)
df['Clothing'].replace(['T-shirt','t-shirts'], value='Shirt', inplace=True)
df['Clothing'].replace(['Vest','sweater','Sweater','blouse'], value='Blouse', inplace=True)
df['Clothing'].replace(['beachwear','Bikini', 'bikini', 'beach wear','swimsuit'], value='Swimsuit', inplace=True)
df['Clothing'].replace(['romper','jumpsuit'], value='Romper', inplace=True)
df['Clothing'].replace(['tank'], value='Tank', inplace=True)
df['Clothing'].replace(np.nan, value='Other', inplace=True)

df['Clothing'].value_counts(dropna=False)
df.drop(columns=['title_orig','tags'],inplace=True)
drops= ['merchant_rating','merchant_rating_count','shipping_is_express','shipping_option_price','product_variation_inventory',
       'badge_fast_shipping','badge_product_quality','badge_local_product','badges_count','shipping_option_name']
df.drop(columns=drops,inplace=True)
discount = ((df['price']-df['retail_price'])/df['retail_price'])*-100
df['Discount']=discount
df[['price','retail_price','Discount','units_sold']]
corr = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.title('Correlation Analysis');
df.isna().sum()
df['rating_five_count'].fillna(df['rating_five_count'].mean(), inplace=True)
df['rating_four_count'].fillna(df['rating_four_count'].mean(), inplace=True)
df['rating_three_count'].fillna(df['rating_three_count'].mean(), inplace=True)
df['rating_two_count'].fillna(df['rating_two_count'].mean(), inplace=True)
df['rating_one_count'].fillna(df['rating_one_count'].mean(), inplace=True)
df['origin_country'].fillna(df['origin_country'].mode()[0],inplace=True)
df.drop(columns='product_variation_size_id',inplace=True)
plt.figure(figsize=(18,10))
sns.distplot(df.price, label="Sale Price")
sns.distplot(df.retail_price, label = "Retail Price")
plt.legend()
plt.xlabel("EUR")
plt.title("Retail and Sale Price Distributions")
plt.show()
plt.figure(figsize=(15,5))
sns.boxplot(x=df.price)
plt.xlabel("EUR")
plt.title("Price Distribution")
plt.show()
plt.figure(figsize=(15,5))
sns.boxplot(x=df.retail_price)
plt.title("Retail Price Distribution")
plt.xlabel("EUR")
plt.show()
result_sold = df.groupby("Clothing")['units_sold'].sum().reset_index().sort_values(by='units_sold')
result_discount = df.groupby("Clothing")['Discount'].mean().reset_index().sort_values(by='Discount')
f,(ax1,ax2) = plt.subplots(1,2, figsize=(20,8))
sns.barplot(x='Clothing',y='units_sold',data=result_sold, order=result_sold['Clothing'], ax=ax1)
ax1.set_xlabel("Clothing Category")
ax1.set_ylabel("Amount Sold")
ax1.set_title("Cummulative Sales per Clothing Category")
sns.barplot(x='Clothing',y='Discount',data=df, order=result_discount['Clothing'], ax=ax2)
ax2.set_xlabel("Clothing Category")
ax2.set_ylabel("Discount %")
ax2.set_title("Discount % per Clothing Category")


df['rating_bins'] = pd.cut(df['rating'],bins=[0,1,2,3,4,5], labels=['1*','2*','3*','4*','5*'])
ratings_sold = df.groupby("rating_bins")['units_sold'].sum().reset_index().sort_values(by='units_sold')
ratings_discount = df.groupby("rating_bins")['Discount'].mean().reset_index().sort_values(by='Discount')
f,(ax1,ax2) = plt.subplots(1,2, figsize=(20,8))
sns.barplot(x='rating_bins',y='units_sold',data=ratings_sold, order=ratings_sold['rating_bins'], ax = ax1)
ax1.set_xlabel("Ratings")
ax1.set_ylabel("Amount Sold")
ax1.set_title("Cummulative Sales by Ratings")
sns.barplot(x='rating_bins', y='Discount', data=ratings_discount, order=ratings_discount['rating_bins'], ax=ax2)
ax2.set_xlabel("Ratings")
ax2.set_ylabel("Discount %")
ax2.set_title("Discount % by Ratings")

price_bins = df.groupby('rating_bins')['price'].mean().reset_index().sort_values('price')
discount_bins = df.groupby('rating_bins')['retail_price'].mean().reset_index().sort_values('retail_price')
f,(ax1,ax2) = plt.subplots(1,2, figsize=(20,8))
sns.barplot(x='rating_bins', y='price', data=price_bins, order=price_bins['rating_bins'], ax=ax1)
ax1.set_xlabel('Ratings')
ax1.set_ylabel('Sale Price')
ax1.set_title("Sale Price vs Ratings")
sns.barplot(x='rating_bins', y='retail_price', data=discount_bins, order=discount_bins['rating_bins'],ax=ax2)
ax2.set_xlabel("Ratings")
ax2.set_ylabel("Retail Price")
ax2.set_title("Retail Price vs Ratings")
sns.distplot(df['Discount'])
df[['Discount']].describe()
df['Sale'] = df['Discount']>df['Discount'].mean()
df['Sale'].replace({False:0, True:1}, inplace=True)
ratings_sold2 = df.groupby(["rating_bins", "uses_ad_boosts"])['units_sold'].sum().reset_index().sort_values(by='units_sold')
ratings_discount2 = df.groupby(["rating_bins","uses_ad_boosts"])['Discount'].mean().reset_index().sort_values(by='Discount')
f,(ax1,ax2) = plt.subplots(1,2, figsize=(20,8))
sns.barplot(x='rating_bins',y='units_sold', hue='uses_ad_boosts', data=ratings_sold2, ax = ax1)
ax1.set_xlabel("Ratings")
ax1.set_ylabel("Amount Sold")
ax1.set_title("Cummulative Sales by Ratings")
sns.barplot(x='rating_bins', y='Discount', hue='uses_ad_boosts', data=ratings_discount2, ax=ax2)
ax2.set_xlabel("Ratings")
ax2.set_ylabel("Discount %")
ax2.set_title("Discount % by Ratings")
price_bins2 = df.groupby(['rating_bins','uses_ad_boosts'])['price'].mean().reset_index().sort_values('price')
discount_bins2 = df.groupby(['rating_bins','uses_ad_boosts'])['retail_price'].mean().reset_index().sort_values('retail_price')
f,(ax1,ax2) = plt.subplots(1,2, figsize=(20,8))
sns.barplot(x='rating_bins', y='price', hue='uses_ad_boosts', data=price_bins2, ax=ax1)
ax1.set_xlabel('Ratings')
ax1.set_ylabel('Sale Price')
ax1.set_title("Sale Price vs Ratings")
sns.barplot(x='rating_bins', y='retail_price', hue='uses_ad_boosts', data=discount_bins2,ax=ax2)
ax2.set_xlabel("Ratings")
ax2.set_ylabel("Retail Price")
ax2.set_title("Retail Price vs Ratings")
ratings_sold3 = df.groupby(["rating_bins", "Sale"])['units_sold'].sum().reset_index().sort_values(by='units_sold')
ratings_discount3 = df.groupby(["rating_bins","Sale"])['Discount'].mean().reset_index().sort_values(by='Discount')
f,(ax1,ax2) = plt.subplots(1,2, figsize=(20,8))
sns.barplot(x='rating_bins',y='units_sold', hue='Sale', data=ratings_sold3, ax = ax1)
ax1.set_xlabel("Ratings")
ax1.set_ylabel("Amount Sold")
ax1.set_title("Cummulative Sales by Ratings")
sns.barplot(x='rating_bins', y='Discount', hue='Sale', data=ratings_discount3, ax=ax2)
ax2.set_xlabel("Ratings")
ax2.set_ylabel("Discount %")
ax2.set_title("Discount % by Ratings")
price_bins3 = df.groupby(['rating_bins','Sale'])['price'].mean().reset_index().sort_values('price')
discount_bins3 = df.groupby(['rating_bins','Sale'])['retail_price'].mean().reset_index().sort_values('retail_price')
f,(ax1,ax2) = plt.subplots(1,2, figsize=(20,8))
sns.barplot(x='rating_bins', y='price', hue='Sale', data=price_bins3, ax=ax1)
ax1.set_xlabel('Ratings')
ax1.set_ylabel('Sale Price')
ax1.set_title("Sale Price vs Ratings")
sns.barplot(x='rating_bins', y='retail_price', hue='Sale', data=discount_bins3,ax=ax2)
ax2.set_xlabel("Ratings")
ax2.set_ylabel("Retail Price")
ax2.set_title("Retail Price vs Ratings")
num_cols=['price','retail_price','uses_ad_boosts','rating','rating_count','countries_shipped_to','Discount','Sale']
cat_cols=['inventory_total','urgency_text','origin_country','Gender','repeat','Clothing','rating_bins']
X = df[cat_cols+num_cols]
data_map = {'inventory_total':{'Full':1, 'Not Full': 0},
            'urgency_text' : {'Y': 1, 'N':0},
            'origin_country': {'CN': 1, 'Other':0},
            'Gender' : {'F':1, 'M':0},
            'repeat' : {'Y' : 1, 'N' : 0},
            'Clothing' : {'Shirt' : 1, 'Dress':2,'Swimsuit':3,'Shorts':4,'Romper':5,'Blouse':6,'Pants':7,'Tank':8,'Other':9,'Skirt':10},
            'rating_bins' : {'1*':1,'2*':2,'3*':3,'4*':4,'5*':5}
}
num_feats = X.select_dtypes(include=["int64","float64"]).columns
num_feats
X.replace(data_map, inplace=True)
y = df[['units_sold']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_feats)])
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]
for classifier in classifiers:
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])
    pipe.fit(X_train, y_train)   
    print(classifier)
    print("model score: %.3f" % pipe.score(X_test, y_test))