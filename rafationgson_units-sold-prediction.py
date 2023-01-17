import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')

df.head()
product_cat = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv')

product_cat.head()
product_cat_results = product_cat.groupby('keyword')['count'].mean().sort_values(ascending=False)

product_cat_results.head(60)
df.info()
import missingno as msno

msno.matrix(df)
df.describe()
## Duplicate values



duplicate_series = df['product_id'].duplicated(keep='first') 

df[duplicate_series]
df.drop_duplicates(subset ="product_id", keep = 'first', inplace = True)
df.isnull().sum()
print(df['units_sold'].value_counts())

df['units_sold'].hist();
df_num = df[['price', 'retail_price', 'units_sold', 'rating_count', 'rating_five_count', 'rating_four_count',

             'rating_three_count', 'rating_two_count', 'rating_one_count', 'badges_count', 'product_variation_inventory', 

             'inventory_total', 'shipping_option_price', 'countries_shipped_to', 'merchant_rating_count']]



df_cat = df[['title', 'title_orig','currency_buyer','uses_ad_boosts', 'rating','badge_local_product', 

             'badge_product_quality', 'badge_fast_shipping','tags', 'product_color', 'product_variation_size_id',

             'shipping_option_name', 'shipping_is_express', 'countries_shipped_to','has_urgency_banner', 

             'urgency_text', 'origin_country', 'merchant_title', 'merchant_name', 'merchant_info_subtitle',

             'merchant_rating','merchant_id', 'merchant_has_profile_picture','product_url', 'product_picture',

             'product_id', 'theme', 'crawl_month']]
df_cat.head()
df_num.head()
corr_matrix = df_num.corr()

fig = plt.figure(figsize=(19, 15))

sns.heatmap(corr_matrix, annot=True);
sns.pairplot(data=df_num,

                  x_vars=['price', 'retail_price', 'rating_count', 'badges_count', 'inventory_total'],

                  y_vars=['units_sold']);
sns.pairplot(data=df_num,

                  x_vars=['product_variation_inventory', 'shipping_option_price',

                          'countries_shipped_to', 'merchant_rating_count' ],

                  y_vars=['units_sold']);
df_num.columns
def hist_num(x):

    print(df_num[x].value_counts())

    df_num[x].hist()
hist_num('price')
hist_num('retail_price')
hist_num('rating_count')
hist_num('rating_five_count')
hist_num('rating_four_count')
hist_num('rating_three_count')
hist_num('rating_two_count')
hist_num('rating_one_count')
hist_num('badges_count')
hist_num('product_variation_inventory')
hist_num('inventory_total')
hist_num('shipping_option_price')
hist_num('countries_shipped_to')
hist_num('merchant_rating_count')
for i in df_cat.columns[2:8]:

    cat_num = df_cat[i].value_counts()

    print("graph for %s: total = %d" % (i, len(cat_num)))

    chart = sns.barplot(x=cat_num.index, y=cat_num)

    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

    plt.show()
for i in df_cat[['product_color', 'product_variation_size_id','shipping_option_name', 'shipping_is_express', 

                'countries_shipped_to', 'origin_country','merchant_has_profile_picture','theme']]:

    cat_num = df_cat[i].value_counts()

    print("graph for %s: total = %d" % (i, len(cat_num)))

    chart = sns.barplot(x=cat_num.index, y=cat_num)

    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

    plt.show()
df_cat.columns
df.groupby('uses_ad_boosts')['units_sold'].mean()
rating_results = df.groupby('rating')['units_sold'].mean().sort_values(ascending=False)

rating_results.head(15)
df.groupby('badge_local_product')['units_sold'].mean().sort_values(ascending=False)
df.groupby('badge_product_quality')['units_sold'].mean().sort_values(ascending=False)
df.groupby('badge_fast_shipping')['units_sold'].mean().sort_values(ascending=False)
color_results = df.groupby('product_color')['units_sold'].mean().sort_values(ascending=False)

color_results.head(30)
variation_size_id_results = df.groupby('product_variation_size_id')['units_sold'].mean().sort_values(ascending=False)

variation_size_id_results.head(20)
df.groupby('shipping_option_name')['units_sold'].mean().sort_values(ascending=False)
for col in df[['countries_shipped_to','has_urgency_banner', 'urgency_text', 'origin_country','merchant_title', 

               'merchant_name', 'merchant_info_subtitle','merchant_rating', 'merchant_id', 'merchant_has_profile_picture',

               'merchant_profile_picture', 'product_url', 'product_picture','product_id']]:

    result = df.groupby(col)['units_sold'].mean().sort_values(ascending=False)

    print(result.head(20))
df.isnull().sum()
df.loc[(df['rating_five_count'].isnull()) & (df['rating_four_count'].isnull()) 

       & (df['rating_three_count'].isnull()) & (df['rating_two_count'].isnull()) 

       & (df['rating_one_count'].isnull()), 'rating'] = 0
df.loc[df['rating'] == 0, 'rating_five_count'] = 0

df.loc[df['rating'] == 0, 'rating_four_count'] = 0

df.loc[df['rating'] == 0, 'rating_three_count'] = 0

df.loc[df['rating'] == 0, 'rating_two_count'] = 0

df.loc[df['rating'] == 0, 'rating_one_count'] = 0
df['rating'].hist()
df.loc[df['has_urgency_banner'].isnull(), 'has_urgency_banner'] = 0
df.groupby('has_urgency_banner')['units_sold'].mean()
df['urgency_text'].value_counts()
df.loc[df['urgency_text'].isnull(), 'urgency_text'] = 'No Text'
df.isnull().sum()
df.groupby('urgency_text')['units_sold'].mean()
product_cat_results.head(60)
product_cat_results.tail(60)
def title_simplifier(title):

    if 'dress' in title.lower() or 'halter' in title.lower():

        return 'Dress'

    elif 'swimwear' in title.lower() or 'swimming' in title.lower() or 'swimsuit' in title.lower() or 'bikini' in title.lower() or 'tankini' in title.lower() or 'swim' in title.lower() or 'beach' in title.lower():

        return 'Swimwear'

    elif 'pant' in title.lower() or 'legging' in title.lower() or 'jean' in title.lower() or 'trouser' in title.lower():

        return 'Pants'

    elif 'short' in title.lower():

        return 'Shorts'

    elif 'skirt' in title.lower():

        return 'Skirt'

    elif 'top' in title.lower() or 'blouse' in title.lower() or 'shirt' in title.lower() or 'sweatshirt' in title.lower() or 'sweater' in title.lower() or 'vest' in title.lower() or 'tank top' in title.lower():

        return 'Top'

    elif 'sport' in title.lower() or 'yoga' in title.lower() or 'fitness' in title.lower() or 'running' in title.lower() or 'athletic' in title.lower(): 

        return 'Sportswear'

    elif 'romper' in title.lower() or 'jumpsuit' in title.lower() or 'overalls' in title.lower() or 'bodysuit' in title.lower():

        return 'Onepiece'

    elif 'shoe' in title.lower() or 'slipper' in title.lower() or 'sneaker' in title.lower():

        return 'Footwear'

    elif 'pajama' in title.lower() or 'pyjama' in title.lower() or 'sleep' in title.lower() or 'sleepwear' in title.lower():

        return 'Sleepwear'

    else:

        return 'Accessories'
df['title_simple'] = df['title_orig'].apply(lambda x: title_simplifier(x))
df['title_simple'].value_counts()
df.groupby('title_simple')['units_sold'].mean().sort_values(ascending=False)
def fashion_category(title):

    if "women's fashion" in title.lower() or 'women fashion' in title.lower() or 'women' in title.lower():

            return "Women's Fashion"

    else:

        return "Men's Fashion"
df['product_category'] = df['title_orig'].apply(lambda x: fashion_category(x))
df['product_category'].value_counts()
color_results.tail(60)
def color_simplify(title):

    if '&' in title.lower():

        return 'two-colors'

    elif 'green' in title.lower() or 'army' in title.lower():

        return 'green'

    elif 'navy' in title.lower() or 'blue' in title.lower():

        return 'blue'

    elif 'burgundy' in title.lower() or 'red' in title.lower() or 'wine' in title.lower():

        return 'red'

    elif 'rosegold' in title.lower() or 'pink' in title.lower():

        return 'pink'

    elif 'white' in title.lower():

        return 'white'

    elif 'black' in title.lower():

        return 'black'

    elif 'grey' in title.lower() or 'gray' in title.lower():

        return 'grey'

    elif 'yellow' in title.lower():

        return 'yellow'

    elif 'orange' in title.lower():

        return 'orange'

    elif 'khaki' in title.lower() or 'beige' in title.lower():

        return 'beige'

    elif 'multicolor' in title.lower() or 'rainbow' in title.lower():

        return 'multicolor'

    elif 'brown' in title.lower() or 'tan' in title.lower() or 'camel' in title.lower() or 'coffee' in title.lower():

        return 'brown'

    elif 'violet' in title.lower():

        return 'violet'

    else:

        return 'others'
df.loc[df['product_color'].isnull(), 'product_color'] = 'others'
df['color_simple'] = df['product_color'].apply(lambda x: color_simplify(x))
df['color_simple'].value_counts()
df.groupby('color_simple')['units_sold'].mean().sort_values(ascending=False)
df['origin_country'] = df['origin_country'].replace(np.nan, 'Other')

df['origin_country'] = df['origin_country'].replace('VE', 'Other')

df['origin_country'] = df['origin_country'].replace('SG', 'Other')

df['origin_country'] = df['origin_country'].replace('GB', 'Other')

df['origin_country'] = df['origin_country'].replace('AT', 'Other')
df['origin_country'].value_counts()
df['units_sold'] = df['units_sold'].replace(1, 10)

df['units_sold'] = df['units_sold'].replace(8, 10)

df['units_sold'] = df['units_sold'].replace(7, 10)

df['units_sold'] = df['units_sold'].replace(3, 10)

df['units_sold'] = df['units_sold'].replace(2, 10)

df['units_sold'] = df['units_sold'].replace(6, 10)
df['units_sold'].value_counts()
predictors = ['price', 'retail_price', 'rating', 'rating_count', 'rating_five_count',

              'rating_four_count','rating_three_count', 'rating_two_count', 'rating_one_count',

              'badges_count', 'product_variation_inventory', 'shipping_option_price',

              'countries_shipped_to', 'inventory_total', 'merchant_rating_count', 'merchant_rating',

              'units_sold', 'uses_ad_boosts','badge_local_product', 'badge_product_quality', 

              'badge_fast_shipping','shipping_is_express', 'has_urgency_banner',

              'origin_country', 'merchant_has_profile_picture','title_simple', 'product_category', 'color_simple']
df[predictors].isnull().sum()
df_model = df[predictors]
df_model.head()
df_model.iloc[:,0:16]
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

df_model_scaled = scaler.fit_transform(df_model.iloc[:,0:16])
df_model_scaled
df_model.iloc[:,0:16] = df_model_scaled
df_model.head()
df_dum = pd.get_dummies(df_model, columns=['uses_ad_boosts', 'badge_local_product', 'badge_product_quality',

                                           'badge_fast_shipping', 'shipping_is_express',

                                           'has_urgency_banner', 'origin_country', 'merchant_has_profile_picture',

                                           'title_simple', 'product_category', 'color_simple'])
df_dum.head()
from sklearn.model_selection import train_test_split



X = df_dum.drop('units_sold', axis =1)

y = df_dum.units_sold.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import Ridge, Lasso

from sklearn.model_selection import cross_val_score



#Linear Regression

ridge_model = Ridge(alpha=1)

ridge_model.fit(X_train, y_train)
np.mean(cross_val_score(ridge_model,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
#Lasso regression (normalizes sparse data)



lasso_model = Lasso(alpha=0.13)

lasso_model.fit(X_train,y_train)
np.mean(cross_val_score(lasso_model,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)

elastic_net.fit(X_train,y_train)
np.mean(cross_val_score(elastic_net,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()



np.mean(cross_val_score(forest_model,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))
from sklearn.ensemble import GradientBoostingRegressor



GB_model = GradientBoostingRegressor()

np.mean(cross_val_score(GB_model,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))
forest_model.fit(X_train,y_train)
test_pred_forest = forest_model.predict(X_test)
from sklearn.metrics import mean_absolute_error



mean_absolute_error(y_test, test_pred_forest)
from sklearn.metrics import r2_score



r2_score(y_test,test_pred_forest)
def feature_importance(model, data):

    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns})

    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]

    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))

    return fea_imp
feature_importance(forest_model, X_train)