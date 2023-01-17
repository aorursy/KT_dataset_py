import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline

import plotly

import plotly.graph_objects as go

import plotly.express as px

import matplotlib.pyplot as plt

from matplotlib_venn import venn2, venn2_circles, venn2_unweighted

from matplotlib_venn import venn3, venn3_circles
import os

print(os.listdir("../input"))
wish=pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")

wish.info()
pd.set_option('display.max_columns', None)

wish.head()
wish.drop(columns=['title_orig','merchant_name','merchant_info_subtitle','merchant_id',

                   'merchant_profile_picture','product_url','product_picture','product_id','crawl_month','theme','currency_buyer'],inplace=True)

wish.isnull().sum()
nan_replace={'has_urgency_banner':0,'urgency_text':'N/A','origin_country':'unknown','product_color':'unknown'}

wish.fillna(nan_replace,inplace=True)

wish_cln=wish.dropna()

wish_cln.info()
price_cmp=wish_cln[['price','retail_price','units_sold']]

price_cmp.describe()

price_cmp.head()
plt.figure(figsize=(20,12))

sns.scatterplot(data=wish_cln,

               x="price",

               y='units_sold')

plt.show()
trace1 = go.Violin(y=price_cmp["price"],name='Price')

trace2 = go.Violin(y=price_cmp["retail_price"],name='Retail price')

fig=go.Figure([trace1, trace2])

fig.update_layout(

title='Comparison between price and retail price',

yaxis_title='Price(EUR)')

fig.show()
price_cmp['price_drops']=price_cmp["retail_price"]-price_cmp["price"]

plt.figure(figsize=(20,12))

sns.regplot(data=price_cmp,

           x='price_drops',

           y='units_sold')

plt.title('Prices drops versus units sold')

plt.show()
country_price=wish_cln[['units_sold','origin_country']]

country_mean_price=country_price.groupby('origin_country')['units_sold'].mean().reset_index()

country_mean_price.rename(columns={'units_sold': 'units_sold_mean'},inplace=True)
to_codes={'CN':'CHN',

         'GB':'GBR',

         'SG':'SGP',

         'US':'USA',

         'VE':'VEN'}

country_mean_price['code']=country_mean_price['origin_country'].map(to_codes)

country_mean_price
country_sales_map=px.choropleth(country_mean_price,

                       color='units_sold_mean',

                       locations='code',

                       hover_name='code',

                       color_continuous_scale=px.colors.sequential.Plasma,

                       title='Sales verses origin country')

country_sales_map.show()
color_sale=wish_cln.groupby('product_color')['units_sold'].sum()

color_sale=color_sale.reset_index().sort_values(by='units_sold',ascending=False)

color_sale
top_10_color_sale=color_sale.head(10)
fig=px.bar(data_frame=top_10_color_sale,

      x='product_color',

      y='units_sold')

fig.update_layout(title='Top 10 color sales')

fig.show()
fig=px.bar(data_frame=color_sale,

      x='product_color',

      y='units_sold')

fig.update_layout(title='All color sales')

fig.show()
rating_cols=['rating_count','rating_five_count','rating_four_count',

             'rating_three_count','rating_two_count','rating_one_count']

ratings_data=wish_cln[rating_cols+['uses_ad_boosts']]



ratings_data.groupby('uses_ad_boosts').describe()
fig = go.Figure()

for col in rating_cols:

    fig.add_trace(go.Box(x=ratings_data['uses_ad_boosts'],

                         y=ratings_data[col],

                         name=col,

                         boxmean=True,

                         boxpoints=False))

fig.update_traces(quartilemethod="exclusive")

fig.update_layout(boxmode='group',

                  title='Relations between ad boosts and rating',

                  xaxis = dict(

                  tickvals = [0,1],

                  ticktext = ['Without add boosts','With add boosts']))

fig.show()
cmp_table=wish_cln[['units_sold','rating','rating_count']]

plt.figure(figsize=(20,12))

sns.jointplot(data=cmp_table,

             x='rating',

             y='units_sold')

plt.show()

line=go.Scatter3d(x=cmp_table['rating'],

                  y=cmp_table['rating_count'],

                  z=cmp_table['units_sold'])

fig=go.Figure(line)

fig.update_layout(title='Impact of rating and rating count to sales',

                  height = 1000,

                  width = 1000,

                  scene = dict(

                  xaxis_title='rating',

                  yaxis_title='rating_count',

                  zaxis_title='units_sold'))

fig.show()
index,name=wish_cln['shipping_option_name'].factorize()

wish_cln['shipping_option_index']=index
corr_map=wish_cln[['badge_fast_shipping','shipping_option_index','shipping_option_price','shipping_is_express','countries_shipped_to']]

corr_map=corr_map.corr()

plt.figure(figsize=(20,12))

sns.heatmap(corr_map,annot=True,cmap='Blues')

plt.xticks(rotation=45,fontsize=14)

plt.yticks(rotation=45,fontsize=14)

plt.show()
from wordcloud import WordCloud



tags_for_count=[]



for x in wish_cln['tags']:

    for word in str(x).split(sep=','):

        word=word.lower()

        tags_for_count.append(word)

tags_for_count       
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(tags_for_count))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
wish_cln[wish_cln['badges_count']!=0].head(10)

badges=wish_cln[['badges_count','badge_local_product', 'badge_product_quality', 'badge_fast_shipping']]



badges_cats=[]



for i in badges.index:

    categories = ['badge_local_product', 'badge_product_quality', 'badge_fast_shipping']

    codes = badges.loc[[i],['badge_local_product', 'badge_product_quality', 'badge_fast_shipping']].values.reshape(3,).tolist()

    zipped = zip(codes,categories)

    my_cats=[]

    for m,n in list(zipped):

        my_cats.append(m*n)

    badges_cats.append(my_cats)

badges_cats = pd.Series((v[0]+v[1]+v[2] for v in badges_cats))
badges.drop(columns=['badge_local_product', 'badge_product_quality', 'badge_fast_shipping'],inplace=True)

badges['badges_cats']=badges_cats.values

badges['records']=np.ones((1514,))

badges_data=badges.groupby(['badges_count','badges_cats']).count().reset_index()

badges_data
badges_cmp=wish_cln[['title','units_sold','badges_count','badge_local_product','badge_product_quality','badge_fast_shipping']]

plt.figure(figsize=(20,20))

sns.pairplot(data=badges_cmp,kind='reg')

plt.show()
merchant_sales=wish_cln[['merchant_title','merchant_rating_count',

                         'merchant_rating','merchant_has_profile_picture','units_sold']]
merchant_sales['merchant_rating'].max()
merchant_sales['merchant_rating'].min()
bins1 = [2.9, 3.5, 4.0, np.inf]

cats1 = pd.cut(merchant_sales['merchant_rating'],bins1)

merchant_sales['merchant_raing_cats']=cats1
bins2 = [0, 250000, 900000, np.inf]

cats2 = pd.cut(merchant_sales['merchant_rating_count'],bins2)

merchant_sales['raing_count_cats']=cats2
merchant_top_50 = merchant_sales.groupby(['merchant_has_profile_picture','merchant_title','merchant_raing_cats','raing_count_cats'])['units_sold'].sum().nlargest(50).reset_index()
fig = px.bar(data_frame = merchant_top_50,

           x = 'merchant_title',

           y = 'units_sold',

           color = 'merchant_raing_cats',

           facet_col = 'merchant_has_profile_picture',

           facet_row = 'raing_count_cats',

           width = 1200, height = 800)

fig.update_layout(title = 'Top 50 merchants')

fig.show()
wish_cln_copy = wish_cln.copy()
color_sale = wish_cln_copy.groupby('product_color')['units_sold'].sum()

color_sale = color_sale.reset_index().sort_values(by = 'units_sold',ascending=False)

top_10_color_sale = color_sale.head(10)

top_10 = list(top_10_color_sale['product_color'])
wish_cln_copy['product_color'][~wish_cln_copy['product_color'].isin(top_10)]='other'
wish_cln_copy['product_color'].unique()
f = lambda x: len(x)

wish_cln_copy['tags_num'] = wish_cln_copy['tags'].apply(f)
wish_cln_copy['rating_count'].hist()
sns.scatterplot(data=wish_cln_copy,x='rating_count',y='units_sold')
wish_cln_copy["rating_count_cat"] = pd.cut(wish_cln_copy["rating_count"],

                               bins=[0, 300, 1000, np.inf],

                               labels=[1, 2, 3])
wish_cln_copy["rating_count_cat"].value_counts()
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(wish_cln_copy, wish_cln_copy["rating_count_cat"]):

    strat_train_set = wish_cln_copy.iloc[train_index]

    strat_test_set = wish_cln_copy.iloc[test_index]
strat_test_set["rating_count_cat"].value_counts() / len(strat_test_set)
strat_train_set["rating_count_cat"].value_counts() / len(strat_train_set)
for set_ in (strat_train_set, strat_test_set):

    set_.drop("rating_count_cat", axis=1, inplace=True)
wish_exp = strat_train_set.copy()
corr_matrix = wish_exp.corr()
corr_matrix["units_sold"].sort_values(ascending=False)
wish_exp['rating_three_count_prop']=wish_exp['rating_three_count']/wish_exp['rating_count']

wish_exp['rating_four_count_prop']=wish_exp['rating_four_count']/wish_exp['rating_count']

wish_exp['rating_five_count_prop']=wish_exp['rating_five_count']/wish_exp['rating_count']

wish_exp['rating_two_count_prop']=wish_exp['rating_two_count']/wish_exp['rating_count']

wish_exp['rating_one_count_prop']=wish_exp['rating_one_count']/wish_exp['rating_count']
wish_exp['drops']=wish_exp["retail_price"]-wish_exp["price"]
corr_matrix = wish_exp.corr()

corr_matrix["units_sold"].sort_values(ascending=False)
wish = strat_train_set.drop("units_sold", axis=1) # drop labels for training set

wish_labels = strat_train_set["units_sold"].copy()
wish.columns
wish_num = wish.drop(['title','tags','product_variation_size_id','product_variation_inventory',

                      'inventory_total','product_color','origin_country','urgency_text',

                      'shipping_option_name','badge_fast_shipping','shipping_option_index',

                      'merchant_title','countries_shipped_to'], axis=1)
wish_num.head()
wish_cat = wish[['product_color','origin_country','shipping_option_name']]
from sklearn.preprocessing import OneHotEncoder



cat_encoder = OneHotEncoder(sparse=False)

wish_cat_1hot = cat_encoder.fit_transform(wish_cat)

wish_cat_1hot
cat_encoder.categories_
from sklearn.preprocessing import FunctionTransformer



one_ix, two_ix, three_ix, four_ix, five_ix, rating_count_ix, retail_price_ix, price_ix= [

    list(wish.columns).index(col)

    for col in ("rating_one_count", "rating_two_count", 

                "rating_three_count", "rating_four_count",

               'rating_five_count', 'rating_count','retail_price','price')]
def add_extra_features(X):

    rating_one_count_prop = X[:,one_ix]/ X[:,rating_count_ix]

    rating_two_count_prop = X[:,two_ix]/ X[:,rating_count_ix]

    rating_three_count_prop = X[:,three_ix]/ X[:,rating_count_ix]

    rating_four_count_prop = X[:,four_ix]/ X[:,rating_count_ix]

    rating_five_count_prop = X[:,five_ix]/ X[:,rating_count_ix]

    drops = X[:,retail_price_ix] - X[:,price_ix]

    return np.c_[X, rating_one_count_prop, rating_two_count_prop, 

                 rating_three_count_prop,rating_four_count_prop, 

                 rating_five_count_prop,drops]



attr_adder = FunctionTransformer(add_extra_features, validate=False)

wish_extra_attribs = attr_adder.transform(wish.values)
attr_adder
wish_extra_attribs = pd.DataFrame(

    wish_extra_attribs,

    columns = list(wish.columns)+["rating_one_count_prop", "rating_two_count_prop",

                               'rating_three_count_prop','rating_four_count_prop',

                               'rating_five_count_prop','drops'],

    index = wish.index)

wish_extra_attribs.head()
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False))])



wish_num_tr = num_pipeline.fit_transform(wish_num)

wish_num_tr
# Building the full pipeline to preprocessing numerical and categorical features.

from sklearn.compose import ColumnTransformer

num_attribs = list(wish_num)

cat_attribs = list(wish_cat)



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])



wish_prepared = full_pipeline.fit_transform(wish)
wish_prepared
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from xgboost.sklearn import XGBClassifier 

from sklearn.model_selection import KFold,cross_val_score
base_models = [('DT_model',DecisionTreeClassifier(random_state=42)),

            ('RF_model',RandomForestClassifier(random_state=42,n_jobs=-1)),

            ('LR_model',LogisticRegression(random_state=42,n_jobs=-1)),

            ("XGB_model", XGBClassifier(random_state=42, n_jobs=-1))]

# split data into 'kfolds' parts for cross validation,

# use shuffle to ensure random distribution of data:

kfolds = 4

split = KFold(n_splits=kfolds,shuffle=True,random_state=42)



# Preprocessing, fitting, making predictions and scoring for every model:

for name,model in base_models:

    model_steps = Pipeline(steps=[('model',model)])

    model_steps.fit(wish_prepared, wish_labels)

    cv_results = cross_val_score(model_steps,wish_prepared,wish_labels,cv=split,scoring='accuracy',

                              n_jobs=-1)

    # output:

    min_score = round(min(cv_results),4)

    max_score = round(max(cv_results),4)

    mean_score = round(np.mean(cv_results),4)

    std_dev = round(np.std(cv_results),4)

    print(f'{name} cross validation accuracy score:{mean_score} +- {std_dev} (std) min:{min_score},max:{max_score}')



XGB_clf = XGBClassifier(random_state=42,n_jobs=-1)



X_test = strat_test_set.drop("units_sold", axis=1)

y_test = strat_test_set["units_sold"].copy()

X_test_prepared = full_pipeline.transform(X_test)



kfolds = 4

split = KFold(n_splits=kfolds,shuffle=True,random_state=42)



cv_results = cross_val_score(XGB_clf,X_test_prepared,y_test,cv=split,scoring='accuracy',

                              n_jobs=-1)

min_score = round(min(cv_results),4)

max_score = round(max(cv_results),4)

mean_score = round(np.mean(cv_results),4)

std_dev = round(np.std(cv_results),4)

print(f'XGB_model cross validation accuracy score:{mean_score} +- {std_dev} (std) min:{min_score},max:{max_score}')

RF_clf = RandomForestClassifier(random_state=42,n_jobs=-1)



X_test = strat_test_set.drop("units_sold", axis=1)

y_test = strat_test_set["units_sold"].copy()

X_test_prepared = full_pipeline.transform(X_test)



kfolds=4

split=KFold(n_splits=kfolds,shuffle=True,random_state=42)



cv_results=cross_val_score(RF_clf,X_test_prepared,y_test,cv=split,scoring='accuracy',

                              n_jobs=-1)

min_score=round(min(cv_results),4)

max_score=round(max(cv_results),4)

mean_score=round(np.mean(cv_results),4)

std_dev=round(np.std(cv_results),4)

print(f'RF_model cross validation accuracy score:{mean_score} +- {std_dev} (std) min:{min_score},max:{max_score}')

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

import numpy as np



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

param_distribs = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



forest_clf = RandomForestClassifier(random_state=42, n_jobs=-1)

rnd_search = RandomizedSearchCV(forest_clf, param_distributions=param_distribs,

                                n_iter=5, cv=4,scoring="accuracy", random_state=42)

rnd_search.fit(wish_prepared, wish_labels)
rnd_search.best_params_
# final_model = grid_search.best_estimator_

final_RF_clf = RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators= 1366,

 min_samples_split= 5,

 min_samples_leaf= 1,

 max_features='sqrt',

 max_depth= 30,

 bootstrap= True)



X_test = strat_test_set.drop("units_sold", axis=1)

y_test = strat_test_set["units_sold"].copy()

X_test_prepared = full_pipeline.transform(X_test)



kfolds=4

split=KFold(n_splits=kfolds,shuffle=True,random_state=42)



cv_results=cross_val_score(final_RF_clf,X_test_prepared,y_test,cv=split,scoring='accuracy',

                              n_jobs=-1)

min_score=round(min(cv_results),4)

max_score=round(max(cv_results),4)

mean_score=round(np.mean(cv_results),4)

std_dev=round(np.std(cv_results),4)

print(f'Final_RF_model cross validation accuracy score:{mean_score} +- {std_dev} (std) min:{min_score},max:{max_score}')
