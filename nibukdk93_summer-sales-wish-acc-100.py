import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from plotly.offline import iplot, init_notebook_mode

import cufflinks as cf
import plotly.graph_objs as go
# import chart_studio.plotly as py

init_notebook_mode(connected=True)
cf.go_offline(connected=True)

# Set global theme
cf.set_config_file(world_readable=True, theme='ggplot')
df = pd.read_csv("/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")
uniuqe_categories = pd.read_csv("/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.csv")
uniuqe_categories_count = pd.read_csv("/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv")
df.head(2)
print(df.info())
print(df.loc[:,df.columns.str.startswith("merchant")].columns.values)
df.drop(['merchant_has_profile_picture', 'merchant_profile_picture','merchant_title' ,'merchant_name', 'merchant_info_subtitle'], inplace=True, axis=1)
df.isnull().sum()
df.loc[:,df.isnull().sum()>0].columns

df.theme.value_counts()
df.drop('theme', axis=1, inplace=True)
df.drop("title", axis=1, inplace=True)

df.drop(['urgency_text','has_urgency_banner'], inplace=True,axis=1)
df.fillna(value="unknown", inplace=True)
df.currency_buyer.unique()
df.drop('currency_buyer', inplace=True, axis=1)
df.crawl_month.unique()
df.drop('crawl_month', inplace=True, axis=1)
df.loc[:,df.columns.str.startswith('badge')].columns
df[['badge_local_product', 'badge_product_quality','badge_fast_shipping']] = df[['badge_local_product', 'badge_product_quality','badge_fast_shipping']].astype(str)
eda_df = df.copy()
eda_df.origin_country = eda_df.origin_country.str.replace( 'CN',"China" )
eda_df.origin_country = eda_df.origin_country.str.replace( "US","United States of America" )
eda_df.origin_country = eda_df.origin_country.str.replace( "unknown","unknown" )
eda_df.origin_country = eda_df.origin_country.str.replace( "VE","Venezuela" )
eda_df.origin_country = eda_df.origin_country.str.replace( 'GB',"Great Britain" )
eda_df.origin_country = eda_df.origin_country.str.replace( 'SG',"Singapore" )
eda_df.origin_country = eda_df.origin_country.str.replace( 'AT',"Austria" )
    
labels = eda_df.origin_country.value_counts(normalize=True).index.values

values  = eda_df.origin_country.value_counts().values

# Create Pie Chart

fig = go.Figure()
fig.add_trace(go.Pie(labels=labels, values=values))
fig.update_layout(title="Country of Origin of Product in Wish", legend_title="Countries", template="plotly_dark")


# Lets create so called discounts column by subtracting the price from  retail_price

eda_df['discounted_price'] = eda_df['retail_price'] - eda_df['price']
prices_by_country = eda_df[['price','discounted_price','retail_price','origin_country']].groupby('origin_country').mean()
fig = go.Figure()

fig.add_trace(go.Bar(x=prices_by_country.index.values, y=prices_by_country.price, name="Price"))
fig.add_trace(go.Scatter(x=prices_by_country.index.values, y=prices_by_country.discounted_price, name="Discounted Price"))
fig.add_trace(go.Bar(x=prices_by_country.index.values, y=prices_by_country.retail_price, name="Retail Price"))
fig.update_layout(title="Prices Categories By Country", xaxis_title="Countries", yaxis_title="Avg Discount Prices", template="plotly_dark", legend_title="Legend")

eda_df[eda_df.origin_country=="China"]['price'].describe()
layout=dict(title="Selling Price Ranges In China", xaxis_title="Prices", yaxis_title="Frequency",)
eda_df[eda_df.origin_country=="China"]['price'].iplot(kind="hist", bins=50 , layout=layout)
eda_df[eda_df.origin_country=="China"]['retail_price'].describe()
layout=dict(title="Original Price Ranges In China", xaxis_title="Prices", yaxis_title="Frequency",)
eda_df[eda_df.origin_country=="China"]['retail_price'].iplot(kind="hist", layout=layout)
eda_df.loc[:,eda_df.columns.str.startswith("shipping")].columns
eda_df['shipping_option_name'].value_counts()

livrasion_prices = eda_df[eda_df.shipping_option_name =='Livraison standard']['shipping_option_price'].value_counts().index.values
livrasion_prices_frquency = eda_df[eda_df.shipping_option_name =='Livraison standard']['shipping_option_price'].value_counts().values

fig = go.Figure()
fig.add_trace(go.Pie(labels=livrasion_prices, values=livrasion_prices_frquency))
fig.update_layout(title="Livrasion Standard Prices", legend_title="Prices In Euros", template="plotly_dark")


eda_df['shipping_is_express'].value_counts()
eda_df.info()
product_cat_columns = eda_df.loc[:,eda_df.columns.str.startswith("product")].columns.values

eda_df[product_cat_columns].info()
eda_df[product_cat_columns].head()
df.drop(['product_picture','product_url'], inplace=True, axis=1)
eda_df.drop(['product_picture','product_url'], inplace=True, axis=1)
eda_df_products = eda_df[['tags', 'price', 'units_sold', 'rating','rating_count', 'product_id','badges_count', 'badge_product_quality']].copy().sort_values(['units_sold','badges_count'], ascending=False)

eda_df_products_by_id = eda_df_products.set_index('product_id')
eda_df_products_by_id.head()
# Top 10 products sold for women
eda_df_products.loc[eda_df_products.tags.str.contains('[Ww]omen')].head(10).index
# Top 10 products in general
eda_df_products.head(10).index 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
eda_df_products_by_id_norm = eda_df_products_by_id.copy()
eda_df_products_by_id_norm.iloc[:,1:] = scaler.fit_transform(eda_df_products_by_id_norm.iloc[:,1:])
fig = go.Figure()

fig.add_trace(go.Bar(x=eda_df_products_by_id_norm.head(20).index.values,y=eda_df_products_by_id_norm.head(20).units_sold,name="Units Sold"  ))
fig.add_trace(go.Scatter(x=eda_df_products_by_id_norm.head(20).index.values,y=eda_df_products_by_id_norm.head(20).price, mode="lines+markers", name="Price" ))
fig.add_trace(go.Scatter(x=eda_df_products_by_id_norm.head(20).index.values,y=eda_df_products_by_id_norm.head(20).rating_count,mode="lines+markers",name="Rating Counts"  ))
fig.add_trace(go.Scatter(x=eda_df_products_by_id_norm.head(20).index.values,y=eda_df_products_by_id_norm.head(20).rating,mode="lines+markers",name="Avg. Rating"  ))

fig.update_layout(title="Top 20 Products Sold", legend_title="Features")
eda_df_products_by_id.head(20).describe()
eda_df_products_by_id.head(10)
dis_rat_slaes = eda_df[['rating', 'product_id', 'units_sold', 'price','discounted_price']]
dis_rat_slaes.set_index('product_id').head()
bins_per_1k= [i for i in range(0,101001,1000)]
labels_bins_per_1k = [str(vals)[:-3]+"k's" for vals in bins_per_1k[1:]]
bins_per_1k_units = pd.cut(dis_rat_slaes.units_sold,bins_per_1k, labels=labels_bins_per_1k )
dis_rat_slaes['bins_per_1k_units'] = bins_per_1k_units
dis_rat_slaes.head()
dis_rat_slaes_per_1k_units_sold = dis_rat_slaes.groupby('bins_per_1k_units').agg('mean')

dis_rat_slaes_per_1k_units_sold
dis_rat_slaes_per_1k_units_sold.dropna(how='all', inplace=True, axis=0)
dis_rat_slaes_per_1k_units_sold
#Plots

fig = go.Figure()


fig.add_trace(go.Bar(x=dis_rat_slaes_per_1k_units_sold.index.values,y=dis_rat_slaes_per_1k_units_sold.price, name="Price" ))
fig.add_trace(go.Scatter(x=dis_rat_slaes_per_1k_units_sold.index.values,y=dis_rat_slaes_per_1k_units_sold.discounted_price,mode="lines+markers",name="Discounted Price"  ))
fig.add_trace(go.Bar(x=dis_rat_slaes_per_1k_units_sold.index.values,y=dis_rat_slaes_per_1k_units_sold.rating,name="Avg. Rating"  ))

fig.update_layout(title="Product Sales Per 1k Bins", legend_title="Features", xaxis_title="Units Sold", yaxis_title="Avg Values per 1000")
import seaborn as sns
import matplotlib.pyplot as plt
def customized_heatmap(corr_df):
    corr_df =corr_df.iloc[1:,:-1].copy()  

    
    # Get only half portion of corr_df to avoid repitition, so create mask    
    mask = np.triu(np.ones_like(corr_df), k=1)
    
     
    # plot a heatmap of the values
    plt.figure(figsize=(20,14))
    plt.title("Heatmap Corrleation")
    ax = sns.heatmap(corr_df, vmin=-1, vmax=1, cbar=False,
                     cmap='rainbow', mask=mask, annot=True)
    
    # format the text in the plot to make it easier to read
    for text in ax.texts:
        t = float(text.get_text())
        if -0.4 < t < 0.4:

#         if -0.5 < t < 0.5:
            text.set_text('')        
        else:
            text.set_text(round(t, 2))
        text.set_fontsize('x-large')
    plt.xticks( size='x-large')
    plt.yticks(rotation=0, size='x-large')
    plt.show()
!pip install dython
# Import dython to check correlations
from dython.nominal import associations

assoc = associations(eda_df,plot=False)
corr_eda_df_dython = assoc['corr']

customized_heatmap(corr_eda_df_dython)
preprocess_df = eda_df.copy()
preprocess_df.loc[:,preprocess_df.columns.str.startswith("rating")].columns
preprocess_df.drop([ 'rating_five_count', 'rating_four_count','rating_three_count','rating_two_count', 'rating_one_count'], axis=1, inplace=True)
def five_rating_to_level_rating(val):
    if val<2.5:
        return "low"
    elif 2.5>= val <3.75:
        return "medium"
    else:
        return "high"
    
    
ratings = preprocess_df.rating.apply(five_rating_to_level_rating)
ratings.value_counts()

preprocess_df.rating = ratings
preprocess_df.drop(['merchant_id', 'product_id'],axis=1, inplace=True)
preprocess_df.drop(['origin_country', 'shipping_option_name'],axis=1, inplace=True)
preprocess_df.columns
# Lets check the proportion of top 20 tags count 
(uniuqe_categories_count['count'].head(20).sum() / uniuqe_categories_count['count'].sum())*100
bag_of_words =uniuqe_categories_count.keyword.head(20).str.lower().tolist()
# bag_of_words_reg_pattern =["\\b{}\\b".format(word) for word in bag_of_words]
# bag_of_words_reg_pattern_str =  "|".join(bag_of_words_reg_pattern)

bag_of_words
for word in bag_of_words:
    # First check if str contains the word
    #If yes converto to 1 , if no convert to 0
    # Again convert 1 and 0 into strings for dummy variables later.
    
    preprocess_df["tag_"+word] = preprocess_df.tags.str.lower().str.contains(word).astype(int).astype(str)
preprocess_df.drop(['title_orig','tags'],axis=1,inplace=True)
preprocess_df.drop('product_color', axis=1, inplace=True)
preprocess_df.drop('discounted_price', axis=1, inplace=True)
final_df = preprocess_df.copy()
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as DC
final_df_dummified = pd.get_dummies(final_df, drop_first=True)
final_df_dummified['rating'] = final_df['rating']
dependent_classes_labels= preprocess_df.rating.value_counts().index.values
dependent_classes_values = preprocess_df.rating.value_counts().values
fig = go.Figure()
fig.add_trace(go.Pie(labels=dependent_classes_labels, values=dependent_classes_values))
fig.update_layout(title="Imbalances in Dependent Classes", legend_title="Target Classes", template="plotly_dark")
from imblearn.over_sampling import SMOTE
X = final_df_dummified.loc[:,final_df_dummified.columns!='rating']
y= final_df_dummified['rating']
sm = SMOTE(sampling_strategy= 'not majority', random_state=101,k_neighbors=2)

X_res,y_res = sm.fit_resample(X,y)
y_res.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,random_state=101)
X_holdout, X_test_final, y_holdout, y_test_final = train_test_split(X_test, y_test,random_state=101)
pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])

pipe2 = Pipeline([('scaler_2', StandardScaler()), ('dc', DC())])
import time
rf_start = time.time()
pipe.fit(X_train,y_train)
rf_end = time.time()
eval_time_rf = rf_end -rf_start

dc_start = time.time()
pipe2.fit(X_train,y_train)
dc_end = time.time()
eval_time_dc = dc_end -dc_start

rf_start_pred = time.time()
pipe.predict(X_test)
rf_end_pred = time.time()
eval_time_rf_pred = rf_end_pred -rf_start_pred

dc_start_pred = time.time()
pipe2.predict(X_test)
dc_end_pred = time.time()
eval_time_dc_pred = dc_end_pred -dc_start_pred

print("Accuracy For Random forest on Validation Set: {}.".format(pipe.score(X_holdout,y_holdout)*100) )

print("Accuracy For Decision tree on Validation Set: {}.".format(pipe2.score(X_holdout,y_holdout)*100))
print("Accuracy For Random forest on Test Set: {}.".format(pipe.score(X_test_final,y_test_final)*100) )

print("Accuracy For Decision tree on Test Set : {}.".format(pipe2.score(X_test_final,y_test_final)*100))
print("Total time taken by RF to fit the model: {:.2f} sec".format(eval_time_rf))
print("Total time taken by Decision Tree to fit the model: {:.2f} sec".format(eval_time_dc))
print("Total time taken by RF to predict the test set: {:.2f} sec".format(eval_time_rf_pred))
print("Total time taken by Decision Tree to predict the test set: {:.2f} sec".format(eval_time_dc_pred))