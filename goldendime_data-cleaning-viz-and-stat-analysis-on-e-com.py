# Import all of the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
%matplotlib inline
#read the csv file and change all columns with date information into datetime objects
olist = pd.read_csv("../input/olist_public_dataset_v2.csv", index_col=0, parse_dates=['order_purchase_timestamp', 
                    'order_aproved_at', 'order_estimated_delivery_date', 'order_delivered_customer_date', 
                    'review_creation_date', 'review_answer_timestamp'])
olist.info()
stat_cat = olist['order_status'].unique().tolist()
stat_cat
status_cat = pd.Categorical(olist['order_status'], categories=stat_cat, ordered=False)
status_cat

#reassign status_cat to original "order_status" column
olist['order_status'] = status_cat
#Do the same operation on the "product_category_name" column
cat_name = olist['product_category_name'].unique().tolist()
olist['product_category_name'] = pd.Categorical(olist['product_category_name'], categories=cat_name, ordered=False)
olist['product_category_name'].describe()
#Make ordered list of qty of photos for future viz
ordered_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
olist['product_photos_qty'] = pd.Categorical(olist['product_photos_qty'], categories = ordered_list, ordered=True)
olist['product_photos_qty'].dtype
#capitalize first character of each word
olist['customer_city'] = olist['customer_city'].str.title()
olist['customer_state'] = olist['customer_state'].str.upper()
olist['customer_state'].value_counts().head(7)
olist['review_comment_title'] = olist['review_comment_title'].str.strip().str.lower()
olist['product_category_name'][1]
#replace underlines with spaces
olist['product_category_name'] = olist['product_category_name'].str.replace('_', ' ').str.lower()
olist['product_category_name'].value_counts().head(13)
to_eng_cat_name = olist['product_category_name'].unique().tolist()
to_eng_cat_name[0]
#translator = Translator(service_urls=[
#      'translate.google.com',
#      'translate.google.com.br',
#    ])
#translations = translator.translate(to_eng_cat_name, dest='en')
(olist['order_estimated_delivery_date'] - olist['order_delivered_customer_date']).describe()
((olist['order_estimated_delivery_date'] - olist['order_delivered_customer_date']) / (np.timedelta64(1, 'D'))).plot(kind='hist', bins=50)
#On average more items came earlier than estimated delivery date 
olist['delivery_accuracy'] = ((olist['order_estimated_delivery_date'] - olist['order_delivered_customer_date']) 
                               / (np.timedelta64(1, 'D')))
olist['total_value'] = olist['order_products_value'].add(olist['order_freight_value'])
#Let's check whether the changes were made successfully (with correct datatype) or not 
olist.info()
olist.describe(include='all')
olist['order_items_qty'].value_counts().sort_index()
olist.groupby('order_items_qty')['total_value'].mean().plot(kind='bar',figsize=(12,5))
sns.set_style("whitegrid")
plt.figure(figsize=(12,6))
sns.distplot(olist['total_value'], bins=800 ,kde=False, color='b')
plt.xlim([0, 600])
state_grouped = (olist.groupby('customer_state')[['order_products_value', 'review_score']]
                             .agg({'review_score': ['mean', 'count'], 'order_products_value':['mean']})
                ).sort_values(by=('review_score','mean'), ascending=False)
                 
state_grouped.head()
state_grouped.plot(kind='barh', figsize=(12,11), logx=True)
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_figheight(8)
fig.set_figwidth(15)


(olist.groupby(olist['order_purchase_timestamp'].dt.month)['order_products_value'].mean()
      .plot(kind='bar', ax=ax1, ylim=(115,140), 
            title='Average Prices for Orders in Brazilian Real Per Month')
)
(olist.groupby(olist['order_purchase_timestamp'].dt.month)['order_products_value'].sum()
      .plot(kind='bar', ax=ax2, ylim=(600000,1350000), sharex=True,
           title='Total Volume of Orders in Brazilian Reals Per Month')
)
olist['review_comment_title'].value_counts().head()
pweekday = olist['order_purchase_timestamp'].dt.weekday
phour = olist['order_purchase_timestamp'].dt.hour
pprice = olist['total_value']
purchase = pd.DataFrame({'day of week': pweekday, 'hour': phour, 'price': pprice})
purchase['day of week'] = purchase['day of week'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
purchase.head()
purchase_count = purchase.groupby(['day of week', 'hour']).count()['price'].unstack()
plt.figure(figsize=(16,6))
sns.heatmap(purchase_count.reindex(index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']), 
            cmap="YlGnBu", annot=True, fmt="d", linewidths=0.5)
dweekday = olist['order_delivered_customer_date'].dt.weekday
dhour = olist['order_delivered_customer_date'].dt.hour
dprice = olist['total_value']
delivery = pd.DataFrame({'day of week': dweekday, 'hour': dhour, 'price': dprice})
delivery['day of week'] = delivery['day of week'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
delivery_count = delivery.groupby(['day of week', 'hour']).count()['price'].unstack()
plt.figure(figsize=(16,6))
sns.heatmap(delivery_count.reindex(index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']), 
            cmap="BuPu", annot=True, fmt="d", linewidths=0.5)
top6 = olist['customer_city'].value_counts().head(6)
top6 = top6.index.tolist()
top6_cities = olist[olist['customer_city'].isin(top6)]
top6_cities['customer_city'].describe()
import warnings
warnings.filterwarnings('ignore')
(
    ggplot(data=top6_cities)
      + aes(y='review_score', x='delivery_accuracy')
      + aes(color='order_status', size='product_description_lenght')
      + geom_point(alpha=0.05)
      + geom_jitter()
      + facet_wrap('~customer_city', nrow=3, ncol=2)
      + theme_classic()
      + theme(figure_size=(18,15))
)

new_df = olist[['order_items_qty', 'product_description_lenght', 'product_photos_qty', 'delivery_accuracy', 'order_products_value', 'review_score']]
new_df.info()
cor = new_df.corr()
sns.heatmap(cor, annot=True, fmt=".2g", linewidths=0.5)
import statsmodels.api as sm
model = sm.OLS.from_formula('review_score ~ order_items_qty + product_description_lenght + product_photos_qty + delivery_accuracy + order_products_value', data=new_df)
result = model.fit()
print(result.summary())