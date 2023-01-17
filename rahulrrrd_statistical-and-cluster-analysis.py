import numpy as np 

import pandas as pd 

import math



import matplotlib.pyplot as plt 

import seaborn as sb

from matplotlib import style

# style.use('fivethirtyeight')

style.use('ggplot')



import plotly.express as px

import plotly.graph_objects as go



import os 
DATADIR = '../input/summer-products-and-sales-in-ecommerce-wish'

os.listdir(DATADIR)
df = pd.read_csv(DATADIR + '/summer-products-with-rating-and-performance_2020-08.csv')
df.info()
df.iloc[1]
df.isna().sum()
def plot_missing_data(df):

    columns_with_null = df.columns[df.isna().sum() > 0]

    null_pct = (df[columns_with_null].isna().sum() / df.shape[0]).sort_values(ascending=False) * 100

    plt.figure(figsize=(8,6));

    sb.barplot(y = null_pct.index, x = null_pct, orient='h')

    plt.title('% Na values in dataframe by columns');
plot_missing_data(df)
df['merchant_profile_picture'].value_counts()
print("Unique values: ", df['has_urgency_banner'].unique())

print("Value counts: ", df['has_urgency_banner'].value_counts())
df['has_urgency_banner'] = df['has_urgency_banner'].replace(np.nan,0)

print("Unique values: ", df['has_urgency_banner'].unique())

print("Value counts: ", df['has_urgency_banner'].value_counts())
df['urgency_text'].unique()
df['urgency_text']=df['urgency_text'].replace({'Quantité limitée !':'QuantityLimited',

                                               'Réduction sur les achats en gros':'WholesaleDiscount',

                                               np.nan:'noText'})

print(df['urgency_text'][:5])

print(df['urgency_text'].value_counts())
rating_columns = ['rating_one_count','rating_two_count','rating_three_count','rating_four_count','rating_five_count']

df[rating_columns] = df[rating_columns].fillna(value=-1)
df.loc[df['rating_five_count']==-1,'rating_count'].value_counts()
df[rating_columns]=df[rating_columns].replace(-1,0)
print(df['origin_country'].unique())

print(df['product_color'].unique())

print(df['product_variation_size_id'].unique())

print(df['merchant_name'].unique())

print(df['merchant_info_subtitle'].unique())
nan_cat_cols = ['origin_country','product_color','product_variation_size_id','merchant_name','merchant_info_subtitle']

df[nan_cat_cols] = df[nan_cat_cols].replace(np.nan,'Unknown')
df.columns[df.isna().sum()>0]
df.duplicated().sum()
df= df.drop_duplicates()

df.duplicated().sum()
print("Duplicate product_id :",df['product_id'].duplicated().sum())
df.describe().T
plt.figure(figsize=(12,6))

sb.distplot(df['price'], color='red', label='Price')

sb.distplot(df['retail_price'], color='blue', label='Retail price')

plt.legend();
kwargs = {'cumulative':True}

f, axes = plt.subplots(1,2, figsize=(14,6))

f.suptitle('CDF of Price and Retail Price')

sb.distplot(df['price'].values,kde_kws=kwargs, hist_kws=kwargs, color='red', label='Price', ax=axes[0]);

sb.distplot(df['retail_price'].values,kde_kws=kwargs, hist_kws=kwargs, color='blue', label='Retail Price', ax=axes[1]);

axes[0].set(xlabel='Price');

axes[1].set(xlabel='Retail Price');
fig = go.Figure()

fig.add_trace(go.Box(x=df['retail_price'], name='Retail Price'))

fig.add_trace((go.Box(x=df['price'], name='Price')))

fig['layout']['title'] = 'Distribution of Price and Retail Price'

fig.show()
df_outliers = df[df['price'] > 18]

print("Number of outliers: ",df_outliers.shape[0])

print("Outlier: ", df_outliers[df_outliers['price']==49])
px.scatter(df, x='units_sold', y='price',marginal_x='box', title='Price vs Units Sold')
#range for units sold

sorted(df['units_sold'].unique())
from sklearn.cluster import KMeans



clusters = {}

for i in range(1,8):

    kmeans = KMeans(n_clusters=i).fit(df[['units_sold']])

    clusters[i] = kmeans.inertia_

    

plt.plot(list(clusters.keys()), list(clusters.values()));

plt.xlabel('no. of clusters');

plt.ylabel('kmeans inertia');   


#order cluster method

def order_cluster(cluster_field_name, target_field_name,df,ascending):

    new_cluster_field_name = 'new_' + cluster_field_name

    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()

    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)

    df_new['index'] = df_new.index

    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)

    df_final = df_final.drop([cluster_field_name],axis=1)

    df_final = df_final.rename(columns={"index":cluster_field_name})

    return df_final
df['units_sold_cluster'] = KMeans(n_clusters=3).fit(df[['units_sold']]).predict(df[['units_sold']])

df = order_cluster('units_sold_cluster','units_sold',df,True)

df.groupby(['units_sold_cluster'])['units_sold'].describe()
px.scatter(df,x='units_sold',y='rating', color='units_sold_cluster', marginal_y ='box',title='Rating vs units sold')
px.scatter(df,x='rating',y='merchant_rating', color='units_sold_cluster', marginal_y ='box',title='Merchant Rating vs units sold', opacity=0.7)
px.scatter(df,x='rating', y='product_variation_inventory', color='units_sold_cluster', title='Product variation vs Rating')
fig = px.scatter(df,x='rating_count',y='rating', color='units_sold_cluster', title='Rating vs Rating count')

fig.add_trace(go.Scatter(x=np.ones((len(df)))*1103,y=df['rating'],name='Threshold 1'))

fig.add_trace(go.Scatter(x=np.ones((len(df)))*7773, y=df['rating'],name='Threshold 2'))

fig.update_layout(showlegend=False)
px.scatter(df,x='retail_price', y='price',color='units_sold_cluster',marginal_y='box')
px.scatter(df, x='price', y='shipping_option_price', color= 'units_sold_cluster', title='Shipping price vs Price')
features= ['price','retail_price','units_sold','rating','rating_count','shipping_option_price','product_variation_inventory','merchant_rating','merchant_rating_count']

corr = df[features].corr(method='spearman')
plt.figure(figsize=(15,8));

sb.heatmap(corr,annot=True);
df['uses_ad_boosts'].value_counts()
df.groupby(['uses_ad_boosts'])['units_sold'].describe()
data = df.query('uses_ad_boosts == 0')['units_sold'].values, df.query('uses_ad_boosts == 1')['units_sold'].values
# calculating the effect size 



def EffectSize(group1, group2):

    diff = group1.mean()- group2.mean() 

    var1 = group1.var()

    var2 = group2.var()

    n1,n2 = len(group1), len(group2)

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)

    d = diff/math.sqrt(pooled_var)

    return d
g1,g2 = data

print("Difference in means: ",g1.mean()-g2.mean())

EffectSize(g1,g2)
class HypothesisTest(object):

    

    def __init__(self, data):

        self.data = data

        self.MakeModel()

        self.actual = self.TestStatistic(data)

        

    def PValue(self, iters=1000):

        self.test_stats = [self.TestStatistic(self.RunModel()) for _ in range(iters)]

        

        count = sum(1 for x in self.test_stats if x > self.actual)

        return count/iters

    

    def TestStatistic(self, data):

        raise UnimplementedMethodException()

    def MakeModel(self):

        pass

    def RunModel(self):

        raise UnimplementedMethodException()
class DiffMeans(HypothesisTest):

    def TestStatistic(self, data):

        group1,group2 =data

#         test_stat = abs(group1.mean() - group2.mean())

        test_stat = abs(EffectSize(group1, group2))

        return test_stat

    def MakeModel(self):

        group1, group2 = self.data

        self.n, self.m = len(group1), len(group2)

        self.pool = np.hstack((group1,group2))

        

    def RunModel(self):

        np.random.shuffle(self.pool)

        data = self.pool[:self.n], self.pool[self.n:]

        return data

    
test = DiffMeans(data)

test.PValue()
df['difference'] = df['retail_price'] - df['price']

df['discount'] = df['difference']/df['retail_price'] *100

plt.figure(figsize=(12,6))

sb.distplot(df['discount']);

plt.title('Distribution of Discount');
px.scatter(df,x='discount', y='rating_count', color='units_sold_cluster')
df['rating_score'] = df['rating']*df['rating_count']

df['rating_score'] =df['rating_score']/df['rating_score'].max()

plt.figure(figsize=(12,6))

sb.distplot(df['rating_score']);

plt.title('Distribution of Rating Score');
px.scatter(df,x='rating_score',y='units_sold', color='units_sold', title='Units Sold vs Rating score')
def make_clusters(df,column):

    clusters = {}

    for i in range(1,8):

        kmeans = KMeans(n_clusters=i).fit(df[[column]])

        clusters[i] = kmeans.inertia_



    plt.plot(list(clusters.keys()), list(clusters.values()));

    plt.title(f'{column} clusters')

    plt.xlabel('no. of clusters');

    plt.ylabel('kmeans inertia');   
make_clusters(df,'rating_score')
kmeans = KMeans(n_clusters=3).fit(df[['rating_score']])

df['rating_score_cluster'] = kmeans.predict(df[['rating_score']])

df= order_cluster(df=df,cluster_field_name='rating_score_cluster',target_field_name='rating_score',ascending=True)

df.groupby('rating_score_cluster')[['rating','rating_count','units_sold']].describe().T
df['overall_score'] = df['rating_score_cluster'] + df['units_sold_cluster']

make_clusters(df,'overall_score');
kmeans= KMeans(n_clusters=2).fit(df[['overall_score']])

df['overall_score_cluster'] = kmeans.predict(df[['overall_score']])

df = order_cluster(df=df,target_field_name='overall_score', cluster_field_name='overall_score_cluster', ascending=True)

df.groupby('overall_score_cluster')[['rating_score','price','units_sold']].describe().T
df[['title_orig','units_sold','price','rating_score','units_sold_cluster','rating_score_cluster','overall_score_cluster']].sample(frac=.25).head(30)