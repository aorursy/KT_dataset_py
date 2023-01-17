# importing libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import timedelta
from scipy import stats
#Loading the data
data_df=pd.read_csv('../input/sample-sales-data/sales_data_sample.csv',
                    encoding = 'unicode_escape',parse_dates=['ORDERDATE'])
pd.set_option('max_columns',30)
#pd.set_option('max_row', 3000)
data_df.head()
data_df.shape
print('-----------NULL VALUES--------------')
data_df.isnull().sum()
print('-----------UNIQUE VALUES--------------')
data_df.nunique()
#droping unnecessary  columns
data_df.drop(['ADDRESSLINE1','ADDRESSLINE2','PHONE','POSTALCODE'],axis=1,inplace=True)
#For some of the rows PRICEEACH was mislabelled as 100(ie: PRICEEACH*QUANTITYORDERED != SALES)
data_df['PRICEEACH']=(data_df['SALES']/data_df['QUANTITYORDERED'])
fg,ax=plt.subplots(figsize=(11,9))
data_df.hist(ax=ax)
plt.show()
data_df.describe()
fig, ax = plt.subplots(figsize=(9,6))   
sns.heatmap(data_df.corr(),cmap='Blues',annot=True,ax=ax)
ax=sns.scatterplot(data_df['MSRP'],data_df['PRICEEACH'])
ax.set_title('MSRP(manufacturer\'s suggested retail price) vs PriceEach')
sns.scatterplot(data_df['QUANTITYORDERED'],data_df['SALES'],hue=data_df['DEALSIZE'])
sns.scatterplot(data_df['QUANTITYORDERED'],data_df['SALES'],hue=data_df['PRICEEACH'])
fg=sns.FacetGrid(data_df,col='YEAR_ID')
fg.map(plt.hist,'SALES')
#State is only given For these COUNTRY. For all other country it is null!
(data_df[data_df['STATE'].notnull()])['COUNTRY'].unique()
top_sales=data_df.groupby('COUNTRY').sum().sort_values(by='SALES',ascending=False).head(10)[['SALES']]
fig,ax=plt.subplots(figsize=(9,5))
sns.barplot(top_sales['SALES'],top_sales.index,ax=ax)
plt.title('Top 10 countries by Sales')
data_df['DEALSIZE'].value_counts().plot(kind='barh')
data_df['PRODUCTLINE'].value_counts().plot(kind='barh')
data_df['STATUS'].value_counts().plot(kind='barh')
pd.crosstab(data_df['TERRITORY'],data_df['DEALSIZE']).plot(kind='barh')
mothly_reveue=data_df.groupby(['YEAR_ID','MONTH_ID'])['SALES'].sum().reset_index()
sns.lineplot(x='MONTH_ID',y='SALES',style='YEAR_ID',data=mothly_reveue)
plt.title('Mothly Reveue')
rfm_df=data_df.groupby('CUSTOMERNAME').agg({'ORDERDATE':'max','CUSTOMERNAME':'count','SALES':'sum'})
last_date=(data_df['ORDERDATE'].max()+timedelta(days=1))
rfm_df['Recency']=(last_date-rfm_df['ORDERDATE']).dt.days
rfm_df.head()
rfm_df.rename(columns={'CUSTOMERNAME':'Frequency','SALES':'Monetry value'},inplace=True)
rfm_df.drop('ORDERDATE',axis=1,inplace=True)
# rfm_df=rfm_df.reset_index()
fig,ax=plt.subplots(figsize=(6,5))
rfm_df.hist(ax=ax)
plt.show()
from scipy import stats
def q_qplot(col,df):
    fig,ax=plt.subplots()
    stats.probplot(df[col],dist="norm",plot=ax)
    ax.set_title(f'Q-Q plot({col})')
    plt.show()
q_qplot('Recency',rfm_df)
q_qplot('Frequency',rfm_df)
q_qplot('Monetry value',rfm_df)
rfm_log=np.log(rfm_df)
q_qplot('Recency',rfm_log)
q_qplot('Frequency',rfm_log)
q_qplot('Monetry value',rfm_log)
fig,ax=plt.subplots(figsize=(6,5))
rfm_log.hist(ax=ax)
plt.show()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
rfm_standarized=scaler.fit_transform(rfm_log)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Fit KMeans and calculate sse,Silhouette_score for each k
sse={}
Silhouette_score={}
for k in range(2,11):
    kmeans=KMeans(n_clusters=k,init='k-means++',random_state=0)
    kmeans.fit(rfm_standarized)
    sse[k]=kmeans.inertia_
    
    labels=kmeans.labels_
    Silhouette_score[k]=silhouette_score(rfm_standarized,labels)
    
fig,(ax1,ax2)=plt.subplots(1,2)
#Plot For The Elbow Method
sns.pointplot(list(sse.keys()),list(sse.values()),ax=ax1)
ax1.set_xlabel('No of Cluster')
ax1.set_title('The Elbow Method')

#Plot For The Silhouette Coefficient 
sns.pointplot(list(Silhouette_score.keys()),list(Silhouette_score.values()),ax=ax2)
ax2.set_title('Silhouette score Coefficient')
ax2.set_xlabel('No of Cluster')
ax2.set_ylabel('Silhouette score')
fig.tight_layout()
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
kmeans.fit(rfm_standarized)
#Assigning value of cluster in original dataframe
labels=kmeans.labels_
rfm_df=rfm_df.assign(Cluster=labels)
#Average value of rfm for each cluster
cluster_summary=rfm_df.groupby(by=['Cluster']).agg({'Recency':'mean',
                                    'Frequency':'mean',
                                    'Monetry value':'mean',
                                    'Cluster':'count'}).rename(columns={'Cluster':'count'})
cluster_summary
#All the cloumns in rfm_log_nrm are in same scale so applying snake plot in it by adding cluster Value

rfm_standarized=pd.DataFrame(rfm_standarized,columns=['Frequency','Monetry value','Recency'])
rfm_standarized['Cluster']=rfm_df['Cluster'].values

# Melting the data  so RFM values and metric names are stored in 1 column each
rfm_melted=pd.melt(rfm_standarized,id_vars=['Cluster'],
        value_vars=['Recency','Monetry value','Frequency'],
        var_name='metrics',
        value_name='Value')
sns.lineplot(x='metrics',y='Value',style='Cluster',data=rfm_melted)
plt.title('Snake Plot')
# %matplotlib notebook    #to make plot interactive
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
for x in range(5):
    ax.scatter(rfm_standarized.loc[ rfm_standarized['Cluster']==x]['Recency'],
               rfm_standarized.loc[ rfm_standarized['Cluster']==x]['Frequency'],
               rfm_standarized.loc[ rfm_standarized['Cluster']==x]['Monetry value'],
               label=f'Cluster {x}')
    ax.legend()
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetry value')
plt.show()