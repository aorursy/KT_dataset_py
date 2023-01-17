import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


app_data=pd.read_csv('../input/AppleStore.csv')
app_data.shape
## columns
app_data.columns
app_data.head()
## function to add data to plot
def annot_plot(ax,w,h):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))
app_data.describe()
plt.figure(figsize=(8,8))
ax=sns.countplot('prime_genre',data=app_data,palette="Set2",order=app_data['prime_genre'].value_counts().index)
plt.xlabel('Genres')
plt.ylabel('Apps')
plt.xticks(rotation=80)
annot_plot(ax,0.03,1)
genre_costly_app=app_data.groupby(['prime_genre'])['price'].max().reset_index()
costly_app=genre_costly_app.merge(app_data,on=['prime_genre','price'],how='left')
costly_app[['prime_genre','track_name','price']].sort_values('price',ascending=False)
## utility functin to convert size from bytes to MB
def byte_2_mb(data):
    return round(data/10**6,2)

app_data['size_mb']=app_data['size_bytes'].apply(lambda x: byte_2_mb(x))

## function for creating category of apps according to size
def size_cat(data):
    if data<=5:
        return '5 MB'
    elif data<=10:
        return '10 MB'
    elif data<=50:
        return '50 MB'
    elif data<=100:
        return '100 MB'
    elif data<=500:
        return '500 MB'
    elif data>500:
        return 'more than 500 MB'

app_data['app_size']=app_data['size_mb'].apply(lambda x : size_cat(x))
plt.figure(figsize=(8,8))
ax=sns.countplot('app_size',data=app_data,palette="Set2",order=app_data['app_size'].value_counts().index)
plt.xlabel('App Size')
plt.ylabel('Apps')
#plt.xticks(rotation=80)
annot_plot(ax,0.2,1)
plt.figure(figsize=(8,8))
ax=sns.countplot('user_rating',data=app_data,palette="Set2")
plt.xlabel('User Rating')
plt.ylabel('Apps')
annot_plot(ax,0.2,1)
free_apps=app_data[app_data['price']==0]
paid_apps=app_data[app_data['price']>0]

labels=['Free Apps','Paid Apps']
sizes = [free_apps.shape[0],paid_apps.shape[0]]
colors = ['lightskyblue','gold']

# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)

plt.title('Apps')
plt.axis('equal')
plt.show()
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
ax1=sns.countplot('user_rating',data=free_apps,palette="Set2")
plt.title('Free Apps Rating')
plt.xlabel('User Rating')
plt.ylabel('Free Apps')
annot_plot(ax1,0.05,1)

plt.subplot(1,2,2)
ax2=sns.countplot('user_rating',data=paid_apps,palette="Set2")
plt.title('Paid Apps Rating')
plt.xlabel('User Rating')
plt.ylabel('Paid Apps')
annot_plot(ax2,0.05,1)
plt.figure(figsize=(8,8))
ax=sns.countplot('cont_rating',data=app_data,palette="Set2",order=app_data['cont_rating'].value_counts().index)
plt.xlabel('Content Rating')
plt.ylabel('Apps')
annot_plot(ax,0.2,1)
for i in app_data['cont_rating'].unique():
    plt.figure(figsize=(12,6))
    ax=sns.countplot('prime_genre',data=app_data[app_data['cont_rating']==i],palette="Set2",order=app_data['prime_genre'].value_counts().index)
    plt.xlabel('Genres')
    plt.ylabel('Apps')
    plt.title(i+' Apps')
    plt.xticks(rotation=80)

paid_apps[(paid_apps['user_rating']==5) &( paid_apps['rating_count_tot']>25000)][['track_name','prime_genre','rating_count_tot']].sort_values('rating_count_tot',ascending=False)
free_apps[(free_apps['user_rating']==5) &( free_apps['rating_count_tot']>25000)][['track_name','prime_genre','rating_count_tot']].sort_values('rating_count_tot',ascending=False)
paid_apps[(paid_apps['user_rating']==5) &( paid_apps['prime_genre']=='Games') &( paid_apps['rating_count_tot']>10000)][['track_name','rating_count_tot']].sort_values('rating_count_tot',ascending=False)
free_apps[(free_apps['user_rating']==5) &( free_apps['prime_genre']=='Games') &( free_apps['rating_count_tot']>10000)][['track_name','rating_count_tot']].sort_values('rating_count_tot',ascending=False)