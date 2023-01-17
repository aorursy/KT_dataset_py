import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly as ply

appdata=pd.read_csv('../input/AppleStore.csv')
appdata.columns
appdata.drop('Unnamed: 0',axis=1)
appdata=appdata.drop(['Unnamed: 0','vpp_lic','currency'],axis=1)
appdata
appdata['Size_GB']=appdata['size_bytes']/(1024*1024)
appdata
appdata.rename(columns={'track_name':'app_name','cont_rating':'content_rate',
                        'prime_genre':'genre','rating_count_tot':'versions_rating',
                       'rating_count_ver':'version_rating','sup_devices.num':'supp_devices','ipadSc_urls.num':'screen_shots_displayed',
                       'lang.num':'supp_lang_num'},inplace=True)
appdata
appdata=appdata.loc[:,['app_name','genre','user_rating_ver','version_rating','price','supp_devices','screen_shots_displayed','size_bytes']]
appdata
appdata.head()
appdata=appdata.sort_values(by=['user_rating_ver','version_rating'],ascending=False)
appdata.head(10)
paidapps=appdata[appdata['price']>0.0]
paidapps.count()
paidapps=paidapps.sort_values(by=['price'],ascending=False)
paidapps.head()
paid_apps=paidapps.groupby(['genre']).count()
paid_apps['app_name'].plot(kind='barh',
                          figsize=(10,6),
                          alpha=0.98)
plt.xlabel('Frequency Count')
plt.ylabel('Category')
plt.title('Paid Apps Category Wise')
plt.show()
games=appdata.loc[appdata['genre']=='Games']
games
gamesapps=games.groupby(['user_rating_ver']).count()
gamesapps['app_name'].plot(kind='barh',
                          figsize=(10,6),
                          alpha=0.98)
plt.xlabel('Frequency Count')
plt.ylabel('Rating')
plt.title('Games Classified by User Rating')
plt.show()
top_rated=appdata.loc[appdata['user_rating_ver']==5.0]
top_rated
paid_apps=top_rated[top_rated['price']>0]
rated_paid_apps=paid_apps.sort_values('version_rating',ascending=False)
top_rated_paid_apps=rated_paid_apps.groupby(by='genre').count()
top_rated_paid_apps=top_rated_paid_apps['app_name']
top_rated_paid_apps
free_apps=top_rated[top_rated['price']==0.0]
rated_free_apps=free_apps.sort_values('version_rating',ascending=False)
top_rated_free_apps=rated_free_apps.groupby(by='genre').count()
top_rated_free_apps=top_rated_free_apps['app_name']
top_rated_free_apps
genre=np.unique(appdata['genre'])
genre

frame={'top_rated_free':top_rated_free_apps,'top_rated_paid':top_rated_paid_apps}
combined=pd.DataFrame(frame,index=genre)
combined.plot(kind='barh',
             figsize=(10,6))
plt.xlabel('Rating Counts')
plt.ylabel('Genre')
plt.show()
four_rated=games.loc[games.user_rating_ver==4.5]
four_rated
four_paid_apps=four_rated[four_rated['price']>0]
four_rated_paid_apps=four_paid_apps.sort_values('version_rating',ascending=False)
four_rated_paid_apps=four_rated_paid_apps.groupby(by='genre').count()
four_rated_paid_apps=four_rated_paid_apps['app_name']
four_rated_paid_apps
four_free_apps=four_rated[four_rated['price']==0.0]
four_rated_free_apps=four_free_apps.sort_values('version_rating',ascending=False)
four_rated_free_apps=four_rated_free_apps.groupby(by='genre').count()
four_rated_free_apps=four_rated_free_apps['app_name']
four_rated_free_apps

