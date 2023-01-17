import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline
df_appstore=pd.read_csv('../input/AppleStore.csv')
df_appstore.head(3) #Getting a feel of data
df_appstore.describe()
df_appstore.info()
df_appstore_desc=pd.read_csv('../input/appleStore_description.csv')
df_appstore_desc.head(3)
df_appstore_desc.info()
plt.figure(figsize=(14,8))
sb.heatmap(df_appstore.corr(),annot=True)
plt.tight_layout()
#as ,we can see user_rating is mostly correlated with user_rating_ver,ipadSc_urls.num and lang.num
df_appstore['prime_genre'].unique()
df_appstore.groupby('prime_genre')['user_rating'].mean()

#Genre wise average user_rating, productivity Genre is highly rated followed by Music
plt.figure(figsize=(20,4))
sb.barplot(x='prime_genre',y='user_rating',data=df_appstore)
plt.tight_layout()
#User_rating vs price
sb.jointplot(x='price',y='user_rating',data=df_appstore)
#We can see Apps having a price tag can also have low rating and free apps can also have best rating
## let's see how many free apps are rated more than 4 and  have received more than the mean rating_count_tot , this is to eliminate apps rated more than 4 by 1 or 2 users 
df_appstore[(df_appstore['user_rating']>4) & (df_appstore['rating_count_tot']>=12892)&(df_appstore['price']==0.0)].count()
# 501 free apps are highly rated and voted by many


#User_rating vs size_bytes, not much coorelation
sb.jointplot(x='size_bytes',y='user_rating',data=df_appstore)
#Let's find out how many apps having user_rating have a history of high rating 
df_appstore[(df_appstore['user_rating']<df_appstore['user_rating_ver']) & (df_appstore['user_rating']>=4)].count()
#seems like 920 such apps 
#merging appstore and details
df_appstore_details=pd.merge(df_appstore,df_appstore_desc,on='id',how='inner')[['id', 'track_name_x', 'size_bytes_x', 'currency', 'price',
       'rating_count_tot', 'rating_count_ver', 'user_rating',
       'user_rating_ver', 'ver', 'cont_rating', 'prime_genre',
       'sup_devices.num', 'ipadSc_urls.num', 'lang.num', 'vpp_lic','app_desc'
]]
df_appstore_details.head(5)
#relationship between most used words  and rating
all_words=''
#collecting all track_names as a single string
for i in range (0,7196):
    all_words += df_appstore_details.iloc[i]['track_name_x']+' '
    
all_tokens=all_words.split(' ')

token_count_dict={}

#Finding the count of each word except the stopwords
for token in all_tokens:
    if token not in ['-','&','for','The','and','of','the','2','A','-','by','My','with','to','in']:
        if token in token_count_dict.keys():
            count_of_token=token_count_dict[token]
            count_of_token+=1
            token_count_dict[token]=count_of_token
        else:
            token_count_dict[token]=1
        
        
#sorting the dictionary of most used words in descending sequence
ten_most_used_words=sorted(((value,key) for (key,value) in token_count_dict.items()),reverse=True)[1:11]
words=[]  
for x,y in ten_most_used_words:
    words.append(y)
words
avg={}
## Average user ratings of apps having these words.
for fuw in words:
    count_fuw=0
    rating=0
    for i in range(0,7096):
        if fuw in df_appstore_details.iloc[i]['track_name_x']:
            count_fuw+=1
            rating+=df_appstore_details.iloc[i]['user_rating']
    avg[fuw]=rating/count_fuw 
avg
#The apps having most frequently used words in track_name are rated higher than the mean overall user_rating of 3.5
## Top Key words used in the name of top 100 most highly rated Apps
#finding the top 100 rated apps
top100=df_appstore_details.sort_values('user_rating',ascending=False).head(100)
all_words=''
#collecting the track_names of all the 100 apps as one string
for i in range (0,99):
    all_words += top100.iloc[i]['track_name_x']+' '
    
all_tokens=all_words.split(' ')

token_count_dict={}

#finding the counts of all the words except the stopwords
for token in all_tokens:
    if token not in ['-','&','for','The','and','of','the','2','A','-','by','My','with','to','in']:
        if token in token_count_dict.keys():
            count_of_token=token_count_dict[token]
            count_of_token+=1
            token_count_dict[token]=count_of_token
        else:
            token_count_dict[token]=1
#sorting the dictionary of most used words in descending sequence
most_used_words=sorted(((value,key) for (key,value) in token_count_dict.items()),reverse=True)
most_used_words[1:10]
## Top Key words used in the description  of top 100 most highly rated Apps
top100=df_appstore_details.sort_values('user_rating',ascending=False).head(100)
all_words=''
for i in range (0,99):
    all_words += top100.iloc[i]['app_desc']+' '
    
all_tokens=all_words.split(' ')

token_count_dict={}

for token in all_tokens:
    if token not in ['-','&','for','when','The','and','one','like','every','about','than','of','the','any','has','out','not','so','You','each','also','get','just','some','but','through','over','I','up','many','most','\n-','-','by','My','with','to','in','a','you','is','on','or','','from','can','as','will','that','are','all','this','it','be','an','at','our','have']:
        if token in token_count_dict.keys():
            count_of_token=token_count_dict[token]
            count_of_token+=1
            token_count_dict[token]=count_of_token
        else:
            token_count_dict[token]=1
token_count_dict
most_used_words=sorted(((value,key) for (key,value) in token_count_dict.items()),reverse=True)
most_used_words[1:16]
#realtionship between content rating and user_rating
df_appstore_details['cont_rating'].unique()
plt.figure(figsize=(20,4))
sb.countplot(x='cont_rating',data=df_appstore_details[df_appstore_details['user_rating']==5])