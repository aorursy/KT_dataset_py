import numpy as np 
import pandas as pd 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
fbdata= pd.read_csv("../input/pseudo_facebook.csv")
fbdata.head()
#Exploring data
fbdata.describe()
fbdata.info()
fbdata['gender'].value_counts()
#nearly 40:60 ratio for female to male
#Counting all vales here
fbdata['gender'].value_counts(dropna=False)
#divided the age into a group of 10. see last column
labels=['10-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','101-110','111-120']
fbdata['age_group'] = pd.cut(fbdata.age,bins=np.arange(10,121,10),labels=labels,right=True)
fbdata.head()
#Counting value in age groups
fbdata.age_group.value_counts()
#Females have more friends than males
sns.barplot(x=fbdata['age_group'],y=fbdata['friend_count'],hue=fbdata.gender)
#No of people having some friends
np.count_nonzero(fbdata.friend_count)
#All the people having zero friends
fc=fbdata.friend_count==0
fc.value_counts()
#plotting the gender vs zero friend count people
#fc=fbdata.friend_count==0
sns.barplot(y=fbdata.friend_count==0,x=fbdata.gender)
fcmale=(fbdata.friend_count==0) & (fbdata.gender=='male')
fcmale.value_counts(dropna=False)
#true:1459
fcfemale=(fbdata.friend_count==0) & (fbdata.gender=='female')
fcfemale.value_counts(dropna=False)
#true:503
fc=fbdata.friend_count==0
fc.value_counts()
#sns.barplot(x=fcmale,y=fcfemale)
fbdata.tenure.interpolate(inplace=True)
tenlabel=['0-1 year','1-2 years','2-3 years','3-4 years','4-5 years','5-6 years','6-7 years','7-8 years','8-9 years']
fbdata['year_group']=pd.cut(fbdata.tenure,bins=np.arange(0,3300,365),labels=tenlabel,right=True)
fbdata.head()
fbdata.year_group.fillna(value='0-1 year',inplace=True)
fbdata.year_group.value_counts(dropna=False)
#Most liked people
fbdata.sort_values(by='likes_received',ascending=False)[:10]
#Calculating likes per day
fbdata['likes_per_day']=fbdata.likes_received/fbdata.tenure.where(fbdata.tenure>0)
fbdata.head()
#Top 10 users getting highest likes received
fbdata.sort_values(by='likes_received',ascending=False)[:10]
#Highest likes received per day
fbdata.sort_values(by='likes_per_day',ascending=False)[:10]
#Extracting famous people
famous=fbdata.sort_values(by='likes_per_day',ascending=False)[:10]
famous.head()
#plt.subplots(figsize=(12,10)
#plt.plot(y='userid',x='likes_per_day',data=famous)
famous.plot(x='userid',y='likes_per_day',kind='bar')
plt.ylabel("Likes per day")
plt.xlabel("User ID")
plt.title("Maximum likes per day")
plt.show()
#pivot table
fbdata.pivot_table(values=['mobile_likes_received','mobile_likes','www_likes_received','www_likes'],index='age_group',columns='gender')
fbdata.pivot_table(values=['mobile_likes_received','mobile_likes','www_likes_received','www_likes'],index='gender').plot()
#Getting those people who are most interested in sending friend requests
fbdata.sort_values(by='friendships_initiated',ascending=False)[:10]
followers=fbdata.sort_values(by='friendships_initiated',ascending=False)[:10]
#plt.subplots(figsize=(12,10)
#plt.plot(y='userid',x='likes_per_day',data=famous)
followers.plot(x='userid',y='friendships_initiated',kind='bar')
plt.ylabel("Friendship_count")
plt.xlabel("User ID")
plt.title("Maximum friendships initiated")
plt.show()
followers['fc_per_day']=followers.friendships_initiated / followers.tenure
followers
#plt.subplots(figsize=(12,10)
#plt.plot(y='userid',x='likes_per_day',data=famous)
followers.plot(x='userid',y='fc_per_day',kind='bar')
plt.ylabel("Friendship_count")
plt.xlabel("User ID")
plt.title('Maximum friendships initiated per day')
plt.show()
