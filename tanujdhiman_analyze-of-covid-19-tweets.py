import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')
data.head()
data.columns
data.shape
data.info()
data.describe()
data.isnull().sum()
print("Values in User_name are ", data["user_name"].unique())
print("________________________________________________________________")
print("Values in User Locations are", data["user_location"].unique())
print("________________________________________________________________")
print("Values in User Description are", data["user_description"].unique())
print("________________________________________________________________")
print("Values in User Created are", data["user_created"].unique())
print("________________________________________________________________")
print("Values in User Followers are", data["user_followers"].unique())
print("________________________________________________________________")
print("Values in User Friends are", data["user_friends"].unique())
print("________________________________________________________________")
print("Values in User Verified are", data["user_verified"].unique())
print("________________________________________________________________")
print("Values in date are", data["date"].unique())
print("________________________________________________________________")
print("Values in text are", data["text"].unique())
print("________________________________________________________________")
print("Values in Hashtags are", data["hashtags"].unique())
print("________________________________________________________________")
print("Values in Source are", data["source"].unique())
print("________________________________________________________________")
print("Values in Retweets are", data["is_retweet"].unique())
data["user_verified"].unique()
is_verify = len(data[data["user_verified"]==True])
is_not_verify = len(data[data["user_verified"]==False])
print("Number of User Verified", is_verify)
print("Number of User Not Verified", is_not_verify)
print("Percentage of User Verified: {:.2f}%".format((is_verify / (len(data.user_verified))*100)))
print("Percentage of User Not verified : {:.2f}%".format((is_not_verify / (len(data.user_verified))*100)))
sns.countplot(data.user_verified)
plt.show()
import plotly.express as px
fig = px.scatter(data, x="date", 
                 color="user_followers",
                 size='user_friends', 
                 hover_data=['user_name', 'user_location', 'user_description', 'user_created',
       'user_followers', 'user_friends', 'user_favourites', 'user_verified',
       'text', 'hashtags', 'source', 'is_retweet'], 
                 title = "Date Plot")
fig.show()
fig1 = px.scatter(data, x="user_name", y="user_followers", 
                 color="user_followers", 
                 title = "User Follower Information")
fig1.show()
fig2 = px.scatter(data, x="user_name", y="user_friends", 
                 color="user_friends", 
                 title = "User Friends Information")
fig2.show()
fig3 = px.scatter(data, x="user_name", y="user_favourites", 
                 color="user_favourites", 
                 title = "User Favourites Information")
fig3.show()