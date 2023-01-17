# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style("whitegrid")

import plotly

import datetime
data = pd.read_csv("/kaggle/input/covid19-tweets/covid19_tweets.csv")
data.info()
data.describe().transpose()
data.head(2)
Per_Null_hashtags = (data["hashtags"].isna().sum()/data["hashtags"].count()) * 100
print("{} percent hastags are Null ".format(Per_Null_hashtags))
data["hashtags"].isna().value_counts()

sns.countplot(data["hashtags"].isna())

plt.title("visualising tweet with and without hashtags")

plt.legend()
data["user_verified"].value_counts()
sns.countplot(data["user_verified"])
data["source"].nunique()

data["source"].isna().value_counts()

Apple_user = data[data["source"] == "Twitter for iPhone"]["source"].count()

Android_user = data[data["source"] == "Twitter for Android"]["source"].count()

Web_user = data[data["source"] == "Twitter Web App"]["source"].count()

others = data[(data["source"] != "Twitter Web App") & 

              (data["source"] != "Twitter for Android") &

              (data["source"] != "Twitter for iPhone")]["source"].count()



plt.figure(figsize=(7,7))

labels = ['Apple_user', 'Android_user', 'Web_app_user', 'others']

plt.pie([Apple_user,Android_user,Web_user,others],labels=labels,autopct='%1.2f%%')

plt.show()
user_name = data.groupby("user_name")["user_location"].count().reset_index()

user_name.columns = ['user_name', 'count']

#user_name.set_index("user_name",inplace = True)

user_name.sort_values(['count'],inplace=True)
plt.figure(figsize=(7,7))

sns.barplot(x = "count",y = "user_name",data = user_name.tail(20),orient = "h")

plt.tight_layout()

plt.title("20 Users of Max tweets")

plt.legend()
user_location = data.groupby("user_location")["user_name"].count().reset_index()

user_location.columns = ['user_location', 'count']

#user_name.set_index("user_name",inplace = True)

user_location.sort_values(['count'],inplace=True)
plt.figure(figsize=(7,7))

sns.barplot(x = "count",y = "user_location",data = user_location.tail(20),orient = "h")

plt.tight_layout()

plt.title("20 Locations of Max tweets")

plt.legend()
data["date"] = pd.to_datetime(data["date"])

time = data['date'].iloc[0]

time.hour
data["Month"] = data["date"].apply(lambda x : x.month)

data["day"] = data["date"].apply(lambda x : x.dayofweek)

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

data["day"] = data["day"].map(dmap)



sns.countplot(data["Month"])
sns.countplot(data["day"])