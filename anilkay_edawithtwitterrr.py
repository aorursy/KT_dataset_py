# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/100-mostfollowed-twitter-accounts-as-of-dec2019/Most_followed_Twitter_acounts_2019.csv")

data.head()
data.dtypes
followers=[]

import re

for follow in data["Followers"]:

    follow=float(re.sub(",","",follow))

    followers.append(follow)
data["Followers"]=followers
following=[]

import re

for follow in data["Following"]:

    follow=float(re.sub(",","",follow))

    following.append(follow)

data["Following"]=following    
data.sort_values(by="Following",ascending=False)[0:20]
ratio=data["Followers"]/data["Following"]

data["FollowRatio"]=ratio
data.sort_values(by="FollowRatio",ascending=True)[0:16]
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,15))

sns.countplot(data=data,x="Nationality/headquarters")
data[data["Nationality/headquarters"]=="U.K"]
data[data["Nationality/headquarters"]=="Germany"]
data[data["Nationality/headquarters"]=="India"]
plt.figure(figsize=(10,10))

ax=sns.countplot(data=data,x="Industry")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()

#plt.show()

plt.savefig("industry.png")
data[data["Industry"]=="Technology "]
data[data["Industry"]=="Publishing Industry"]
import numpy as np



tweets=[]

for twitcont in data["Tweets"]:

    withoutcomma=re.sub(",","",twitcont)

    tweets.append(withoutcomma)

data["Tweets"]=tweets

data["Tweets"]=np.asarray(data["Tweets"]).astype(float)
data[["Followers","Tweets"]].corr()
data[["Following","Tweets"]].corr()
data.to_csv("data2.csv",index=False)