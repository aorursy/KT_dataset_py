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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import statsmodels.api as sm
US_video=pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")
US_video.isnull().sum()
print(US_video.columns)
print(US_video.dtypes)
print(US_video["video_id"].value_counts())
print(US_video["publish_time"].value_counts())
US_video["publish_time"]=pd.to_datetime(US_video["publish_time"])
US_video=US_video.assign(publish_day=US_video.publish_time.dt.day,publish_month=US_video.publish_time.dt.month,publish_year=US_video.publish_time.dt.year)
sns.countplot(x=US_video["publish_year"],data=US_video)
US_video.drop(US_video[US_video["publish_year"]<2017].index,inplace=True)
sns.barplot(x=US_video["publish_year"],y=US_video["comment_count"])
US_video.groupby("publish_year")["likes","dislikes","views","comment_count"].mean()

sns.barplot(x=US_video["publish_year"],y=US_video["views"])
sns.barplot(x=US_video["publish_year"],y=US_video["likes"])

sns.barplot(x=US_video["publish_year"],y=US_video["dislikes"])
sns.countplot(x=US_video["publish_month"],data=US_video)
US_video.drop(US_video[(US_video["publish_month"]<11) & (US_video["publish_month"]>7)].index,inplace=True)
sns.barplot(x=US_video["publish_month"],y=US_video["likes"])
sns.barplot(x=US_video["publish_month"],y=US_video["dislikes"])
sns.barplot(x=US_video["publish_month"],y=US_video["views"])
sns.barplot(x=US_video["publish_month"],y=US_video["comment_count"])
US_video.groupby("publish_month")["likes","dislikes","views","comment_count"].mean()
print(US_video["trending_date"].value_counts())
US_video["trending_date"]=pd.to_datetime(US_video["trending_date"],format="%y.%d.%m")
US_video=US_video.assign(trending_day=US_video.trending_date.dt.day,trending_month=US_video.trending_date.dt.month,trending_year=US_video.trending_date.dt.year)

sns.countplot(x=US_video["trending_year"],data=US_video)
sns.barplot(x=US_video["trending_year"],y=US_video["likes"])
sns.barplot(x=US_video["trending_year"],y=US_video["dislikes"])
sns.barplot(x=US_video["trending_year"],y=US_video["views"])
sns.barplot(x=US_video["trending_year"],y=US_video["comment_count"])
US_video.groupby("trending_year")["likes","dislikes","views","comment_count"].mean()
sns.countplot(x=US_video["trending_month"],data=US_video)
sns.barplot(x=US_video["trending_month"],y=US_video["likes"])
sns.barplot(x=US_video["trending_month"],y=US_video["dislikes"])
sns.barplot(x=US_video["trending_month"],y=US_video["views"])
sns.barplot(x=US_video["trending_month"],y=US_video["comment_count"])
US_video.groupby("trending_month")["likes","dislikes","views","comment_count"].mean()
print(US_video["title"].value_counts())
print(US_video["channel_title"].value_counts())
US_video["category_id"]=US_video["category_id"].astype("object")
print(US_video["category_id"].value_counts())
sns.barplot(x=US_video["category_id"],y=US_video["views"],data=US_video,hue="publish_year")

sns.barplot(x=US_video["category_id"],y=US_video["likes"],data=US_video,hue="publish_year")


sns.barplot(x=US_video["category_id"],y=US_video["dislikes"],data=US_video, hue="publish_year")


sns.barplot(x=US_video["category_id"],y=US_video["comment_count"],data=US_video,hue="publish_year")

US_video.groupby("category_id")["views","likes","dislikes","comment_count"].mean()
print(US_video["views"].describe())

print(US_video["likes"].describe())

print(US_video["dislikes"].describe())

print(US_video["comment_count"].describe())
x1=np.log(US_video["views"])
y1=np.log(US_video["likes"])
y2=np.log(US_video["dislikes"])
y3=np.log(US_video["comment_count"])

sns.regplot(x=x1,y=y1,fit_reg=False)
sns.regplot(x=x1,y=y2,fit_reg=False)
sns.regplot(x=x1,y=y3,fit_reg=False)
num_data=US_video.select_dtypes(exclude=["object","bool"])
num_data.corr()
sns.regplot(x=y1,y=y3,fit_reg=False)
sns.regplot(x=y2,y=y3,fit_reg=False)
print(US_video["comments_disabled"].value_counts())
sns.countplot(x="comments_disabled",data=US_video)
sns.barplot(x="comments_disabled",y="likes",data=US_video,hue="publish_year")


sns.barplot(x="comments_disabled",y="dislikes",data=US_video,hue="publish_year")


sns.barplot(x="comments_disabled",y="views",data=US_video,hue="publish_year")



US_video.groupby("comments_disabled")["views","likes","dislikes"].mean()
print(US_video["ratings_disabled"].value_counts())
sns.countplot(x="ratings_disabled",data=US_video)
sns.barplot(x="ratings_disabled",y="likes",data=US_video)
sns.barplot(x="ratings_disabled",y="dislikes",data=US_video)
sns.barplot(x="ratings_disabled",y="views",data=US_video,hue="publish_year")
sns.barplot(x="ratings_disabled",y="comment_count",data=US_video,hue="publish_year")

US_video.groupby("ratings_disabled")["views","likes","dislikes","comment_count"].mean()
sns.countplot(x="video_error_or_removed",data=US_video)
print(US_video["video_error_or_removed"].value_counts())

sns.barplot(x="video_error_or_removed",y="comment_count",data=US_video,hue="publish_year")
US_video.groupby("video_error_or_removed")["views","likes","dislikes","comment_count"].mean()
sns.barplot(x="video_error_or_removed",y="likes",data=US_video,hue="publish_year")
sns.barplot(x="video_error_or_removed",y="dislikes",data=US_video,hue="publish_year")
sns.barplot(x="video_error_or_removed",y="views",data=US_video,hue="publish_year")
print(US_video["tags"].value_counts())
print(US_video["description"].value_counts())
US_video.drop(["tags","description","title","channel_title","publish_time","video_id","thumbnail_link","trending_date"],axis=1,inplace=True)
data1=US_video.copy(deep=True)
data1["comments_disabled"]=(data1["comments_disabled"]=="True").astype(int)
data1["ratings_disabled"]=(data1["ratings_disabled"]=="True").astype(int)
data1["video_error_or_removed"]=(data1["video_error_or_removed"]=="True").astype(int)
data1["likes"]=np.log(data1["likes"])
data1["dislikes"]=np.log(data1["dislikes"])
data1["views"]=np.log(data1["views"])
data1["comment_count"]=np.log(data1["comment_count"])
data1["views"]=data1["views"].replace([np.inf,-np.inf],np.nan)
data1["dislikes"]=data1["dislikes"].replace([np.inf,-np.inf],np.nan)
data1["likes"]=data1["likes"].replace([np.inf,-np.inf],np.nan)
data1["comment_count"]=data1["comment_count"].replace([np.inf,-np.inf],np.nan)
data1.isnull().sum()
data1.dropna(axis=0,how='any',inplace=True)
data1=pd.get_dummies(data1,drop_first=True)
x=data1.drop("likes",axis=1,inplace=False)
x=x.values
y=data1["likes"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


lgr=LinearRegression(fit_intercept=True)
fit_model=lgr.fit(x_train,y_train)
prediction=lgr.predict(x_test)

print(fit_model.score(x_test,y_test))
print(r2_score(y_test,prediction))
residual=y_test-prediction
print(residual.mean())
sns.regplot(prediction,residual,fit_reg=False)
base_pred=np.repeat(np.mean(y_test),len(y_test))
base_rms=np.sqrt(mean_squared_error(y_test,base_pred))
rms=np.sqrt(mean_squared_error(y_test,prediction))
print(base_rms)
print(rms)
x1=data1.drop("likes",axis=1,inplace=False)
x1=sm.add_constant(x1)
y1=data1["likes"]

model=sm.OLS(y,x).fit()
print(model.summary())