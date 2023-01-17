# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path="//kaggle//input//Data.csv"
# First Original Data
df=pd.read_csv(path)
df.head()
df=df.drop(columns=["order_id","line_item_type_id"])
df.head()
df.info()
df["site_id"].unique()
site351=df[df["site_id"]==351]
df=df.dropna()
#site351.plot.bar(x="geo_id",y="ad_type_id")
site351.head()
second=site351[:100]
second.plot.bar(x="ad_type_id",y="geo_id")
df[:1000].plot.bar(x="monetization_channel_id",y="geo_id")
df["monetization_channel_id"].unique()
temp1=[]
temp2=[]
for each in df["monetization_channel_id"].unique():
    val=df[df["monetization_channel_id"]==each].count()
    temp1.append(each)
    temp2.append(val[0])
    print(each,val[0])
plt.bar(temp1,temp2)
plt.show()
# 19> 4> 1> 2==21
df[:1000].plot.line(x="monetization_channel_id",y="total_impressions",rot=0)
# Most Impressions through Monetization ID 19
df[:1000].plot.line(x="monetization_channel_id",y="total_revenue",rot=0)
# More revenue from 19 > 4
df[:1000].plot.line(x="total_impressions",y="total_revenue",rot=0)
df["total_impressions"].corr(df["total_revenue"])
df.corr()
((df["total_revenue"]/df["total_impressions"])*1000).unique()[:100]
pd.get_dummies(df["site_id"],prefix="siteid")
pd.get_dummies(df["os_id"],prefix="os")
pd.get_dummies(df["ad_type_id"],prefix="adid")
#Here I'm trying to figure out the correlations between different Columns
cl="total_impressions"
df["ad_type_id"].corr(df[cl]) #Neg
df["site_id"].corr(df[cl]) #0.01
df["geo_id"].corr(df[cl]) #0.12
df["device_category_id"].corr(df[cl]) #Neg
df["advertiser_id"].corr(df[cl]) #0.0
df["os_id"].corr(df[cl])  #0.0
df["integration_type_id"].corr(df[cl]) #Nan
df["monetization_channel_id"].corr(df[cl]) #0.0
df["ad_unit_id"].corr(df[cl]) #Neg
df["total_revenue"].corr(df[cl]) #7.38
df["revenue_share_percent"].corr(df[cl]) #Nan
#Here I'm trying to figure out the correlations between different Columns
cl="total_revenue"
# df["ad_type_id"].corr(df[cl]) #Neg
# df["site_id"].corr(df[cl]) #Neg
# df["geo_id"].corr(df[cl]) #0.12
# df["device_category_id"].corr(df[cl]) #Neg
# df["advertiser_id"].corr(df[cl]) #Neg
# df["os_id"].corr(df[cl])  #0.0
# df["integration_type_id"].corr(df[cl]) #Nan
# df["monetization_channel_id"].corr(df[cl]) #0.0
# df["ad_unit_id"].corr(df[cl]) #Neg
# df["total_impressions"].corr(df[cl]) #7.38
# df["revenue_share_percent"].corr(df[cl]) #Nan
df[(df["revenue_share_percent"]==1)].nunique()
df[df["total_impressions"]==14452.0]
data=df.nlargest(10000,["total_impressions"])["site_id"]
data=list(data)
ans={}
for each in data:
    ans.update({each:data.count(each)})
ans
df.nlargest(20,["total_revenue"])[["site_id","total_revenue","total_impressions","ad_type_id"]]

"""
Here I'm plotting all the relational Values between total_revenue to inference what drive more revenue
"""

def plotter(one,two,c):
    first1=[]
    second1=[]
    for each in df[one].unique():
        first1.append(each)
        temp=df[df[one]==each][two].sum()
        second1.append(temp)
    #     print(each,temp)
    print("Relation between {} and {}".format(one,two))
    plt.bar(first1,second1,color=c,align="center")
    # ax.autoscale(tight=True)
    plt.show()
#Relations between the Site_id and Total_impressions
plotter("site_id","total_impressions","b")
#Relations between the Site_id and Total_revenue

plotter("site_id","total_revenue","g")

plotter("geo_id","total_revenue","y")
plotter("device_category_id","total_revenue","y")
plotter("advertiser_id","total_revenue","g")

# plotter("advertiser_id","total_revenue","y")

plotter("os_id","total_revenue","b")
plotter("ad_type_id","total_revenue","b")
first=df[df["ad_type_id"]==10]
second=first[first["site_id"]==349]
second.describe()

df["ad_type_id"].unique()
