import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization
%matplotlib inline


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/montcoalert/911.csv')
df.info()
df.head()
df["zip"].value_counts().head(5)
df["twp"].value_counts().head(5)
df["Reason"]=df["title"].apply(lambda x: x.split(":")[0])
df.head()
df["Reason"].value_counts()
sns.countplot(x="Reason",data=df)
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week']=df["Day of Week"].map(dmap)
sns.countplot(x="Day of Week",data=df,hue="Reason")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.9))
sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
df["Date"]=df["timeStamp"].apply(lambda time: time.date())
df.head()
plt.figure(figsize=(16,4))
dd=df.groupby("Date").count()
dd["e"].plot()
plt.tight_layout()
plt.figure(figsize=(16,4))
Traffic=df[df["Reason"]=="Traffic"].groupby("Date").count()
Traffic["e"].plot().set_title("Traffic")
plt.figure(figsize=(16,4))
Fire=df[df["Reason"]=="Fire"].groupby("Date").count()
Fire["e"].plot().set_title("Fire")
plt.figure(figsize=(16,4))
EMS=df[df["Reason"]=="EMS"].groupby("Date").count()
EMS["e"].plot().set_title("EMS")
df.groupby(by=["Day of Week","Hour"]).count()["e"].unstack(level=-1)
newdf=df.groupby(by=["Day of Week","Hour"]).count()["e"].unstack(level=-1)
plt.figure(figsize=(12,6))
sns.heatmap(data=newdf,cmap="viridis")
sns.clustermap(newdf)
df.groupby(by=["Day of Week","Month"]).count()["e"].unstack()
md=df.groupby(by=["Day of Week","Month"]).count()["e"].unstack()
sns.heatmap(md)
sns.clustermap(md)
