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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("../input/online retail.csv")
df.head()
df.shape
df.describe()


df = df.loc[df["Quantity"] > 0]
df.shape
df.info()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df.info()
df["Sale"] = df.Quantity * df.UnitPrice
df

monetory = df.groupby("CustomerID").Sale.sum()
monetory  = monetory.reset_index()
monetory
frequency = df.groupby("CustomerID").InvoiceNo.count()
frequency = frequency.reset_index()
frequency
lastdate = max(df.InvoiceDate)
lastdate = lastdate + pd.DateOffset(days = 1)
df["Diff"] = lastdate - df.InvoiceDate
df["Diff"] = df["Diff"].dt.days
df.info()
recency = df.groupby("CustomerID").Diff.min()
recency = recency.reset_index()
recency

rmf = monetory.merge(frequency , on = "CustomerID")
rmf = rmf.merge(recency, on = "CustomerID")
rmf
rmf.columns = ["CustomerID", "Monetory", "Frequency", "Recency"]
rmf

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
rmf1 = ss.fit_transform(rmf)
rmf1 = pd.DataFrame(rmf1 , columns = ["CustomerID", "Monetory", "Frequency", "Recency"])
rmf1
rmf1 = rmf1.drop("CustomerID",axis =1)
rmf1.head()
from sklearn.cluster import KMeans
sse = []
n = []
for i in range(1,20):
    n.append(i)
    model = KMeans(n_clusters = i)
    model.fit(rmf1)
    sse.append(model.inertia_)
    

plt.plot(n , sse, color = "green")
plt.scatter(n,sse , color = "black")
plt.show()
model = KMeans(n_clusters = 5)
ClusterID = model.fit_predict(rmf1)
rmf["ClusterID"] = ClusterID
rmf
monetory = rmf.groupby("ClusterID").Monetory.mean()
frequency = rmf.groupby("ClusterID").Frequency.mean()
recency = rmf.groupby("ClusterID").Recency.mean()
import seaborn as sns
fig , axis = plt.subplots(1,3, figsize = (20,10))
sns.barplot(x = np.arange(0,5), y = monetory, ax = axis[0])
sns.barplot(x = np.arange(0,5), y = frequency, ax = axis[1])
sns.barplot(x = np.arange(0,5), y = recency, ax = axis[2])
fig,axis = plt.subplots(1,3 , figsize = (20,10))
ax1 = fig.add_subplot(1, 3, 1)
plt.title("Sale Mean")
ax1.pie(monetory, labels = [0,1,2,3,4], colors = ["red","blue", "green", "brown", "black"])

ax1 = fig.add_subplot(1, 3, 2)
plt.title("Frequency Mean")
ax1.pie(frequency, labels = [0,1,2,3,4],  colors = ["red","blue", "green", "brown", "black"])

ax1 = fig.add_subplot(1, 3, 3)
plt.title("Recency Mean")
ax1.pie(recency, labels = [0,1,2,3,4],  colors = ["red","blue", "green", "brown", "black"])
plt.axis("off")
plt.show()

