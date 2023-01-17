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
import pandas as pd

data = pd.read_csv("../input/customer-segmentation/Online Retail.csv")

data.head()
data.shape
data.keys()
data.describe()
data = data.loc[data["Quantity"]>0]

data.shape
data.info()
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

data.info()  #verifying the result
data["Sale"] = data.Quantity * data.UnitPrice

data.head()
monetary = data.groupby("CustomerID").Sale.sum()

monetary = monetary.reset_index()

monetary.head()
frequency = data.groupby("CustomerID").InvoiceNo.count()

frequency = frequency.reset_index()

frequency.head()
LastDate = max(data.InvoiceDate)

LasDate = LastDate + pd.DateOffset(days = 1)

data["Diff"] = LastDate - data.InvoiceDate

data["Diff"] = data["Diff"].dt.days

recency = data.groupby("CustomerID").Diff.min()

recency = recency.reset_index()

recency.head()
RMF = monetary.merge(frequency, on = "CustomerID")

RMF = RMF.merge(recency, on = "CustomerID")

RMF.columns = ["CustomerID", "Monetary", "Frequency", "Recency"]

RMF
RMF1 = pd.DataFrame(RMF,columns= ["Monetary","Frequency","Recency"])

RMF1 

'''from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

RMF = ss.fit_transform(RMF)

RMF1 = pd.DataFrame(RMF, columns=["Monetary", "Frequency", "Recency"])

RMF1'''
from sklearn.cluster import KMeans

ssd = []

for k in range(1,20):

    km = KMeans(n_clusters = k)

    km.fit(RMF1)

    ssd.append(km.inertia_)

import matplotlib.pyplot as plt

import numpy as np

plt.plot(np.arange(1,20), ssd, color = "blue")

plt.scatter(np.arange(1,20), ssd, color = "red")

plt.show()


model = KMeans(n_clusters = 5)

ClusterID = model.fit_predict(RMF1)

RMF1["ClusterID"] = ClusterID

RMF1
km_cluster_Sale = RMF1.groupby("ClusterID").Monetary.mean()

km_cluster_Recency = RMF1.groupby("ClusterID").Recency.mean()

km_cluster_Frequency = RMF1.groupby("ClusterID").Frequency.mean()


import seaborn as sns

fig, axs = plt.subplots(1,3, figsize = (15,5))

sns.barplot(x = [0,1,2,3,4], y= km_cluster_Sale,ax = axs[0])

sns.barplot(x = [0,1,2,3,4], y= km_cluster_Frequency,ax = axs[1])

sns.barplot(x = [0,1,2,3,4], y= km_cluster_Recency,ax = axs[2])

fig, axs = plt.subplots(1,3, figsize = (15, 5))

ax1 = fig.add_subplot(1,3,1)

ax1.pie(km_cluster_Sale,labels = [0,1,2,3,4])

plt.title("Monetary Mean")



ax2 = fig.add_subplot(1,3,2)

ax2.pie(km_cluster_Frequency,labels = [0,1,2,3,4])

plt.title("Frequency Mean")



ax3 = fig.add_subplot(1,3,3)

ax3.pie(km_cluster_Recency,labels = [0,1,2,3,4])

plt.title("Recency Mean")

plt.axis("off")

plt.show()


