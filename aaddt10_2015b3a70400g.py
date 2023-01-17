import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv("../input/dataset.csv")

df.head()
pd.set_option('display.max_columns', 500)
df_now = df
df_now = df_now.replace({"?": np.nan})
df_now.tail()
df_now = df_now.fillna(df.mode().iloc[0])
df_now = df_now.replace({"?": np.nan})
df_now = df_now.fillna(0)
df_now= df_now.drop(columns=["id"])
cl = df["Class"].values
df_now=df_now.drop(columns = ["Class"])
df_cat = pd.get_dummies(df_now, columns=['Account1', 'History', 'Motive', 'Account2', 'Employment Period', 'Gender&Type', 'Sponsors','Plotsize', 'Plan', 'Housing','Post', 'Expatriate'])
df_cat=df_cat.drop(columns=["Phone"])
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(df_cat)

dataN1 = pd.DataFrame(np_scaled)

dataN1.head()
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression
model = LogisticRegression() 

rfe = RFE(model, 18)
len(dataN1.columns)
X=dataN1.values
X.shape
fit = rfe.fit(X[:175], cl[:175])
fit.support_
dataN2 = dataN1
len(dataN2.columns)
to_be_dropped = []

for i in range(len(fit.support_)):

  if (fit.support_[i]==False):

    to_be_dropped.append(i)
len(to_be_dropped)
dataN2= dataN2.drop(columns=to_be_dropped)
dataN2.head()
from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(dataN2)

T1 = pca1.transform(dataN2)
plt.scatter(T1[:,0],T1[:, 1])
from sklearn.cluster import KMeans

preds1=[]

kmean = KMeans(n_clusters = 7, random_state = 42)

kmean.fit(dataN2)

pred = kmean.predict(dataN2)

preds1.append(pred)

plt.scatter(T1[:, 0], T1[:, 1], c=pred)
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan', 'black']
pred_colors=[]

for i in pred:

  pred_colors.append(colors[i])
df_up=dataN2[:175]

df_down=dataN2[175:]
pred = kmean.predict(dataN2)



plt.scatter(T1[175:, 0], T1[175:, 1], c=pred_colors[175:])
pred = kmean.predict(dataN2)



plt.scatter(T1[:175, 0], T1[:175, 1], c=pred_colors[:175])
label_colors=[]

for i in cl[:175]:

  label_colors.append(colors[int(i)])
pred = kmean.predict(dataN2)



plt.scatter(T1[:175, 0], T1[:175, 1], c=label_colors)
hdict={'red':2,'green': 0,'blue':2,'yellow':0,'purple':1,'pink':1,'palegreen':2,'violet':9,'cyan':9}

#red=0 green = 1 blue =2                                             try
curr_preds = pred_colors
curr_nums=[]

for i in curr_preds:

  curr_nums.append(hdict[i])
corr=0

for i in range(175):

  if curr_nums[i]==cl[i]:

    corr=corr+1
corr/175
idx=df["id"].values
len(idx)
dicty= {"id" : idx[175:], "Class" : curr_nums[175:]}
sub=pd.DataFrame.from_dict(dicty)
sub=sub[['id', 'Class']]
sub.head()
sub.to_csv("2015B3A70400G.csv", index=False)

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(sub)