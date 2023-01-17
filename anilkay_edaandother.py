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
data=pd.read_csv("../input/hfi_cc_2018.csv")

data.head()
freedomscore=data["hf_score"].groupby(data["countries"]).mean()

freedomscore
freedomscore.argmax()
freedomscore.argmin()
print("min:"+str(freedomscore.min()))

print("max:"+str(freedomscore.max()))

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

freedomscore.plot()
plt.figure(figsize=(12,12))

freedomscore.plot(kind="hist")
least=data[data["hf_score"]<5]

set(least["countries"])
most=data[data["hf_score"]>=8]

set(most["countries"])
set(data.columns)

#sns.relplot(data=data,x=)
plt.figure(figsize=(12,12))

sns.relplot(data=data,x="ef_legal_crime",y="hf_score")
plt.figure(figsize=(12,12))

sns.lmplot(data=data,x="ef_legal_crime",y="hf_score")
plt.figure(figsize=(12,12))

sns.relplot(data=data,x="ef_score",y="hf_score")
plt.figure(figsize=(12,12))

sns.lmplot(data=data,x="ef_score",y="hf_score")
plt.figure(figsize=(12,12))

sns.relplot(data=data,x="pf_ss_women",y="hf_score")
plt.figure(figsize=(12,12))

sns.countplot(data=data,x="pf_ss_women")
set(data["pf_religion"])
plt.figure(figsize=(12,12))

sns.relplot(data=data,x="pf_religion",y="hf_score")
plt.figure(figsize=(12,12))

sns.lmplot(data=data,x="pf_religion",y="hf_score")
from sklearn.cluster import KMeans

subset=data[["countries","pf_religion","hf_score","pf_ss_women","ef_score"]]

subset.head()
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

subset2=imp.fit_transform(subset.iloc[:,1:])

sc=StandardScaler()

scaled=sc.fit_transform(subset2)

kmn=KMeans(n_clusters=3, random_state=111)

clusters=kmn.fit_predict(scaled)
print(clusters)

print(len(clusters))
sondata=pd.DataFrame({

    "countries":data["countries"],

    "rel":data["pf_religion"],

    "hf":data["hf_score"],

    "clusters":clusters,

    "woman":data["pf_ss_women"]

})





sns.relplot(x="rel", y="hf", hue="clusters", data=sondata);
sns.relplot(x="woman", y="hf", hue="clusters", data=sondata);
which=sondata[["countries","clusters"]]
which[which["countries"]=="Turkey"]
which[which["countries"]=="Brazil"]
which[which["countries"]=="Germany"]
which[which["countries"]=="United States"]
which[which["countries"]=="Iraq"]
which[which["countries"]=="Zimbabwe"]
whichwithyear=pd.DataFrame({

    "countries":data["countries"],

    "clusters":clusters,

    "year":data["year"]

})
whichwithyear[whichwithyear["countries"]=="Iraq"]
whichwithyear[whichwithyear["countries"]=="France"]
whichwithyear[whichwithyear["countries"]=="Slovenia"]
whichwithyear[whichwithyear["countries"]=="Israel"]
whichwithyear[whichwithyear["countries"]=="Georgia"]
whichwithyear[whichwithyear["countries"]=="Iran"]