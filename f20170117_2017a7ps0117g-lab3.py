## This is the Submission with best score
#Importing the required libraries

import pandas as pd

import numpy as np

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
#reading required csv files into respective dataframes

df1 = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

df20 = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")
#Quering information of dataframe of train dataset

df1.info()
#Quering information of dataframe of train dataset

df1.head(15)
df1.replace(r'^\s*$', np.nan, regex=True, inplace = True)

df1.TotalCharges.fillna(df1.MonthlyCharges, inplace=True)
df1["TotalCharges"]=df1.TotalCharges.astype("float64")
le = LabelEncoder()

df1['gender']= le.fit_transform(df1['gender'])

df1['Married']= le.fit_transform(df1['Married'])

df1['Children']= le.fit_transform(df1['Children'])

df1['TVConnection']= le.fit_transform(df1['TVConnection'])

df1['Channel1']= le.fit_transform(df1['Channel1'])

df1['Channel2']= le.fit_transform(df1['Channel2'])

df1['Channel3']= le.fit_transform(df1['Channel3'])

df1['Channel4']= le.fit_transform(df1['Channel4'])

df1['Channel5']= le.fit_transform(df1['Channel5'])

df1['Channel6']= le.fit_transform(df1['Channel6'])

df1['Internet']= le.fit_transform(df1['Internet'])

df1['HighSpeed']= le.fit_transform(df1['HighSpeed'])

df1['AddedServices']= le.fit_transform(df1['AddedServices'])

df1['Subscription']= le.fit_transform(df1['Subscription'])

df1['PaymentMethod']= le.fit_transform(df1['PaymentMethod'])
df2 = df1

ss = StandardScaler()

df1 = pd.DataFrame(ss.fit_transform(df1), columns = df1.columns)
df1
kmeans = KMeans(n_clusters=2, random_state=0, n_init=500)
df1 = df1.drop(columns=["custId", "Satisfied"])

df = df1
#plot heat map

plt.subplots(figsize=(20,9))

g=sns.heatmap(df.corr(),annot=True,cmap="RdYlGn")
# delete a column and calulate accuracy score

y_true = df2.Satisfied

scores = []

for x in df2.columns:

    if (x != "Satisfied") and (x != "custId"):

        df = df1

        df = df.drop(columns=[x])

        kmeans.fit(df)

        y_pred = kmeans.labels_

        scores.append(accuracy_score(y_true, y_pred))

        print(x)

scores
## Dropping TotalCharges column

y_true = df2.Satisfied

df = df1

df = df.drop(columns=["TotalCharges"])

kmeans.fit(df)

y_pred = kmeans.labels_

accuracy_score(y_true, y_pred)
#df2 now contains test.csv

df2 = df20
df2.head(20)
df2.info()
le = LabelEncoder()

df2['gender']= le.fit_transform(df2['gender'])

df2['Married']= le.fit_transform(df2['Married'])

df2['Children']= le.fit_transform(df2['Children'])

df2['TVConnection']= le.fit_transform(df2['TVConnection'])

df2['Channel1']= le.fit_transform(df2['Channel1'])

df2['Channel2']= le.fit_transform(df2['Channel2'])

df2['Channel3']= le.fit_transform(df2['Channel3'])

df2['Channel4']= le.fit_transform(df2['Channel4'])

df2['Channel5']= le.fit_transform(df2['Channel5'])

df2['Channel6']= le.fit_transform(df2['Channel6'])

df2['Internet']= le.fit_transform(df2['Internet'])

df2['HighSpeed']= le.fit_transform(df2['HighSpeed'])

df2['AddedServices']= le.fit_transform(df2['AddedServices'])

df2['Subscription']= le.fit_transform(df2['Subscription'])

df2['PaymentMethod']= le.fit_transform(df2['PaymentMethod'])
df2.head(20)
df2.replace(r'^\s*$', np.nan, regex=True, inplace = True)

df2.TotalCharges.fillna(df2.MonthlyCharges, inplace=True)
df2["TotalCharges"]=df1.TotalCharges.astype("float64")

df4 = df2
ss = StandardScaler()

df2 = pd.DataFrame(ss.fit_transform(df2), columns = df2.columns)
df2 = df2.drop(columns=["custId","TotalCharges"])

df2.info()
arr1 = kmeans.predict(df2)
df4["Satisfied"] = arr1
df4.head(20)
df5 = df4[["custId", "Satisfied"]]

df5.to_csv("Result14.csv", index=False)
## This is the Second Submission
#Importing the required libraries

import pandas as pd

import numpy as np

import sklearn

import seaborn as sns

from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
#reading required csv files into respective dataframes

df1 = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

df2 = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")
#Quering information of dataframe of train dataset

df1.info()
#Quering information of dataframe of train dataset

df1.head(15)
df1.replace(r'^\s*$', np.nan, regex=True, inplace = True)

df1.TotalCharges.fillna(df1.MonthlyCharges, inplace=True)
df1["TotalCharges"]=df1.TotalCharges.astype("float64")
le = LabelEncoder()

df1['gender']= le.fit_transform(df1['gender'])

df1['Married']= le.fit_transform(df1['Married'])

df1['Children']= le.fit_transform(df1['Children'])

df1['TVConnection']= le.fit_transform(df1['TVConnection'])

df1['Channel1']= le.fit_transform(df1['Channel1'])

df1['Channel2']= le.fit_transform(df1['Channel2'])

df1['Channel3']= le.fit_transform(df1['Channel3'])

df1['Channel4']= le.fit_transform(df1['Channel4'])

df1['Channel5']= le.fit_transform(df1['Channel5'])

df1['Channel6']= le.fit_transform(df1['Channel6'])

df1['Internet']= le.fit_transform(df1['Internet'])

df1['HighSpeed']= le.fit_transform(df1['HighSpeed'])

df1['AddedServices']= le.fit_transform(df1['AddedServices'])

df1['Subscription']= le.fit_transform(df1['Subscription'])

df1['PaymentMethod']= le.fit_transform(df1['PaymentMethod'])
ss = StandardScaler()

df1 = pd.DataFrame(ss.fit_transform(df1), columns = df1.columns)
agglomerativeClustering = AgglomerativeClustering(n_clusters=2)
df = df1
df.head(15)
df = df.drop(columns=["custId", "Satisfied"])
#plot heat map

plt.subplots(figsize=(20,9))

g=sns.heatmap(df.corr(),annot=True,cmap="RdYlGn")
agglomerativeClustering.fit(df)
arr = agglomerativeClustering.labels_
df["predicted"] = arr
df2.head(20)
df2.info()
le = LabelEncoder()

df2['gender']= le.fit_transform(df2['gender'])

df2['Married']= le.fit_transform(df2['Married'])

df2['Children']= le.fit_transform(df2['Children'])

df2['TVConnection']= le.fit_transform(df2['TVConnection'])

df2['Channel1']= le.fit_transform(df2['Channel1'])

df2['Channel2']= le.fit_transform(df2['Channel2'])

df2['Channel3']= le.fit_transform(df2['Channel3'])

df2['Channel4']= le.fit_transform(df2['Channel4'])

df2['Channel5']= le.fit_transform(df2['Channel5'])

df2['Channel6']= le.fit_transform(df2['Channel6'])

df2['Internet']= le.fit_transform(df2['Internet'])

df2['HighSpeed']= le.fit_transform(df2['HighSpeed'])

df2['AddedServices']= le.fit_transform(df2['AddedServices'])

df2['Subscription']= le.fit_transform(df2['Subscription'])

df2['PaymentMethod']= le.fit_transform(df2['PaymentMethod'])
df2.head(20)
df2.replace(r'^\s*$', np.nan, regex=True, inplace = True)

df2.TotalCharges.fillna(df2.MonthlyCharges, inplace=True)
df2["TotalCharges"]=df2.TotalCharges.astype("float64")

df4 = df2
ss = StandardScaler()

df2 = pd.DataFrame(ss.fit_transform(df2), columns = df2.columns)
df2 = df2.drop(columns=["custId"])
arr1 =agglomerativeClustering.fit_predict(df2)
df4["Satisfied"] = arr1
df4.head(20)
df5 = df4[["custId", "Satisfied"]]

df5.to_csv("Result19.csv", index=False)