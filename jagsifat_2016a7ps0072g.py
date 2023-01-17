##Importing packages##

import pandas as pd

import numpy as np

from sklearn.feature_selection import VarianceThreshold

import sklearn.preprocessing as pp

from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import Birch

from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings('ignore')
##Creation of dataframe##

df = pd.read_csv("../input/dataset.csv")

df.iloc[:10,:]
df.iloc[:10,:17]
df.iloc[:10,17:]
##details of attributes##

df.info()
##Storing columns with atleast one '?'##

missing = set()

for i in range (len(df)):

    for j in range (len(df.columns)):

        if df.columns[j] == "#Authorities" or df.columns[j] == "#Credits" or df.columns[j] == "Class" or df.columns[j] == "Expatriate":

            continue

        if df.iloc[i,j] == "?":

            missing.add(df.columns[j])

            

missing
##Dropping duplicates##

df = df.drop_duplicates()

df.info()
#Replacing all '?'s with '0's##

df = df.replace("?","0")
df.iloc[:10,:17]
df.iloc[:10,17:]
##Conversion of objects with numerals to data type 'int'##

df["InstallmentRate"] = df["InstallmentRate"].astype(int)

df["Expatriate"] = df["Expatriate"].astype(int)

df["Credit1"] = df["Credit1"].astype(int)

df["#Credits"] = df["#Credits"].astype(int)

df["Tenancy Period"] = df["Tenancy Period"].astype(int)

df["Age"] = df["Age"].astype(int)

df["Yearly Period"] = df["Yearly Period"].astype(float)

df["#Authorities"] = df["#Authorities"].astype(int)

df["Monthly Period"] = df["Monthly Period"].astype(int)

df["InstallmentCredit"] = df["InstallmentCredit"].astype(float)
df.info()
##Unique values of columns with data type 'object'##

for column in missing :

    if df[column].dtype=="object" :

        print(df[column].value_counts())
##Replacing '0's with the maximum occuring entry##

x = df[df["Account1"]=="0"].index[0]

df["Account1"][x] = "ad"

x = df[df["Motive"]=="0"].index[0]

df["Motive"][x] = "p3"

x = df[df["History"]=="0"].index[0]

df["History"][x] = "c2"
##Unique values of columns with data type other than 'object'##

for column in missing :

    if df[column].dtype!="object" :

        print(df[column].value_counts())
##Replacing '0's with the average value of the numeric entries##

x = df[df["InstallmentCredit"]==0].index

for i in x :

    df["InstallmentCredit"][i] = df["InstallmentCredit"].mean()



x = df[df["Age"]==0].index

for i in x :

    df["Age"][i] = 24



x = df[df["InstallmentRate"]==0].index[0]

df["InstallmentRate"][x] = 4



x = df[df["Credit1"]==0].index

for i in x :

    df["Credit1"][i] = 1262



x = df[df["Monthly Period"]==0].index

for i in x :

    df["Monthly Period"][i] = 24

    

x = df[df["Yearly Period"]==0].index

for i in x :

    df["Yearly Period"][i] = df["Yearly Period"].mean()



x = df[df["Tenancy Period"]==0].index[0]

df["Tenancy Period"][x] = 4



for column in missing :

    if df[column].dtype!="object" :

        print(df[column].value_counts())
##Unique counts - categorical attributes##

col_list = [column for column in df.columns if df[column].dtype == "object"]

col_list.append("InstallmentRate")

col_list.append("Tenancy Period")

col_list.append("#Credits")

col_list.append("#Authorities")

col_list.append("Expatriate")

col_list.append("Class")

for column in col_list:

    print("{} : {}".format(column,df[column].unique()))

    print()
##Saving the dataframe##

df.to_csv("dataset_preprocessing_done.csv")
##Dropping the 'id' column##

idd = df["id"]

df = df.drop("id",axis=1)
##Creating a copy of df##

cp = df.copy()
##One Hot encoding##

cp = pd.get_dummies(cp)
cp.head()
##Creating a copy of cp##

cp_1 = cp.copy()
##Creating x_test and x_train##

y_train = cp_1["Class"][:175].values

cp_1 = cp_1.drop("Class",axis=1)

x = cp_1.values

print(x.shape)
##Applying MinMax##

x = pp.MinMaxScaler().fit_transform(x)
##Applying PCA with 0.966 variance##

model = PCA(.966)

x = model.fit_transform(x)

x_test  = x[175:]

x_train = x[:175] 

print(x_test.shape, x_train.shape)
##Agglomerative, cosine##

from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, affinity = 'cosine',linkage="average")

model.fit(x_test)

pr = model.labels_.astype("int")

final = pd.DataFrame(columns=["id","Class"])

final["id"] = idd[175:]

final["Class"] = pr
from IPython.display import HTML

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(final)