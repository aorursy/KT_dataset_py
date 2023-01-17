## importing the packages needed



import pandas as pd

import numpy as np

from sklearn.feature_selection import VarianceThreshold

import sklearn.preprocessing as prep

from sklearn.decomposition import PCA



from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import DBSCAN



import warnings

warnings.filterwarnings('ignore')
data_orig = pd.read_csv("../input/dataset.csv", sep = ",")

data = data_orig
## inspecting the data



data.head()
data.iloc[:,:13].head()
data.iloc[:,13:].head()
data.info()
null_columns = data.columns[data.isnull().any()]
null_columns
for col in data.columns:

    if data[col].dtype=="object" and col != "id":

        print(data[col].value_counts())
data.info()
## findinhg the "?"'s present in the data'



rep = set()

for i in range(len(data)) :

    for j in range(len(data.columns)) :

        if data.columns[j] == "#Credits" or data.columns[j]=="#Authorities" or data.columns[j]=="Expatriate" or data.columns[j]=="Class":

            continue

        if data.iloc[i,j] == "?" :

            rep.add(data.columns[j])

rep
## replacing them with 0



data = data.replace("?","0")
data.iloc[:,:13].head()
data.iloc[:,13:].head()
## typecasting to (numeric)



data["Credit1"] = data["Credit1"].astype(int)

data["Monthly Period"] = data["Monthly Period"].astype(int)



data["InstallmentRate"] = data["InstallmentRate"].astype(int)



data["Age"] = data["Age"].astype(int)

data["#Credits"] = data["#Credits"].astype(int)

data["Tenancy Period"] = data["Tenancy Period"].astype(int)

data["#Authorities"] = data["#Authorities"].astype(int)



data["Expatriate"] = data["Expatriate"].astype(int)



data["Yearly Period"] = data["Yearly Period"].astype(float)

data["InstallmentCredit"] = data["InstallmentCredit"].astype(float)
data.info()
## unique values of type(object)



for col in rep:

    if data[col].dtype=="object" :

        print(data[col].value_counts())
## Replacing all cells with "0" with most occuring values in that column ##



idx = data[data["History"]=="0"].index[0]

data["History"][idx] = "c2"

idx = data[data["Account1"]=="0"].index[0]

data["Account1"][idx] = "ad"

idx = data[data["Motive"]=="0"].index[0]

data["Motive"][idx] = "p3"



## Unique value counts of type("numeric") ##





for col in rep :

    if data[col].dtype!="object" :

        print(data[col].value_counts())


## Replacing entries having "0" with average values of that attribute



idx = data[data["InstallmentRate"]==0].index[0]

data["InstallmentRate"][idx] = 4



idx = data[data["Monthly Period"]==0].index

for i in idx :

    data["Monthly Period"][i] = 24



idx = data[data["Tenancy Period"]==0].index[0]

data["Tenancy Period"][idx] = 4



idx = data[data["InstallmentCredit"]==0].index

for i in idx :

    data["InstallmentCredit"][i] = data["InstallmentCredit"].mean()



idx = data[data["Age"]==0].index

for i in idx :

    data["Age"][i] = 24



idx = data[data["Credit1"]==0].index

for i in idx :

    data["Credit1"][i] = 1262



idx = data[data["Yearly Period"]==0].index

for i in idx :

    data["Yearly Period"][i] = data["Yearly Period"].mean()
for col in rep :

    if data[col].dtype!="object" :

        print(data[col].value_counts())
cols = ["Account2","Sponsors","Plotsize"]



for col in cols :

    print("{} : {}".format(col,data[col].unique()))

    print()
## Fixing columns with duplicate feature values ##







idx = data[data["Sponsors"]=="g1"].index

for i in idx :

    data["Sponsors"][i] = "G1"





idx = data[data["Account2"]=="Sacc4"].index

for i in idx :

    data["Account2"][i] = "sacc4"    

    

    

idx = data[data["Plotsize"]=="sm"].index

for i in idx :

    data["Plotsize"][i] = "SM"



idx = data[data["Plotsize"]=="me"].index

for i in idx :

    data["Plotsize"][i] = "ME"



idx = data[data["Plotsize"]=="M.E."].index

for i in idx :

    data["Plotsize"][i] = "ME"



idx = data[data["Plotsize"]=="la"].index

for i in idx :

    data["Plotsize"][i] = "LA"
cls_cols = [col for col in data.columns if data[col].dtype=="object" and col!="id"]

cls_cols.append("InstallmentRate")

cls_cols.append("Tenancy Period")

cls_cols.append("#Credits")

cls_cols.append("#Authorities")



cls_cols.append("Expatriate")

cls_cols.append("Class")





for col in cls_cols:

    print("{} : {}".format(col,data[col].unique()))

    print()
## Dropping id column ##



idd = data["id"]

data = data.drop("id",axis=1)
## Encoding columns with type("object") to one-hot encoder ##



data = pd.get_dummies(data)
data
data.head()
data.info()
data.describe()
## Correlation factor of Class with other attributes 



cols = ["Monthly Period","Credit1","InstallmentRate","Tenancy Period","Age","#Credits","#Authorities","Expatriate","InstallmentCredit","Yearly Period","Class"]

corr = pd.DataFrame(columns=cols)

for col in cols :

    corr[col] = data[col][:175]

corr = corr.corr()

corr["Class"]
data1 = data.copy()
## Creating y_train and X



y_train = data1["Class"][:175].values

data1 = data1.drop("Class",axis=1)

X = data1.values

print(X.shape)
## Applying MinMax feature Scaling ##



X = prep.MinMaxScaler().fit_transform(X)
## Applying PCA with variance 0.968 ##



model   = PCA(.968,random_state=42)

X = model.fit_transform(X)

X_test  = X[175:]

X_train = X[:175] 

print(X_test.shape, X_train.shape)
###### AgglomerativeClustering(affinity:cosine) : score = 0.50000 on last try, highest 0.51869 (did not save the code for this one) #######

from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, affinity = 'cosine',linkage="average")

model.fit(X_test)

pred = model.labels_.astype("int")
ans = pd.DataFrame(columns=["id","Class"])

ans["id"] = idd[175:]

ans["Class"] = pred
from IPython.display import HTML

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(ans)