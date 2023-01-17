import numpy as np

import pandas as pd
df = pd.read_csv("/kaggle/input/dmassign1/data.csv", sep=",")
df.info()
df.head()
df.tail()
pd.set_option('display.max_rows', 200)

df.dtypes
## Replace ? with NaN

df = df.replace('?', np.NaN)

df.to_csv( "Formatted.csv", index=False)
# Listing Columns with NaN

null_columns = df.columns[df.isnull().any()]

null_columns
##Filling Int Column's NaN with Mean

Int_Columns = ['Col30', 'Col31', 'Col34', 'Col36', 'Col37', 'Col38', 'Col39', 'Col40',

       'Col43', 'Col44', 'Col46', 'Col47', 'Col48', 'Col49', 'Col50', 'Col51',

       'Col53', 'Col56', 'Col138', 'Col139', 'Col140', 'Col141', 'Col142',

       'Col143', 'Col144', 'Col145', 'Col146', 'Col147', 'Col148', 'Col149',

       'Col151', 'Col152', 'Col153', 'Col154', 'Col155', 'Col156', 'Col157',

       'Col158', 'Col159', 'Col160', 'Col161', 'Col162', 'Col173', 'Col174',

       'Col175']

for x in Int_Columns:

        df[x] = df[x].astype(float)

        df[x].fillna(df[x].mean(), inplace=True)
##Filling Float Column's NaN with Mean

Float_columns = ['Col179', 'Col180', 'Col181', 'Col182', 'Col183', 'Col184',

       'Col185', 'Col186', 'Col187']

for x in Float_columns:

    df[x] = df[x].astype(float)

    df[x].fillna(df[x].mean(), inplace=True)
null_columns = df.columns[df.isnull().any()]

null_columns
##Filling String Column's NaN with Mode

Str_columns = ['Col192', 'Col193', 'Col194', 'Col195',

       'Col196', 'Col197']

for x in Str_columns:

    df[x] = df[x].astype(str)

    df[x] = df[x].replace('nan', df[x].mode()[0])
## Converting Categorical Data into Numeric Labels

from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()



df["Col189"]= le.fit_transform(df["Col189"])

df["Col190"]= le.fit_transform(df["Col190"])

df["Col191"]= le.fit_transform(df["Col191"])

df["Col192"]= le.fit_transform(df["Col192"])

df["Col193"]= le.fit_transform(df["Col193"])

df["Col194"]= le.fit_transform(df["Col194"])

df["Col195"]= le.fit_transform(df["Col195"])

df["Col196"]= le.fit_transform(df["Col196"])

df["Col197"]= le.fit_transform(df["Col197"])
df1 = df.copy()
df1 = df1.drop(columns=["ID", "Class"])
## Run the Kmeans algo

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
kmeans.fit(df1)
predicted_labels_0_indexed = kmeans.labels_
predicted_labels = []

for x in predicted_labels_0_indexed:

    y = x + 1

    predicted_labels.append(y)
df2 = df[["ID"]]
df2["Class"] = predicted_labels

df3 = df2.copy()

df2["Truth"] = df["Class"]

df2 = df2.iloc[0:1300,:]
# Calculating accuracy to assigned labels, 120 possiblities to arrange the labels of predicted_labels

from itertools import permutations

from sklearn.metrics import accuracy_score



y_true = df2["Truth"].to_numpy()

l = [1, 2, 3, 4, 5]

predicted_labels_0_to_1299 = predicted_labels[0:1300]

l = list(permutations(l))

scores = []



for x in l:

    mapper_dict = { 1: x[0], 2: x[1], 3: x[2], 4: x[3], 5: x[4]}

    new_labels = [ mapper_dict[x] for x in predicted_labels_0_to_1299]

    scores.append([accuracy_score(y_true, new_labels), [x[0], x[1], x[2], x[3], x[4]]])
# Sort according to the score

scores.sort(key=lambda x: x[0])

scores
## # Best score of Kmeans along with cluster label assignment

scores[-1]
## Run the aggolo algo

from sklearn.cluster import AgglomerativeClustering

agglomerativeClustering = AgglomerativeClustering(n_clusters=5)
agglomerativeClustering.fit(df1)
predicted_labels_0_indexed = agglomerativeClustering.labels_
predicted_labels = []

for x in predicted_labels_0_indexed:

    y = x + 1

    predicted_labels.append(y)
df2 = df[["ID"]]
df2["Class"] = predicted_labels

df3 = df2.copy()

df2["Truth"] = df["Class"]

df2 = df2.iloc[0:1300,:]
# Calculating accuracy to assigned labels, 120 possiblities to arrange the labels of predicted_labels

from itertools import permutations

from sklearn.metrics import accuracy_score



y_true = df2["Truth"].to_numpy()

l = [1, 2, 3, 4, 5]

predicted_labels_0_to_1299 = predicted_labels[0:1300]

l = list(permutations(l))

scores = []



for x in l:

    mapper_dict = { 1: x[0], 2: x[1], 3: x[2], 4: x[3], 5: x[4]}

    new_labels = [ mapper_dict[x] for x in predicted_labels_0_to_1299]

    scores.append([accuracy_score(y_true, new_labels), [x[0], x[1], x[2], x[3], x[4]]])
# Sort according to the score

scores.sort(key=lambda x: x[0])

scores
# Best score of AgglomerativeClustering along with cluster label assignment

scores[-1]
## Run the Birch algo

from sklearn.cluster import Birch

birch = Birch(n_clusters=5)
birch.fit(df1)
predicted_labels_0_indexed = birch.labels_
predicted_labels = []

for x in predicted_labels_0_indexed:

    y = x + 1

    predicted_labels.append(y)
df2 = df[["ID"]]
df2["Class"] = predicted_labels

df3 = df2.copy()

df2["Truth"] = df["Class"]

df2 = df2.iloc[0:1300,:]
# Calculating accuracy to assigned labels, 120 possiblities to arrange the labels of predicted_labels

from itertools import permutations

from sklearn.metrics import accuracy_score



y_true = df2["Truth"].to_numpy()

l = [1, 2, 3, 4, 5]

predicted_labels_0_to_1299 = predicted_labels[0:1300]

l = list(permutations(l))

scores = []



for x in l:

    mapper_dict = { 1: x[0], 2: x[1], 3: x[2], 4: x[3], 5: x[4]}

    new_labels = [ mapper_dict[x] for x in predicted_labels_0_to_1299]

    scores.append([accuracy_score(y_true, new_labels), [x[0], x[1], x[2], x[3], x[4]]])
# Sort according to the score

scores.sort(key=lambda x: x[0])

scores
# Best score of Birch along with cluster label assignment

scores[-1]
df4 = df[["ID"]]
#Assigning acccording to best scores

mapper_dict = { 1: scores[-1][1][0], 

                2: scores[-1][1][1], 

                3: scores[-1][1][2], 

                4: scores[-1][1][3], 

                5: scores[-1][1][4]}

new_labels = [ mapper_dict[x] for x in predicted_labels]

df4["Class"] = new_labels
df4 = df4.iloc[1300:13000,:]
df4.to_csv("2017A7PS0117G.csv", index=False)
# ---------------------Modification Start--------------------------

# This notebook is representative of     2017A7PS0117G_psc.py   and     2017A7PS0117G_psc.pdf

# This code snippet was not added by me in the 2017A7PS0117G_psc.py 



# An additional formatted.csv [original data.csv without missing data] is outputted by mistake, 

# which is not used for scores on kaggle, and it is not selected for scoring by me on kaggle. Sorry for inconvenience!!



# Furthermore, this is the final notebook which contains Model1 ,Model2 & Model3 compared to 2017A7PS0117G_fsc.pdf

# that only has Model3, BUT final predictions of both the notebooks are the same (using only Model3). Sorry for inconvenience!



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

create_download_link(df4)

# ------------------------Modification End-----------------------------