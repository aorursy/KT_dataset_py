import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import cross_val_score

from sklearn import cluster 

from sklearn.metrics import accuracy_score

#import data from csv to dataframe df

df = pd.read_csv("../input/dmassign1/data.csv", sep=",",na_values='?') #dataframe object
df['Col189'] = df['Col189'].apply(lambda x: 1 if x=='yes' else '0')

#df['Col197']=df['Col197'].str.lower()

#replacing m.e. with me in col 197

#df["Col197"]= df["Col197"].replace('m.e.', "me") 
num_nul_col = ['Col30', 'Col31', 'Col34', 'Col36', 'Col37', 'Col38', 'Col39', 'Col40',

       'Col43', 'Col44', 'Col46', 'Col47', 'Col48', 'Col49', 'Col50', 'Col51',

       'Col53', 'Col56', 'Col138', 'Col139', 'Col140', 'Col141', 'Col142',

       'Col143', 'Col144', 'Col145', 'Col146', 'Col147', 'Col148', 'Col149',

       'Col151', 'Col152', 'Col153', 'Col154', 'Col155', 'Col156', 'Col157',

       'Col158', 'Col159', 'Col160', 'Col161', 'Col162', 'Col173', 'Col174',

       'Col175', 'Col179', 'Col180', 'Col181', 'Col182', 'Col183', 'Col184',

       'Col185', 'Col186', 'Col187']

cat_nul_col = ['Col192', 'Col193', 'Col194', 'Col195',

       'Col196', 'Col197']

cat_columns = ['Col190','Col191','Col192', 'Col193', 'Col194', 'Col195',

       'Col196', 'Col197']

classes = df['Class']
df[num_nul_col]=df[num_nul_col].fillna(value=df[num_nul_col].mean())

df[cat_nul_col]=df[cat_nul_col].fillna(value = df[cat_nul_col].mode().iloc[0])
df.columns[df.isnull().any()]
#one hot encoding

df_onehot = df.copy()

df_onehot = pd.get_dummies(df_onehot, columns=cat_columns)

df_onehot=df_onehot.drop(['ID','Class'],axis=1)

df_onehot.head()
list(df_onehot.columns.values)

# df.to_csv("processed_data.csv")

# df = pd.read_csv("processed_data.csv")
#standard scaling

scaler=StandardScaler()

scaled_data=scaler.fit(df_onehot).transform(df_onehot)

df1=pd.DataFrame(scaled_data,columns=df_onehot.columns)

df1.tail()
k=50

data=scaled_data

model = cluster.AgglomerativeClustering(n_clusters=k,affinity="cosine",linkage='average')

model.fit(data)

clustnum = model.fit_predict(data)

map_class= np.ndarray((k,),int);

for i in range(k):

    if len(classes[clustnum==i].mode())!=0: 

        map_class[i]= classes[clustnum==i].mode().iloc[0]

    else:

        map_class[i]=5

predict=map_class[clustnum]

#checking accuracy of the train

predict = map_class[clustnum]

freq= np.unique(predict,return_counts=True)

accuracy_train = accuracy_score(classes[:1300],predict[:1300])

res = {'K':k,'acc':accuracy_train,'map':map_class,'freq':freq}   

print(res)
submit = df['ID'].iloc[1300:]

submit = submit.to_frame()

submit['Class']=predict[1300:]

submit=submit.reset_index().drop('index',axis=1)

submit.to_csv("submit_csv.csv",index=False)
from IPython.display import HTML

import base64



def create_download_link(data_orig, title = "Download CSV file", filename = "data.csv"): 

    csv = data_orig.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html =  '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(submit)