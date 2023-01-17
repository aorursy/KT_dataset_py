import numpy as np

import pandas as pd

import matplotlib.pyplot as pl

%matplotlib inline
data_original = pd.read_csv("/kaggle/input/dmassign1/data.csv", na_values={'?'})

#data_original = pd.read_csv("../input/dmassign1/data.csv", na_values={'?'})

data = data_original
data_null = data.columns[data.isnull().any()]

data_cat = data.iloc[:,-10:-1]

data_nullnum = data_null[:-7]

data_nullcat = data_null[-7:-1]
#Filling numerical null values with mean



data[data_nullnum] = data[data_nullnum].fillna(data[data_nullnum].mean())
#Cleaning and filling categorical null values with mode



data_original['Col197'].replace(['sm', 'me', 'M.E.', 'la'], ['SM', 'ME', 'ME', 'LA'], inplace = True)

data[data_nullcat] = data[data_nullcat].fillna(data[data_nullcat].mode().loc[0, :])
#One Hot Encoding the data



data = pd.get_dummies(data, columns = data_cat.columns)

class_col = data['Class']

data.drop(labels=['Class'],axis=1,inplace=True)

data.insert(226,'Class',class_col)
data.iloc[:1300,:].corr()['Class'].abs().sort_values(ascending=False)[:20]
#Scaling numerical data

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

data.iloc[:,1:189] = scaler.fit_transform(data.iloc[:,1:189].values)
data_train = data.iloc[:,1:-1]

x_train = data_train.iloc[:,1:-1]

y_train = data.iloc[:1300,-1]
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

x_train_labelled = data_train.iloc[:1300,:]

plot = pca.fit_transform(x_train_labelled.values)

plot.shape
data2 = data.loc[:,['Col152','Col153','Col151','Col85','Col43','Col84','Col154','Col86','Col42','Col44','Col45','Col150','Col41','Col72','Col179','Col184','Col71','Col172']]
#Plotting clusters of the labelled data



%matplotlib inline

color_map = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

for i in np.arange(1,6):

    pl.scatter(plot[y_train == i,0],plot[y_train == i,1], c=color_map[i])

pl.show()
from sklearn.cluster import AgglomerativeClustering



agg = AgglomerativeClustering(n_clusters=5,affinity='cosine',linkage='complete')

agg_pred = agg.fit_predict(data2)

agg_pred = agg_pred + 1

print("Predictions:\n")

print(agg_pred[:300])

print("Actual values:\n")

print(np.asarray(y_train[:300].astype(int).tolist()))
#Plotting clusters using Agglomerative Clustering



%matplotlib inline

color_map = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

for i in np.arange(1,6):

    pl.scatter(plot[agg_pred[:1300] == i,0],plot[agg_pred[:1300] == i,1], c=color_map[i])

pl.show()
final_sub = pd.DataFrame(agg_pred[1300:], index = data_original.loc[1300:, 'ID'], columns=['Class'])



final_sub.replace([1,2,3,4,5],[4,2,3,1,5],inplace=True)
final_sub.to_csv("submission.csv")
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

create_download_link(final_sub)