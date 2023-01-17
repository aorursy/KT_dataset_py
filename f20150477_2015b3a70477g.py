import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv("../input/dataset.csv")
data.info()
data = data.replace('?',np.nan)
datanew = data.drop(columns=['id', 'Class'])
datanew['Plotsize'] = datanew['Plotsize'].replace('sm','SM')

datanew['Plotsize'] = datanew['Plotsize'].replace('me','ME')

datanew['Plotsize'] = datanew['Plotsize'].replace('M.E.','ME')

datanew['Plotsize'] = datanew['Plotsize'].replace('la','LA')

datanew['Sponsors'] = datanew['Sponsors'].replace('g1','G1')
datanew['Age'] = pd.to_numeric(datanew['Age'])

datanew['Credit1'] = pd.to_numeric(datanew['Credit1'])

datanew['InstallmentCredit'] = pd.to_numeric(datanew['InstallmentCredit'])

datanew['Yearly Period'] = pd.to_numeric(datanew['Yearly Period'])

datanew['InstallmentRate'] = pd.to_numeric(datanew['InstallmentRate'])

datanew['Tenancy Period'] = pd.to_numeric(datanew['Tenancy Period'])

datanew['Monthly Period'] = pd.to_numeric(datanew['Monthly Period'])
for column in datanew.columns:

    if (datanew[column].dtype == np.object or datanew[column].dtype == np.bool) and (datanew[column].dtype != np.int64) and(datanew[column].dtype != np.float64):

        datanew[column].fillna(datanew[column].mode()[0], inplace=True)
for column in datanew.columns:

    if datanew[column].dtype == np.int64 or datanew[column].dtype == np.float64:

        datanew[column].fillna(datanew[column].mean(), inplace=True)
rep_col = []

for column in datanew.columns:

    if (datanew[column].dtype == np.object or datanew[column].dtype == np.bool) and (datanew[column].dtype != np.int64) and(datanew[column].dtype != np.float64):

        rep_col.append(column)

print(rep_col)
one_hot = []

for column in rep_col:

    one_hot.append(pd.get_dummies(datanew[column], prefix = column))

print(one_hot)
datanew = datanew.drop(rep_col,axis=1)
for i in range(0,len(one_hot)):

    datanew = datanew.join(one_hot[i])
datanew.info()
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(datanew)

datanewscaled = pd.DataFrame(np_scaled)

datanewscaled.head()
from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.ensemble.forest import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.tree import DecisionTreeClassifier
rand_model = RandomForestClassifier(random_state = 42)

y_train = []

for i in range(0,175):

    y_train.append(data['Class'][i])

rand_model.fit(datanewscaled[:175],y_train)
important_features_dict = {}

for x,i in enumerate(rand_model.feature_importances_):

    important_features_dict[x]=i





important_features_list = sorted(important_features_dict,

                                 key=important_features_dict.get,

                                 reverse=True)
selected_feat = important_features_list[0:13]
non_selected_feat = []

for i in range(0,64):

    if i not in selected_feat:

        non_selected_feat.append(i)
datanewscaled_sel = datanewscaled.drop(non_selected_feat,axis = 1)
datanewscaled_sel
from sklearn.decomposition import PCA

pca2 = PCA(n_components=2)

pca2.fit(datanewscaled_sel)

T2 = pca2.transform(datanewscaled_sel)
colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan']
plt.figure(figsize=(16, 8))

from sklearn.cluster import KMeans



kmean = KMeans(n_clusters = 3, random_state = 42)

kmean.fit(datanewscaled_sel)

pred = kmean.predict(datanewscaled_sel)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()



for i in arr:

    meanx = 0

    meany = 0

    count = 0

    for j in range(len(pred)):

        if i == pred[j]:

            count+=1

            meanx+=T2[j,0]

            meany+=T2[j,1]

            plt.scatter(T2[j, 0], T2[j, 1], c=colors[i])

    meanx = meanx/count

    meany = meany/count

    plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black', backgroundcolor=colors[i])
from sklearn.metrics import confusion_matrix

conf  = confusion_matrix(data['Class'][:175],pred[:175])
conf
from sklearn.metrics import accuracy_score

print(accuracy_score(data['Class'][:175],pred[:175]))
from sklearn.metrics import confusion_matrix

conf  = confusion_matrix(data['Class'][:175],pred[:175])
conf
from sklearn.metrics import accuracy_score

print(accuracy_score(data['Class'][:175],pred[:175]))
for i in range(0,len(pred)):

    if pred[i] == 0:

        pred[i] = 2

    elif pred[i] == 2:

        pred[i] = 0
from sklearn.metrics import confusion_matrix

conf  = confusion_matrix(data['Class'][:175],pred[:175])
from sklearn.metrics import accuracy_score

print(accuracy_score(data['Class'][:175],pred[:175]))
from sklearn.metrics import mean_squared_error

print(mean_squared_error(data['Class'][:175],pred[:175]))
id_arr = []

Class = []

for i in range(175,len(pred)):

    id_arr.append(data['id'][i])

    Class.append(pred[i])
new_row = []

new_row.append('abcd')

new_row.append(0)

final = pd.DataFrame(columns = ['id','Class'])

for i in range(175,len(pred)):

    final.loc[len(final)] = new_row
for i in range(0,len(Class)):

    final.at[i,'id'] = id_arr[i]

    final.at[i,'Class'] = Class[i]
final
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