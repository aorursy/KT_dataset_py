import numpy as np

import sys

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn import preprocessing
data_orig = pd.read_csv("../input/dataset.csv")

data = data_orig
data.info()
#The Dataset contains lot of columns with missing values which are represented by '?'

#Replacing all the '?' with NaN

data = data.replace({'?':np.nan})
data.head()
#checking uniqueness of column values

for col in data.columns:

    print(data[col].value_counts())
#Making column enteries uniform 

#for eg. all "me" ,"ME", and "M.E." corresponds to the same value

data['Sponsors'] = data['Sponsors'].replace('g1','G1')

data['Plotsize'] = data['Plotsize'].replace('me','ME')

data['Plotsize'] = data['Plotsize'].replace('M.E.','ME')

data['Plotsize'] = data['Plotsize'].replace('la','LA')

data['Plotsize'] = data['Plotsize'].replace('sm','SM')
#Checking for columns with NaN values

null_columns = data.columns[data.isna().any()]

null_columns
#Columns with Catigorical Data can be replaced with their mode value

#Columns where NaN will be replaced by the column mode

cols_mode = ['Account1', 'History', 'Motive', 'InstallmentRate', 'Tenancy Period']
#Replacing NaN with column mode

for c in cols_mode:

    data[c] = data[c].fillna(data[c].mode()[0])
#Checking for columns with NaN values

null_columns_new = data.columns[data.isna().any()]

null_columns_new
#Columns where NaN will be replaced by the column mean

cols_mean = ['Monthly Period', 'Credit1', 'Age', 'Yearly Period','InstallmentCredit']
#Replacing NaN with column mean

data[['Monthly Period', 'Credit1', 'Age', 'Yearly Period','InstallmentCredit']]=data[['Monthly Period', 'Credit1', 'Age', 'Yearly Period','InstallmentCredit']].astype(float,errors='ignore')

new_data = data

for col in cols_mean:

    new_data[col] = new_data[col].astype(float)

    new_data[col] = new_data[col].fillna(new_data[col].mean())
null_columns_new = new_data.columns[new_data.isna().any()]

null_columns_new
#Creating dummy features for categorical features

new_data = pd.get_dummies(new_data, columns=['Account1', 'History', 'Motive',

        'Account2', 'Employment Period', 'Gender&Type',

        'Sponsors', 'Plotsize', 'Plan', 'Housing', 'Post', 'Phone', 'Expatriate'])
complete_data = new_data

complete_data = complete_data.drop(columns='Class')

complete_data = complete_data.drop(columns='id')

complete_data['InstallmentRate'] = complete_data['InstallmentRate'].fillna(0)
#Performing Min_Max Normalization on the data so that it can be easily handeled by the algorithm

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(complete_data)

dataN1 = pd.DataFrame(np_scaled)

dataN1 = pd.DataFrame(dataN1).fillna(dataN1.mean())
dataN1.columns = complete_data.columns
train_data = new_data.iloc[:175,:]

test_data = new_data.iloc[175:,:]

#test_data.to_csv("processed_data.csv",index=False)
y_train = train_data['Class']

X_train = train_data.drop(columns ='Class')

X_train = X_train.drop(columns='id')



y_test = test_data['Class']

X_test = test_data.drop(columns ='Class')

X_test = X_test.drop(columns='id')

dat1 = dataN1.iloc[:175,:]
from sklearn.ensemble import RandomForestClassifier
#Using RandomForestClassifier for feature selection

plt.figure(figsize=(40,40))

model = RandomForestClassifier(random_state=42)

model = model.fit(dat1,y_train)

features = dat1.columns

importances = model.feature_importances_

impfeatures_index = np.argsort(importances)

#print([features[i] for i in impfeatures_index])

sns.barplot(x = [importances[i] for i in impfeatures_index], y = [features[i] for i in impfeatures_index])

plt.xlabel('value', fontsize=28)

plt.ylabel('parameter', fontsize=26)

plt.tick_params(axis='both', which='major', labelsize=32)

plt.tick_params(axis='both', which='minor', labelsize=32)

plt.show()
#Selecting top features based on their importance according to the above graph

#And creating a new dataframe from it

impfeatures = features[impfeatures_index[-15:]]

X_new = dataN1[[features for features in impfeatures]]
X_new.head()
from sklearn.cluster import Birch

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture
#Applying PCA for Visualization

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X_new)

pca_data = pd.DataFrame(data = principalComponents, columns = ['PC 1', 'PC 2'])
km2 = KMeans(n_clusters=3,random_state=37,max_iter=50000,n_init=10)

km2.fit(X_new)

y_test2 = km2.labels_
# Hclustering=AgglomerativeClustering(n_clusters=3,affinity='cosine',linkage='average',connectivity=None,compute_full_tree=False)

# Q=Hclustering.fit(X_new)

# y_test2 = Q.labels_
mat=pca_data.values

plt.figure(figsize=(6,6)) 

plt.scatter(mat[:,0],mat[:,1],c=y_test2,cmap='rainbow')
y_train_pred = y_test2[:175]
from sklearn import metrics

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_train_pred)
#replacing 0 and 1

y_test2[y_test2 == 1] = 4

y_test2[y_test2 == 0] = 1

y_test2[y_test2 == 4] = 0
y_train_pred = y_test2[:175]

confusion_matrix(y_train, y_train_pred)
#replacing 0 and 2

y_test2[y_test2 == 2] = 4

y_test2[y_test2 == 0] = 2

y_test2[y_test2 == 4] = 0



y_train_pred = y_test2[:175]

confusion_matrix(y_train, y_train_pred)
y_test = y_test2[175:]

y_test.shape
df = pd.DataFrame(columns=['id', 'Class'])

df['id']=test_data['id']

df['Class']=y_test

#df.head()
from sklearn.metrics import accuracy_score

accuracy_score(y_train, y_train_pred)
df.to_csv('2015B3S70395G.csv',index=False)
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



create_download_link(df)