import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import time

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans

from collections import Counter

from sklearn.cluster import DBSCAN

from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans
# from google.colab import drive

# drive.mount('/content/gdrive')
# # load the csv file

# %cd gdrive/My\ Drive/lastLab/

df = pd.read_csv('TEST.csv')

df_train = pd.read_csv('TRAIN.csv')
print('Number of samples:',len(df))
df.info()
df.head()
df_train = df_train.loc[~df_train.discharge_disposition_id.isin([11,13,14,19,20,21])]
df_train['OUTPUT_LABEL'] = (df_train.readmitted_NO == 0).astype('int')
for c in list(df.columns):

    

    n = df[c].unique()

    

    # if number of unique values is less than 30, print the values. Otherwise print the number of unique values

    if len(n)<30:

        print(c)

        print(n)

    else:

        print(c + ': ' +str(len(n)) + ' unique values')

        

for c in list(df_train.columns):

    

    n = df_train[c].unique()

    

    # if number of unique values is less than 30, print the values. Otherwise print the number of unique values

    if len(n)<30:

        print(c)

        print(n)

    else:

        print(c + ': ' +str(len(n)) + ' unique values')
# replace ? with nan

df = df.replace('?',np.nan)

df_train = df_train.replace('?',np.nan)
#Numerical Features

cols_num = ['time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications',

       'number_outpatient', 'number_emergency', 'number_inpatient','number_diagnoses']
df_train[cols_num].isnull().sum()
cols_cat = ['race', 'gender', 

       'max_glu_serum', 'A1Cresult',

       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',

       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',

       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',

       'tolazamide', 'insulin',

       'glyburide-metformin', 'glipizide-metformin',

       'glimepiride-pioglitazone', 'metformin-rosiglitazone',

       'metformin-pioglitazone', 'change', 'diabetesMed','payer_code']
df_train[cols_cat].isnull().sum()
df['race'] = df['race'].fillna('UNK')

df['payer_code'] = df['payer_code'].fillna('UNK')

df['medical_specialty'] = df['medical_specialty'].fillna('UNK')



df_train['race'] = df_train['race'].fillna('UNK')

df_train['payer_code'] = df_train['payer_code'].fillna('UNK')

df_train['medical_specialty'] = df_train['medical_specialty'].fillna('UNK')
print('Number medical specialty:', df.medical_specialty.nunique())

df.groupby('medical_specialty').size().sort_values(ascending = False)
#Keep only top 10

top_10 = ['UNK','InternalMedicine','Emergency/Trauma',\

          'Family/GeneralPractice', 'Cardiology','Surgery-General' ,\

          'Nephrology','Orthopedics',\

          'Orthopedics-Reconstructive','Radiologist']



# make a new column with duplicated data

df_train['med_spec'] = df_train['medical_specialty'].copy()

df['med_spec'] = df['medical_specialty'].copy()





# replace all specialties not in top 10 with 'Other' category

df_train.loc[~df_train.med_spec.isin(top_10),'med_spec'] = 'Other'

df.loc[~df.med_spec.isin(top_10),'med_spec'] = 'Other'

df_train.groupby('med_spec').size()
cols_cat_num = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']



df[cols_cat_num] = df[cols_cat_num].astype('str')

df_train[cols_cat_num] = df_train[cols_cat_num].astype('str')
df_cat = pd.get_dummies(df[cols_cat + cols_cat_num + ['med_spec']],drop_first = True)

df_cat_train = pd.get_dummies(df_train[cols_cat + cols_cat_num + ['med_spec']],drop_first = True)
df_cat.head()
df = pd.concat([df,df_cat], axis = 1)

df_train = pd.concat([df_train,df_cat_train], axis = 1)
cols_all_cat = list(df_cat.columns)

cols_all_cat_train = list(df_cat_train.columns)
df[['age', 'weight']].head()

df_train[['age', 'weight']].head()
df.groupby('age').size()
age_id = {'[0-10)':0, 

          '[10-20)':10, 

          '[20-30)':20, 

          '[30-40)':30, 

          '[40-50)':40, 

          '[50-60)':50,

          '[60-70)':60, 

          '[70-80)':70, 

          '[80-90)':80, 

          '[90-100)':90}

df['age_group'] = df.age.replace(age_id)

df_train['age_group'] = df_train.age.replace(age_id)
df.weight.notnull().sum()
df['has_weight'] = df.weight.notnull().astype('int')

df_train['has_weight'] = df_train.weight.notnull().astype('int')
cols_extra = ['age_group','has_weight']
print('Total number of features:', len(cols_num + cols_all_cat + cols_extra))

print('Numerical Features:',len(cols_num))

print('Categorical Features:',len(cols_all_cat))

print('Extra features:',len(cols_extra))
df[cols_num + cols_all_cat + cols_extra].isnull().sum().sort_values(ascending = False).head(10)

df[cols_num + cols_all_cat + cols_extra].isnull().sum().sort_values(ascending = False).head(10)
col2use = cols_num + cols_all_cat + cols_extra

df_data = df[col2use]

col2use_train = cols_num + cols_all_cat_train + cols_extra

df_data_train = df_train[col2use_train]
scaled_data = StandardScaler().fit_transform(df_data)

scaled_data_train = StandardScaler().fit_transform(df_data_train)



scaled_df=pd.DataFrame(scaled_data,columns=df_data.columns)

scaled_df_train=pd.DataFrame(scaled_data_train,columns=df_data_train.columns)



scaled_df.head()
pca = PCA(80)

pca.fit(scaled_df)

T1 = pca.transform(scaled_df)



pca.fit(scaled_df_train)

T1_train = pca.transform(scaled_df_train)
# #ALGO 1 : Takes too much space, not so good results,  lite

# from sklearn.cluster import AgglomerativeClustering



# cluster = AgglomerativeClustering(n_clusters=2)  

# cluster.fit(T1_train[:20000])

# cluster.fit(T1_train[20000:30000])

# print('a')

# cluster.fit(T1_train[30000:40000])

# print('b')

# cluster.fit(T1_train[40000:50000])

# print('c')

# cluster.fit(T1_train[50000:60000])

# print('d')

# cluster.fit(T1_train[60000:])
# ans = cluster.fit_predict(T1)
# ans.sum()
# #Standard KMeans

# kmeans = KMeans(n_clusters = 2, random_state=10)

# kmeans.fit(T1_train , )

# pred = kmeans.fit_predict(T1)

# pred.sum()
# #DBSCAN, not good results

# cluster = DBSCAN(eps = 0.05, min_samples = 10)  

# cluster.fit(T1_train[:20000])

# cluster.fit(T1_train[20000:30000])

# print('a')

# cluster.fit(T1_train[30000:40000])

# print('b')

# cluster.fit(T1_train[40000:50000])

# print('c')

# cluster.fit(T1_train[50000:60000])

# print('d')

# cluster.fit(T1_train[60000:])

# ans = cluster.fit_predict(T1)

# ans.sum()
#Code for grid search of hyperparams for KMeans and MiniBatchKMeans  

mx, mr, mn = 0, 0, 0



# Random centroid initialization and n_clusters simulation

num = 2

for s in range(4,20):

    for r in range(1,50):

        kmeans = MiniBatchKMeans(n_clusters = 2, random_state=r, batch_size=s)

        pred = kmeans.fit_predict(T1_train)

        print(pred.shape)



        a = {}

        for item in range(2):

            a[item] = []



        for index, p in enumerate(pred[:60000]):

            a[p].append(index)



        subs = {}

        for item in range(num):

            subs[item] = int(Counter(df_train['readmitted_NO'].iloc[a[item]]

                               .dropna()).most_common(1)[0][0])



        test = [subs.get(n, n) for n in pred[:60000]]

        pred1 = [subs.get(n, n) for n in pred[60000:]]



        correct, total = 0,0

        for i,j in zip(test, df_train.iloc[:60000,-1]):

            if i==int(j):

                correct+=1

            total+=1



        if correct/total>mx:

            mx = correct/total

            mn = s

            mr = r

          

        print('Iteration :', r, 'best : ', mr)



print('Found optimal hyperparameters ->')

print('Number of clusters: ', mn)

print('Random State: ', mr)

# #ALGO 4

# from sklearn.cluster import SpectralClustering

# cluster = SpectralClustering(n_clusters=2)

# cluster.fit(T1_train[:10000])

# print('z')

# cluster.fit(T1_train[10000:20000])

# print('x')

# cluster.fit(T1_train[20000:30000])

# print('a')

# cluster.fit(T1_train[30000:40000])

# print('b')

# cluster.fit(T1_train[40000:50000])

# print('c')

# cluster.fit(T1_train[50000:60000])

# print('d')

# cluster.fit(T1_train[60000:])



# ans = cluster.fit_predict(T1)

# ans.sum()
#ALGO 5

for i in range(50):

    kmeans = MiniBatchKMeans(n_clusters=2,

           random_state=i,

           batch_size=6)

    kmeans = kmeans.fit(T1_train)



    ans = kmeans.predict(T1)

    print(ans.sum()-15000, i)

# kmeans = kmeans.partial_fit(X[6:12,:])

# kmeans.predict([[0, 0], [4, 4]])
kmeans = MiniBatchKMeans(n_clusters=2,

         random_state=18,

         batch_size=6)

kmeans = kmeans.fit(T1_train)



ans = kmeans.predict(T1)

print(ans.sum(), i)
#EXPORT TO CSV

count = []

for x in range(0,len(ans)):

    count.append(x)



penul = pd.DataFrame({'index': count, 'target': ans})



len(penul)

penul.head()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = penul.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(penul)