# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.cluster import KMeans

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/PM2.5 Global Air Pollution 2010-2017.csv')
df.head()
df.shape
df.describe()
df.isnull().sum().sum()
mean_val = pd.Series([],dtype='float')

#The dataframe has 240 rows
for i in range(0,df.shape[0]):
    mean = pd.Series(df.iloc[i][2:10].mean())
    mean_val = mean_val.append(mean,ignore_index=True)
df['mean_val'] = mean_val
df_pol = df.sort_values(['mean_val'],axis=0,ascending = False)
df_pol.reset_index(inplace=True)
df_pol.drop("index",axis=1,inplace=True)

df_pol.astype( {'2010': 'float64' , '2011': 'float64' , '2012' : 'float64' , '2013': 'float64' , '2014': 'float64',
                  '2015': 'float64' , '2016': 'float64' , '2017': 'float64' , 'mean_val': 'float64'}  ).dtypes
df_pol.head()
#Arranging the data
c_names = pd.Series(['Years'],dtype='object')
for i in range (0,10):
    ser = pd.Series(df_pol.iloc[i][0] + " (" + df_pol.iloc[i][1] + ")")
    c_names = c_names.append(ser)
    
df_arr10 = pd.DataFrame( columns = c_names )
df_arr10['Years'] = pd.Series(['2010','2011','2012','2013','2014','2015','2016','2017'],dtype='int')
for i in range(0,10):
    df_arr10[df_arr10.columns[i+1]] = df_pol.iloc[i][2:10].values

df_arr10 = df_arr10.apply(pd.to_numeric, errors='coerce')
df_arr10.head()
color_list = ['#e6b0aa','#78281f','#d7bde2','#4a235a','#196f3d','#3498db','#a3e4d7','#e74c3c','#f9e79f','#2c3e50']
#Running this code cell with Shift+Enter once might result in the axis labels being black. 
#Run this code cell again to observe the correct plot. 

plt.figure(figsize=(12,6))
sns.set(font_scale=1.15)

for i in range(1,11):
    ax = sns.lineplot(x = 'Years' , y = df_arr10.columns[i] , color = color_list[i-1], data = df_arr10)
    
ax.set(xlabel = 'Years', ylabel = 'PM 2.5 mean annual exposure' , title = 'Air pollution in 10 most polluted countries')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0,
           labels=['Nepal','India','Qatar','Saudi Arabia','Egypt','Niger','Bahrain','Bangladesh','Cameroon','Iraq'])

plt.show()
df['Change in Pollution'] = df['2010'] - df['2017']
df.head()
df_ch = df.sort_values(['Change in Pollution'],axis=0,ascending = False)
df_ch.reset_index(inplace=True)
df_ch.drop("index",axis=1,inplace=True)
df_ch.tail()
world_avg = []
for i in range(2,10):
    world_avg.append(df[df.columns[i]].mean())

df_world = pd.DataFrame(columns=['Years','World Average'])
df_world['Years'] = df.columns[2:10]
df_world['World Average'] = pd.Series(world_avg,dtype=float)

sns.lineplot(x = 'Years', y = 'World Average', data=df_world, color='Red')
df_cluster = df.drop(['Country Name', 'Country Code'], 1)

sse = []
list_k = list(range(1, 10, 1))

for k in list_k:
    km = KMeans(n_clusters=k,random_state=0)
    km.fit(df_cluster)
    sse.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(list_k, sse, '-o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distance');
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(df_cluster)
df['Cluster Label'] = kmeans.labels_.astype(int)
df.head() 
df_cluster0 = df[df['Cluster Label'] == 0]
df_cluster1 = df[df['Cluster Label'] == 1]
df_cluster2 = df[df['Cluster Label'] == 2]

print(df_cluster0.shape)
print(df_cluster1.shape)
print(df_cluster2.shape)
#Cluster 0
print('Cluster0')
print(df_cluster0['mean_val'].min())
print(df_cluster0['mean_val'].max(),"\n")

#Cluster 1
print('Cluster1')
print(df_cluster1['mean_val'].min())
print(df_cluster1['mean_val'].max(),"\n")

#Cluster 2
print('Cluster2')
print(df_cluster2['mean_val'].min())
print(df_cluster2['mean_val'].max())
features = ['2010','2011','2012','2013','2014','2015','2016','2017','mean_val','Change in Pollution']
X = df[features]
y = df['Cluster Label']
#Random Forest Classification

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

Rfc_model = RandomForestClassifier(n_estimators = 400,max_depth=10,n_jobs= -1,random_state=0)
scores = cross_val_score(Rfc_model,X,y,cv=5,scoring='f1_macro')
print('macro average - ',scores.mean())
scores = cross_val_score(Rfc_model,X,y,cv=5,scoring='f1_micro')
print('micro average - ',scores.mean())
#K-Nearest Neighbours (KNN)

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5,leaf_size = 50,n_jobs=-1)
scores_KNN = cross_val_score(knn_model,X,y,cv=5,scoring='f1_macro')
print('macro average - ',scores_KNN.mean())
scores_KNN = cross_val_score(knn_model,X,y,cv=5,scoring='f1_micro')
print('micro average - ',scores_KNN.mean())
#Logistic Regression

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

LoReg_model = LogisticRegression(random_state = 3,n_jobs = -1)
scores_LR = cross_val_score(LoReg_model,X,y,cv=5,scoring='f1_macro')
print('macro average - ',scores_LR.mean())
scores_LR = cross_val_score(LoReg_model,X,y,cv=5,scoring='f1_micro')
print('micro average - ',scores_LR.mean())
final_model = LogisticRegression(random_state = 3,n_jobs = -1)
final_model.fit(X,y);
#Replace this data with the country data
data_dict = {'Country Name' : 'Vatican City',
             'Country Code' : 'VCY',
             '2010' : 25.4,
             '2011' : 22.4,
             '2012' : 21.3,
             "2013" : 14.85,
             '2014' : 13.6,
             '2015' : 10.85,
             '2016' : 9.45,
             '2017' : 11.25}
df_test = pd.DataFrame(columns = df.columns[:12])
df_test = df_test.append(data_dict,ignore_index=True)
df_test['Change in Pollution'] = df_test['2010'] - df_test['2017']
df_test['mean_val'] = df_test.iloc[0][2:10].mean()
df_test
X_test = df_test[features]
clu = final_model.predict(X_test)
if(clu == 0):
    print('Lowly Polluted Country')
if(clu == 1):
    print('Highly Polluted Country')
if(clu == 2):
    print('Averagely Polluted Country')