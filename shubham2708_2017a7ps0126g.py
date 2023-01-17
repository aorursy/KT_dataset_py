#------------------KMEANS( With Kmeans And Correlation)-----------------------





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import seaborn as sns; sns.set()

from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering as AC

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



df = pd.read_csv('../input/dmassign1/data.csv')

temp = df.copy()



for i in df.columns:

    if i == "Col189":

        print("breakk")

        break

    if i!="ID" and df[i].dtype == object:

#         df[i].fillna((df[i].mean()), inplace=True)

        df[i] = pd.to_numeric(df[i],errors='coerce')

    

for i in df.columns:

    if i == "Col189":

        print("breakk")

        break

    if df[i].dtype != object:

        df[i].fillna((df[i].mean()), inplace=True)

        

for i in df.columns:

    if df[i].dtype == object:

        df[i].fillna(df[i].mode())

        

le = LabelEncoder()

for i in df.columns:

    if(df[i].dtype == object):

        df[i] = df[i].astype('str')

        df[i] = le.fit_transform(df[i])

        

scaler=StandardScaler()

df1=scaler.fit(df).transform(df)

df1=pd.DataFrame(df1,columns=df.columns)

del df1['Class']

del df1['ID']



corr = df1.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.948:

            if columns[j]:

                columns[j] = False

selected_columns = df1.columns[columns]

df1 = df1[selected_columns]

print("no. of columns after removing more correlated : " + str(len(df1.columns)))



pca = PCA().fit(df1)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.show()



pca1 = PCA(n_components=50)

pca1.fit(df1)

df1 = pca1.transform(df1)





clusters = 45

kmeans = KMeans(clusters, random_state = 30)

y_kmeans = kmeans.fit_predict(df1)

for i in range(0, len(y_kmeans)):

    y_kmeans[i] += 1

    

class_cnt = [[0 for x in range(clusters)] for y in range(5)]

for i in range(1300):

    class_cnt[df['Class'][i].astype(np.int64) - 1][int(y_kmeans[i])-1] += 1

class_mapp = np.zeros(clusters)

for i in range(clusters):

    maxx = 0

    ind_maxx = 0

    for j in range(5):

        if class_cnt[j][i] > maxx:

            maxx = class_cnt[j][i]

            ind_maxx = j

    class_mapp[i] = ind_maxx

for i in range(len(y_kmeans)):

    y_kmeans[i] = class_mapp[y_kmeans[i]-1]

for i in range(len(y_kmeans)):

    y_kmeans[i] += 1

for i in range(5):

    print(class_cnt[i])

    

cm = confusion_matrix(df['Class'][:1300].astype(np.int64), y_kmeans[:1300], labels=[1,2,3,4,5])

summ = 0

cnt = 0



for i in range(5):

    for j in range(5):

        if i == j:

            summ += cm[i][j]

        cnt += cm[i][j]        

print(summ / cnt)



class_label_cnt = [0 for x in range(5)]

for i in range(1300,13000):

    class_label_cnt[y_kmeans[i]-1] += 1

for i in range(5):

    print(str(i+1) + " : " + str(class_label_cnt[i]))





final = pd.DataFrame({'ID': temp['ID'][1300:]})

final['Class'] = y_kmeans[1300:]

print(final['Class'][:20])



from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "predictions.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(final)