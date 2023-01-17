import pandas as pd

pd.set_option('display.max_rows',500)

import numpy as np

from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA



import seaborn as sns
data_frame = pd.read_csv('../input/dataset.csv')
data_frame.head()
for column in data_frame.columns.tolist() :

    list_of_vals = data_frame[column].unique().tolist()

    if '?' in list_of_vals :

        print(column)
data_frame['Account1'].mode()
data_frame['Account1'].replace({'?' : 'ad'},inplace=True)
data_frame['Monthly Period'].replace({'?' : np.nan},inplace=True)
data_frame['Monthly Period'].fillna(int(data_frame['Monthly Period'].astype('float64').mean()),inplace=True)
data_frame['History'].mode()
data_frame['History'].replace({'?' : 'c2'},inplace=True)
data_frame['Motive'].mode()
data_frame['Motive'].replace({'?' : 'p3'},inplace=True)
data_frame['Credit1'].replace({'?' : np.nan},inplace=True)

data_frame['Credit1'].fillna(int(data_frame['Credit1'].astype('float64').mean()),inplace=True)
data_frame['InstallmentRate'].replace({'?' : np.nan},inplace=True)

data_frame['InstallmentRate'].fillna(int(data_frame['InstallmentRate'].astype('float64').mean()),inplace=True)
data_frame['Tenancy Period'].replace({'?' : np.nan},inplace=True)

data_frame['Tenancy Period'].fillna(int(data_frame['Tenancy Period'].astype('float64').mean()),inplace=True)
data_frame['Age'].replace({'?' : np.nan},inplace=True)

data_frame['Age'].fillna(int(data_frame['Tenancy Period'].astype('float64').mean()),inplace=True)
data_frame['InstallmentCredit'].replace({'?' : np.nan},inplace=True)

data_frame['InstallmentCredit'].fillna(data_frame['InstallmentCredit'].astype('float64').mean(),inplace=True)
data_frame['Yearly Period'].replace({'?' : np.nan},inplace=True)

data_frame['Yearly Period'].fillna(data_frame['Yearly Period'].astype('float64').mean(),inplace=True)
list_of_cols = data_frame.columns.tolist()

for col in list_of_cols :

    try :

        data_frame[col] = data_frame[col].astype('float64')

    except :

        pass
data_frame.head()

plot_data_frame = data_frame.drop(['id','Class'],1)
f, ax = plt.subplots(figsize=(10,8))

corr = plot_data_frame.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True)
def replace_value(given_list,given_dict) :

    new_altered_list = list()

    for ele in given_list :

        new_altered_list.append(given_dict[ele])

    return new_altered_list
data_frame = pd.get_dummies(data_frame,columns=["Account1","Motive","History","Account2","Employment Period","Gender&Type","Sponsors","Plotsize","Plan","Housing","Post","Phone","Expatriate"])

data_frame.head()
data_frame = data_frame.drop(['InstallmentCredit','Yearly Period'],1)
X = data_frame.drop(['id','Class'],1)

X = pd.DataFrame(MinMaxScaler().fit_transform(X),columns=X.columns)

train_x = X[:175]

y_train = data_frame['Class'][:175]

test_x = X[175:]
pca = PCA(n_components=2)

pca.fit(X)

T1 = pca.transform(X)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,random_state=55,max_iter=1000).fit(train_x,y_train)

kmeans.cluster_centers_

train_preds = kmeans.predict(train_x)
plt.plot()

plt.title("Plotting Training clusters")

plt.scatter(T1[:, 0][:175], T1[:, 1][:175], c=train_preds)



centroids = kmeans.cluster_centers_

centroids = pca.transform(centroids)

plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3,\

        c = ['k', 'g', 'b'])
dict_list = [{0:0, 1:1, 2:2},{0:0, 1:2,2:1},{0:1,1:0,2:2},{0:1,1:2,2:0},{0:2,1:0,2:1},{0:2,1:1,2:0}]

best_dic = dict_list[0]

best_acc = 0

for dic in dict_list :

    train_modified_list = replace_value(train_preds.tolist(),dic)

    train_acc = accuracy_score(train_modified_list,y_train.values.tolist())

    if train_acc > best_acc :

        best_acc = train_acc

        best_dic = dic

print('best training accuracy is %.6f'%best_acc)
test_preds = kmeans.predict(test_x)
plt.plot()

plt.title("Test clusters")

plt.scatter(T1[:, 0][175:], T1[:, 1][175:], c=test_preds)



centroids = kmeans.cluster_centers_

centroids = pca.transform(centroids)

plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3,\

        c = ['k', 'g', 'b'])
test_vals = test_preds.tolist()

test_new_vals = replace_value(test_vals,best_dic)

ids = data_frame['id'].values.tolist()[175:]

columns = ['id','class']

new_data_frame = pd.DataFrame({'id' : ids,'class' : test_vals})

new_data_frame.to_csv('submission.csv',index=False)
from sklearn.cluster import AffinityPropagation
clustering = AffinityPropagation().fit(train_x)

train_preds = clustering.predict(train_x)

accuracy_score(train_preds.tolist(),y_train.values.tolist())
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=2, min_samples=10)

train_pred = dbscan.fit_predict(train_x)

plt.scatter(T1[:, 0][:175], T1[:, 1][:175], c=train_pred)
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



create_download_link(data_frame)