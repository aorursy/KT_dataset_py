import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings  

warnings.filterwarnings('ignore')
clients_data = pd.read_csv('../input/UCI_Credit_Card.csv',index_col='ID')

clients_data.head()
clients_data.isnull().sum()
clients_data = clients_data.replace({'SEX' :[1,2],'MARRIAGE' : [1,2,3,0],'EDUCATION' : [0,1,2,3,4,5,6]},{'SEX' :['Male','Female'], 'MARRIAGE' :['Married','Single','Others','Others'],

                                    'EDUCATION' : ['Unknown','Graduate school','University','High school','Others','Unknown','Unknown']})
f,ax = plt.subplots(1,2,figsize=(18,8))

sns.countplot(x='SEX',data=clients_data,ax=ax[0])

sns.stripplot(x='SEX',y='LIMIT_BAL',data=clients_data,ax=ax[1])
sns.catplot(x='SEX',y='AGE',col='MARRIAGE',data=clients_data,kind='box')
sns.catplot(x='SEX',data=clients_data,kind='count',col='EDUCATION',row='MARRIAGE', aspect=1)
points = np.array([

    [clients_data[clients_data['EDUCATION'] == 'University']['LIMIT_BAL'].max(),

    clients_data[clients_data['EDUCATION'] == 'University']['LIMIT_BAL'].min()],

    [clients_data[clients_data['EDUCATION'] == 'Graduate school']['LIMIT_BAL'].max(),

    clients_data[clients_data['EDUCATION'] == 'Graduate school']['LIMIT_BAL'].min()],

    [clients_data[clients_data['EDUCATION'] == 'High school']['LIMIT_BAL'].max(),

    clients_data[clients_data['EDUCATION'] == 'High school']['LIMIT_BAL'].min()],

    [clients_data[clients_data['EDUCATION'] == 'Others']['LIMIT_BAL'].max(),

    clients_data[clients_data['EDUCATION'] == 'Others']['LIMIT_BAL'].min()],

    [clients_data[clients_data['EDUCATION'] == 'Graduate school']['LIMIT_BAL'].max(),

    clients_data[clients_data['EDUCATION'] == 'Graduate school']['LIMIT_BAL'].min()]

])

#types = ['max','min','max','min','max','min','max','min','max','min']

rate = pd.DataFrame(points,columns=['max','min'])



f,ax = plt.subplots(3,2,figsize=(18,10))

sns.boxenplot(x='EDUCATION',y='LIMIT_BAL',data=clients_data[clients_data['EDUCATION'] == 'University'],ax=ax[0,0])

sns.boxenplot(x='EDUCATION',y='LIMIT_BAL',data=clients_data[clients_data['EDUCATION'] == 'Graduate school'],ax=ax[0,1])

sns.boxenplot(x='EDUCATION',y='LIMIT_BAL',data=clients_data[clients_data['EDUCATION'] == 'High school'],ax=ax[1,0])

sns.boxenplot(x='EDUCATION',y='LIMIT_BAL',data=clients_data[clients_data['EDUCATION'] == 'Others'],ax=ax[1,1])

sns.boxenplot(x='EDUCATION',y='LIMIT_BAL',data=clients_data[clients_data['EDUCATION'] == 'Graduate school'],ax=ax[2,0])

sns.lineplot(x=rate.index,y='min',data=rate,ax=ax[2,1])

ax3 = ax[2,1].twinx()

sns.lineplot(x=rate.index,y='max',data=rate,ax=ax3,color='r')
clients_data['PAY_0'][clients_data['PAY_0'] <= 0] = 0

clients_data['PAY_0'][(clients_data['PAY_0'] > 0) & (clients_data['PAY_0'] <= 3) ] = 1 

clients_data['PAY_0'][clients_data['PAY_0'] > 3 ] = 2



clients_data['PAY_2'][clients_data['PAY_2'] <= 0] = 0

clients_data['PAY_2'][(clients_data['PAY_2'] > 0) & (clients_data['PAY_2'] <= 3) ] = 1 

clients_data['PAY_2'][clients_data['PAY_2'] > 3 ] = 2



clients_data['PAY_3'][clients_data['PAY_3'] <= 0] = 0

clients_data['PAY_3'][(clients_data['PAY_3'] > 0) & (clients_data['PAY_3'] <= 3) ] = 1 

clients_data['PAY_3'][clients_data['PAY_3'] > 3 ] = 2



clients_data['PAY_4'][clients_data['PAY_4'] <= 0] = 0

clients_data['PAY_4'][(clients_data['PAY_4'] > 0) & (clients_data['PAY_4'] <= 3) ] = 1 

clients_data['PAY_4'][clients_data['PAY_4'] > 3 ] = 2



clients_data['PAY_5'][clients_data['PAY_5'] <= 0] = 0

clients_data['PAY_5'][(clients_data['PAY_5'] > 0) & (clients_data['PAY_5'] <= 3) ] = 1 

clients_data['PAY_5'][clients_data['PAY_5'] > 3 ] = 2



clients_data['PAY_6'][clients_data['PAY_6'] <= 0] = 0

clients_data['PAY_6'][(clients_data['PAY_6'] > 0) & (clients_data['PAY_6'] <= 3)] = 1 

clients_data['PAY_6'][clients_data['PAY_6'] > 3 ] = 2



clients_data = clients_data.replace({'PAY_0' : [0,1,2],'PAY_2' : [0,1,2],'PAY_3' : [0,1,2],

                                     'PAY_4' : [0,1,2],'PAY_5' : [0,1,2],'PAY_6' : [0,1,2]},

                                     {'PAY_0' : ['Early','On time','Late'],'PAY_2' : ['Early','On time','Late'],

                                      'PAY_3' : ['Early','On time','Late'],'PAY_4' : ['Early','On time','Late'],

                                      'PAY_5' : ['Early','On time','Late'],'PAY_6' : ['Early','On time','Late']})

clients_data.head()
f,ax= plt.subplots(2,3,figsize=(16,6))

sns.countplot(x='PAY_0',data=clients_data,ax=ax[0,0])

sns.countplot(x='PAY_2',data=clients_data,ax=ax[0,1])

sns.countplot(x='PAY_3',data=clients_data,ax=ax[0,2])

sns.countplot(x='PAY_4',data=clients_data,ax=ax[1,0])

sns.countplot(x='PAY_5',data=clients_data,ax=ax[1,1])

sns.countplot(x='PAY_6',data=clients_data,ax=ax[1,2])
clients_data['TOTAL_BILL'] = [value.sum() for value in clients_data.iloc[:,11:17].values]

clients_data = clients_data.drop(['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'],axis=1)

clients_data['TOTAL_PAY_AMT'] = [value.sum() for value in clients_data.iloc[:,11:17].values]

clients_data = clients_data.drop(['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'],axis=1)

clients_data.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

clients_data['SEX'] = le.fit_transform(clients_data['SEX'])

educations = pd.get_dummies(clients_data['EDUCATION'])

marriages = pd.get_dummies(clients_data['MARRIAGE'])

pay_0 = pd.get_dummies(clients_data['PAY_0'],prefix='PAY_0')

pay_2 = pd.get_dummies(clients_data['PAY_2'],prefix='PAY_2')

pay_3 = pd.get_dummies(clients_data['PAY_3'],prefix='PAY_3')

pay_4 = pd.get_dummies(clients_data['PAY_4'],prefix='PAY_4')

pay_5 = pd.get_dummies(clients_data['PAY_5'],prefix='PAY_5')

pay_6 = pd.get_dummies(clients_data['PAY_6'],prefix='PAY_6')

clients_data = pd.concat([clients_data,educations],axis=1)

clients_data = pd.concat([clients_data,marriages],axis=1)

clients_data = pd.concat([clients_data,pay_0],axis=1)

clients_data = pd.concat([clients_data,pay_2],axis=1)

clients_data = pd.concat([clients_data,pay_3],axis=1)

clients_data = pd.concat([clients_data,pay_4],axis=1)

clients_data = pd.concat([clients_data,pay_5],axis=1)

clients_data = pd.concat([clients_data,pay_6],axis=1)

clients_data = clients_data.drop(['EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'],axis=1)

clients_data.head()
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from mpl_toolkits import mplot3d

from sklearn.metrics import davies_bouldin_score
pca = PCA(n_components=3)

X_pca = pca.fit_transform(clients_data)
wcss = []

for i in range(1,11):

    k_mean = KMeans(n_clusters=i, init='k-means++',random_state=0)

    k_mean.fit(X_pca)

    wcss.append(k_mean.inertia_)



sns.lineplot(x=range(1,11),y=wcss)
k_mean = KMeans(n_clusters=3,init='k-means++')

k_mean.fit(X_pca)

k_mean_predict = k_mean.predict(X_pca)

%config InlineBackend.print_figure_kwargs = {'bbox_inches':None}

plt.figure(figsize=(16,8))

ax = plt.axes(projection='3d')

ax.scatter(X_pca[k_mean_predict == 0,0],X_pca[k_mean_predict == 0,1],X_pca[k_mean_predict == 0,2], 'blue')

ax.scatter(X_pca[k_mean_predict == 1,0],X_pca[k_mean_predict == 1,1],X_pca[k_mean_predict == 1,2], 'red')

ax.scatter(X_pca[k_mean_predict == 2,0],X_pca[k_mean_predict == 2,1],X_pca[k_mean_predict == 2,2], 'green')
dbs = davies_bouldin_score(X_pca,k_mean.labels_)

print(dbs)