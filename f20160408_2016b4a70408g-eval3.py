import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # for plot styling
df = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

df.info()
df_test = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

# df_test.head()
df.isnull().sum().sum()
df.groupby('Children')['Satisfied'].value_counts()
df.corr()
hs_map = {"Yes":2, "No":1, "No internet":0}

df['HighSpeed'] = df['HighSpeed'].map(hs_map)

df_test['HighSpeed'] = df_test['HighSpeed'].map(hs_map)
# df['TVConnection'].unique()

# tv_map = {"No":0, "Cable":1, "DTH":2}

# df['TVConnection'] = df['TVConnection'].map(tv_map)

# df_test['TVConnection'] = df_test['TVConnection'].map(tv_map)
ad_map = {"No":0, "Yes":1}

list1 = ['Children', 'Married', 'AddedServices', 'Internet']

df[list1] = df[list1].replace(ad_map)

df_test[list1] = df_test[list1].replace(ad_map)



ch_map = {"No tv connection":0, "No":1, "Yes":2}

list2 = ['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6' ]

df[list2] = df[list2].replace(ch_map)

df_test[list2] = df_test[list2].replace(ch_map)
# try 2

df['Subscription'].unique()

sub_map = {"Monthly":1, "Biannually":6, "Annually":12}

df['Subscription'] = df['Subscription'].map(sub_map)

df_test['Subscription'] = df_test['Subscription'].map(sub_map)
df['gender'].unique()

# swap 1/0 try 2

gen_map = {"Female":1, "Male":0}

df['gender'] = df['gender'].map(gen_map)

df_test['gender'] = df_test['gender'].map(gen_map)
# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()

# le.fit(df['PaymentMethod'].unique())

# le.transform(df['PaymentMethod'])



# TV connection try 2

df = pd.get_dummies(data = df, columns = ['PaymentMethod', 'TVConnection'])

df_test = pd.get_dummies(data = df_test, columns = ['PaymentMethod', 'TVConnection'])
df['TotalCharges'] = df['TotalCharges'].apply(pd.to_numeric, errors='coerce')

df['TotalCharges'].isnull().sum()
df = df[df['TotalCharges'].notnull()]

df = df.reset_index(drop = True)

# tc_mean = df['TotalCharges'].mean()

# df['TotalCharges'].fillna(tc_mean, inplace = True)

df.shape
tc_mean = df['TotalCharges'].mean()

df_test['TotalCharges'] = df_test['TotalCharges'].apply(pd.to_numeric, errors='coerce')

df_test['TotalCharges'].fillna(tc_mean, inplace = True)
# df.info()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

cols = ['TotalCharges', 'MonthlyCharges', 'tenure']

df[cols] = scaler.fit_transform(df[cols])

df_test[cols] = scaler.transform(df_test[cols])
X = df.drop(['custId','Satisfied','Internet'], axis = 1)

y = df['Satisfied']

id_test = df_test['custId']

X_test = df_test.drop(['custId','Internet'], axis = 1)



# feat = ['TotalCharges', 'MonthlyCharges', 'tenure', 'Subscription', 'HighSpeed']

# X = df[feat]

# X_test = df_test[feat]



X.head()
from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans

# Method 1

clusters = 10

# Method 2

# clusters = 25

kmeans = KMeans(n_clusters = clusters, n_init = 100, random_state = 10).fit(X)

y_pred = kmeans.labels_

pd.Series(y_pred).unique()
ones = [0 for x in range(clusters)] 

zeros = [0 for x in range(clusters)] 

n = len(y)

for i in range(n):

    if (y[i] == 0):

        zeros[y_pred[i]] += 1

    else:

        ones[y_pred[i]] += 1



y_map = {}

for x in range(clusters):

    if (ones[x]>= (0.75*(zeros[x]+ones[x]))):

        y_map[x] = 1

    else:

        y_map[x] = 0

#     print(ones[x], zeros[x], y_map[x])        
# y_map
y_pred = pd.Series(y_pred).map(y_map)

acc = roc_auc_score(y,y_pred)



print(acc)
y.value_counts()
df_test.isnull().sum().sum()
y_test = kmeans.predict(X_test)

prediction = pd.DataFrame(id_test)

y_test = pd.Series(y_test).map(y_map)

prediction['Satisfied'] = y_test

prediction.head(15)

prediction['Satisfied'].value_counts()
# from sklearn.cluster import AgglomerativeClustering

# agg = AgglomerativeClustering().fit(X)

# y_pred = 1-agg.labels_



# acc = roc_auc_score(y,y_pred)



# print(acc)
# y_test = 1-agg.fit_predict(X_test)

# prediction = pd.DataFrame(id_test)

# prediction['Satisfied'] = y_test

# prediction.head(15)

# prediction['Satisfied'].value_counts()
prediction.to_csv('pred.csv', index=False)

prediction.shape