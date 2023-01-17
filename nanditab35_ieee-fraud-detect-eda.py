# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import gc

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split

from numpy import cov

from numpy.linalg import eig

import xgboost as xgb

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve

import scikitplot as skplt



pd.set_option('display.max_columns', 500)

print(os.listdir('/kaggle/input/ieee-fraud-detection'))



# Any results you write to the current directory are saved as output.
PATH = '/kaggle/input/ieee-fraud-detection'
df_train_tran = pd.read_csv(PATH + '/train_transaction.csv')

df_train_tran.head()
df_train_tran.describe()
print(df_train_tran['isFraud'].value_counts())

# 1 is 3.6% of 0
plt.figure(figsize=(5,5))

ax = sns.countplot(df_train_tran['isFraud'])

del ax

gc.collect()
# Bubble plot

'''plt.figure(figsize=(10,10))

plt.scatter(x=df_train_tran['isFraud'], y=df_train_tran['isFraud'].value_counts(), s=z*1000, alpha=0.5)

plt.show()'''
# yet to code
# yet to code
plt.figure(figsize=(5,5))

ax = sns.scatterplot(x='isFraud', y='TransactionAmt', data=df_train_tran)

del ax

gc.collect()
#df_train_tran.fillna('0', inplace=True)
plt.figure(figsize=(5,5))

ax = sns.boxplot(x='isFraud', y='ProductCD', data=df_train_tran)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax = sns.boxplot(x='isFraud', y='card1', data=df_train_tran)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax = sns.boxplot(x='isFraud', y='card2', data=df_train_tran)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax = sns.boxplot(x='isFraud', y='card3', data=df_train_tran)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax = sns.boxplot(x='isFraud', y='card4', data=df_train_tran)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax = sns.boxplot(x='isFraud', y='card5', data=df_train_tran)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax = sns.boxplot(x='isFraud', y='card6', data=df_train_tran)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax = sns.boxplot(x='isFraud', y='addr1', data=df_train_tran)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_tran['addr1'])

plt.xticks(rotation=45)

del ax

gc.collect()
plt.figure(figsize=(20,5))

ax=sns.barplot(x=df_train_tran['addr2'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('addr2')

plt.ylabel('isFraud')

plt.title('isFraud Given addr2')

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_tran['addr2'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(20,5))

ax=sns.barplot(x=df_train_tran['P_emaildomain'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('P_emaildomain')

plt.ylabel('isFraud')

plt.title('isFraud Given P_emaildomain')

del ax

gc.collect()
plt.figure(figsize=(25,8))

ax=sns.countplot(df_train_tran['P_emaildomain'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(20,5))

ax=sns.barplot(x=df_train_tran['R_emaildomain'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('R_emaildomain')

plt.ylabel('isFraud')

plt.title('isFraud Given R_emaildomain')

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_tran['R_emaildomain'])

plt.xticks(rotation=45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.barplot(x=df_train_tran['M1'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('M1')

plt.ylabel('isFraud')

plt.title('isFraud Given M1')

del ax

gc.collect()
df_train_tran.groupby('M1').mean()
plt.figure(figsize=(5,5))

ax=sns.barplot(x=df_train_tran['M2'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('M2')

plt.ylabel('isFraud')

plt.title('isFraud Given M2')

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.barplot(x=df_train_tran['M3'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('M3')

plt.ylabel('isFraud')

plt.title('isFraud Given M3')

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.barplot(x=df_train_tran['M4'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('M4')

plt.ylabel('isFraud')

plt.title('isFraud Given M4')

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.barplot(x=df_train_tran['M5'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('M5')

plt.ylabel('isFraud')

plt.title('isFraud Given M5')

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.barplot(x=df_train_tran['M6'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('M6')

plt.ylabel('isFraud')

plt.title('isFraud Given M6')

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.barplot(x=df_train_tran['M7'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('M7')

plt.ylabel('isFraud')

plt.title('isFraud Given M7')

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.barplot(x=df_train_tran['M8'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('M8')

plt.ylabel('isFraud')

plt.title('isFraud Given M8')

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.barplot(x=df_train_tran['M9'], y=df_train_tran['isFraud'])

plt.xticks(rotation= 45)

plt.xlabel('M9')

plt.ylabel('isFraud')

plt.title('isFraud Given M9')

del ax

gc.collect()
# C1 to C14

C_cols = ['C1', 'C2' ] #, 'C3' ,'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']

gc.collect()

sns.pairplot(df_train_tran[C_cols])

plt.show()
C_cols = ['C2', 'C3' ]

sns.pairplot(df_train_tran[C_cols])

plt.show()
X = df_train_tran.drop('isFraud', axis=1)

Y = df_train_tran[['isFraud']]
print(X.shape, Y.shape)
C_cols12 = []

for i in range(1,16):

    C_cols1.append('V'+str(i))

df = pd.DataFrame(data = df_train_tran, columns = C_cols1)



plt.figure(figsize=(25,8))

ax=sns.boxplot(x="variable", y="value", data=pd.melt(df))

plt.xticks(rotation=45)

plt.show()



del ax

gc.collect()
C_cols2 = []

for i in range(16,51):

    C_cols2.append('V'+str(i))

    

df = pd.DataFrame(data = df_train_tran, columns = C_cols2)



plt.figure(figsize=(30,8))

ax=sns.boxplot(x="variable", y="value", data=pd.melt(df))

plt.xticks(rotation=45)

plt.show()



del ax

gc.collect()
C_cols2 = []

for i in range(51,101):

    C_cols2.append('V'+str(i))

df = pd.DataFrame(data = df_train_tran, columns = C_cols2)



plt.figure(figsize=(35,10))

ax=sns.boxplot(x="variable", y="value", data=pd.melt(df))

plt.xticks(rotation=45)

plt.show()



del ax

gc.collect()
C_cols2 = []

for i in range(101,151):

    C_cols2.append('V'+str(i))

    

df = pd.DataFrame(data = df_train_tran, columns = C_cols2)



plt.figure(figsize=(30,10))

ax=sns.boxplot(x="variable", y="value", data=pd.melt(df))

plt.xticks(rotation=45)

plt.show()



del ax

gc.collect()
C_cols2 = []

for i in range(151,201):

    C_cols2.append('V'+str(i))

    

df = pd.DataFrame(data = df_train_tran, columns = C_cols2)



plt.figure(figsize=(30,10))

ax=sns.boxplot(x="variable", y="value", data=pd.melt(df))

plt.xticks(rotation=45)

plt.show()



del ax

gc.collect()
C_cols2 = []

for i in range(201,251):

    C_cols2.append('V'+str(i))

    

df = pd.DataFrame(data = df_train_tran, columns = C_cols2)



plt.figure(figsize=(30,10))

ax=sns.boxplot(x="variable", y="value", data=pd.melt(df))

plt.xticks(rotation=45)

plt.show()



del ax

gc.collect()
C_cols2 = []

for i in range(251,301):

    C_cols2.append('V'+str(i))

    

df = pd.DataFrame(data = df_train_tran, columns = C_cols2)



plt.figure(figsize=(30,10))

ax=sns.boxplot(x="variable", y="value", data=pd.melt(df))

plt.xticks(rotation=45)

plt.show()



del ax

gc.collect()
C_cols2 = []

for i in range(301,340):

    C_cols2.append('V'+str(i))

    

df = pd.DataFrame(data = df_train_tran, columns = C_cols2)



plt.figure(figsize=(30,10))

ax=sns.boxplot(x="variable", y="value", data=pd.melt(df))

plt.xticks(rotation=45)

plt.show()



del ax

gc.collect()
del df

gc.collect()
df_train_id = pd.read_csv(PATH + '/train_identity.csv')
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['DeviceType'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['DeviceInfo'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_12'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_13'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_14'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_14'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_15'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_16'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_17'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_18'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_19'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(30,5))

ax=sns.countplot(df_train_id['id_20'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_21'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_22'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_23'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_24'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_25'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_26'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_27'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_28'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_29'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(30,5))

ax=sns.countplot(df_train_id['id_30'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(30,5))

ax=sns.countplot(df_train_id['id_31'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_32'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(25,5))

ax=sns.countplot(df_train_id['id_33'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_34'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_35'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_36'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_37'])

plt.xticks(rotation= 45)

del ax

gc.collect()
plt.figure(figsize=(5,5))

ax=sns.countplot(df_train_id['id_38'])

plt.xticks(rotation= 45)

del ax

gc.collect()
C_cols2 = []

lbl = ''

for i in range(1,12):

    if i < 10:

        lbl = 'id_0'+str(i)

    else:

        lbl = 'id_'+str(i)

    C_cols2.append(lbl)

    

df = pd.DataFrame(data = df_train_id, columns = C_cols2)



plt.figure(figsize=(30,10))

ax=sns.boxplot(x="variable", y="value", data=pd.melt(df))

plt.xticks(rotation=45)

plt.show()



del ax

gc.collect()
'''df_train = pd.merge(df_train_tran, df_train_id, on='TransactionID')

del df_train_tran

del df_train_id

gc.collect()'''
df_train_tran.describe()
#df_train_tran.interpolate(method ='linear', limit_direction ='forward') 
df_train_tran.head()
gc.collect()
cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

n_cat = len(cat_cols)

print(n_cat)
all_cols = set(df_train_tran.columns).difference(set(['TransactionID','isFraud']))

type(all_cols)

len(all_cols)
non_cat_cols = list(all_cols.difference(set(cat_cols)))

len(non_cat_cols)
df_train_tran.head(5)
for col in cat_cols:

    #print(col)

    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    imp.fit(df_train_tran[[col]])

    df_train_tran[[col]]= imp.transform(df_train_tran[[col]])

    df_train_tran[[col]] = LabelEncoder().fit_transform(df_train_tran[[col]])
print(len(non_cat_cols))

for col in non_cat_cols:

    #print(col)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    imp.fit(df_train_tran[[col]])

    df_train_tran[[col]]= imp.transform(df_train_tran[[col]])

    #df_train_tran[[col]] = LabelEncoder().fit_transform(df_train_tran[[col]])
df_train_tran.head(10)
df_train_tran = df_train_tran.values
print(type(df_train_tran))

print(df_train_tran)
train_tran_x = df_train_tran[:,2:]

train_tran_y = df_train_tran[:,1]

del df_train_tran

gc.collect()

print(np.shape(train_tran_x))

print(np.shape(train_tran_y))
for i in range(0,np.shape(train_tran_x)[1]):

    min1 = np.min(train_tran_x[:,i])

    max1 = np.max(train_tran_x[:,i])

    mean = np.mean(train_tran_x[:,i])

    train_tran_x[:,i] = (train_tran_x[:,i] - min1)/(max1-min1)

    #train_tran_x = MinMaxScaler().fit_transform(train_tran_x)
print(np.shape(train_tran_x))
rs = RandomUnderSampler(random_state=42)

X, y = rs.fit_resample(train_tran_x, train_tran_y)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

m_train = np.shape(y_train)[0]

m_valid = np.shape(y_valid)[0]



y_train = y_train.reshape(m_train, 1)

y_valid = y_valid.reshape(m_valid, 1)



print(np.shape(X_train))

print(np.shape(y_train))

print(np.shape(X_valid))

print(np.shape(y_valid))
reduced_features_no = 200
# calculate the mean of each column

M = np.mean(X_train, axis=1) 

#print(M)

M = M.reshape(np.shape(M)[0],1)

# center columns by subtracting column means

C = X_train - M

# calculate covariance matrix of centered matrix

V = cov(C.T)

#print(V)

# eigendecomposition of covariance matrix

values, vectors = eig(V)



# Sorting eigenvectors based on eigen values

sort_index = np.argsort(values)[::-1]

values = values[sort_index]

vectors = vectors[:, sort_index]

print(sort_index)



#print("eigen vectors shape: ", np.shape(vectors))

#print("eigen values: ",values)



# project data

P = vectors.T.dot(C.T)

P = P.T

print("transformed data size: ", np.shape(P))



# take top 3 for visualizing if we can do outlier detection 

tran_x_viz = P[:,:reduced_features_no]

print(np.shape(tran_x_viz))
# calculate the mean of each column

M = np.mean(X_valid, axis=1) 

#print(M)

M = M.reshape(np.shape(M)[0],1)

# center columns by subtracting column means

C = X_valid - M

# calculate covariance matrix of centered matrix

V = cov(C.T)

#print(V)

# eigendecomposition of covariance matrix

values, vectors = eig(V)



# Sorting eigenvectors based on eigen values

sort_index = np.argsort(values)[::-1]

values = values[sort_index]

vectors = vectors[:, sort_index]



print(sort_index)

#print("eigen vectors shape: ", np.shape(vectors))

#print("eigen values: ",values)



# project data

P = vectors.T.dot(C.T)

P = P.T

print(np.shape(P))



# take top 3 for visualizing if we can do outlier detection 

valid_x_viz = P[:,:reduced_features_no]

print(np.shape(valid_x_viz))
idx_1 = (y_train.T==1)

idx_0 = (y_train.T==0)

print(idx_0[0])

print(idx_1[0])

np.shape(tran_x_viz[idx_1[0]])



pos_tran_x = tran_x_viz[idx_1[0]]

neg_tran_x = tran_x_viz[idx_0[0]]

print(np.shape(pos_tran_x))

print(np.shape(neg_tran_x))
print(np.max(np.max(pos_tran_x)))

print(np.max(np.max(neg_tran_x)))
print(np.shape(tran_x_viz[0]))

print(tran_x_viz[0])
import seaborn as sns

from mpl_toolkits.mplot3d import axes3d

sns.set()



fig=plt.figure(figsize=(20,20))

ax = fig.gca(projection='3d')



print(np.shape(tran_x_viz))

print(np.shape(y_train))

for data, y in zip(tran_x_viz, y_train):

    c = 'b'

    #print(data)

    y = int(y)

    if y == 1:

        c = 'r' 

    x = data[0]

    y = data[1]

    z = data[2]

    ax.scatter(x, y, z, alpha=0.8, c=c, s=30)



ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')



plt.show()
# 2D plot



fig=plt.figure(figsize=(20,20))

ax = fig.gca()



print(np.shape(tran_x_viz))

print(np.shape(y_train))

for data, y in zip(tran_x_viz, y_train):

    c = 'b'

    #print(data)

    y = int(y)

    if y == 1:

        c = 'r' 

    x = data[0]

    y = data[1]

    ax.scatter(x, y, alpha=0.8, c=c, s=30)



ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')



plt.show()

print(np.shape(tran_x_viz))

print(np.shape(y_train))

print(np.shape(valid_x_viz))

print(np.shape(y_valid))
params ={

        'booster':'gbtree', 

        'objective':'binary:logistic', 

        'n_estimators':2000,

        'rate_drop': 0.2 

        }
gbm = xgb.XGBClassifier(**params)

model = gbm.fit(tran_x_viz, y_train)

                                                                                

y_pred = model.predict(valid_x_viz)

y_pred_proba = model.predict_proba(valid_x_viz)
tran_x_viz
# metrics

print(confusion_matrix(y_valid,y_pred))

print(classification_report(y_valid,y_pred))

print(accuracy_score(y_valid, y_pred))

skplt.metrics.plot_roc_curve(y_valid, y_pred_proba)

plt.show()