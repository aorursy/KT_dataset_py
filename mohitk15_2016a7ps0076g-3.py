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
train_data = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

test_data = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")
test_data.shape
train_data.head()
train_data.info()
train_data['Satisfied'].value_counts()
train_data=train_data.drop('custId',axis=1)
train_data.drop_duplicates(inplace=True)

train_data.shape
train_data.replace('',np.nan, inplace=True)

test_data.replace('',np.nan, inplace=True)

train_data.replace(' ',np.nan, inplace=True)

test_data.replace(' ',np.nan, inplace=True)
train_data['TotalCharges'] = pd.to_numeric(train_data['TotalCharges'])

test_data['TotalCharges'] =  pd.to_numeric(test_data['TotalCharges'])
train_data.fillna(train_data.mean(), inplace=True)

test_data.fillna(test_data.mean(), inplace=True)
train_data.isnull().sum()
df = train_data
allfeatures = ["SeniorCitizen","Married","Children","TVConnection","Internet","HighSpeed","AddedServices","tenure","MonthlyCharges","TotalCharges","Channel1","Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]

numfeatures = ["tenure", "MonthlyCharges", "TotalCharges"]

typefeatures = ["SeniorCitizen","Married","Children","TVConnection","Internet","HighSpeed","AddedServices","Channel1","Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]

X = df[allfeatures].copy()

Y = df['Satisfied'].copy()

X_test = test_data[allfeatures].copy()

X_id = test_data['custId'].copy()
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X[numfeatures])

X_test_scaled = scaler.fit_transform(X_test[numfeatures])
# Hot encoding

X_encoded = pd.get_dummies(X[typefeatures])

X_test_encoded = pd.get_dummies(X_test[typefeatures])
X_new = np.concatenate([X_scaled,X_encoded.values],axis=1)

X_test_new =  np.concatenate([X_test_scaled,X_test_encoded.values],axis=1)

X_new_df = pd.DataFrame(X_new)

X_new_df.to_csv("X_new.csv")
X_test = X_test_new

X_test.shape, X_new.shape

X_new=pd.DataFrame(X_new)

X_new.head()
# X_new = pd.DataFrame(X_new)

# corr = X_new.corr()



# # Generate a mask for the upper triangle

# mask = np.zeros_like(corr, dtype=np.bool)

# mask[np.triu_indices_from(mask)] = True



# # Set up the matplotlib figure

# f, ax = plt.subplots(figsize=(12, 9))



# # Generate a custom diverging colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)



# # Draw the heatmap with the mask and correct aspect ratio

# sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

#             square=True, linewidths=.5, cbar_kws={"shrink": .5})



# plt.show()
from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val = train_test_split(X_new,Y,test_size=0.3,random_state=0)
Y_train.value_counts(),Y_val.value_counts()
X_val = X_val=pd.DataFrame(X_val)
from sklearn.cluster import DBSCAN

from sklearn.cluster import KMeans
K = range(1,10) 



inertias = []

for k in K: 

    #Building and fitting the model 

    kmeans = KMeans(n_clusters=k).fit(X_val)

    kmeans.fit(X_val)

    inertias.append(kmeans.inertia_)

    

plt.plot(K, inertias, 'bx-')

plt.xlabel('Values of K') 

plt.ylabel('Inertia') 

plt.title('The Elbow Method using inertia') 

plt.show() 
X_test = pd.DataFrame(X_test)
kmeans = KMeans(n_clusters=6,random_state=0)

kmeans.fit(X_test)

y_kmeans = kmeans.predict(X_test)

set(y_kmeans)
y_kmeans = pd.DataFrame(y_kmeans,columns=['kf1'])

X_test = pd.concat([X_test,y_kmeans],axis=1)
kmeans = KMeans(n_clusters=2,random_state=0)

kmeans.fit(X_test)

y_kmeans = kmeans.predict(X_test)

X_test.head()
# kmeans = KMeans(n_clusters=2,random_state=0)

# kmeans.fit(X_val)

# y_kmeans = kmeans.predict(X_val)

# accuracy_score(y_kmeans,Y_val)
# kmeans = KMeans(n_clusters=5,random_state=0)

# kmeans.fit(X_test)

# y_kmeans = kmeans.predict(X_test)

# set(y_kmeans)
# y_kmeans = pd.DataFrame(y_kmeans,columns=['kf1'])

# X_test = pd.DataFrame(X_test)

# X_test = pd.concat([X_test,y_kmeans],axis=1)
# kmeans = KMeans(n_clusters=2,random_state=0)

# kmeans.fit(X_test)

# y_kmeans = kmeans.predict(X_test)
Y_out = pd.DataFrame(y_kmeans,columns=['Satisfied'])

Y_out = pd.concat([X_id,Y_out],axis=1)
Y_out["Satisfied"].value_counts()
Y_out.to_csv('output3.csv',index=False)
# db = DBSCAN(min_samples=14,eps=0.99)

# y_db = db.fit(X_test)

# labels = y_db.labels_

# labels = labels+1

# print(set(labels))

# i=114

# temp = copy.copy(labels)

# for j in range(len(labels)):

#     if ((1<<(labels[j]))&i)!=0:

#         labels[j]=0

#     else:

#         labels[j]=1

# print(labels)
# labels.shape
# Y_out = pd.DataFrame(labels,columns=['Satisfied'])

# Y_out = pd.concat([X_id,Y_out],axis=1)
# Y_out["Satisfied"].value_counts()
# Y_out.to_csv('output2.csv',index=False)