import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Data Visualization 

import seaborn as sns

from sklearn import datasets
train = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

test = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')
train.head()
train['gender']=train['gender'].astype('category').cat.codes

train['Married']=train['Married'].astype('category').cat.codes

train['Children']=train['Children'].astype('category').cat.codes

train['TVConnection']=train['TVConnection'].astype('category').cat.codes

train['Channel1']=train['Channel1'].astype('category').cat.codes

train['Channel2']=train['Channel2'].astype('category').cat.codes

train['TotalCharges']=train['TotalCharges'].replace(r'^\s+$', 0, regex=True).astype(np.float64)

train['Channel3']=train['Channel3'].astype('category').cat.codes

train['Channel4']=train['Channel4'].astype('category').cat.codes

train['Channel5']=train['Channel5'].astype('category').cat.codes

train['Channel6']=train['Channel6'].astype('category').cat.codes

train['Internet']=train['Internet'].astype('category').cat.codes

train['HighSpeed']=train['HighSpeed'].astype('category').cat.codes

train['AddedServices']=train['AddedServices'].astype('category').cat.codes

train['Subscription']=train['Subscription'].astype('category').cat.codes

train['PaymentMethod']=train['PaymentMethod'].astype('category').cat.codes
test['gender']=test['gender'].astype('category').cat.codes

test['Married']=test['Married'].astype('category').cat.codes

test['Children']=test['Children'].astype('category').cat.codes

test['TVConnection']=test['TVConnection'].astype('category').cat.codes

test['Channel1']=test['Channel1'].astype('category').cat.codes

test['Channel2']=test['Channel2'].astype('category').cat.codes

test['TotalCharges']=test['TotalCharges'].replace(r'^\s+$', 0, regex=True).astype(np.float64)

test['Channel3']=test['Channel3'].astype('category').cat.codes

test['Channel4']=test['Channel4'].astype('category').cat.codes

test['Channel5']=test['Channel5'].astype('category').cat.codes

test['Channel6']=test['Channel6'].astype('category').cat.codes

test['Internet']=test['Internet'].astype('category').cat.codes

test['HighSpeed']=test['HighSpeed'].astype('category').cat.codes

test['AddedServices']=test['AddedServices'].astype('category').cat.codes

test['Subscription']=test['Subscription'].astype('category').cat.codes

test['PaymentMethod']=test['PaymentMethod'].astype('category').cat.codes
train.corr()
corr = train.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
features=['SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','AddedServices','Subscription','PaymentMethod','tenure']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# features=['gender','SeniorCitizen','Children','TVConnection','Internet','AddedServices','Subscription','PaymentMethod','tenure']

x1 = scaler.fit_transform(train[features])

y1 = scaler.fit_transform(train[['Satisfied']])





x2 = scaler.fit_transform(test[features])

x3=test['custId']



from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(x1,y1,test_size=0.10,random_state=42) 
from sklearn.cluster import KMeans

# # from sklearn.cluster import KMeans



kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)

kmeans.fit(X_train,y_train)

y_pred=kmeans.predict(X_val)
#from sklearn.cluster import Birch, SpectralClustering

# from sklearn.cluster import Birch

 

# from sklearn.cluster import KMeans

#kmeans= SpectralClustering(2,n_init=100)

# kmeans = Birch(branching_factor=50,n_clusters=2,threshold=1.5)

# kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)

#kmeans.fit(X_train,y_train)

#y_pred=kmeans.predict(X_val)
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_pred,y_val)

accuracy

# y_pred.to_csv('temp.csv')

#from sklearn.metrics import mean_absolute_error

#mae_lr = mean_absolute_error(y_pred,y_val)

#print("Mean Absolute Error of K-means: {}".format(mae_lr))
y2=kmeans.predict(x2)
test.head()
output = test

output['Satisfied'] = y2

output['custId']=test[['custId']].copy()



ans=output[['custId','Satisfied']].copy()

ans.to_csv('tempo.csv',index=False)

ans.head()