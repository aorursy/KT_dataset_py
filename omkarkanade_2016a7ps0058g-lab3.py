import warnings

warnings.filterwarnings("ignore")

import numpy as np

import math

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # for plot styling
df = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')
df.head()
df.info()
# df.isnull().sum()
tcs = df['TotalCharges']

mcs = df['MonthlyCharges']

for t in range(len(tcs)):

#     if math.isnan(tcs[t]):

    if tcs[t] == " ":

        df['TotalCharges'][t] = mcs[t]

#         print(df.iloc[t, :])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
# df.isnull().sum()
df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
# categorical_features = ['gender', 'SeniorCitizen', 'Married', 'Children',

#        'TVConnection', 'Channel1', 'Channel2', 'Channel3', 'Channel4',

#        'Channel5', 'Channel6', 'Internet', 'HighSpeed', 'AddedServices',

#        'Subscription', 'PaymentMethod']

# df.columns
# new_df = 

new_df = pd.get_dummies(df)

new_df.head()
corr = new_df.corr()

# sns.heatmap(corr, vmin=0, vmax=1, linewidth=0.5)

corr.style.background_gradient(cmap='coolwarm')
new_df.columns
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# categorical_features = ['gender_Female', 'gender_Male', 'Married_No',

#        'Married_Yes', 'Children_No', 'Children_Yes', 'TVConnection_Cable',

#        'TVConnection_DTH', 'TVConnection_No', 'Channel1_No',

#        'Channel1_No tv connection', 'Channel1_Yes', 'Channel2_No',

#        'Channel2_No tv connection', 'Channel2_Yes', 'Channel3_No',

#        'Channel3_No tv connection', 'Channel3_Yes', 'Channel4_No',

#        'Channel4_No tv connection', 'Channel4_Yes', 'Channel5_No',

#        'Channel5_No tv connection', 'Channel5_Yes', 'Channel6_No',

#        'Channel6_No tv connection', 'Channel6_Yes', 'Internet_No',

#        'Internet_Yes', 'HighSpeed_No', 'HighSpeed_No internet',

#        'HighSpeed_Yes', 'AddedServices_No', 'AddedServices_Yes',

#        'Subscription_Annually', 'Subscription_Biannually',

#        'Subscription_Monthly', 'PaymentMethod_Bank transfer',

#        'PaymentMethod_Cash', 'PaymentMethod_Credit card',

#        'PaymentMethod_Net Banking']



# categorical_features = ['gender_Female', 'gender_Male', 'Married_No',

#        'Married_Yes', 'Children_No', 'Children_Yes', 'TVConnection_Cable',

#        'TVConnection_DTH', 'TVConnection_No', 'Channel1_No',

#        'Channel1_Yes', 'Channel2_No',

#        'Channel2_Yes', 'Channel3_No',

#        'Channel3_Yes', 'Channel4_No',

#        'Channel4_Yes', 'Channel5_No',

#        'Channel5_Yes', 'Channel6_No',

#        'Channel6_Yes', 'Internet_No',

#        'Internet_Yes', 'HighSpeed_No', 'HighSpeed_No internet',

#        'HighSpeed_Yes', 'AddedServices_No', 'AddedServices_Yes',

#        'Subscription_Annually', 'Subscription_Biannually',

#        'Subscription_Monthly', 'PaymentMethod_Bank transfer',

#        'PaymentMethod_Cash', 'PaymentMethod_Credit card',

#        'PaymentMethod_Net Banking']



# categorical_features = ['Married_No',

#        'Married_Yes', 'Children_No', 'Children_Yes', 'TVConnection_Cable',

#        'TVConnection_DTH', 'TVConnection_No', 'Channel1_No',

#        'Channel1_No tv connection', 'Channel1_Yes', 'Channel2_No',

#        'Channel2_No tv connection', 'Channel2_Yes', 'Channel3_No',

#        'Channel3_No tv connection', 'Channel3_Yes', 'Channel4_No',

#        'Channel4_No tv connection', 'Channel4_Yes', 'Channel5_No',

#        'Channel5_No tv connection', 'Channel5_Yes', 'Channel6_No',

#        'Channel6_No tv connection', 'Channel6_Yes',

#        'AddedServices_No', 'AddedServices_Yes',

#        'Subscription_Annually', 'Subscription_Biannually',

#        'Subscription_Monthly', 'PaymentMethod_Bank transfer',

#        'PaymentMethod_Cash', 'PaymentMethod_Credit card',

#        'PaymentMethod_Net Banking']



categorical_features = ['Married_No',

       'Married_Yes', 'Children_No', 'Children_Yes', 'TVConnection_Cable',

       'TVConnection_DTH', 'TVConnection_No', 'Channel1_No',

       'Channel1_No tv connection', 'Channel2_No',

       'Channel2_No tv connection', 'Channel3_No',

       'Channel3_No tv connection', 'Channel4_No',

       'Channel4_No tv connection', 'Channel5_No',

       'Channel5_No tv connection', 'Channel5_Yes', 'Channel6_No',

       'Channel6_No tv connection', 'Channel6_Yes',

       'AddedServices_No', 'AddedServices_Yes',

       'Subscription_Annually', 'Subscription_Biannually',

       'Subscription_Monthly', 'PaymentMethod_Bank transfer',

       'PaymentMethod_Credit card',

       'PaymentMethod_Net Banking']
X = new_df[numerical_features+categorical_features]

y = new_df['Satisfied']
from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X[numerical_features] = scaler.fit_transform(X[numerical_features])



X.head()
# X.isnull().sum()
# from sklearn.decomposition import PCA



# new_dim = 5

# pca = PCA(n_components=new_dim)

# principalComponents = pca.fit_transform(X)
# principalDf = pd.DataFrame(data = principalComponents

#              , columns = ['pc'+str(i) for i in range(1, new_dim+1)])
# from sklearn.model_selection import RandomizedSearchCV



# threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# branching_factor = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
# random_grid = {'threshold': threshold,

#                'branching_factor': branching_factor}



# print(random_grid)
from sklearn.cluster import MiniBatchKMeans

from sklearn.cluster import KMeans

from sklearn.cluster import SpectralClustering

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import Birch



# kmeans = MiniBatchKMeans(n_clusters=2, batch_size=100).fit(X)

# kmeans = KMeans(n_clusters=2, random_state=42).fit(X)

# kmeans = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(X)

# kmeans = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(principalDf)

kmeans = Birch(branching_factor=45, n_clusters=2, threshold=0.5, compute_labels=True).fit(X)
preds = kmeans.predict(X)

# preds = kmeans.labels_

# sat_vals = kmeans.cluster_centers_[preds]
# kmeans.labels_
correct = 0

for p in range(len(y)):

    if preds[p] == y[p]:

        correct += 1

print(float(correct)/len(y))
tst = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

temp = tst.copy()

tst.head()
# X_tst.isnull().sum()
tst_tcs = tst['TotalCharges']

tst_mcs = tst['MonthlyCharges']

for t in range(len(tst_tcs)):

#     if math.isnan(tst_tcs[t]):

    if tst_tcs[t] == " ":

        tst['TotalCharges'][t] = tst_mcs[t]
tst['TotalCharges'] = pd.to_numeric(tst['TotalCharges'])
tst.isnull().sum()
new_tst = pd.get_dummies(tst)

new_tst.head()
X_tst = new_tst[numerical_features+categorical_features]

# y_tst = new_df['Satisfied']
from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_tst[numerical_features] = scaler.fit_transform(X_tst[numerical_features])



X_tst.head()
# from sklearn.decomposition import PCA



# new_dim = 5

# pca = PCA(n_components=new_dim)

# tst_principalComponents = pca.fit_transform(X_tst)



# tst_principalDf = pd.DataFrame(data = tst_principalComponents

#              , columns = ['pc'+str(i) for i in range(1, new_dim+1)])
# tst_pred = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X_tst)

# tst_pred = tst_pred.labels_

tst_pred = kmeans.predict(X_tst)
# for e in range(len(tst_pred)):

#     tst_pred[e] = 1-tst_pred[e]
tst_pred = pd.Series(tst_pred)

frame = {'custId':temp.custId, 'Satisfied':tst_pred}

res = pd.DataFrame(frame)

res
# export_csv = res.to_csv (r'/home/omkar/Desktop/ml_result.csv', index = None, header=True)