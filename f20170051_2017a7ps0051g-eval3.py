import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

def label(data, label_encoder=False, one_hot=False, categorical_features=None):

    data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    data.replace(r'^\s+$', np.nan, regex=True, inplace=True)

    if 'TotalCharges' in data.columns:

        data['TotalCharges'] = data['TotalCharges'].astype(float)

    if 'MonthlyCharges' in data.columns:

        data['MonthlyCharges'] = data['MonthlyCharges'].astype(float)

    if 'tenure' in data.columns:

        data['tenure'] = data['tenure'].astype(float)

#     data['TotalCharges'] = np.log(data['TotalCharges']+1)

#     data['MonthlyCharges'] = np.log(data['MonthlyCharges']+1)

#     data['tenure'] = np.log(data['tenure']+1)

    if one_hot:

        data = onehotencode(data, categorical_features)

    elif label_encoder:

        data = data.apply(LabelEncoder().fit_transform)

    return data
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.cluster import DBSCAN, SpectralClustering

from sklearn.cluster import MiniBatchKMeans

#import hdbscan

from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler

def preprocess(x, normalize=True):

    x.fillna(value = x.mean(), inplace=True)

    print(x.columns)

    if normalize:

        x = sklearn.preprocessing.normalize(x,norm='l1')

    #scaler = RobustScaler()

    #x = scaler.fit_transform(x)

    #print('After Normalizing')

    #print(x)

    x = np.array(x)

    x = x.reshape((x.shape[0],x.shape[1]))

    return x
data = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

data.drop('custId', axis=1, inplace=True)

categorical_features = ['Internet','gender', 'SeniorCitizen', 'Married', 'Children', 'TVConnection', 'Channel1', 'Channel2', 'Channel3', 'Channel4','Channel5', 'Channel6','HighSpeed', 'AddedServices', 'Subscription', 'PaymentMethod']



y = data['Satisfied']

data.drop(['Satisfied','Subscription','PaymentMethod','SeniorCitizen','AddedServices','Internet','gender','HighSpeed','Married','Channel4','Channel5','Channel3','Channel6','Channel1','Channel2','Children','TVConnection'], axis=1, inplace=True)

#data.drop(categorical_features,axis=1,inplace=True)

data = label(data, label_encoder=True, categorical_features=categorical_features)

data.columns

x = preprocess(data, normalize=True)



x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.001,random_state=69)



clf = KMeans(2,n_init = 200, random_state=42)

#clf = AgglomerativeClustering(2,linkage='complete')

clf.fit(x_train, y_train)

#pred = clf.fit_predict(test)
pred_train = clf.predict(x_train)
pred_train
pred_test = clf.predict(x_test)
print(roc_auc_score(y_train, pred_train))

print(accuracy_score(y_train, pred_train))
print(roc_auc_score(y_train, 1-pred_train))

print(accuracy_score(y_train, 1-pred_train))
print(roc_auc_score(y_test, pred_test))

print(accuracy_score(y_test, pred_test))
print(roc_auc_score(y_test, 1-pred_test))

print(accuracy_score(y_test, 1-pred_test))
# pred = clf.fit_predict(x_test)

# print(roc_auc_score(y_test, 1-pred))

# print(accuracy_score(y_test, pred))
# pred
test_df = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

test_df.drop(['custId'], axis=1, inplace=True)

test_df.drop(['Subscription','PaymentMethod','SeniorCitizen','AddedServices','Internet','gender','HighSpeed','Married','Channel4','Channel5','Channel3','Channel6','Channel1','Channel2','Children','TVConnection'], axis=1, inplace=True)

test = label(test_df, label_encoder=True, categorical_features=categorical_features)

test = preprocess(test,normalize=True)

pred = clf.fit_predict(test)
test_df = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

submission = pd.concat([test_df['custId'], pd.Series(1-pred)], axis=1)

submission.columns = ['custId', 'Satisfied']

submission.to_csv('submission.csv', index=False)
np.unique(1-pred, return_counts=True)
#Should look like

np.unique(1-pred, return_counts=True)
from sklearn.feature_selection import chi2

chi_scores = chi2(x, y)
x.shape
data.columns
p_values = pd.Series(chi_scores[1], index=data.columns)

p_values.sort_values(ascending=False, inplace=True)



p_values.plot.bar()
chi_scores[1]
data.columns
def onehotencode(df, categorical_features):

    df = pd.get_dummies(df, columns=categorical_features)

    return df
# data
data = label(data, one_hot=True, categorical_features=categorical_features)
y = data['Satisfied']

X = preprocess(data)



clf = AgglomerativeClustering(2)

clf.fit(X, y)
plt.figure(figsize=(10,10))

plt.subplot(221)

plt.scatter(x_train[:,0],x_train[:,1],c=y_train)

plt.subplot(222)

plt.scatter(x_train[:,0],x_train[:,1],c=pred_train)
from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

from matplotlib import style


%matplotlib notebook

style.use('ggplot')



fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(111, projection='3d')



x = x_train[:,0]

y = x_train[:,1]

z = x_train[:,2]



ax1.scatter(x, y, z, c=y_train, marker='o')



ax1.set_xlabel('x axis')

ax1.set_ylabel('y axis')

ax1.set_zlabel('z axis')



plt.show()
fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(111, projection='3d')



x = x_train[:,0]

y = x_train[:,1]

z = x_train[:,2]



ax1.scatter(x, y, z, c=pred_train, marker='o')



ax1.set_xlabel('x axis')

ax1.set_ylabel('y axis')

ax1.set_zlabel('z axis')



plt.show()