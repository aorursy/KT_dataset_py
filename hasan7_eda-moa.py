import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train_path = '/kaggle/input/lish-moa/train_features.csv'
df_train_features = pd.read_csv(train_path)
test_path = '/kaggle/input/lish-moa/test_features.csv'
df_test_features = pd.read_csv(test_path)
train_targets_path = '/kaggle/input/lish-moa/train_targets_scored.csv'
df_train_target = pd.read_csv(train_targets_path)
df_train_features.head()
print("Total Records for train : ",df_train_features.sig_id.nunique())
print("Total Features for train : ", df_train_features.shape[1])
print("Total Records for test : ",df_test_features.sig_id.nunique())
print("Total Features for test : ", df_test_features.shape[1])
print("Number of Nan in train : ",df_train_features.isnull().sum().sum())
print("Number of Nan in test : ",df_test_features.isnull().sum().sum())
print("Count plot for categorical feature in Train data")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))
sns.countplot(x="cp_type", data=df_train_features, ax=ax1)
sns.countplot(x="cp_time", data=df_train_features, ax=ax2)
sns.countplot(x="cp_dose", data=df_train_features, ax=ax3)
print("Count plot for categorical feature in test data")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))
sns.countplot(x="cp_type", data=df_test_features, ax=ax1)
sns.countplot(x="cp_time", data=df_test_features, ax=ax2)
sns.countplot(x="cp_dose", data=df_test_features, ax=ax3)
df_train_features['cp_type'] = df_train_features['cp_type'].map({'trt_cp':0,'ctl_vehicle':1})
df_train_features['cp_time'] = df_train_features['cp_time'].map({24:0,48:1,72:2})
df_train_features['cp_dose'] = df_train_features['cp_dose'].map({'D1':0,'D2':1})

df_test_features['cp_type'] = df_test_features['cp_type'].map({'trt_cp':0,'ctl_vehicle':1})
df_test_features['cp_time'] = df_test_features['cp_time'].map({24:0,48:1,72:2})
df_test_features['cp_dose'] = df_test_features['cp_dose'].map({'D1':0,'D2':1})
df_train_target.head()
df_train_target_temp = df_train_target.drop('sig_id', axis=1) 
df_count = df_train_target_temp.apply(pd.Series.value_counts)
df_count = df_count.sort_values(by = 1, axis = 1, ascending = False) 
df_T = df_count.T
df_T['Index'] = df_T.index
df_T
fig, (ax) = plt.subplots(1, 1, figsize=(15,20))
ax = sns.barplot(x=1, y="Index", data=df_T.head(40), ax = ax)
g_col = [col for col in df_train_features if col.startswith('g-')]
c_col = [col for col in df_train_features if col.startswith('c-')]
from sklearn.decomposition import PCA # import.
pca = PCA(n_components=5) #instantiate.
pca.fit(df_train_features[c_col])
fig = plt.figure(figsize = (20,5))
ax = plt.subplot(121)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('principal components')
plt.ylabel('explained variance')

ax2 = plt.subplot(122)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

plt.show()
