import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Visualization
folder_name = "/kaggle"
train_features = pd.read_csv(folder_name + "/input/lish-moa/train_features.csv")
test_features = pd.read_csv(folder_name +  "/input/lish-moa/test_features.csv")
train_features.head(5)
test_features.head(5)
print('We have {} training rows and {} test rows.'.format(train_features.shape[0], test_features.shape[0]))
print('We have {} training columns and {} test columns.'.format(train_features.shape[1], test_features.shape[1]))
print(f'We have {train_features.isnull().values.sum()} missing values in train data')
print(f'We have {test_features.isnull().values.sum()} missing values in test data')
print(train_features.describe())
train_features.info()
indexName = train_features.columns.tolist()
type(indexName)
gIndexName = [name for name in indexName if 'g-' in name]
gIndexData = train_features.loc[:,gIndexName]
gIndexData.head(5)
g_mean_df = []
g_std_df = [] 
for index in gIndexName:
    mData = gIndexData[index].mean()
    gData = gIndexData[index].std()
    g_mean_df.append(mData) #print(gIndexData.std(axis=0));
    g_std_df.append(gData)
plt.plot( gIndexName , g_mean_df  ,linestyle='--', linewidth=1) 
plt.plot(gIndexName, g_std_df, linestyle='-', linewidth=2)
plt.title('Line Graph w/ different linestyles and linewidths', fontsize=20) 
plt.ylabel('Cummulative Num', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.show()

cIndexName = [name for name in indexName if 'c-' in name]
cIndexData = train_features.loc[:,cIndexName]
cIndexData.head(5)
c_mean_df = []
c_std_df = [] 
for index in cIndexName:
    mData = cIndexData[index].mean()
    cData = cIndexData[index].std()
    c_mean_df.append(mData) #print(gIndexData.std(axis=0));
    c_std_df.append(cData)
plt.plot( cIndexName , c_mean_df  ,linestyle='--', linewidth=1) 
plt.plot(cIndexName, c_std_df, linestyle='-', linewidth=2)
plt.title('Line Graph w/ different linestyles and linewidths', fontsize=20) 
plt.ylabel('Cummulative Num', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.show()

plt.figure(figsize=(16,6))
ax=sns.countplot(train_features['cp_type'],palette="hls",order = train_features['cp_type'].value_counts().index)
plt.title("Pitch (quality of sound - how high/low the tone is)", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel("Frequency", fontsize=14)
plt.xlabel("");
plt.figure(figsize=(16,6))
ax=sns.countplot(train_features['cp_dose'],palette="hls",order = train_features['cp_dose'].value_counts().index)
plt.title("Pitch (quality of sound - how high/low the tone is)", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel("Frequency", fontsize=14)
plt.xlabel("");
plt.figure(figsize=(16,6))
ax=sns.countplot(train_features['cp_time'],palette="hls",order = train_features['cp_time'].value_counts().index)
plt.title("Pitch (quality of sound - how high/low the tone is)", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel("Frequency", fontsize=14)
plt.xlabel("");