import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score
import os
print(os.listdir("../input"))
df_athlete = pd.read_csv('../input/athlete_events.csv')
df_regions = pd.read_csv('../input/noc_regions.csv')
df_athlete.head()
df_athlete.info()
df_athlete.describe()
df_athlete.isna().any()
sns.heatmap(data=df_athlete.isna())
df_regions.head()
df_regions.info()
df_regions.describe()
df_regions.isna().any()
sns.heatmap(data=df_regions.isna())
merge_2DF = pd.merge(df_athlete, df_regions, how='left', on = 'NOC')
merge_2DF.head()
merge_2DF.info()
merge_2DF.describe()
sns.heatmap(data=merge_2DF.isna())
merge_2DF_1=merge_2DF[(merge_2DF['Medal']=='Gold') | (merge_2DF['Medal']=='Silver') | (merge_2DF['Medal']=='Bronze')]
merge_2DF_1.head()
merge_2DF_1.isnull().any()
merge_2DF_1.info()
merge_2DF_1.head()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(merge_2DF_1['Medal'])
merge_2DF_1['Medal_1'] = le.transform(merge_2DF_1['Medal'])
keys = le.classes_
values = le.transform(le.classes_)
dictionary = dict(zip(keys, values))
print(dictionary)
sns.heatmap(merge_2DF_1.corr())
#Exploring counts
import seaborn as sns
sns.set(rc={'figure.figsize':(8,4)})
sns.countplot("Sex", data=merge_2DF_1)
merge_2DF_1[merge_2DF_1['Sex']=='M']['Medal'].count()
merge_2DF_1[merge_2DF_1['Sex']=='F']['Medal'].count()
sns.catplot(y="Sex",hue="Medal", data=merge_2DF_1, kind="count")
GM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Gold')]['Medal'].count()
print('Count of (Male) Gold medals-',GM)
SM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Silver')]['Medal'].count()
print('Count of (Male) Silver medals-',SM)
BM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Bronze')]['Medal'].count()
print('Count of (Male) Bronze medals-',BM)
GM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Gold')]['Medal'].count()
print('Count of (Female) Gold medals-',GM)
SM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Silver')]['Medal'].count()
print('Count of (Female) Silver medals-',SM)
BM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Bronze')]['Medal'].count()
print('Count of (Female) Bronze medals-',BM)
#Exploring Ages

import seaborn as sns
sns.set(rc={'figure.figsize':(19,10)})
sns.countplot(merge_2DF_1['Age'])
# age of medalists (Male)
A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Gold')]['Age'].hist()
print('Mean age of gold medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Gold')]['Age'].mean())
print('Median age of gold medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Gold')]['Age'].median())
A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Silver')]['Age'].hist()
print('Mean age of silver medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Silver')]['Age'].mean())
print('Median age of silver medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Silver')]['Age'].median())
A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Bronze')]['Age'].hist()
print('Mean age of bronze medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Bronze')]['Age'].mean())
print('Median age of bronze medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Bronze')]['Age'].median())
# age of  medalists (Female)
A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Gold')]['Age'].hist()
print('Mean age of gold medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Gold')]['Age'].mean())
print('Median age of gold medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Gold')]['Age'].median())
A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Silver')]['Age'].hist()
print('Mean age of silver medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Silver')]['Age'].mean())
print('Median age of silver medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Silver')]['Age'].median())
A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Bronze')]['Age'].hist()
print('Mean age of bronze medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Bronze')]['Age'].mean())
print('Median age of bronze medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Bronze')]['Age'].median())
# stats/medals team - Russia
A_GM=merge_2DF_1[(merge_2DF_1['Team']=='Russia') & (merge_2DF_1['Medal']=='Gold')]['Age'].hist()
A_GM=merge_2DF_1[(merge_2DF_1['Team']=='Russia') & (merge_2DF_1['Medal']=='Gold')]['ID'].count()
print('Count of Russia gold medalists-',A_GM)
# trends
merge_2DF_1_M = merge_2DF_1[(merge_2DF_1.Sex == 'M') & (merge_2DF_1.Team == 'Russia')]
sns.countplot(x='Year', data=merge_2DF_1_M)
merge_2DF_1_F = merge_2DF_1[(merge_2DF_1.Sex == 'F') & (merge_2DF_1.Team == 'Russia')]
sns.countplot(x='Year', data=merge_2DF_1_F)
sns.countplot(x='Year', data=merge_2DF_1)
