# This Python 3 environment comes with many helpful analytics libraries installed
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from IPython.display import display
import gc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
dtypes = {
    'Src IP': 'category',
    'Src Port': 'uint16',
    'Dst IP': 'category',
    'Dst Port': 'uint16',
    'Protocol': 'category',
    'Flow Duration': 'uint32',
    'Tot Fwd Pkts': 'uint32',
    'Tot Bwd Pkts': 'uint32',
    'TotLen Fwd Pkts': 'float32',
    'TotLen Bwd Pkts': 'float32',
    'Fwd Pkt Len Max': 'float32',
    'Fwd Pkt Len Min': 'float32',
    'Fwd Pkt Len Mean': 'float32',
    'Fwd Pkt Len Std': 'float32',
    'Bwd Pkt Len Max': 'float32',
    'Bwd Pkt Len Min': 'float32',
    'Bwd Pkt Len Mean': 'float32',
    'Bwd Pkt Len Std': 'float32',
    'Flow Byts/s': 'float32',
    'Flow Pkts/s': 'float32',
    'Flow IAT Mean': 'float32',
    'Flow IAT Std': 'float32',
    'Flow IAT Max': 'float32',
    'Flow IAT Min': 'float32',
    'Fwd IAT Tot': 'float32',
    'Fwd IAT Mean': 'float32',
    'Fwd IAT Std': 'float32',
    'Fwd IAT Max': 'float32',
    'Fwd IAT Min': 'float32',
    'Bwd IAT Tot': 'float32',
    'Bwd IAT Mean': 'float32',
    'Bwd IAT Std': 'float32',
    'Bwd IAT Max': 'float32',
    'Bwd IAT Min': 'float32',
    'Fwd PSH Flags': 'category',
    'Bwd PSH Flags': 'category',
    'Fwd URG Flags': 'category',
    'Bwd URG Flags': 'category',
    'Fwd Header Len': 'uint32',
    'Bwd Header Len': 'uint32',
    'Fwd Pkts/s': 'float32',
    'Bwd Pkts/s': 'float32',
    'Pkt Len Min': 'float32',
    'Pkt Len Max': 'float32',
    'Pkt Len Mean': 'float32',
    'Pkt Len Std': 'float32',
    'Pkt Len Var': 'float32',
    'FIN Flag Cnt': 'category',
    'SYN Flag Cnt': 'category',
    'RST Flag Cnt': 'category',
    'PSH Flag Cnt': 'category',
    'ACK Flag Cnt': 'category',
    'URG Flag Cnt': 'category',
    'CWE Flag Count': 'category',
    'ECE Flag Cnt': 'category',
    'Down/Up Ratio': 'float32',
    'Pkt Size Avg': 'float32',
    'Fwd Seg Size Avg': 'float32',
    'Bwd Seg Size Avg': 'float32',
    'Fwd Byts/b Avg': 'uint32',
    'Fwd Pkts/b Avg': 'uint32',
    'Fwd Blk Rate Avg': 'uint32',
    'Bwd Byts/b Avg': 'uint32',
    'Bwd Pkts/b Avg': 'uint32',
    'Bwd Blk Rate Avg': 'uint32',
    'Subflow Fwd Pkts': 'uint32',
    'Subflow Fwd Byts': 'uint32',
    'Subflow Bwd Pkts': 'uint32',
    'Subflow Bwd Byts': 'uint32',
    'Init Fwd Win Byts': 'uint32',
    'Init Bwd Win Byts': 'uint32',
    'Fwd Act Data Pkts': 'uint32',
    'Fwd Seg Size Min': 'uint32',
    'Active Mean': 'float32',
    'Active Std': 'float32',
    'Active Max': 'float32',
    'Active Min': 'float32',
    'Idle Mean': 'float32',
    'Idle Std': 'float32',
    'Idle Max': 'float32',
    'Idle Min': 'float32',
    'Label': 'category'
}
df = pd.read_csv(
    '/kaggle/input/ddos-datasets/ddos_balanced/final_dataset.csv',
    dtype=dtypes,
    parse_dates=['Timestamp'],
    usecols=[*dtypes.keys(), 'Timestamp'],
    engine='c',
    low_memory=True
)
del dtypes
gc.collect()
df.shape
df.describe(include='all')
mb = df.memory_usage().sum() / 1024**2
print('Memory usage of dataframe is {:.2f} MB'.format(mb))
colsToDrop = np.array(['Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'])
gc.collect()
# counting unique values and checking for skewness in the data
rowbuilder = lambda col: {'col': col, 'unique_values': df[col].nunique(), 'most_frequent_value': df[col].value_counts().index[0],'frequency': df[col].value_counts(normalize=True).values[0]}
frequency = [rowbuilder(col) for col in df.select_dtypes(include=['category']).columns]
skewed = pd.DataFrame(frequency)
skewed = skewed[skewed['frequency'] >= 0.95]
colsToDrop = np.union1d(colsToDrop, skewed['col'].values)
colsToDrop
del skewed
del rowbuilder
del frequency
gc.collect()
missing = df.isna().sum()
missing = pd.DataFrame({'count': missing, '% of total': missing/len(df)*100}, index=df.columns)
colsToDrop = np.union1d(colsToDrop, missing[missing['% of total'] >= 50].index.values)
dropnaCols = missing[(missing['% of total'] > 0) & (missing['% of total'] <= 5)].index.values
df['Flow Byts/s'].replace(np.inf, np.nan, inplace=True)
df['Flow Pkts/s'].replace(np.inf, np.nan, inplace=True)
dropnaCols = np.union1d(dropnaCols, ['Flow Byts/s', 'Flow Pkts/s'])
colsToDrop
dropnaCols
# perform actual drop
df.drop(columns=colsToDrop, inplace=True)
df.dropna(subset=dropnaCols, inplace=True)
gc.collect()
negValCols = ['Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Min']
for col in negValCols:
    df = df[df[col] >= 0]
mb = df.memory_usage().sum() / 1024**2
print('Memory usage of dataframe is {:.2f} MB'.format(mb))
df.describe()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["Label"]):
    traindf = df.iloc[train_index]
del df
gc.collect();
# plotting the target variable
labelCount = traindf['Label'].value_counts(normalize=True)*100
ax = sns.barplot(x=labelCount.index, y=labelCount.values)
ax1 = ax.twinx()
ax.set_ylabel('Frequency [%]')
ax1.set_ylabel("Count (in millions)")
ax1.set_ylim(0, len(traindf)/10**6)
ax.set_ylim(0, 100)
plt.title('Target Variable')
cnt = pd.crosstab(traindf['Protocol'], traindf['Label'])
cnt = cnt.stack().reset_index().rename(columns={0: 'Count'})
sns.barplot(x=cnt['Protocol'], y=cnt['Count'], hue=cnt['Label'])
def getNetworkClass(col):
    networkClasses = traindf[col].str.split('.',n=1, expand=True)[0]
    networkClasses = networkClasses.astype('uint8')
    networkClasses = pd.cut(
        networkClasses,
        bins=[0, 127, 191, 223, 239, np.inf],
        labels=['A', 'B', 'C', 'D', 'E'],
        include_lowest=True
    )
    return networkClasses
srcNetworkClass = getNetworkClass('Src IP')
dstNetworkClass = getNetworkClass('Dst IP')
cnt = pd.crosstab(srcNetworkClass, traindf['Label'], rownames=['Class'])
cnt = cnt.stack().reset_index().rename(columns={0: 'Count'})
sns.barplot(x=cnt['Class'], y=cnt['Count'], hue=cnt['Label'])
cnt = pd.crosstab(dstNetworkClass, traindf['Label'], rownames=['Class'])
cnt = cnt.stack().reset_index().rename(columns={0: 'Count'})
sns.barplot(x=cnt['Class'], y=cnt['Count'], hue=cnt['Label'])
sns.scatterplot(x='Tot Fwd Pkts', y='Tot Bwd Pkts', hue='Label', data=traindf.sample(100000, random_state=42))
sns.scatterplot(x='TotLen Fwd Pkts', y='TotLen Bwd Pkts', hue='Label', data=traindf.sample(100000, random_state=42))
num_cols = traindf.select_dtypes(exclude=['category', 'datetime64[ns]']).columns
fwd_cols = [col for col in num_cols if 'Fwd' in col]
bwd_cols = [col for col in num_cols if 'Bwd' in col]
corr = traindf[fwd_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
plt.subplots(figsize=(10,10))
sns.heatmap(corr, mask=mask)
def getCorrelatedFeatures(corr):
    correlatedFeatures = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                correlatedFeatures.add(corr.columns[i])
    return correlatedFeatures
correlatedFeatures = set()
correlatedFeatures = correlatedFeatures | getCorrelatedFeatures(corr)
corr = traindf[bwd_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
plt.subplots(figsize=(10,10))
sns.heatmap(corr, mask=mask)
correlatedFeatures = correlatedFeatures | getCorrelatedFeatures(corr)
correlatedFeatures
traindf.drop(columns=correlatedFeatures, inplace=True)
num_cols = traindf.select_dtypes(exclude=['category', 'datetime64[ns]']).columns
cols = [col for col in num_cols if 'Fwd' in col or 'Bwd' in col]
corr = traindf[cols].corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
plt.subplots(figsize=(10,10))
sns.heatmap(corr, mask=mask)
correlatedFeatures = correlatedFeatures | getCorrelatedFeatures(corr)
traindf.drop(columns=getCorrelatedFeatures(corr), inplace=True)
traindf.describe()
traindf.shape
# sns.scatterplot(x='Fwd IAT Tot', y='Bwd IAT Tot', hue='Label', data=traindf.sample(100000, random_state=42))
# sns.scatterplot(x='Fwd Header Len', y='Bwd Header Len', hue='Label', data=traindf.sample(100000, random_state=42))
# sns.scatterplot(x='Fwd Pkts/s', y='Bwd Pkts/s', hue='Label', data=traindf.sample(100000, random_state=42))