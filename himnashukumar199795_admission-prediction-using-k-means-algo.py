import pandas as pd

import numpy as np

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base_dataset1=pd.read_csv("../input/admissionpredictioncsv/Admission_Predict.csv")
base_dataset=base_dataset1
base_dataset.head()
base_dataset.isna().sum()
base_dataset.columns
base_dataset.shape
base_dataset.var()
base_dataset=base_dataset[['GRE Score','TOEFL Score']]
sns.boxplot(base_dataset['GRE Score'])
sns.boxplot(base_dataset['TOEFL Score'])
base_dataset.describe()
import pandas_profiling
base_dataset.profile_report(style={'full_width':True})
base_dataset.head()
from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler()

mn.fit(base_dataset)

test=mn.transform(base_dataset)

test=pd.DataFrame(test,columns=['GRE Score','TOEFL Score'])
test.describe()
from sklearn.cluster import KMeans

km=KMeans(n_clusters=3)

km.fit(test)

km.labels_
base_dataset1['cluster']=km.labels_
base_dataset1[base_dataset1['cluster']==0].shape
base_dataset1[base_dataset1['cluster']==1].shape
base_dataset1[base_dataset1['cluster']==2].shape
base_dataset1[base_dataset1['cluster']==0]['GRE Score'].min(),base_dataset1[base_dataset1['cluster']==0]['GRE Score'].max()
avg_c1=base_dataset1[base_dataset1['cluster']==0]['GRE Score'].mean()

cluster1=base_dataset1[base_dataset1['cluster']==0]['GRE Score']

cluster1[cluster1<avg_c1].index
x=[]

for i in cluster1.values:

    if (i-cluster1.values.mean())<0:

        x.append(abs(i-cluster1.values.mean()))
x=np.array(x)
x
len(x)