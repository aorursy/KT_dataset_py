import pandas as pd

import numpy as np
data=pd.read_csv('../input/Patient Data.csv')

data.head()
data.isna().sum()
data.shape
data['smoking history'].unique()
#import pandas_profiling
#pandas_profiling.ProfileReport(data)
#import matplotlib.pyplot as plt

#%matplotlib inline
#plt.scatter(data.index,data.BMI)
data['smoking history']=data['smoking history'].fillna('Unknown')
data['smoking history'].value_counts()
data['smoking history'].unique()
data.dtypes
data.isna().sum()
train_data_cat=data.select_dtypes(include=[object])

train_data_num=data.select_dtypes(exclude=[object])

train_data_cat=pd.get_dummies(train_data_cat, columns=train_data_cat.columns, drop_first=True)

data_df=pd.concat([train_data_cat,train_data_num],axis=1)
data_df.head()
data_df.shape
from fancyimpute import IterativeImputer
XY_incomplete = data_df.copy()
data_complete_df = pd.DataFrame(IterativeImputer(n_iter=5, sample_posterior=True, random_state=8).fit_transform(XY_incomplete))
data_complete_df.columns=data_df.columns
data_complete_df.head()
data_complete_df.isna().sum()
data_complete_df['diabetes & hypertension']=data_complete_df['diabetes']*data_complete_df['hypertension']

data_complete_df['diabetes & srtoke']=data_complete_df['diabetes']*data_complete_df['stroke']

data_complete_df['diabetes & heart disease']=data_complete_df['diabetes']*data_complete_df['heart disease']

data_complete_df['hypertension & stroke']=data_complete_df['hypertension']*data_complete_df['stroke']

data_complete_df['hypertension & heart disease']=data_complete_df['hypertension']*data_complete_df['heart disease']

data_complete_df['stroke & heart disease']=data_complete_df['stroke']*data_complete_df['heart disease']
data_complete_df['diabetes,hypertension,stroke']=data_complete_df['diabetes']*data_complete_df['hypertension']*data_complete_df['stroke']

data_complete_df['diabetes,hypertension,heart disease']=data_complete_df['diabetes']*data_complete_df['hypertension']*data_complete_df['heart disease']

data_complete_df['hypertension,stroke,heart disease']=data_complete_df['hypertension']*data_complete_df['stroke']*data_complete_df['heart disease']
data_complete_df.shape
data_complete_df['diabetes & hypertension'].value_counts()
data_complete_df.head()
dd=data_complete_df[data_complete_df['diabetes']==1]
dd1=dd[dd['hypertension']==1]
dd1.shape
data_complete_df['hypertension,stroke,heart disease'].value_counts()
dataaa=data_complete_df[data_complete_df['hypertension,stroke,heart disease']==1]
dataaa.shape
dataaa.head(10)
dataaa['age'].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt
plt.hist(dataaa['age'])
plt.hist(dataaa['BMI'])
data_complete_df['diabetes,hypertension,heart disease'].value_counts()
data_complete_df['diabetes,hypertension,stroke'].value_counts()
data_complete_df['diabetes & heart disease'].value_counts()
data_complete_df['diabetes & hypertension'].value_counts()
data_complete_df['diabetes & srtoke'].value_counts()
data_complete_df['hypertension & heart disease'].value_counts()
data_complete_df['hypertension & stroke'].value_counts()
data_complete_df['stroke & heart disease'].value_counts()
from sklearn.model_selection import train_test_split
X_stroke=data_complete_df.drop(['stroke'],axis=1)

X_stroke.head()
Y_stroke=data_complete_df['stroke']

Y_stroke.head()
x_train,x_test,y_train,y_test=train_test_split()