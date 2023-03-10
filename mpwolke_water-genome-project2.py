#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/buildingdatagenomeproject2/solar_cleaned.csv')

df.head()
df.isna().sum()
import missingno as msno

# checking null values

msno.bar(df, figsize=(16, 4))
%%time

a = msno.heatmap(df, sort='ascending')

a
%%time

a2 = msno.dendrogram(df)

a2
corrs = df.corr()

corrs
plt.figure(figsize = (20, 8))



# Heatmap of correlations

sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap');
def plot_dist_col(column):

    pos__df = df[df['Bobcat_education_Dylan'] ==1]

    neg__df = df[df['Bobcat_education_Dylan'] ==0]



    '''plot dist curves for train and test weather data for the given column name'''

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.distplot(pos__df[column].dropna(), color='green', ax=ax).set_title(column, fontsize=16)

    sns.distplot(neg__df[column].dropna(), color='purple', ax=ax).set_title(column, fontsize=16)

    plt.xlabel(column, fontsize=15)

    plt.legend(['Bobcat_education_Dylan', 'Bobcat_other_Timothy'])

    plt.show()

plot_dist_col('Bobcat_other_Timothy')
ts=df.groupby(["timestamp"])["Bobcat_education_Dylan"].sum()
import statsmodels.api as sm

# Additive model

res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")

fig = res.plot()
from sklearn.preprocessing import LabelEncoder, StandardScaler

#fill in mean for floats

for c in df.columns:

    if df[c].dtype=='float16' or  df[c].dtype=='float32' or  df[c].dtype=='float64':

        df[c].fillna(df[c].mean())



#fill in -999 for categoricals

df = df.fillna(-999)

# Label Encoding

for f in df.columns:

    if df[f].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(df[f].values))

        df[f] = lbl.transform(list(df[f].values))

        

print('Labelling done.')
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
%%time

missing_data(df)
# Numerical features

Numerical_feat = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('Total numerical features: ', len(Numerical_feat))

print('\nNumerical Features: ', Numerical_feat)
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
for col in ('Bobcat_education_Dylan', 'Bobcat_education_Alissa', 'Bobcat_education_Coleman', 'Bobcat_other_Timothy', 'Bobcat_office_Justine'):

    df[col] = df[col].fillna(0)

    

#for col in ['BCG Strain' ,'Location of Administration of BCG Vaccine', 'BCG Supply Company', 'Additional Comments']:

   # df[col] = df[col].fillna('None')
from scipy.stats import norm, skew

num_features = df.dtypes[df.dtypes != 'object'].index

skewed_features = df[num_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_features})

skewness.head(15)
numerical_df = df.select_dtypes(exclude='object')



for i in range(len(numerical_df.columns)):

    f, ax = plt.subplots(figsize=(7, 4))

    fig = sns.distplot(numerical_df.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(numerical_df.columns[i])
numerical_df = df.select_dtypes(exclude='object')



for i in range(len(numerical_df.columns)):

    f, ax = plt.subplots(figsize=(7, 4))

    fig = sns.distplot(numerical_df.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(numerical_df.columns[i])