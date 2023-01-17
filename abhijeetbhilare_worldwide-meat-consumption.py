import numpy as np

import pandas as pd

import seaborn as sns

import plotly.graph_objs as go

import missingno as msno

from scipy import stats

sns.set(color_codes=True)

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("/kaggle/input/worldwide-meat-consumption/meat_consumption.csv")

print(df.shape)

df.head()
df.isna().any()
df.info()
df.describe()
df.describe(include="all")
df.LOCATION.unique()
df.SUBJECT.unique()
df.MEASURE.value_counts()
df.nunique()


dfKgCap = df.loc[(df['MEASURE'] == 'KG_CAP') & (df['TIME'] >= 1999) & (df['TIME'] <= 2020)]



print('Country that has the highest Consumption : ', dfKgCap.groupby(by = ['LOCATION']).Value.sum().idxmax())

print('Country that has the lowest Consumption : ', dfKgCap.groupby(by = ['LOCATION']).Value.sum().idxmin())



print('\nHigest Consumption of meat on : ', dfKgCap.groupby(by = ['TIME']).Value.sum().idxmax())

print('Lowest Consumption of meat on : ', dfKgCap.groupby(by = ['TIME']).Value.sum().idxmin())
df1 = pd.get_dummies(df,columns=['SUBJECT'])

df1.head()
sns.heatmap(df1.corr(),annot=True,cmap='coolwarm')
df2 = df.copy()

df2.shape
def remove_outliers(df):

    df_out = pd.DataFrame()

    for key, subdf in df.groupby('LOCATION'):

        m = np.mean(subdf.Value)

        st = np.std(subdf.Value)

        reduced_df = subdf[(subdf.Value>(m-st)) & (subdf.Value<=(m+st))]

        df_out = pd.concat([df_out,reduced_df],ignore_index=True)

    return df_out



df2 = remove_outliers(df2)

df2.shape
import plotly.express as px

df3 = df.groupby('LOCATION')[['Value']].sum().reset_index().sort_values('Value',ascending=False)

px.bar(df3,df3['LOCATION'],df3['Value'])
df3 = df.groupby('SUBJECT')[['Value']].sum().reset_index().sort_values('Value',ascending=False)

px.bar(df3,df3['SUBJECT'],df3['Value'])
dfIndia = df2.loc[(df['LOCATION'] == "IND"), ["SUBJECT", "Value"]]

dfIndia = dfIndia.groupby('SUBJECT')[['Value']].sum().reset_index().sort_values('Value',ascending=False)

px.bar(dfIndia,dfIndia['SUBJECT'],dfIndia['Value'])