# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv('../input/us-accidents/US_Accidents_May19.csv')
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
df.head()
df.describe()
listItem = []

for col in df.columns :

    listItem.append([col, df[col].dtype, df[col].isna().sum(), round((df[col].isna().sum()/len(df[col])) * 100,2),

                    df[col].nunique(), list(df[col].unique()[:2])]);



dfDesc = pd.DataFrame(columns=['dataFeatures', 'dataType', 'null', 'nullPct', 'unique', 'uniqueSample'],

                     data=listItem)

dfDesc
df[df['TMC'].isnull()]['Source'].value_counts()
len(df[df['Source'] == 'Bing'])
df['TMC'] = df['TMC'].fillna('Unknown')
df.drop(['End_Lat','End_Lng'],axis=1,inplace=True)
df['Description'] = df['Description'].fillna('Unknown')
df.drop('Number',axis=1,inplace=True)
df.dropna(subset=['City'],inplace=True)
def remove_column_or_row(name_col):

    null = df[name_col].isnull().sum()

    perc = null/len(df) * 100

    if(perc > 50):

        df.drop(name_col,axis=1,inplace=True)

    else:

        df.dropna(subset=[name_col],inplace=True)
for item in df.columns:

    remove_column_or_row(item)
df.isnull().sum()
df.columns
df.drop('ID' , inplace=True,axis=1)
diff = pd.to_datetime(df['End_Time']) - pd.to_datetime(df['Start_Time'])

df['Minutes'] = diff.dt.total_seconds().div(60).astype(int)
df.columns
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

sns.countplot(df['Severity'],ax=ax[0])

sns.countplot(df[(df['Severity'] == 0) | (df['Severity']==1)]['Severity'],ax=ax[1])
df['Severity'].value_counts(normalize=True)
import plotly.graph_objects as go



labels = df['State'].value_counts().head(10).index

values = df['State'].value_counts().head(10).values



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
# import plotly.graph_objects as go



labels = df['TMC'].value_counts().head(10).index

values = df['TMC'].value_counts().head(10).values



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
df.sort_values(by='Minutes',ascending=False).head(10)[['Severity','County','Description','Distance(mi)','Minutes']]
sns.distplot(df['Minutes'])

plt.tight_layout()
df.sort_values(by='Distance(mi)',ascending=False).head(10)[['Severity','County','Description','Distance(mi)','Minutes']]
sns.distplot(df['Distance(mi)'])

plt.tight_layout()
import plotly.express as px

fig = px.scatter(x=df['Minutes'], y=df['Distance(mi)'])

fig.show()
df.groupby('State').mean()[['Distance(mi)','Minutes']].sort_values(by='Distance(mi)',ascending=False).head(10)
df.groupby('State').mean()[['Distance(mi)','Minutes']].sort_values(by='Minutes',ascending=False).head(10)