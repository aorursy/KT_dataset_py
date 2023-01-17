import pandas as pd

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/sample-data/google_play_apps_data.csv")

df.head()
df.shape
df.columns
df.dtypes
df.isnull().sum()
df.info()
df.describe()
sliced_df = df.iloc[:,:-3]

sliced_df.head()
dropped_df = df.drop(["Last Updated","Current Ver","Android Ver"],axis=1)

dropped_df.head()
required_df = df[['Category','App','Installs','Rating','Reviews','Type']]
required_df['Installs'] = required_df['Installs'].str.replace('+','')

required_df['Installs'] = required_df['Installs'].str.replace(',','')
required_df.head()
df.nunique()
required_df.Category.value_counts()
required_df.groupby("Category").count().head()
required_df.groupby(['Category','Type']).count().head(15)
required_df.groupby("Category").first().head()
required_df['Type'].fillna('Free',inplace=True)
required_df['Rating'].fillna((required_df['Rating'].mean()), inplace=True)
required_df.dropna(inplace=True)
df = pd.get_dummies(required_df)
df.head()