import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv('../input/us-police-shootings/shootings.csv')
df.head()
df.dtypes
df.shape
df.isnull().sum()
# 死亡の種別
print(df['manner_of_death'].unique())
print(df['manner_of_death'].value_counts())
# 武装の有無とその種別
print(df['armed'].unique())
print(df['armed'].value_counts())
# 武装のカテゴリ
print(df['arms_category'].value_counts())
# 性別と人種
print(df['gender'].value_counts())
print(df['race'].value_counts())
df['date'] = pd.to_datetime(df['date'])
df.dtypes
df['year'] = df['date'].dt.strftime('%Y')
df['year_month'] = df['date'].dt.strftime('%Y%m')
df.head()
# 年別
df['year'].value_counts()
# 年別の平均年齢
df.groupby(['year'])['age'].mean()
# 都市の数
df['city'].value_counts()
# 人種別×都市別
df.groupby(['race','city'])['id'].count()
# 逃げたか？　flee=逃げた
df['flee'].value_counts()
# 攻撃したか
df['threat_level'].value_counts()
df.groupby(['threat_level','arms_category'])['id'].count()
df['body_camera'].value_counts()
# 精神疾患
df['signs_of_mental_illness'].value_counts()
# 人種×性別
df.groupby(['race','gender'])['id'].count()
# 人種×性別×平均年齢
df.groupby(['race','gender'])['age'].mean()
# 人種×脅迫レベル
df.groupby(['race','threat_level'])['id'].count()
df.groupby(['race','arms_category'])['id'].count()
pd.set_option('display.max_rows', 500)
df.groupby(['state','race'])['id'].count()
df['state'].unique()
df['state'].replace(['PA','NY','VT','ME', 'NH', 'MA', 'RI', 'CT', 'NJ'],'E', inplace=True)
df['state'].replace(['DE','MD','DC','VA', 'WV', 'NC', 'SC', 'GA', 'FL','KY','TN','AL','MS', 'AR', 'LA', 'OK', 'TX'],'S', inplace=True)
df['state'].replace(['OH','MI','IN','WI', 'IL', 'MN', 'IA', 'MO', 'ND','SD','NE','KS'],'MW', inplace=True)
df['state'].replace(['WA','MT','OR','ID', 'WY', 'CA', 'NV', 'UT', 'CO','AZ','NM','AK', 'HI'],'W', inplace=True)
print(df['state'].unique())
print(df['state'].value_counts())
# 東部、南部、中西部、西部ごとの分析
df.groupby(['state','race'])['id'].count()
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
fig = px.histogram(df, "age", nbins=20, title = 'Age')
fig.show()
sns.distplot(df['age'],kde=True,color='r')
sns.boxplot(x="gender", y="age", palette=["b", "m"],data=df,)
sns.despine(offset=10, trim=True)
sns.boxplot(x="race", y="age",data=df)
sns.despine(offset=10, trim=True)
city = df.groupby('city')['name'].count().reset_index().sort_values('name', ascending=True).tail(20)
fig = px.bar(city, x="name", y="city", orientation='h',title="City")
fig.show()
