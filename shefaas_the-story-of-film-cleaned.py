!pip install chart_studio

!pip install --upgrade pip
%matplotlib inline

from IPython.display import Image, HTML

import json

import datetime

import ast

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy import stats

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier, XGBRegressor

from wordcloud import WordCloud, STOPWORDS

import plotly

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

import chart_studio

chart_studio.tools.set_credentials_file(username='DemoAccount', api_key='lr1c37zw81')

import chart_studio.plotly as py

from chart_studio.plotly import plot, iplot

#plotly.tools.set_credentials_file(username='rounakbanik', api_key='xTLaHBy9MVv5szF4Pwan')



sns.set_style('whitegrid')

sns.set(font_scale=1.25)

pd.set_option('display.max_colwidth', 50)
df = pd.read_csv('../input/movies_metadata.csv')

df.head().transpose()
df.columns
df.shape
df.info()
# حذف العنوان الأصلي للفيلم والاعتماد على العنوان المترجم

df = df.drop('original_title', axis=1)
# التحقق من الأفلام التي تحمل إيرادات غير مسجلة (صفر)

df[df['revenue'] == 0].shape
# استبدال الأفلام ذات الإيرادات صفر إلى قيمة مفقودة

df['revenue'] = df['revenue'].replace(0, np.nan)
# تحويل الميزانية من قيمة object 

# إلى قيمة رقمية

# ثم استبدال كل قيمة صفرية غير مسجلة بقيمة مفقودة

df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

df['budget'] = df['budget'].replace(0, np.nan)

df[df['budget'].isnull()].shape
df['return'] = df['revenue'] / df['budget']
def clean_numeric(x):

    try:

        return float(x)

    except:

        return np.nan
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

df['year'] = df['year'].replace('NaT', np.nan)

df['year'] = df['year'].apply(clean_numeric)
df['title'] = df['title'].astype('str')

title_corpus = ' '.join(df['title'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)

plt.figure(figsize=(16,8))

plt.imshow(title_wordcloud)

plt.axis('off')

plt.show()
df_fran = df[df['belongs_to_collection'].notnull()]

df_fran['belongs_to_collection'] = df_fran['belongs_to_collection'].apply(ast.literal_eval).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)

df_fran = df_fran[df_fran['belongs_to_collection'].notnull()]
fran_pivot = df_fran.pivot_table(index='belongs_to_collection', values='revenue', aggfunc={'revenue': ['mean', 'sum', 'count']}).reset_index()
fran_pivot.sort_values('mean', ascending=False).head(10)
df['production_companies'] = df['production_companies'].fillna('[]').apply(ast.literal_eval)

df['production_companies'] = df['production_companies'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
s = df.apply(lambda x: pd.Series(x['production_companies']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'companies'
com_df = df.drop('production_companies', axis=1).join(s)
com_sum = pd.DataFrame(com_df.groupby('companies')['revenue'].sum().sort_values(ascending=False))

com_sum.columns = ['Total']

com_mean = pd.DataFrame(com_df.groupby('companies')['revenue'].mean().sort_values(ascending=False))

com_mean.columns = ['Average']

com_count = pd.DataFrame(com_df.groupby('companies')['revenue'].count().sort_values(ascending=False))

com_count.columns = ['Number']



com_pivot = pd.concat((com_sum, com_mean, com_count), axis=1)
com_pivot[com_pivot['Number'] >= 15].sort_values('Average', ascending=False).head(10)
df['original_language'].drop_duplicates().shape[0]
lang_df = pd.DataFrame(df['original_language'].value_counts())

lang_df['language'] = lang_df.index

lang_df.columns = ['number', 'language']

lang_df.head()
plt.figure(figsize=(12,5))

sns.barplot(x='language', y='number', data=lang_df.iloc[1:11])

plt.show()
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
def get_month(x):

    try:

        return month_order[int(str(x).split('-')[1]) - 1]

    except:

        return np.nan
df['month'] = df['release_date'].apply(get_month)
# انشاء جدول بيانات جديد يحتوي على الأفلام ذات متوسط الايرادات أعلى من مئة مليون دولار

# مجمعة حسب شهر اصدارها

month_mean = pd.DataFrame(df[df['revenue'] > 1e8].groupby('month')['revenue'].mean())



# تحديد عمود الشهر ليكون هو العمود الأساسي للترقيم

month_mean['mon'] = month_mean.index



# الرسم

plt.figure(figsize=(12,6))

plt.title("Average Gross by the Month for Blockbuster Movies")



# الشهر هو المحور السيني

# الايرادات هي المحور الصادي

sns.barplot(x='mon', y='revenue', data=month_mean, order=month_order)
df['budget'].describe()
df[df['budget'].notnull()][['title', 'budget', 'revenue', 'return', 'year']].sort_values('budget', ascending=False).head(10)
df['genres'] = df['genres'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'genre'
# إنشاء جدول بيانات جديد يحمل عمود genres.

gen_df = df.drop('genres', axis=1).join(s)
# لإيجاد إجمالي عدد التصنيفات لدينا

gen_df['genre'].value_counts().shape[0]
# إنشاء جدول بيانات جديد يحمل عدد الأفلام في كل تصنيف

pop_gen = pd.DataFrame(gen_df['genre'].value_counts()).reset_index()

pop_gen.columns = ['genre', 'movies']

pop_gen.head(10)
plt.figure(figsize=(18,8))

sns.barplot(x='genre', y='movies', data=pop_gen.head(15))

plt.show()
genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Adventure', 'Science Fiction', 'Mystery', 'Fantasy', 'Mystery', 'Animation']
# لجلب الأفلام في العقد العشرين فقط

pop_gen_movies = gen_df[(gen_df['genre'].isin(genres)) & (gen_df['year'] >= 2000) & (gen_df['year'] <= 2017)]

# a cross-tabulation table that can show the frequency of each genrein a cerain year

ctab = pd.crosstab([pop_gen_movies['year']], pop_gen_movies['genre']).apply(lambda x: x/x.sum(), axis=1)

ctab[genres].plot(kind='bar', stacked=True, colormap='jet', figsize=(12,8)).legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.title("Stacked Bar Chart of Movie Proportions by Genre")

plt.show()
violin_genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Science Fiction', 'Fantasy', 'Animation']

violin_movies = gen_df[(gen_df['genre'].isin(violin_genres))]
plt.figure(figsize=(18,8))

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))

# لرسم بوكس بلوت يمثل التصنيف فيه المحور السيني والإيرادات المحور الصادي

sns.boxplot(x='genre', y='revenue', data=violin_movies, palette="muted", ax =ax)

# لتحديد مقياس المحور الصادي (من صفر إلى ٣٠٠ مليون دولار)

ax.set_ylim([0, 3e8])

plt.show()
credits_df = pd.read_csv('../input/credits.csv')

credits_df.head()
def convert_int(x):

    try:

        return int(x)

    except:

        return np.nan
df['id'] = df['id'].apply(convert_int)
df[df['id'].isnull()]
df = df.drop([19730, 29503, 35587])
df['id'] = df['id'].astype('int')
df = df.merge(credits_df, on='id')

df.shape
df['cast'] = df['cast'].apply(ast.literal_eval)

df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
s = df.apply(lambda x: pd.Series(x['cast']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'actor'

cast_df = df.drop('cast', axis=1).join(s)
sns.set_style('whitegrid')

plt.title('Actors with the Highest Total Revenue')

cast_df.groupby('actor')['revenue'].sum().sort_values(ascending=False).head(10).plot(kind='bar')

plt.show()