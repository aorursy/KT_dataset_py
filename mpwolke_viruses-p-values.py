# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/ai4all-project/results/viral_calls/viruses_with_pval.csv')

df.head()
print(f"data shape: {df.shape}")
df.describe()
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
missing_data(df)
df.isnull().sum()
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    return(np.transpose(tt))
unique_values(df)
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count("name", "name", df,4)
#Correlation map to see how features are correlated with each other and with target

corrmat = df.corr(method='kendall')

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
#https://www.kaggle.com/ilyabiro/visual-analysis-or-how-to-find-out-who-escaped-1st/notebook

fig, axarr = plt.subplots(1, 2, figsize=(14,8))

axarr[0].set_title('P-Values distribution')

f = sns.distplot(df['p_val'], color='g', bins=15, ax=axarr[0])

axarr[1].set_title('P-values distribution for the two subpopulations')

g = sns.kdeplot(df['p_val'].loc[df['genus_nt_count'] == 1], 

                shade= True, ax=axarr[1], label='genus_nt_count').set_xlabel('p_val')

g = sns.kdeplot(df['p_val'].loc[df['genus_nt_count'] == 0], 

                shade=True, ax=axarr[1], label='genus_nt_count')
##https://www.kaggle.com/ilyabiro/visual-analysis-or-how-to-find-out-who-escaped-1st/notebook

fig, ax = plt.subplots(figsize=(10, 8))

ax.set_title('P Values distribution')

g = sns.kdeplot(df['p_val'].loc[df['nr_e_value'] == 1], 

                shade= True, ax=ax, label='nr_e_value').set_xlabel('p_val')

g = sns.kdeplot(df['p_val'].loc[df['nr_e_value'] == 0], 

                shade=True, ax=ax, label='nr_e_valuet')

ax.grid()
covid = df[(df['name']=='Wuhan seafood market pneumonia virus')].reset_index(drop=True)

covid.head()
fig = px.parallel_categories(covid, color="p_val", color_continuous_scale=px.colors.sequential.OrRd)

fig.show()
fig = px.line(covid, x="p_val", y="genus_nt_count", color_discrete_sequence=['darksalmon'], 

              title="Covid-19 P Values")

fig.show()
fig = px.bar(covid, 

             x='name', y='p_val',color_discrete_sequence=['blue'],

             title='Wuhan Pneumonia Virus P-Values', text='expected_bg_count')

fig.show()
fig = px.bar(covid, 

             x='name', y='nt_count', color_discrete_sequence=['crimson'],

             title='Wuhan Pneumonia Virus', text='p_val')

fig.show()
fig = px.bar(covid,

             y='name',

             x='p_val',

             orientation='h',

             color='genus_nt_count',

             title='Wuhan Pneumonia Virus P-Values',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Temps,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.line(covid, x="p_val", y="expected_bg_count", color_discrete_sequence=['darksalmon'], 

              title="P-Values vs expected_bg_count")

fig.show()
fig = px.scatter(covid, x="name", y="p_val",color_discrete_sequence=['#4257f5'], title="Wuhan Pneumonia Virus P-Values" )

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in covid.name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set2', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()