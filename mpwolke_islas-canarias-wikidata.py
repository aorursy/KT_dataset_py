# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

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
df = pd.read_csv('../input/corpus-bics-canarias-wikidata-commons-eswiki/corpus_bics_wikidata_commons_eswiki.csv', encoding='ISO-8859-2')

df.head()
df.isnull().sum()
#Code from Gabriel Preda



def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(2*size,2))

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
plot_count("wikidata_etiquetas", "Wikidata Etiquetas", df,4)
plot_count("wikidata_declaraciones", "Wikidata Declaraciones", df,4)
import matplotlib.gridspec as gridspec

from scipy.stats import skew

from sklearn.preprocessing import RobustScaler,MinMaxScaler

from scipy import stats

import matplotlib.style as style

style.use('seaborn-colorblind')
def plotting_3_chart(df, feature): 

    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(10,6))

    ## crea,ting a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    #gs = fig3.add_gridspec(3, 3)



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

 



print('Skewness: '+ str(df['wikidata_declaraciones'].skew())) 

print("Kurtosis: " + str(df['wikidata_declaraciones'].kurt()))

plotting_3_chart(df, 'wikidata_declaraciones')
stats.probplot(df['wikidata_etiquetas'].values, dist="norm", plot=plt)

plt.show()
stats.probplot(df['wikidata_descripciones'].values, dist="norm", plot=plt)

plt.show()
stats.probplot(df['wikidata_declaraciones_referencias_P143'].values, dist="norm", plot=plt)

plt.show()
stats.probplot(df['wikidata_declaraciones_referencias'].values, dist="norm", plot=plt)

plt.show()
stats.probplot(df['wikidata_identificadores_externos'].values, dist="norm", plot=plt)

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)



ax1.plot(df.discusion_tamanho_bytes, df.discusion_tamanho_palabras, c = 'green')

ax1.set_title('Discussion Bytes/Words', c = 'green')

ax2.scatter(df.tamano_bytes, df.tamano_palabras, c='red')

ax2.set_title('Bytes/Words Size', c ='red')



plt.ylabel('Bytes Size', fontsize = 20)



plt.show()
sns.boxplot(x=df['commons_archivos'], color = 'cyan')

plt.title('Commons Archives', fontsize = 20)

plt.show()
sns.boxplot(x=df['commons_subcats'], color = 'cyan')

plt.title('Commons Subcategories', fontsize = 20)

plt.show()
sns.boxplot(x=df['editores_anonimos'], color = 'magenta')

plt.title('Anonymous Editors', fontsize = 20)

plt.show()
sns.boxplot(x=df['editores_registrados'], color = 'cyan')

plt.title('Registered Editors', fontsize = 20)

plt.show()
#Code from Mario Filho

from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['commons_categoria']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['wikidata_interwiki']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
ax = df.groupby('wikidata_interwiki')['commons_archivos'].mean().plot(kind='barh', figsize=(12,8),

                                                           title='Mean Wikidata Interwiki')

plt.xlabel('Mean Wikidata')

plt.ylabel('Commons Archivos')

plt.show()
ax = df.groupby('wikidata_interwiki')['wikidata_etiquetas', 'wikidata_declaraciones'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Wikidata')

plt.xlabel('Wikidata')

plt.ylabel('Log')



plt.show()
ax = df.groupby('wikidata_interwiki')['wikidata_descripciones', 'wikidata_declaraciones_referencias_P143'].sum().plot(kind='barh', figsize=(14,8),

                                                                 title='Wikidata', logx=True, linewidth=3)

plt.xlabel('Log')

plt.ylabel('Wikidata')

plt.show()
fig=sns.lmplot(x='wikidata_interwiki', y="wikidata_identificadores_externos",data=df)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.articulo)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
fig = px.parallel_categories(df, color="wikidata_interwiki", color_continuous_scale=px.colors.sequential.Viridis)

fig.show()
fig = px.pie(df, values=df['wikidata_interwiki'], names=df['wikidata_id'],

             title='Wikidata Id',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = go.Figure(data=[go.Bar(

            x=df['tamano_bytes'][0:10], y=df['tamano_palabras'][0:10],

            text=df['tamano_palabras'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Wikidata',

    xaxis_title="Bytes Size",

    yaxis_title="Words Size",

)

fig.show()