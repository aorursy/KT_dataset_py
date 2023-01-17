%matplotlib inline

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

plt.style.use('ggplot')



import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

sns.set(rc={'figure.figsize':(25,15)})



import plotly

# connected=True means it will download the latest version of plotly javascript library.

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import plotly.figure_factory as ff

import cufflinks as cf



import warnings

warnings.filterwarnings('ignore')



import pylab as pl
import pandas as pd

df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

reviews_df = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")
df.drop_duplicates(subset='App', inplace=True)

df = df[df['Android Ver'] != np.nan]

df = df[df['Android Ver'] != 'NaN']

df = df[df['Installs'] != 'Free']

df = df[df['Installs'] != 'Paid']
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)

df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)

df['Installs'] = df['Installs'].apply(lambda x: int(x))
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)



df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)

df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)

df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)





df['Size'] = df['Size'].apply(lambda x: float(x))

df['Installs'] = df['Installs'].apply(lambda x: float(x))



df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))

df['Price'] = df['Price'].apply(lambda x: float(x))



df['Reviews'] = df['Reviews'].apply(lambda x: int(x))
x = df['Rating'].dropna()

y = df['Size'].dropna()

z = df['Installs'][df.Installs!=0].dropna()

p = df['Reviews'][df.Reviews!=0].dropna()

t = df['Type'].dropna()

price = df['Price']



p = sns.pairplot(pd.DataFrame(list(zip(x, y, np.log(z), np.log10(p), t, price)), 

                        columns=['Rating','Size', 'Installs', 'Reviews', 'Type', 'Price']), hue='Type', palette="Set2")



p.fig.suptitle("Analisis Eksplorasi Data", y=1.025)
import scipy.stats as stats

f = stats.f_oneway(df.loc[df.Category == 'BUSINESS']['Rating'].dropna(), 

               df.loc[df.Category == 'FAMILY']['Rating'].dropna(),

               df.loc[df.Category == 'GAME']['Rating'].dropna(),

               df.loc[df.Category == 'PERSONALIZATION']['Rating'].dropna(),

               df.loc[df.Category == 'LIFESTYLE']['Rating'].dropna(),

               df.loc[df.Category == 'FINANCE']['Rating'].dropna(),

               df.loc[df.Category == 'EDUCATION']['Rating'].dropna(),

               df.loc[df.Category == 'MEDICAL']['Rating'].dropna(),

               df.loc[df.Category == 'TOOLS']['Rating'].dropna(),

               df.loc[df.Category == 'PRODUCTIVITY']['Rating'].dropna()

              )



groups = df.groupby('Category').filter(lambda x: len(x) > 286).reset_index()

array = groups['Rating'].hist(by=groups['Category'], sharex=True, figsize=(20,20))





pl.suptitle("Rata-rata Nilai Rating Pada Setiap Kategori", y=0.935)
merged_df = pd.merge(df, reviews_df, on = "App", how = "inner")

merged_df = merged_df.dropna(subset=['Sentiment', 'Translated_Review'])
number_of_apps_in_category = df['Category'].value_counts().sort_values(ascending=True)



data = [go.Pie(

        labels = number_of_apps_in_category.index,

        values = number_of_apps_in_category.values,

        hoverinfo = 'label+value',

        title='Jumlah Download Aplikasi Pada Masing-Masing Kategori'

    

)]



plotly.offline.iplot(data, filename='active_category')
corrmat = df.corr()

f, ax = plt.subplots()

p =sns.heatmap(corrmat, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

title = ax.set_title('Heatmap korelasi antar atribut')
subset_df = df[df.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY', 'MEDICAL', 'TOOLS', 'FINANCE',

                                 'LIFESTYLE','BUSINESS'])]

sns.set_style('whitegrid')

fig, ax = plt.subplots()

fig.set_size_inches(15, 8)

p = sns.stripplot(x="Price", y="Category", data=subset_df, jitter=True, linewidth=1)

title = ax.set_title('Tren Harga Aplikasi pada Masing-masing Kategori')
sns.set_style('ticks')

sns.set_style("whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

ax = sns.boxplot(x='Type', y='Sentiment_Polarity', data=merged_df)

title = ax.set_title('Distribusi Polaritas Sentiment')
from wordcloud import WordCloud

wc = WordCloud(background_color="white", max_words=200, colormap="Set2")

# generate word cloud



from nltk.corpus import stopwords

stop = stopwords.words('english')

stop = stop + ['app', 'APP' ,'ap', 'App', 'apps', 'application', 'browser', 'website', 'websites', 'chrome', 'click', 'web', 'ip', 'address',

            'files', 'android', 'browse', 'service', 'use', 'one', 'download', 'email', 'Launcher']



#merged_df = merged_df.dropna(subset=['Translated_Review'])

merged_df['Translated_Review'] = merged_df['Translated_Review'].apply(lambda x: " ".join(x for x in str(x).split(' ') if x not in stop))

#print(any(merged_df.Translated_Review.isna()))

merged_df.Translated_Review = merged_df.Translated_Review.apply(lambda x: x if 'app' not in x.split(' ') else np.nan)

merged_df.dropna(subset=['Translated_Review'], inplace=True)





free = merged_df.loc[merged_df.Type=='Free']['Translated_Review'].apply(lambda x: '' if x=='nan' else x)

wc.generate(''.join(str(free)))

plt.figure(figsize=(10, 10))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.title("Wordcloud Mengenai Review Pengguna Terhadap Aplikasi", y=1.05)

plt.show()


