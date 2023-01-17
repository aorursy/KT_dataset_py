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
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

import seaborn as sns
df_orig = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

df = df_orig.copy()

df.head()
df.count()
df.isna().sum()
df = df.dropna()

df.count()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='white',

                          stopwords=stopwords,

                          max_words=2000,

                          max_font_size=50, collocations=False).generate(' '.join(df.Category.values))

fig = plt.figure(figsize=(15,5))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
#rcParams['figure.figsize'] = 11.7,8.27

g = sns.kdeplot(df.Rating)

g.set_xlabel("App Rating")

g.set_ylabel("Frequency")

plt.title('Distribution of App Ratings')

plt.show()
df['Reviews'] = df['Reviews'].astype(int)

plt.scatter(x=np.arange(0,9360), y = np.log10(df['Reviews'].sort_values()))

plt.ylabel('log 10 of amount of reviews')

plt.title('Distribution of Review Counts log 10')

plt.show()
df['Size'].value_counts()
df['new_Size'] = df['Size'].apply(lambda x: float(x.replace('M', ''))*1000000 if 'M' in x else float(x.replace('k', ''))*1000 if 'k' in x else x)

df['new_Size'] = df['new_Size'][df.new_Size != 'Varies with device']

med = df['new_Size'].median()

df['Size'] = df['Size'].apply(lambda x: float(x.replace('M', ''))*1000000 if 'M' in x else float(x.replace('k', ''))*1000 if 'k' in x else med)

df['Size'] = df['Size'].astype(int)

df = df.drop(['new_Size'], axis = 1)

df['Size']
plt.boxplot(np.log10(df['Size']))

plt.show()
df['Installs'] = df['Installs'].apply(lambda x: x[:-1] if x[-1] == '+' else x).apply(lambda y: int(y.replace(',', '')))

df['Installs']
np.log10(df['Installs']).plot(kind='hist')

plt.title('Histogram of Amount of Installs Log10')

plt.xlabel('Amount of Installs Log10')

plt.show()
plt.pie(df['Type'].value_counts(), autopct='%1.1f%%', labels = df['Type'].value_counts().index)

plt.show()
df['Type'] = df['Type'].apply(lambda x: 1 if x == 'Paid' else 0)

df['Type'].value_counts()
df['Price'] = df['Price'].apply(lambda x: float(x.replace('$', '')))
df[df.Price > 0].Price.describe()
df['Content Rating'].value_counts()
fig1, ax1 = plt.subplots(figsize=(7,7))

ax1.pie(df['Content Rating'].value_counts(), autopct='%1.1f%%', labels = df['Content Rating'].value_counts().index)

plt.legend()

plt.show()
dict = {'Everyone': 0, 'Everyone 10+': 1, 'Teen':2, 'Mature 17+':3, 'Adults Only 18+':4, 'Unrated':5}

df['Content Rating'] = df['Content Rating'].apply(lambda x: dict.get(x))

df = df.dropna()

df['Content Rating'].value_counts()
text = ';'.join(df.Genres.values).replace(' ', '_').replace('&', 'and').split(';')

text = ' '.join(text)

stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='white',

                          stopwords=stopwords,

                          max_words=2000,

                          max_font_size=50, collocations=False).generate(text)

fig = plt.figure(figsize=(15,5))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
df['Last Updated'] = pd.to_datetime(df['Last Updated'])

df['Last Updated'].dt.year.value_counts().plot(kind='bar')

plt.show()
df.isna().sum()
plt.figure(figsize = (7,7))

sns.regplot(x="Size", y="Rating",data=df, line_kws = {'color':'red'})

plt.title('Rating VS Size',size = 20)

plt.show()
plt.figure(figsize = (7,7))

sns.regplot(x="Reviews", y="Rating",data=df, line_kws = {'color':'red'}, order=2)

plt.title('Rating VS Reviews',size = 20)

plt.show()
plt.figure(figsize = (7,7))

sns.regplot(x="Installs", y="Rating",data=df, line_kws = {'color':'red'}, order = 2)

plt.title('Rating VS Installs',size = 20)

plt.show()
df['Year'] = df['Last Updated'].dt.year

df['Month'] = df['Last Updated'].dt.month

df = df.drop(['Last Updated'], axis = 1)
plot = sns.catplot(x="Year",y="Rating",data=df, kind="box")

plt.title('Rating VS Year of Last Update Boxplot')

plt.show()
df['Main Genre'] = df['Genres'].apply(lambda x: x.split(';')[0])

df['Secondary Genre'] = df['Genres'].apply(lambda x: x.split(';')[-1] if ';' in x else 0)

df = df.drop(['Genres'], axis = 1)
numeric = df[['Reviews', 'Size', 'Installs', 'Type', 'Price', 'Year', 'Month', 'Content Rating']]

categorical = df[['Category', 'Main Genre', 'Secondary Genre']]
for cat in categorical:

    dummy_cr = pd.get_dummies(df[cat])

    df = pd.concat([df, dummy_cr], axis=1)

    df = df.drop([cat], axis = 1)
df = df.drop(['Current Ver', 'Android Ver'], axis = 1)

df = df.reset_index(drop=True)
X = df.drop(['App', 'Rating'], axis = 1)

y = df['Rating'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



sc = StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)



reg = LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))