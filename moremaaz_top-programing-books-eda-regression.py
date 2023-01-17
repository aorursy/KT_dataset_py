# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS



sns.set(rc= {'figure.figsize': (12,10)})

plt.style.use('ggplot')



from plotly.offline import init_notebook_mode, iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')

import plotly.graph_objs as go

import plotly

import plotly.express as px

import plotly.figure_factory as ff
df = pd.read_csv('/kaggle/input/top-270-rated-computer-science-programing-books/prog_book.csv')

df2 = df.copy()
df.info()
df.isna().sum()
df.head()
df.columns = [i.lower() for i in df.columns]
df['reviews'] = df.reviews.apply(lambda x: x.replace(',', '') if ',' in x else x)

df['reviews'] = pd.to_numeric(df.reviews)
df.describe()
def avg_book(x):

    count = 0

    words = x.split(' ')

    for i in words:

        count += len(i)

    return count / len(words)



def avg_description(x):

    count = 0

    words = x.split(' ')

    for i in words:

        count += len(i)

    return count / len(words)
df['avg_title'] = df['book_title'].apply(avg_book)

df['avg_desc'] = df['description'].apply(avg_description)
df['len_title'] = [len(i) for i in df['book_title']]

df['len_desc'] = [len(i) for i in df['description']]
def histo(cols):

    for i, x in enumerate(cols):

        plt.figure(figsize=(9,5))

        plt.figure(i)

        sns.distplot(df[x])

        print(f'Mean of {x} is: {round(df[x].mean(), 2)}')

        print(f'Skew is: {round(df[x].skew(), 2)}')

        print(f'Skew is: {round(df[x].kurtosis(), 2)}')

        print('****' * 10)
histo(df.select_dtypes([np.int, np.float]).drop(['len_title', 'len_desc'], axis= 1))
cor = df.corr()



mask = np.zeros_like(cor, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.set_style('whitegrid')

plt.subplots(figsize = (15,12))

sns.heatmap(cor, 

            annot=True,

            mask = mask,

            cmap = 'RdBu_r', ## in order to reverse the bar replace "RdBu" with "RdBu_r"

            linewidths=.9, 

            linecolor='white',

            fmt='.2g',

            center = 0,

            square=True)

plt.title("Correlations Among Numeric Dtypes", y = 1.03,fontsize = 20, pad = 40)
cor['price']
fig, axes = plt.subplots(2, 2, figsize=(10, 7))



sns.scatterplot(data= df, x='price', y='number_of_pages', ax= axes[0,0], color= 'blue')

sns.scatterplot(data= df, x='price', y='reviews', ax= axes[0,1], color= 'orange')

sns.scatterplot(data= df, x='price', y='avg_title', ax= axes[1,0], color= 'green')

sns.scatterplot(data= df, x='price', y='avg_desc', ax= axes[1,1], color= 'purple')
def box(cols):

    for i, x in enumerate(cols):

        plt.figure(figsize=(9,5))

        plt.figure(i)

        sns.boxplot(df[x])
box(df.select_dtypes([np.int, np.float]))
pie = df.type.value_counts()



pie_df = pd.DataFrame({'index':pie.index, 'values': pie.values})

pie_df.iplot(kind='pie', labels= 'index', values= 'values', hole= .5)
dums = pd.get_dummies(df['type'])
df = df.merge(dums, left_index= True, right_index= True)

df.drop('type', axis= 1, inplace= True)
def get_text(column):

    words = ''

    for text in column:

        words += text

    return words
text1 = get_text(df['book_title'])



stopwords = set(STOPWORDS)

wc = WordCloud(background_color= 'black', stopwords= stopwords,

              width=1600, height=800)



wc.generate(text1)

plt.figure(figsize=(20,10), facecolor='k')

plt.axis('off')

plt.tight_layout(pad=0)

plt.imshow(wc)

plt.show()
text2 = get_text(df['description'])



stopwords = set(STOPWORDS)

wc = WordCloud(background_color= 'black', stopwords= stopwords,

              width=1600, height=800)



wc.generate(text1)

plt.figure(figsize=(20,10), facecolor='k')

plt.axis('off')

plt.tight_layout(pad=0)

plt.imshow(wc)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score
X = df.drop(['book_title', 'description', 'price'], axis= 1)

y = np.log2(df.price)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .2, random_state= 42)
rfm = RandomForestRegressor(n_estimators = 50, random_state = 42)



rfm.fit(X_train, y_train)

y_pred= rfm.predict(X_test)
errors = abs(y_pred - y_test)

print('Mean Absolute Error:', round(np.mean(errors), 2))



# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
scores = cross_val_score(rfm, X, y, cv=5, scoring='neg_mean_squared_error')

print(f'Cross validated random forest MAE: {abs(round(np.mean(scores), 2))}')
feat_importances = pd.Series(rfm.feature_importances_, index=df.drop(['book_title', 'description', 'price'], axis=1).columns).sort_values(ascending= False)

feat_importances.iplot(kind='bar', labels= 'index', values= 'values')
X2 = df[['number_of_pages', 'rating', 'reviews']]

y2 = np.log2(df['price'])



X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size= .2, random_state= 42)
sc = StandardScaler()



X2_train = sc.fit_transform(X2_train)

X2_test = sc.transform(X2_test)
lm = LinearRegression()



lm.fit(X2_train, y2_train)

y2_pred = lm.predict(X2_test)



errors2 = abs(y2_pred - y2_test)

print('Mean Absolute Error:', round(np.mean(errors2), 2))



mape2 = 100 * (errors2 / y2_test)

# Calculate and display accuracy

accuracy2 = 100 - np.mean(mape2)

print('Accuracy:', round(accuracy2, 2), '%.')