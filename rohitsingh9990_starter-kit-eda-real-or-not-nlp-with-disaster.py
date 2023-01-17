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
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
print('Size of train data', train.shape)

print('Size of test data', test.shape)

print('Size of sample submission', sample.shape)

train.head()

train.columns

train.info()
test.head()

test.columns

test.info()
sample.head()
# util func to check null values in a dataframe



def report_nulls(df):

    '''

    Show a fast report of the DF.

    '''

    rows = df.shape[0]

    columns = df.shape[1]

    null_cols = 0

    list_of_nulls_cols = []

    for col in list(df.columns):

        null_values_rows = df[col].isnull().sum()

        null_rows_pcn = round(((null_values_rows)/rows)*100, 2)

        col_type = df[col].dtype

        if null_values_rows > 0:

            print("The column {} has {} null values. It is {}% of total rows.".format(col, null_values_rows, null_rows_pcn))

            print("The column {} is of type {}.\n".format(col, col_type))

            null_cols += 1

            list_of_nulls_cols.append(col)

    null_cols_pcn = round((null_cols/columns)*100, 2)

    print("The DataFrame has {} columns with null values. It is {}% of total columns.".format(null_cols, null_cols_pcn))

    return list_of_nulls_cols
report_nulls(train)
report_nulls(test)
import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec 

from gensim.models import KeyedVectors 

import pickle

from tqdm import tqdm

import os

import matplotlib.pyplot as plt

import seaborn as sns



from collections import Counter

import warnings

warnings.simplefilter("ignore")
# https://matplotlib.org/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py



y_value_counts = train['target'].value_counts()

print("Number of tweet are about real disaster", y_value_counts[1], ", (", (y_value_counts[1]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")

print("Number of tweet are not about real disaster", y_value_counts[0], ", (", (y_value_counts[0]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")



fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))

recipe = ["Real Disaster", "Not Real Disaster"]



data = [y_value_counts[1], y_value_counts[0]]



wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)



bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),

          bbox=bbox_props, zorder=0, va="center")



for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})

    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),

                 horizontalalignment=horizontalalignment, **kw)



ax.set_title("Nmber of tweets that are about Real Disaster or not")



plt.show()

train.keyword.unique()
#stacked bar plots matplotlib: https://matplotlib.org/gallery/lines_bars_and_markers/bar_stacked.html

def stack_plot(data, xtick, col2='target', col3='total'):

    ind = np.arange(data.shape[0])

    

    plt.figure(figsize=(20,5))

    p1 = plt.bar(ind, data[col3].values)

    p2 = plt.bar(ind, data[col2].values)



    plt.ylabel('Projects')

    plt.title('Number of real disaster tweets vs fake disaster tweets')

    plt.xticks(ind, list(data[xtick].values), rotation='vertical')

    

    plt.legend((p1[0], p2[0]), ('total tweets', 'Real Disaster tweets'))

    plt.show()





def univariate_barplots(data, col1, col2='target', top=False):

    # Count number of zeros in dataframe python: https://stackoverflow.com/a/51540521/4084039

    temp = pd.DataFrame(data.groupby(col1)[col2].agg(lambda x: x.eq(1).sum())).reset_index()

    

    # Pandas dataframe grouby count: https://stackoverflow.com/a/19385591/4084039

    temp['total'] = pd.DataFrame(data.groupby(col1)[col2].agg({'total':'count'})).reset_index()['total']

    temp['Avg'] = pd.DataFrame(data.groupby(col1)[col2].agg({'Avg':'mean'})).reset_index()['Avg']

    

    temp.sort_values(by=['total'],inplace=True, ascending=False)

    

    if top:

        temp = temp[0:top]

        

    stack_plot(temp, xtick=col1, col2=col2, col3='total')
univariate_barplots(train, 'keyword', top=50)
train.location.unique()[:100]
#How to calculate number of words in a string in DataFrame: https://stackoverflow.com/a/37483537/4084039

word_count = train['text'].str.split().apply(len).value_counts()

word_dict = dict(word_count)

word_dict = dict(sorted(word_dict.items(), key=lambda kv: kv[1]))





ind = np.arange(len(word_dict))

plt.figure(figsize=(20,5))

p1 = plt.bar(ind, list(word_dict.values()))



plt.ylabel('Numeber of Tweets')

plt.xlabel('Number words in tweet text')

# plt.title('Words for each  of the project')

plt.xticks(ind, list(word_dict.keys()))

plt.show()
real_dis_tweet_word_count = train[train['target']==1]['text'].str.split().apply(len)

real_dis_tweet_word_count = real_dis_tweet_word_count.values



fake_dis_tweet_word_count = train[train['target']==0]['text'].str.split().apply(len)

fake_dis_tweet_word_count = fake_dis_tweet_word_count.values
# https://glowingpython.blogspot.com/2012/09/boxplot-with-matplotlib.html

plt.boxplot([real_dis_tweet_word_count, fake_dis_tweet_word_count])

plt.xticks([1,2],('Real Disaster Tweets','Fake Disaster Tweets'))

plt.ylabel('Words in Tweet')

plt.grid()

plt.show()
plt.figure(figsize=(10,3))

sns.kdeplot(real_dis_tweet_word_count,label="Real Disaster Tweets", bw=0.6)

sns.kdeplot(fake_dis_tweet_word_count,label="Fake Disaster Tweets", bw=0.6)

plt.legend()

plt.show()