# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.stem.snowball import SnowballStemmer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
stemmer = SnowballStemmer('english')
df_train = pd.read_csv('../input/train.csv', encoding = "ISO-8859-1")

df_test = pd.read_csv('../input/test.csv', encoding = "ISO-8859-1")

df_prodDesc = pd.read_csv('../input/product_descriptions.csv', encoding = "ISO-8859-1")

df_train.info()

df_prodDesc.info()
def getStemmedText(text):

    return ' '.join([stemmer.stem(x) for x in text.lower().split()])
numTrain = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_prodDesc, how='left', on='product_uid')
df_all.head()
df_all['search_term'] = df_all['search_term'].map(lambda x:getStemmedText(x))
df_all.head()
df_all['product_title'] = df_all['product_title'].map(lambda x:getStemmedText(x))

df_all.head()
df_all.head()
df_all['product_description'] = df_all['product_description'].map(lambda x:getStemmedText(x))

df_all.head()
def getCommonWordsCount(text1, text2):

    words1 = set(text1.lower().split())

    words2 = text2.lower().split()

    sum = 0

    for word in words1:

        sum += int(word in words2)

    return sum



def getCommonWordsTitleCount(row):

    return getCommonWordsCount(row['search_term'], row['product_title'])



def getCommonWordsDescriptionCount(row):

    return getCommonWordsCount(row['search_term'], row['product_description'])
df_all['common_words_title_count'] = df_all.apply(getCommonWordsTitleCount, axis=1)
df_all['common_words_description_count'] = df_all.apply(getCommonWordsDescriptionCount, axis=1)

df_all.head()
df_all.drop(['search_term', 'product_description', 'product_title'], axis=1, inplace=True)
df_all.head()

df_train = df_all.iloc[:numTrain]

df_test = df_all.iloc[numTrain:]

id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values

X_test = df_test.drop(['id','relevance'],axis=1).values

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor



rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)

clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)

clf.fit(X_train, y_train)
clf.score(X_train, y_train)