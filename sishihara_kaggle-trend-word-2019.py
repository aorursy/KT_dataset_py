import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

import collections

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style='darkgrid')

sns.set_context('poster')
kernels = pd.read_csv('../input/meta-kaggle/Kernels.csv')
kernels.head()
kernels.shape
kernels = kernels.query('TotalVotes > 0')
kernels.shape
kernels['Date'] = pd.to_datetime(kernels['MadePublicDate'])

# kernels['Date'] = pd.to_datetime(kernels['CreationDate'])
kernels['Date'].dt.year.value_counts().plot.bar()
kernel2019 = kernels[kernels['Date'].dt.year == 2019.0].reset_index()

kernel2018 = kernels[kernels['Date'].dt.year == 2018.0].reset_index()

kernel2017 = kernels[kernels['Date'].dt.year == 2017.0].reset_index()

kernel2016 = kernels[kernels['Date'].dt.year == 2016.0].reset_index()

kernel2015 = kernels[kernels['Date'].dt.year == 2015.0].reset_index()
words2019 = []

words2018 = []

words2017 = []

words2016 = []

words2015 = []



for _ in (kernel2019['CurrentUrlSlug']):

    words2019 += _.split("-")



for _ in (kernel2018['CurrentUrlSlug']):

    words2018 += _.split("-")



for _ in (kernel2017['CurrentUrlSlug']):

    words2017 += _.split("-")



for _ in (kernel2016['CurrentUrlSlug']):

    words2016 += _.split("-")



for _ in (kernel2015['CurrentUrlSlug']):

    words2015 += _.split("-")
c2019 = collections.Counter(words2019)

c2018 = collections.Counter(words2018)

c2017 = collections.Counter(words2017)

c2016 = collections.Counter(words2016)

c2015 = collections.Counter(words2015)
data = [

    ' '.join(words2015),

    ' '.join(words2016),

    ' '.join(words2017),

    ' '.join(words2018),

    ' '.join(words2019)

]
stopWords = stopwords.words("english")
vectorizer = TfidfVectorizer(stop_words=stopWords)

X = vectorizer.fit_transform(data).toarray()



df = pd.DataFrame(X.T, index=vectorizer.get_feature_names(),

                  columns=['words2015', 'words2016', 'words2017', 'words2018', 'words2019'])



forplot = df.sort_values('words2019', ascending=False).head(20)

forplot
plt.rcParams['font.size'] = 18

forplot.T.plot(figsize=(20, 30))
stopWordsAdd = ['data', 'analysis', 'model', 'simple', '2019', 'ashrae', 'ieee',

                'using', 'prediction', 'ml', 'classification', 'regression',

                'machine', 'learning', 'exercise', 'detection', 'kernel', 'dataset']



for sw in stopWordsAdd:

    stopWords.append(sw)
vectorizer = TfidfVectorizer(stop_words=stopWords)

X = vectorizer.fit_transform(data).toarray()



df = pd.DataFrame(X.T, index=vectorizer.get_feature_names(),

                  columns=['words2015', 'words2016', 'words2017', 'words2018', 'words2019'])



forplot = df.sort_values('words2019', ascending=False).head(8)

forplot
plt.rcParams['font.size'] = 18

forplot.T.plot(figsize=(20, 30))