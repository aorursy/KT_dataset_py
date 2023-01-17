import pandas as pd

df = pd.read_csv('./../input/winemag-data_first150k.csv')

df.head()
df.drop(df.columns[[0]], axis=1, inplace=True) # ditch that unnamed row numbers column

df.describe(include='all')
dups = df[df.duplicated('description')]

dups.sort_values('description', ascending=False).iloc[3:5]
dedupped_df = df.drop_duplicates(subset='description')

print('Total unique reviews:', len(dedupped_df))

print('\nVariety description \n', dedupped_df['variety'].describe())
varieties = dedupped_df['variety'].value_counts()

varieties
varieties.describe()
top_wines_df = dedupped_df.loc[dedupped_df['variety'].isin(varieties.axes[0][:20])]

top_wines_df['variety'].describe()
# our labels, as numbers. 



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(top_wines_df['variety'])
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

vect = TfidfVectorizer()

x = vect.fit_transform(top_wines_df['description'])

x
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)
import time

start = time.time()

clf = LogisticRegression()

clf.fit(x_train, y_train)

pred = clf.predict(x_test)

print(time.time() - start)
accuracy_score(y_test, pred)
wine_stop_words = []

for variety in top_wines_df['variety'].unique():

    for word in variety.split(' '):

        wine_stop_words.append(word.lower())

wine_stop_words = pd.Series(data=wine_stop_words).unique()

wine_stop_words
vect2 = TfidfVectorizer(stop_words=list(wine_stop_words))

x2 = vect2.fit_transform(top_wines_df['description'])

x2
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y, test_size=0.25, random_state=10)

clf = LogisticRegression()

clf.fit(x_train2, y_train2)

pred = clf.predict(x_test2)

accuracy_score(y_test, pred)
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



x_train3, x_test3, y_train3, y_test3 = train_test_split(

    top_wines_df['description'].values, y, test_size=0.25, random_state=10)



pipe = Pipeline([

    ('vect', TfidfVectorizer(stop_words=list(wine_stop_words))), 

    ('clf', LogisticRegression(random_state=0))

])



param_grid = [

  {

    'vect__ngram_range': [(1, 2)],

    'clf__penalty': ['l1', 'l2'],

    'clf__C': [1.0, 10.0, 100.0]

  },

  {

    'vect__ngram_range': [(1, 2)],

    'vect__use_idf':[False],

    'clf__penalty': ['l1', 'l2'],

    'clf__C': [1.0, 10.0, 100.0]

    },

]



grid = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

from sklearn.decomposition import LatentDirichletAllocation as LDA

from sklearn.feature_extraction import text

stop_words = text.ENGLISH_STOP_WORDS.union(wine_stop_words)

vect = TfidfVectorizer(stop_words=stop_words)

x = vect.fit_transform(top_wines_df['description'])

lda = LDA(learning_method='batch')

topics = lda.fit_transform(x)
import numpy as np

def print_topics(topics, feature_names, sorting, topics_per_chunk=5,

                 n_words=10):

    for i in range(0, len(topics), topics_per_chunk):

        # for each chunk:

        these_topics = topics[i: i + topics_per_chunk]

        # maybe we have less than topics_per_chunk left

        len_this_chunk = len(these_topics)

        # print topic headers

        print(("topic {!s:<8}" * len_this_chunk).format(*these_topics))

        print(("-------- {0:<5}" * len_this_chunk).format(""))

        # print top n_words frequent words

        for i in range(n_words):

            try:

                print(("{!s:<14}" * len_this_chunk).format(

                    *feature_names[sorting[these_topics, i]]))

            except:

                pass

        print("\n")

        



sorting = np.argsort(lda.components_, axis=1)[:, ::-1]

feature_names = np.array(vect.get_feature_names())

        

print_topics(topics=range(10), feature_names=feature_names, sorting=sorting)
top_wines_df[top_wines_df['description'].str.contains('cheeseburger')].values
dedupped_df[dedupped_df['description'].str.contains('cheeseburger')]['points'].describe()

dedupped_df['points'].describe()
