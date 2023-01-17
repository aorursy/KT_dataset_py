import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from nltk import wordpunct_tokenize          

from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):

    """Custom tokenizer class that stems tokens"""

    def __init__(self):

        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):

        return [self.wnl.lemmatize(t) for t in wordpunct_tokenize(doc) if len(t.strip()) > 1]

    

def show_topn(classifier, vectorizer, categories, n):

    """Returns the top n features that characterize each category"""

    feature_names = np.asarray(vectorizer.get_feature_names())

    for i, category in enumerate(categories):

        topn = np.argsort(classifier.coef_[i])[-n:] #argsort sorts in asc order

        print('{}: {}'.format(category, ", ".join(feature_names[topn])))
# read in a few columns from the data and show the top of the resulting dataframe

df = pd.read_csv('../input/listings.csv', usecols = ['id', 'name', 'space', 'description', 'neighborhood_overview', 'neighbourhood_cleansed'])



df.head()
# let's combine the name, space, description, and neighborhood_overview into a new column

df['combined_description'] = df.apply(lambda x: '{} {} {} {}'.format(x['name'], x['space'], x['description'], x['neighborhood_overview']), axis=1)
df.groupby(by='neighbourhood_cleansed').count()[['id']].sort_values(by='id', ascending=False)
pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english', tokenizer=LemmaTokenizer())),

                     ('clf', SGDClassifier(loss='hinge', penalty='l2',

                                           alpha=1e-3, n_iter=5, random_state=42)),

])





le = LabelEncoder()

new_target = le.fit_transform(df['neighbourhood_cleansed'])

model = pipeline.fit(df['combined_description'], new_target)
show_topn(model.named_steps['clf'], model.named_steps['tfidf'], le.inverse_transform(model.named_steps['clf'].classes_), 5)