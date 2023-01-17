import numpy as np

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer()

docs = np.array([

    'The sun is shining',

    'The weather is sweet',

    'The sun is shining and the weather is sweet'

])



bag = count.fit_transform(docs)
from sklearn.feature_extraction.text import TfidfTransformer



tfidf = TfidfTransformer()

np.set_printoptions(precision=2)



print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
import pandas as pd



df = pd.read_csv('../input/movie_data.csv')



df.loc[49941, 'review'][-50:]
import re



def preprocessor(text):

    text = re.sub('<[^>]*>', '', text)

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    text = re.sub('[\W]+', ' ', text.lower())

    text = text + " ".join(emoticons).replace('-', '')

    return text



preprocessor("</a>This :) is :( a test :-)!")
df['review'] = df['review'].apply(preprocessor)



print(df.tail())
def tokenizer(text):

    return text.split()



tokenizer('runners like running and thus they run')
from nltk.stem.porter import PorterStemmer



porter = PorterStemmer()



def tokenizer_porter(text):

    return [porter.stem(word) for word in text.split()]



tokenizer_porter('runners like running and thus they run')
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords



stop = stopwords.words('english')



[w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop]
X = df.review

y = df.sentiment



from sklearn.model_selection import train_test_split 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)



param_grid = [{'vect__ngram_range': [(1,1)],

              'vect__stop_words': [stop, None],

              'vect__tokenizer': [tokenizer, tokenizer_porter],

              'clf__penalty': ['l1', 'l2'],

              'clf__C': [1.0, 10.0, 100.0]},

             {'vect__ngram_range': [(1,1)],

              'vect__stop_words': [stop, None],

              'vect__tokenizer': [tokenizer, tokenizer_porter],

              'vect__use_idf': [False],

              'vect__norm': [None],

              'clf__penalty': ['l1', 'l2'],

              'clf__C': [1.0, 10.0, 100.0]}]



lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])



gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)



gs_lr_tfidf.fit(X_train, y_train)



print('Best parameter set: %s' % gs_lr_tfidf.best_params_)