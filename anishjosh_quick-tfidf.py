import pandas as pd



train = pd.read_csv('../input/train.csv', header = 0)

train.head()
train.info()
train.describe()
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import words

from sklearn.model_selection import train_test_split



X, y = train[['comment_text']], train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)
vectorizer = CountVectorizer(stop_words = 'english',\

                             lowercase = True,\

                             max_df = 0.95,\

                             min_df = 0.05,\

                             vocabulary = set(words.words()))



vectorized_text = vectorizer.fit_transform(X_train.comment_text)
transformer = TfidfTransformer(smooth_idf = False)

tfidf = transformer.fit_transform(vectorized_text)
from sklearn.feature_selection import SelectKBest, chi2



ch2 = SelectKBest(chi2, k = 200)

best_features = ch2.fit_transform(tfidf, y_train.toxic)
filth = [feature for feature, mask in\

         zip(vectorizer.get_feature_names(), ch2.get_support())\

         if mask == True]



print(filth)
analyzer = CountVectorizer(lowercase = True,\

                             vocabulary = filth)
def get_features(frame):

    result = pd.DataFrame(\

                transformer.fit_transform(\

                analyzer.fit_transform(\

                frame.comment_text)\

                                         ).todense(),\

                                            index = frame.index)

    return result
feature_frames = {}



for frame in ('train', 'test'):

    feature_frames[frame] = get_features(eval('X_%s' % frame))



feature_frames['train'].info()
from sklearn.neighbors import KNeighborsClassifier



knc = KNeighborsClassifier(n_neighbors = 10)

knc.fit(feature_frames['train'], y_train.toxic)
from sklearn.metrics import log_loss



result = pd.DataFrame(knc.predict_proba(feature_frames['test']), index = feature_frames['test'].index)



result['actual'] = y_test.toxic

result['text'] = X_test.comment_text



print(log_loss(y_test.toxic, result[1]))
pd.set_option('max_colwidth', 100)

result[[1, 'actual', 'text']][(result.actual == 1) & (result[1] > 0.5)][:10]