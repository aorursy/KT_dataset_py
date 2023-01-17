%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin 
from nltk.stem.snowball import SnowballStemmer
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import wordnet
train_data = pd.read_csv('../input/train.csv')
train_data.head()
train_data.info()
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
def combined_cv_scores(X, ys, params):
    """ CV scores for different set of labels (ys) """
    scores = {}
    for col in ys.columns:
        clf = LogisticRegression(C=params[col])
        s = cross_val_score(clf, X, ys[col], scoring='roc_auc')
        scores[col] = np.mean(s)
        print('{}: {}, mean {}'.format(col, s, np.mean(s)))
    return scores
vectorizer_unigram = TfidfVectorizer(sublinear_tf=True)
X_unigram = vectorizer_unigram.fit_transform(train_data['comment_text'])
X_unigram.shape
# 'C' parameter of logistic regression, obtained by GridSearchCV 
params_unigram = {
    'toxic': 4,
    'severe_toxic': 2,
    'obscene': 3,
    'threat': 4,
    'insult': 3,
    'identity_hate': 3,
}
%%time
scores_unigram = combined_cv_scores(X_unigram, train_data[label_names], params_unigram)
vectorizer_bigram = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
X_bigram = vectorizer_bigram.fit_transform(train_data['comment_text'])
X_bigram.shape
# 'C' parameter of logistic regression, obtained by GridSearchCV 
params_bigram = {
    'toxic': 50,
    'severe_toxic': 4,
    'obscene': 30,
    'threat': 40,
    'insult': 12,
    'identity_hate': 12,
}
%%time
scores_bigram = combined_cv_scores(X_bigram, train_data[label_names], params_bigram)
stemmer = SnowballStemmer('english', ignore_stopwords=False)

class StemmedTfidfVectorizer(TfidfVectorizer):
    
    def __init__(self, stemmer, *args, **kwargs):
        super(StemmedTfidfVectorizer, self).__init__(*args, **kwargs)
        self.stemmer = stemmer
        
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(word) for word in analyzer(doc.replace('\n', ' ')))
vectorizer_stem_u = StemmedTfidfVectorizer(stemmer=stemmer, sublinear_tf=True)
X_train_stem_u = vectorizer_stem_u.fit_transform(train_data['comment_text'])
X_train_stem_u.shape
# 'C' parameter of logistic regression, obtained by GridSearchCV 
params_stem_u = {
    'toxic': 3,
    'severe_toxic': 1,
    'obscene': 3,
    'threat': 4,
    'insult': 2,
    'identity_hate': 2,
}
%%time
scores_stem_u = combined_cv_scores(X_train_stem_u, train_data[label_names], params_stem_u)
vectorizer_stem_b = StemmedTfidfVectorizer(stemmer=stemmer, ngram_range=(1,2), sublinear_tf=True)
X_train_stem_b = vectorizer_stem_b.fit_transform(train_data['comment_text'])
X_train_stem_b.shape
# 'C' parameter of logistic regression, obtained by GridSearchCV 
params_stem_b = {
    'toxic': 20,
    'severe_toxic': 3,
    'obscene': 20,
    'threat': 40,
    'insult': 8,
    'identity_hate': 12,
}
%%time
scores_stem_b = combined_cv_scores(X_train_stem_b, train_data[label_names], params_stem_b)
def lemmatize(text):
    """ Tokenize text and lemmatize word tokens """
    def get_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN
    
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(token, get_pos(tag)) for token, tag in pos_tag(word_tokenize(text))]
vectorizer_lemma_u = TfidfVectorizer(tokenizer=lemmatize, sublinear_tf=True)
X_train_lemma_u = vectorizer_lemma_u.fit_transform(train_data['comment_text'])
X_train_lemma_u.shape
# 'C' parameter of logistic regression, obtained by GridSearchCV 
params_lemma_u = {
    'toxic': 4,
    'severe_toxic': 2,
    'obscene': 4,
    'threat': 4,
    'insult': 3,
    'identity_hate': 3,
}
%%time
scores_stem_b = combined_cv_scores(X_train_lemma_u, train_data[label_names], params_lemma_u)
models = {
    'toxic': {'classifier': LogisticRegression(C=20), 'features': 'stem_b'},
    'severe_toxic': {'classifier': LogisticRegression(C=3), 'features': 'stem_b'},
    'obscene': {'classifier': LogisticRegression(C=3), 'features': 'stem_u'},
    'threat': {'classifier': LogisticRegression(C=40), 'features': 'stem_b'},
    'insult': {'classifier': LogisticRegression(C=8), 'features': 'stem_b'},
    'identity_hate': {'classifier': LogisticRegression(C=2), 'features': 'stem_u'},
}
def fit_predict_results(models, train_features, test_features, train_data, test_data):
    result = pd.DataFrame(columns=(['id']+list(models.keys())))
    result.id = test_data['id']
    for label, model in models.items():
        clf = model['classifier']
        X = train_features[model['features']]
        clf.fit(X, train_data[label])
        X = test_features[model['features']]
        predicts = clf.predict_proba(X)
        result[label] = predicts[:,1]
    return result
# read test data
test_data = pd.read_csv('../input/test.csv')
test_data.head()
# Test unigrams with stemming
X_test_stem_u = vectorizer_stem_u.transform(test_data['comment_text'])
X_test_stem_u.shape
# Test bigrams with stemming
X_test_stem_b = vectorizer_stem_b.transform(test_data['comment_text'])
X_test_stem_b.shape
%%time
train_features = {
    'stem_u': X_train_stem_u,
    'stem_b': X_train_stem_b
}
test_features = {
    'stem_u': X_test_stem_u,
    'stem_b': X_test_stem_b
}

result = fit_predict_results(models, train_features, test_features, train_data, test_data)
result[result.threat>0.5][:5]
test_data['comment_text'][1053]
result.to_csv('submission.csv', index=False)