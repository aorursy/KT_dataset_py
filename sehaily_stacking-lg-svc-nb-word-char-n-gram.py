import numpy as np
import pandas as pd
import seaborn as sns
import nltk
import nltk.sentiment
from nltk.stem.porter import PorterStemmer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

train = pd.read_csv('../input/train.tsv', sep='\t')
train.head()
sns.countplot(x='Sentiment', data=train)
train['word_count'] = train['Phrase'].apply(lambda x: len(x.split()))
sns.boxplot(x='Sentiment', y='word_count', data=train)
train['avg_word_lenght'] = train['Phrase'].apply(lambda x:np.mean([len(word) for word in x.split()]))
sns.boxplot(x='Sentiment', y='avg_word_lenght', data=train)
group_split = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
train_index, validation_index = list(group_split.split(train['PhraseId'],
                                    y=train['Sentiment'], groups=train['SentenceId']))[0]
train, validation = train.iloc[train_index], train.iloc[validation_index]
log_reg_countvec_pl = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='word',ngram_range=[1,3], stop_words=nltk.corpus.stopwords.words('english'))),
        ('clf',LogisticRegression())
])
log_reg_countvec_pl.fit(train['Phrase'], train['Sentiment'])
accuracy_score(validation['Sentiment'], log_reg_countvec_pl.predict(validation['Phrase']))
#concating tran and validation to calculate the features only once
data = pd.concat([train, validation], keys=['train', 'validation'])
#tokenizing the phrases to use it as input to calculate another features
data['tokenized_words'] = data['Phrase'].apply(nltk.word_tokenize)
# mark words after negation word with _NEG tag
data['negated_phrase_tokenized'] = data['tokenized_words'].apply(nltk.sentiment.util.mark_negation)
data['negated_phrase'] = data['negated_phrase_tokenized'].apply(lambda x: " ".join(x))
#returns 1 if the text contains negation word
data['negated_flag'] = (data['tokenized_words'].apply(nltk.sentiment.vader.negated)).astype('int8')
# get_numeric_data = preprocessing.FunctionTransformer(lambda a: a[['negated_flag']], validate=False)
get_text_data = preprocessing.FunctionTransformer(lambda a: a['negated_phrase'], validate=False)
lg_pl = Pipeline([
        ('union', FeatureUnion( #unites both text and numeric arrays into one array
            transformer_list = [
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(analyzer='word',ngram_range=[1,3],
                                                   stop_words=nltk.corpus.stopwords.words('english')))
                ])),
             ]
        )), 
        ('clf',LogisticRegression(penalty ='l1'))
    ])
lg_pl.fit(data.loc['train'], data.loc['train']['Sentiment'])
accuracy_score(data.loc['validation']['Sentiment'], lg_pl.predict(data.loc['validation']))
lg_pl = Pipeline([
        ('union', FeatureUnion( #unites both text and numeric arrays into one array
            transformer_list = [
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(analyzer='word',ngram_range=[1,3],
                                                   stop_words=nltk.corpus.stopwords.words('english')))
                ])),
                ('tfidf', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer(analyzer='word',ngram_range=[1,3],
                                                   stop_words=nltk.corpus.stopwords.words('english')))
                ])),
                ('char_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer(analyzer='char',ngram_range=[3,5], \
                                                   stop_words=nltk.corpus.stopwords.words('english')))
                ])),
             ]
        )), 
        ('clf',LogisticRegression(penalty ='l1'))
])
lg_pl.fit(data.loc['train'], data.loc['train']['Sentiment'])
accuracy_score(data.loc['validation']['Sentiment'], lg_pl.predict(data.loc['validation']))
train_data = data.loc['train']
train_index, validation_index = list(group_split.split(train_data['PhraseId'],
                                    y=train_data['Sentiment'], groups=train_data['SentenceId']))[0]
train, validation_stack = train_data.iloc[train_index], train_data.iloc[validation_index]
nb_pl = Pipeline([
        ('union', FeatureUnion( #unites both text and numeric arrays into one array
            transformer_list = [
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(analyzer='word',ngram_range=[1,3],
                                                   stop_words=nltk.corpus.stopwords.words('english')))
                ])),
                ('tfidf', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer(analyzer='word',ngram_range=[1,3],
                                                   stop_words=nltk.corpus.stopwords.words('english')))
                ])),
                ('char_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer(analyzer='char',ngram_range=[3,5], \
                                                   stop_words=nltk.corpus.stopwords.words('english')))
                ])),
             ]
        )), 
        ('clf',MultinomialNB())
    ])
nb_pl.fit(train, train['Sentiment'])
svc_pl = Pipeline([
        ('union', FeatureUnion( #unites both text and numeric arrays into one array
            transformer_list = [
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(analyzer='word',ngram_range=[1,3],
                                                   stop_words=nltk.corpus.stopwords.words('english')))
                ])),
                ('tfidf', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer(analyzer='word',ngram_range=[1,3],
                                                   stop_words=nltk.corpus.stopwords.words('english')))
                ])),
                ('char_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer(analyzer='char',ngram_range=[3,5], \
                                                   stop_words=nltk.corpus.stopwords.words('english')))
                ])),
             ]
        )), 
        ('clf',LinearSVC())
])
svc_pl.fit(train, train['Sentiment'])
lg_stack = LogisticRegression()
lg_stack.fit(
    np.column_stack(
        (
            nb_pl.predict_proba(validation_stack),
            lg_pl.predict_proba(validation_stack),
            svc_pl.decision_function(validation_stack)
        )
    )
    ,validation_stack['Sentiment']
)
accuracy_score(data.loc['validation']['Sentiment'], lg_stack.predict(
    np.column_stack(
        (
            nb_pl.predict_proba(data.loc['validation']),
            lg_pl.predict_proba(data.loc['validation']),
            svc_pl.decision_function(data.loc['validation'])

            
        )
    )
))
test = pd.read_csv('../input/test.tsv', sep='\t')
#tokenizing the phrases to use it as input to calculate another features
test['tokenized_words'] = test['Phrase'].apply(nltk.word_tokenize)
# mark words after negation word with _NEG tag
test['negated_phrase_tokenized'] = test['tokenized_words'].apply(nltk.sentiment.util.mark_negation)
test['negated_phrase'] = test['negated_phrase_tokenized'].apply(lambda x: " ".join(x))
#returns 1 if the text contains negation word
test['negated_flag'] = (test['tokenized_words'].apply(nltk.sentiment.vader.negated)).astype('int8')
lg_pl.fit(data.loc['train'], data.loc['train']['Sentiment'])
nb_pl.fit(data.loc['train'], data.loc['train']['Sentiment'])
svc_pl.fit(data.loc['train'], data.loc['train']['Sentiment'])
lg_stack.fit(
    np.column_stack(
        (
            nb_pl.predict_proba(data.loc['validation']),
            lg_pl.predict_proba(data.loc['validation']),
            svc_pl.decision_function(data.loc['validation'])
        )
    )
    ,data.loc['validation']['Sentiment']
)
test['Sentiment'] = lg_stack.predict(
                        np.column_stack(
                            (
                                nb_pl.predict_proba(test),
                                lg_pl.predict_proba(test),
                                svc_pl.decision_function(test)
                            )
                        )
                    )
test[['PhraseId', "Sentiment"]].to_csv('test_predictions.csv', index=False)
