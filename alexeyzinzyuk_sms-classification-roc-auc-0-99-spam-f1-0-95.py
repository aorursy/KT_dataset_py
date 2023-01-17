import numpy as np

import pandas as pd

import seaborn as sns

from textblob import TextBlob

import matplotlib.pyplot as plt

import lightgbm as lgbm

from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.metrics import classification_report, accuracy_score, make_scorer, roc_auc_score, roc_curve, precision_recall_fscore_support, f1_score

from sklearn.utils import class_weight

from eli5 import show_weights

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

import re



%config InlineBackend.figure_format = 'retina'

sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)



from IPython.core.display import display, HTML, clear_output



def print_df(df, index=True):

    display(HTML(df.to_html(index=index)))
df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')

df = df.drop(df.columns.difference(['v1','v2']), 1).rename({'v1': 'label', 'v2': 'text'}, axis=1)



df.groupby(by='label').describe().T
def get_labels_df(df=df):

    return df[df.label=='spam'], df[df.label=='ham']
df['sms_length'] = df.text.map(lambda x: len(x))



fig1, axes = plt.subplots(1,2, figsize=(20 ,5))



spam, ham = get_labels_df()



sns.distplot(ham.sms_length, hist=False, rug=False, label='ham', ax=axes[0])

sns.distplot(spam.sms_length, hist=False, rug=False, label='spam', ax=axes[0])



sns.boxplot(x='label', y='sms_length', data=df, ax=axes[1])

plt.show()
text_blob = df['text'].map(lambda x: TextBlob(x))

df['sentiment'] = text_blob.map(lambda x: x.sentiment.polarity)

df['subjectivity'] = text_blob.map(lambda x: x.sentiment.subjectivity)
fig1, axes = plt.subplots(1,2, figsize=(20 ,5))

spam, ham = get_labels_df(df)



sns.distplot(ham.sentiment, hist=False, rug=False, label='ham', ax=axes[0])

sns.distplot(spam.sentiment, hist=False, rug=False, label='spam', ax=axes[0])



sns.distplot(ham.subjectivity, hist=False, rug=False, label='ham', ax=axes[1])

sns.distplot(spam.subjectivity, hist=False, rug=False, label='spam', ax=axes[1])



plt.show()



print('spam mean sentiment: {0:.2}, abs mean: {1:.2}'.format(spam.sentiment.mean(), spam.sentiment.abs().mean()))

print('ham mean abs sentiment: {0:.2}, abs mean: {1:.2}'.format(ham.sentiment.mean(), ham.sentiment.abs().mean()))



print('\n'+'spam mean subjectivity: {0:.2}'.format(spam.subjectivity.mean()))

print('ham mean abs subjectivity: {0:.2}'.format(ham.subjectivity.mean()))
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer



def get_tf_ids(text, top_N):

    vect = TfidfVectorizer()

    matrix = vect.fit_transform(text)

    freqs = zip(vect.get_feature_names(),matrix.sum(axis=0).tolist()[0])

   

    return sorted(freqs, key=lambda x: -x[1])[:top_N]



N = 50

tf_idf_df = pd.DataFrame()

tf_idf_df['top_spam'] = get_tf_ids(spam.text, N)

tf_idf_df['top_ham'] = get_tf_ids(ham.text, N)



tf_idf_df
print_df(spam.sample(100)[['text']])
print_df(ham.sample(100)[['text']])
def classify_text(df, clf, max_features):

    text_feature_name = 'text'

    target = 'label'

    df = df[(df[text_feature_name].notna()) & (df[target].notna())]

    

    vect = TfidfVectorizer(max_features=max_features)

    text_clf_pipe = Pipeline([

        ('vect', vect),

        ('clf', clf)

    ])

    

    # Fit on splited data & print classification report

    train_df, test_df = train_test_split(df, test_size=0.20, stratify=df[target])

    text_clf_pipe.fit(train_df[text_feature_name], train_df[target].values)

    

    test_predicted_proba = text_clf_pipe.predict_proba(test_df[text_feature_name])[:,1]

    test_predicted = text_clf_pipe.predict(test_df[text_feature_name])

    

    auc_score = roc_auc_score(test_df[target], test_predicted_proba)

    print('ROC AUC: {0:.4f}'.format(auc_score))

    

    print(classification_report(test_df[target].values, test_predicted)) 

    

    columns = vect.get_feature_names()

    return clf, columns
# MultinomialNB baseline

clf, columns = classify_text(df, clf = MultinomialNB(), max_features=10000)
# LGBMClassifier baseline

clf, columns = classify_text(df, clf = lgbm.LGBMClassifier(n_jobs=-1, importance_type='gain'), max_features=10000)



fi_percent = (100*clf.feature_importances_ / clf.feature_importances_.sum()).round(2)

fi =  pd.DataFrame(fi_percent, 

                   index = columns, 

                   columns=['feature_importance'])



fi.sort_values(by='feature_importance', inplace=True, ascending=False)

print_df(fi.head(20))
def get_text_df(df, text_feature_name, target, max_features=None, regex=None):

    vect = TfidfVectorizer(max_features=max_features)

    matrix = vect.fit_transform(df.text)



    text_df = pd.DataFrame(matrix.toarray(), columns=vect.get_feature_names())

    text_df[target] = df[target]

    text_df[text_feature_name] = df[text_feature_name]

    return text_df, vect







text_feature_name = 'text'

target = 'label'



INF_COLUMNS = [text_feature_name, target]



def get_X_df(df):

    return df[df.columns.difference(INF_COLUMNS)]



def get_X_columns(df):

    return df.columns.difference(INF_COLUMNS)





custom_df, _ = get_text_df(df,text_feature_name, target, max_features=1000)
# Remove all numbers from text features, instead - use one feature "numbers_count"

regex = r"(?u)[0-9]+"

custom_df = custom_df[custom_df.columns.drop(list(custom_df.filter(regex=r"[0-9]+|[^\x00-\x7f]")))]
#custom_df['__grammar_error_count'] = text_blob.map(lambda x: len([w for w in x.words if len(w.spellcheck())>0 ]))
custom_df['__sms_length'] = df['sms_length']

custom_df['__sentiment'] = df['sentiment']

custom_df['__subjectivity'] = df['subjectivity']



# SPAM specific

#1. links (http, www, .com)

custom_df['__url_symbols_count'] = df['text'].str.count(pat='http|www|.com')

#2. a lot of numbers

custom_df['__numbers_count'] = df['text'].str.count(pat="(?u)[0-9]+")



#3. currencies (£, $, €)

custom_df['__currencies_count'] = df['text'].str.count(pat="£|€")



#4. ALL_CAPITAL_LETTERS like this

custom_df['__all_capital_words'] = df['text'].astype(str).map(lambda x: len([y for y in x.split(' ') if y.isupper()]))





# HAM specific

#1. A lot of slangs usage (try to use spelling check for finding slangs?)

#custom_df['__grammar_error_count'] = text_blob.map(lambda x: len([w for w in x.words if len(w.spellcheck())>0 ]))



#2. Smiles! Normal people use smiles ^) ( ':)' , ':(', '=D', ':-)', ':-(', 

custom_df['__smiles_count'] = df['text'].str.count(pat="=D|:-\)|:-\(|:\)|:\(")



#3. '...' usage

custom_df['__ellipsis_count'] = df['text'].str.count(pat="\.\.\.")



#4. Not all text ends with punctuation marks ( !, ?, .)

# pandas endswith not accept regex

custom_df['__ends_with_punctuation'] = df['text'].str.endswith(pat="?") | df['text'].str.endswith(pat="!") | df['text'].str.endswith(pat=".")

fig1, axes = plt.subplots(2,2, figsize=(20 ,10))

spam, ham = get_labels_df(custom_df)



sns.distplot(ham.__numbers_count, hist=True, kde=False, label='ham', ax=axes[0, 0])

sns.distplot(spam.__numbers_count, hist=True, kde=False, label='spam', ax=axes[0, 0])



sns.distplot(ham.__ellipsis_count, hist=True, kde=False, label='ham', ax=axes[1, 0])

sns.distplot(spam.__ellipsis_count, hist=True, kde=False, label='spam', ax=axes[1, 0])



sns.distplot(ham.__currencies_count, hist=True, kde=False, label='ham', ax=axes[0, 1])

sns.distplot(spam.__currencies_count, hist=True, kde=False, label='spam', ax=axes[0, 1])



sns.distplot(ham.__all_capital_words, hist=True, kde=False, label='ham', ax=axes[1, 1])

sns.distplot(spam.__all_capital_words, hist=True, kde=False, label='spam', ax=axes[1, 1])



plt.show()
train_df, test_df = train_test_split(custom_df, test_size=0.20, stratify=custom_df[target])





clf = lgbm.LGBMClassifier(n_jobs=-1, importance_type='gain')



X_train = get_X_df(train_df)

clf.fit(X_train, train_df[target].values)

    

X_test = get_X_df(test_df)

test_predicted_proba = clf.predict_proba(X_test)[:,1]

test_predicted = clf.predict(X_test)

    

auc_score = roc_auc_score(test_df[target], test_predicted_proba)

print('ROC AUC: {0:.4f}'.format(auc_score))

    

print(classification_report(test_df[target].values, test_predicted)) 



fi_percent = (100*clf.feature_importances_ / clf.feature_importances_.sum()).round(2)

fi =  pd.DataFrame(fi_percent, 

                   index = X_test.columns, 

                   columns=['feature_importance'])



fi.sort_values(by='feature_importance', inplace=True, ascending=False)

print_df(fi.head(30))
from sklearn import preprocessing

from sklearn.pipeline import make_pipeline



lb = preprocessing.LabelBinarizer()



def score_classifier(df, clf, preprocess_objects=None):

    X = get_X_df(df)

    y = df[target]

  

    #clf = clf if preprocess_objects is None else make_pipeline(preprocess_objects + [clf])



    roc_auc_scores = cross_val_score(clf, X=X, y=y, scoring='roc_auc', cv=5)

    print('ROC AUC = {0} +- {1}'.format(roc_auc_scores.mean(), roc_auc_scores.std()))



    f1_scores =  cross_val_score(clf, X=X, y=y, scoring=make_scorer(f1_score, pos_label='spam'), cv=5)

    print('spam F1 = {0} +- {1}'.format(f1_scores.mean(), f1_scores.std()))

    

    return roc_auc_scores, f1_scores





_,_ = score_classifier(clf = lgbm.LGBMClassifier(n_jobs=-1, importance_type='gain'), df=custom_df)
_,_ = score_classifier(clf = lgbm.LGBMClassifier(n_jobs=-1, importance_type='gain', class_weight="balanced"), df=custom_df)
#for undersampler in 

from imblearn.under_sampling import *

from imblearn.pipeline import Pipeline as PipelineImblearn

for undersampler in [

                     #CondensedNearestNeighbour(), 

                     #EditedNearestNeighbours(), 

                     #AllKNN(), 

                     #InstanceHardnessThreshold(), 

                     #NeighbourhoodCleaningRule(), 

                     OneSidedSelection(),

                     RandomUnderSampler(),

                     TomekLinks()]:

    

    print(type(undersampler))

    clf = lgbm.LGBMClassifier(n_jobs=-1, importance_type='gain')

    pipeline = PipelineImblearn([('undersampler', undersampler), ('clf', clf)])

    _,_ = score_classifier(clf = pipeline,

                     df = custom_df)
print_df(df.groupby(by='text')

        .agg(['count', 'max'])

        .label

        .sort_values(by='count', ascending=False)

        .head(100))
from imblearn.pipeline import Pipeline as PipelineImblearn

from imblearn import FunctionSampler



def remove_duplicates(X, y):

    _, unique_indexes = np.unique(X.astype('float'), return_index=True, axis=0)



    return X[unique_indexes], y[unique_indexes]



duplicates_remover = FunctionSampler(func=remove_duplicates)

clf = lgbm.LGBMClassifier(n_jobs=-1, importance_type='gain', class_weight="balanced")



pipeline = PipelineImblearn([('remove_duplicates', duplicates_remover), ('undersampler', TomekLinks()), ('clf', clf)])



_,_ = score_classifier(clf = pipeline,

                 df = custom_df)

X = get_X_df(custom_df)

y = df[target]

  

pipeline = PipelineImblearn([('undersampler', TomekLinks()), ('clf', clf)])



predicted = cross_val_predict(clf, X=X, y=y, cv=5)



wrong_prediction = custom_df[predicted != y][['text', 'label']]

#wrong_prediction['predicted_label'] = predicted

print_df(wrong_prediction)