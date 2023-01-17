import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

%config InlineBackend.figure_format = 'svg' # plots look nicer in svg

%matplotlib inline

sns.set()

pd.options.display.max_colwidth = 160
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
train.shape, test.shape
train.head(2)
sns.countplot(x=train.author);
import spacy

import nltk
nlp = spacy.en.English()
# add custom stop words

spacy.en.STOP_WORDS.add("'s")

for word in spacy.en.STOP_WORDS:

    lexeme = nlp.vocab[word]

    lexeme.is_stop = True
def replace_ents(doc):

    prefix = 'ent__'

    text = str(doc.doc)

    for ent in doc.ents:

        text = text.replace(ent.orth_, prefix + ent.label_)

    return text
def preprocess(df):

    print('Started parsing...')

    doc = df.text.apply(nlp)

    print('Text parsed')

    

    df['n_char']   = df.text.apply(len)

    df['n_words']  = doc.apply(lambda x: len([t for t in x if not t.is_punct]))

    df['n_punct']  = doc.apply(lambda x: len([t for t in x if t.is_punct]))

    df['n_ents']   = doc.apply(lambda x: len(x.ents))

    df['n_chunks'] = doc.apply(lambda x: len(list(x.noun_chunks)))

    df['n_unique_words'] = doc.apply(lambda x: len(set([t.lower_ for t in x if not t.is_punct])))

    df['n_stop_words']   = doc.apply(lambda x: len([t for t in x if t.is_stop]))

    df['char_by_word']   = doc.apply(lambda x: np.mean([len(t.orth_) for t in x if not t.is_punct]))

    print('Features created')

    

    df['text_ent_repl'] = doc.apply(replace_ents)

    print('Entities replaced')

    

    clean_and_lemmatize = lambda x: ' '.join([t.lemma_ for t in x if not t.is_punct and not t.is_stop])

    df['text_cleaned'] = doc.apply(clean_and_lemmatize)

    print('Text cleaned')
%%time

preprocess(train)
%%time

preprocess(test)
eap = train.loc[train.author == 'EAP']

hpl = train.loc[train.author == 'HPL']

mws = train.loc[train.author == 'MWS']



eap_t = eap.text_cleaned

hpl_t = hpl.text_cleaned

mws_t = mws.text_cleaned
fd = nltk.FreqDist([y for x in train.text_cleaned.str.split() for y in x])



fd_eap = nltk.FreqDist([y for x in eap_t.str.split() for y in x])

fd_hpl = nltk.FreqDist([y for x in hpl_t.str.split() for y in x])

fd_mws = nltk.FreqDist([y for x in mws_t.str.split() for y in x])
fd.plot(30, title='Overall')

fd_eap.plot(30, title='EAP')

fd_hpl.plot(30, title='HPL')

fd_mws.plot(30, title='MWS')
from wordcloud import WordCloud

sns.set_style({'axes.grid' : False})
WordCloud(min_font_size=8, width=1000, height=400).generate(' '.join(eap_t)).to_image()
WordCloud(min_font_size=8, width=1000, height=400, colormap='cubehelix_r').generate(' '.join(hpl_t)).to_image()
WordCloud(min_font_size=8, width=1000, height=400, colormap='hot').generate(' '.join(mws_t)).to_image()
sns.set()
def drop_outliers(s):

    med = s.mean()

    std = s.std()

    return s[(med - 3*std <= s) & (s <= med + 3*std)]
f, ax = plt.subplots(figsize=(7,4))

sns.kdeplot(drop_outliers(eap.n_char), shade=True, color="r");

sns.kdeplot(drop_outliers(hpl.n_char), shade=True, color="g");

sns.kdeplot(drop_outliers(mws.n_char), shade=True, color="b");

ax.legend(labels=['EAP', 'HPL', 'MWS']);
f, ax = plt.subplots(figsize=(7,4))

sns.kdeplot(drop_outliers(eap.n_punct), shade=True, color="r");

sns.kdeplot(drop_outliers(hpl.n_punct), shade=True, color="g");

sns.kdeplot(drop_outliers(mws.n_punct), shade=True, color="b");

ax.legend(labels=['EAP', 'HPL', 'MWS']);
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import LabelEncoder



from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB



from sklearn.pipeline import Pipeline
y = train.author
vectorizer = CountVectorizer(

    token_pattern=r'\w{1,}',

    ngram_range=(1, 2), stop_words='english'

)

X = vectorizer.fit_transform(train.text)
%%time

scores = cross_val_score(LogisticRegression(), X, y, cv=10, n_jobs=-1, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
%%time

scores = cross_val_score(MultinomialNB(), X, y, cv=10, n_jobs=-1, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
vectorizer = CountVectorizer(ngram_range=(1,2))

X = vectorizer.fit_transform(train.text_cleaned)
%%time

scores = cross_val_score(LogisticRegression(), X, y, cv=10, n_jobs=-1, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
%%time

scores = cross_val_score(MultinomialNB(), X, y, cv=10, n_jobs=-1, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
vectorizer = CountVectorizer(token_pattern=r'\w{1,}', stop_words='english', ngram_range=(1,2))

X = vectorizer.fit_transform(train.text_ent_repl)
%%time

scores = cross_val_score(LogisticRegression(), X, y, cv=10, n_jobs=-1, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
%%time

scores = cross_val_score(MultinomialNB(), X, y, cv=10, n_jobs=-1, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
vectorizer = TfidfVectorizer(

    token_pattern=r'\w{1,}', ngram_range=(1, 1), 

    use_idf=True, smooth_idf=True, sublinear_tf=True,

)

X = vectorizer.fit_transform(train.text_cleaned)
%%time

scores = cross_val_score(LogisticRegression(), X, y, cv=10, n_jobs=-1, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
%%time

scores = cross_val_score(MultinomialNB(), X, y, cv=10, n_jobs=-1, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
import xgboost as xgb

import lightgbm as lgb
drop = ['id', 'text', 'text_cleaned', 'text_ent_repl']
X_meta = train.drop(drop + ['author'], axis=1)
%%time

lgbc = lgb.LGBMClassifier(objective='multiclass', n_estimators=100)

scores = cross_val_score(lgbc, X_meta, y, cv=4, n_jobs=1, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
%%time

xgbc = xgb.XGBClassifier(objective='multi:softprob', n_estimators=200)

scores = cross_val_score(xgbc, X_meta, y, cv=4, n_jobs=4, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
xgbc.fit(X_meta, y);

xgb.plot_importance(xgbc);
from sklearn.model_selection import StratifiedKFold

from sklearn import metrics
def add_prob_features(vectorizer, col, model, prefix, cv=5):

    vectorizer.fit(train[col].append(test[col]))

    X = vectorizer.transform(train[col])

    X_test = vectorizer.transform(test[col])

    

    cv_scores = []

    pred_test = 0

    pred_train = np.zeros([train.shape[0], 3])

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=123)

    

    print('CV started')

    for train_index, dev_index in kf.split(X, y):

        X_train, X_dev = X[train_index], X[dev_index]

        y_train, y_dev = y[train_index], y[dev_index]

        

        model.fit(X_train, y_train)

        pred_dev   = model.predict_proba(X_dev)

        pred_test += model.predict_proba(X_test)

    

        pred_train[dev_index, :] = pred_dev

        cv_scores.append(metrics.log_loss(y_dev, pred_dev))

        print('.', end='')

        

    print('')

    print("Mean CV LogLoss: %.3f" % (np.mean(cv_scores)))

    pred_test /= cv



    train[prefix+'eap'] = pred_train[:, 0]

    train[prefix+'hpl'] = pred_train[:, 1]

    train[prefix+'mws'] = pred_train[:, 2]

    

    test[prefix+'eap'] = pred_test[:, 0]

    test[prefix+'hpl'] = pred_test[:, 1]

    test[prefix+'mws'] = pred_test[:, 2]
vectorizer = CountVectorizer(

    token_pattern=r'\w{1,}',

    ngram_range=(1, 2), stop_words='english'

)

add_prob_features(vectorizer, 'text', MultinomialNB(), 'nb_ctv_', cv=40)
vectorizer = TfidfVectorizer(

    token_pattern=r'\w{1,}', ngram_range=(1, 1), 

    use_idf=True, smooth_idf=True, sublinear_tf=True,

)

add_prob_features(vectorizer, 'text_cleaned', MultinomialNB(), 'nb_tfv_', cv=40)
vectorizer = TfidfVectorizer(

    ngram_range=(1, 5), analyzer='char'

)

add_prob_features(vectorizer, 'text', MultinomialNB(), 'nb_char_', cv=40)
vectorizer = CountVectorizer(

    token_pattern=r'\w{1,}',

    ngram_range=(1, 2), stop_words='english'

)

add_prob_features(vectorizer, 'text_ent_repl', MultinomialNB(), 'nb_ent_', cv=40)
X = train.drop(drop + ['author'], axis=1)
%%time

lgbc = lgb.LGBMClassifier(objective='multiclass', n_estimators=150, num_leaves=10)

scores = cross_val_score(lgbc, X, y, cv=4, n_jobs=1, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
%%time

xgbc = xgb.XGBClassifier(objective='multi:softprob', n_estimators=150)

scores = cross_val_score(xgbc, X, y, cv=4, n_jobs=1, scoring='neg_log_loss')

print('LogLoss: %.3f +- %.3f' % (-np.mean(scores), 2*np.std(scores)))
%%time

xgbc.fit(X, y);
xgb.plot_importance(xgbc);
def sub(est, name='sub.csv'):

    sub = pd.DataFrame(est.predict_proba(test.drop(drop, axis=1)), columns=['EAP', 'HPL', 'MWS'])

    sub.insert(0, 'id', test.id)

    sub.to_csv(name, index=False)
from sklearn import model_selection
clf = xgb.XGBClassifier(objective = 'multi:softprob', nthread=1)



parameters = {

    'n_estimators': [150],

    'max_depth': [3],

    'subsample': [0.65],

    'colsample_bytree': [0.95],

    'min_child_weight': [1],

}



clf = model_selection.GridSearchCV(clf, parameters, n_jobs=4, verbose=1, scoring='neg_log_loss', refit=True)  
clf.fit(X, y);
-clf.best_score_
sub(clf.best_estimator_)