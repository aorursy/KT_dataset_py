import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import re

from scipy import sparse



from sklearn.model_selection import train_test_split 

from sklearn.model_selection import cross_val_score 

from sklearn.model_selection import StratifiedKFold 

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline



from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.cluster import KMeans





from nltk.corpus import stopwords

from gensim.models.word2vec import Word2Vec

from gensim.models.doc2vec import Doc2Vec, TaggedDocument





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('/kaggle/input/ocrv-intent-classification/train.csv', index_col='id')

df_test = pd.read_csv('/kaggle/input/ocrv-intent-classification/test.csv', index_col='id')
print(df_train.info())

print(df_train.head(10))

print(df_train.describe())
print(df_test.info())

print(df_test.head(10))

print(df_test.describe())
print('Количество строк с пропусками в обучащих данных: {}'.format(df_train['text'].isnull().sum()))

print('Количество строк с пропусками в тестовых данных: {}'.format(df_test['text'].isnull().sum()))
df_train['label'].value_counts()
sns.set()

plt.figure(figsize = (12,8))

_=sns.countplot(df_train['label'])

_.set_xticklabels(_.get_xticklabels(),rotation=45, horizontalalignment='right', fontweight='light', fontsize='large')

plt.title('Распределение классов в обучающей выборке', fontsize=18)

plt.xlabel('')

plt.ylabel('Количество запросов')

plt.show()
f, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

sns.distplot(df_train['text'].apply(lambda x: len(str(x))), hist=True, ax=axes[0])

axes[0].set_title('Распределение длин текстов в обучающей выборке')

axes[0].set_xlabel('Количество символов')

sns.distplot(df_test['text'].apply(lambda x: len(str(x))), hist=True, ax=axes[1])

axes[1].set_title('Распределение длин текстов в тестовой выборке')

axes[1].set_xlabel('Количество символов')

plt.tight_layout()

plt.show()
plt.figure(figsize = (12,8))

_=sns.boxplot(y = df_train['text'].apply(lambda x: len(str(x))), x='label', data=df_train)

_.set_xticklabels(_.get_xticklabels(),rotation=45, horizontalalignment='right', fontweight='light', fontsize='large')

plt.title('Распределение длин запросов в зависимости от категории', fontsize=16)

plt.xlabel('')

plt.ylabel('Количество запросов')

plt.show()
pd.set_option('display.max_colwidth', -1)
df_train[df_train['text'].isnull()].label.value_counts()
df_train[df_train['text'].duplicated()].count()
df_test[df_test['text'].duplicated()].count()
f, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

df_train.groupby('text')['text'].count().sort_values(ascending=False)[:15].plot(kind='bar', ax=axes[0])

axes[0].set_title('Дубликаты в обучающих данных')

axes[0].set_xlabel('')

df_test.groupby('text')['text'].count().sort_values(ascending=False)[:15].plot(kind='bar', ax=axes[1])

axes[1].set_title('Дубликаты в тестовых данных')

axes[1].set_xlabel('')

plt.tight_layout()

plt.show()
(df_train.groupby('text')['label'].nunique()>1).sum()
df_train['text'] = df_train['text'].fillna(' ')

df_test['text'] = df_test['text'].fillna(' ')
def clean(string):

    '''Clean text from special charachters and lowercase string'''

    string = str(string)

    clean_text = re.sub(r'[,.!-?_]+', ' ', string)

    clean_text = ' '.join(re.findall(r"\w+", string.lower()))

    return clean_text 
def make_clean(df):

    '''Create a column with cleaned text'''

    df['text_clean'] = df['text'].apply(clean)

    return df
stop_words = stopwords.words('russian')

words_add=['который', 'очень', 'еще', 'это', 'здравствовать','здрасте','здраствуйте']

for word in words_add:

    stop_words.append(word)

words_excep = ['нет', 'другой', 'какой', 'какая', 'мой', 'как', 'для', 'без', 'на', 'где', 'ничего']

stop_words = [word for word in stop_words if word not in words_excep]

print('Число стоп-слов: {}'.format(len(stop_words)))
def del_stops(df):

    '''Create text column in df cleaned from stop words'''    

    df['text_stop'] = df['text_clean'].apply(lambda string: [word for word in string.split() if word not in stop_words])

    return df
def stem(string):

    '''Remove some word endings'''

    words = string.split() 

    res = list()

    words_stop = ['какой','какая','какое','какие', 'какую','какого','каким','каком','какому', 'каких']  # not to confuse with 'как' which may be important

    

    pattern1 = re.compile("(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$")    

    pattern2 = re.compile(u"(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я)$")   

    pattern3 = re.compile(u"((ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)|((?<=[ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|нно)))$")

    

    for word in words:   

        if re.search(pattern1, word):  #adjective

            if word not in words_stop:

                res.append(re.sub(pattern1,'', word))

            else:                 

                res.append(re.sub(pattern1,'_', word))

                

        elif re.search(pattern2, word):  #noun

            res.append(re.sub(pattern2,'', word))

            

        elif re.search(pattern3, word):  #verb

            res.append(re.sub(pattern3,'', word))

            

        else:  #anything else

            res.append(word)

            

    return res
def make_stems(df):

    '''Create text column in df with list of stemmed words'''

    df['text_stop'] = df['text_stop'].apply(lambda x: ' '.join(x))

    df['text_stem'] = df['text_stop'].apply(lambda x: stem(x))

    return df
def preprocess(df):

    '''Preprocess text data in a dataframe'''

    df_prep = df.copy()

    df_prep = make_clean(df_prep) 

    df_prep = del_stops(df_prep) 

    df_prep = make_stems(df_prep)                                                                               

        

    return df_prep
df_train = preprocess(df_train)

df_train.head(5)
df_test = preprocess(df_test)

df_test.head(5)
def make_split(test_size = 0.2):

    '''Returns X_train, X_test, y_train, y_test'''    

    X_train, X_test, y_train, y_test = train_test_split(df_train['text_stem'], df_train['label'], test_size = 0.2, random_state = 42, stratify = df_train['label'])    

    X_train_joint = X_train.apply(lambda x: ' '.join(x))

    X_test_joint = X_test.apply(lambda x: ' '.join(x))

    return X_train_joint, X_test_joint, y_train, y_test
X_train, X_test, y_train, y_test = make_split()

print(u'Размер обучающей выборки: {}'.format(X_train.shape))

print(u'Размер тестовой выборки: {}'.format(X_test.shape))
def tfidf_vectors(vectorizer, X_train, X_test, printed=True):

    '''Return tf-idf features'''

    features_train = vectorizer.fit_transform(X_train)

    features_test = vectorizer.transform(X_test)

    if printed:

        print(u'Размер словаря: {} слов'.format(features_train.shape[1]))

    return features_train, features_test
# Word Bigrams

vectorizer = TfidfVectorizer(analyzer='word', lowercase = False, ngram_range=(1,2), min_df = 1, dtype=np.float32)

features_train, features_test = tfidf_vectors(vectorizer, X_train, X_test)
# Word Trigrams

vectorizer1 = TfidfVectorizer(max_features=30000,analyzer='word', lowercase = False, ngram_range=(1,3), dtype=np.float32)

features_train1, features_test1 = tfidf_vectors(vectorizer1, X_train, X_test)
# Word Trigrams + Chars

vectorizer_word = TfidfVectorizer(max_features=1000, analyzer='word', lowercase = False, ngram_range=(1,3), dtype=np.float32)

vectorizer_char = TfidfVectorizer(max_features=40000, lowercase=False, analyzer='char', ngram_range=(3,6),dtype=np.float32)

features_train2, features_test2 = tfidf_vectors(vectorizer_word, X_train, X_test)

charfeat_train, charfeat_test = tfidf_vectors(vectorizer_char, X_train, X_test)

full_feattrain = sparse.hstack([features_train2, charfeat_train])

full_feattest= sparse.hstack([features_test2, charfeat_test])
# Just to compare with the baseline:

vectorizer_word2 = TfidfVectorizer(max_features=1000, analyzer='word', lowercase = True, ngram_range=(1,3), dtype=np.float32)

vectorizer_char2 = TfidfVectorizer(max_features=40000, lowercase= True, analyzer='char', ngram_range=(3,6),dtype=np.float32)

features_train3, features_test3 = tfidf_vectors(vectorizer_word2, df_train.loc[X_train.index,'text'], df_train.loc[X_test.index,'text'])

charfeat_train3, charfeat_test3 = tfidf_vectors(vectorizer_char2, df_train.loc[X_train.index,'text'], df_train.loc[X_test.index,'text'])

full_feattrain3 = sparse.hstack([features_train3, charfeat_train3])

full_feattest3= sparse.hstack([features_test3, charfeat_test3])
kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 42)
def cross_val(model, features_train=features_train, features_test=features_test):

    '''Train ML model and get cross val score'''

    cv_results = cross_val_score(model, features_train, y_train, scoring='f1_macro', cv=kf)

    print(cv_results)

    print('F1-score: {}'.format(cv_results.mean()))
naiv_model = MultinomialNB(0.005)

cross_val(naiv_model, features_train, features_test)
naiv_model = MultinomialNB(0.005)

cross_val(naiv_model, features_train1, features_test1)
naiv_model = MultinomialNB(0.005)

cross_val(naiv_model, full_feattrain, full_feattest)
naiv_model = MultinomialNB(0.005)

cross_val(naiv_model, full_feattrain3, full_feattest3)
lg_model = LogisticRegression(C=5, random_state = 123, solver='lbfgs', multi_class='multinomial')
cross_val(lg_model, features_train, features_test)
cross_val(lg_model, features_train1, features_test1)
cross_val(lg_model, full_feattrain, full_feattest)
lg_model = LogisticRegression(C=23, n_jobs = 5, random_state = 123, solver='lbfgs', multi_class='multinomial', warm_start=True)

cross_val(lg_model, full_feattrain3, full_feattest3)
#rf_model = RandomForestClassifier()

#cross_val(rf_model, features_train, features_test)

#cross_val(rf_model, features_train1, features_test1)

#cross_val(rf_model, full_feattrain, full_feattest)

#cross_val(rf_model, full_feattrain3, full_feattest3)
sgd_model = SGDClassifier(loss='modified_huber', penalty='l2', random_state =123, alpha=0.0001)
cross_val(sgd_model, features_train, features_test)
cross_val(sgd_model, features_train1, features_test1)
cross_val(sgd_model, full_feattrain, full_feattest)
cross_val(sgd_model, full_feattrain3, full_feattest3)
def grid_model(model, parameters, features_train=features_train, features_test=features_test):

    '''Train ML model, get predictions and score'''

    cv = GridSearchCV(model, param_grid=parameters)

    cv.fit(features_train, y_train)

    y_pred = cv.predict(features_test)

    print('Лучшие найденные параметры: {}'.format(cv.best_params_))

    print('Лучшая оценка: {}'.format(cv.best_score_))

    score = f1_score(y_test, y_pred, average='macro')

    print('F1-score of {}: {}'.format(cv, score))

    print(classification_report(y_test, y_pred))

    return y_pred
lg_model = LogisticRegression(random_state = 123, solver='lbfgs', multi_class='multinomial')

lg_parameters = [{'C':[1,5,10,15]}]

y_pred_lg = grid_model(lg_model, lg_parameters, full_feattrain, full_feattest)
lg_model = LogisticRegression(C=20,random_state = 123, n_jobs=5,solver='lbfgs', multi_class='multinomial')

cross_val(lg_model, full_feattrain, full_feattest)

lg_model.fit(full_feattrain, y_train)

y_pred = lg_model.predict(full_feattest)

f1_score(y_test, y_pred, average='macro')
sgd_model = SGDClassifier(loss='modified_huber', penalty='l2', random_state =123)

sgd_parameters = [{'alpha': np.logspace(-6, 2, 20)}]

y_pred_sgd = grid_model(sgd_model, sgd_parameters, full_feattrain, full_feattest)
sgd_model = SGDClassifier(loss='modified_huber', penalty='l2', random_state =123, alpha = 0.0001)

sgd_model.fit(full_feattrain, y_train)

y_pred = sgd_model.predict(full_feattest)

f1_score(y_test, y_pred, average='macro')
clf = OneVsRestClassifier(LogisticRegression(C=5, n_jobs=5, random_state = 123, solver='lbfgs', multi_class='multinomial'))

cross_val(clf, full_feattrain, full_feattest)

clf.fit(full_feattrain, y_train)

y_pred = clf.predict(full_feattest)

f1_score(y_test, y_pred, average='macro')
clf = OneVsRestClassifier(LogisticRegression(C=20,random_state = 123, n_jobs=5,solver='lbfgs', multi_class='multinomial'))

cross_val(clf, full_feattrain, full_feattest)

clf.fit(full_feattrain, y_train)

y_pred = clf.predict(full_feattest)

f1_score(y_test, y_pred, average='macro')
# Визуализируем матрицу столкновений в относительных величинах для каждого класса.

cnf_matrix = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cnf_matrix, index = y_test.value_counts().index, columns = y_test.value_counts().index)

df_cmr = df_cm.apply(lambda x: x/np.sum(x), axis=1)

df_cmr = df_cmr.apply(lambda x: round(x,2), axis=1)

plt.figure(figsize = (10,7))

sns.heatmap(df_cmr, annot=True)

plt.xlabel('Predicted class')

plt.ylabel('True class')

plt.show()
clf = OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='l2', random_state =123, alpha = 0.0001))

cross_val(clf, full_feattrain, full_feattest)

clf.fit(full_feattrain, y_train)

y_pred = clf.predict(full_feattest)

f1_score(y_test, y_pred, average='macro')
model_lg = OneVsRestClassifier(LogisticRegression(C=20,random_state = 123, n_jobs=5,solver='lbfgs', multi_class='multinomial'))

model_lg.fit(full_feattrain, y_train)

pred_lg = model_lg.predict(full_feattrain)

pred_lg_test = model_lg.predict(full_feattest)
model_sgd = SGDClassifier(loss='modified_huber', penalty='l2', random_state =123, alpha = 0.0001)

model_sgd.fit(full_feattrain, y_train)

pred_sgd = model_sgd.predict(full_feattrain)

pred_sgd_test = model_sgd.predict(full_feattest)
model_base = LogisticRegression(C=23, n_jobs = 5, random_state = 123, solver='lbfgs', multi_class='multinomial', warm_start=True)

model_base.fit(full_feattrain3, y_train)

pred_base = model_base.predict(full_feattrain3)

pred_base_test = model_base.predict(full_feattest3)
df_stack2 = pd.DataFrame({'pred_lg': pred_lg, 'pred_sgd': pred_sgd, 'pred_base': pred_base,'label': y_train})

print(df_stack2.shape)

print(df_stack2.head(5))

df_stack2 = pd.get_dummies(df_stack2.drop('label', axis=1))

print(df_stack2.shape)
df_stack_test2 = pd.DataFrame({'pred_lg_test': pred_lg_test, 'pred_sgd_test': pred_sgd_test, 'pred_base': pred_base_test, 'label': y_test})

print(df_stack_test2.shape)

print(df_stack_test2.head(15))

df_stack_test2 = pd.get_dummies(df_stack_test2.drop('label', axis=1))

print(df_stack_test2.shape)
X_train_stack = sparse.hstack([full_feattrain, df_stack2])

X_test_stack = sparse.hstack([full_feattest, df_stack_test2])

print(X_train_stack.shape)

print(X_test_stack.shape)
clf = OneVsRestClassifier(LogisticRegression(C=20,random_state = 123, n_jobs=5,solver='lbfgs', multi_class='multinomial'))

clf.fit(X_train_stack, y_train)

y_pred = clf.predict(X_test_stack)

f1_score(y_test, y_pred, average='macro')
vectorizer_word = TfidfVectorizer(max_features=1000, analyzer='word', lowercase = False, ngram_range=(1,3), dtype=np.float32)

vectorizer_char = TfidfVectorizer(max_features=40000, lowercase=False, analyzer='char', ngram_range=(3,6),dtype=np.float32)



features_train_all = vectorizer_word.fit_transform(df_train['text_stem'].apply(lambda x: ' '.join(x)))

features_test_all = vectorizer_word.transform(df_test['text_stem'].apply(lambda x: ' '.join(x)))



charfeat_train_all = vectorizer_char.fit_transform(df_train['text_stem'].apply(lambda x: ' '.join(x)))

charfeat_test_all = vectorizer_char.transform(df_test['text_stem'].apply(lambda x: ' '.join(x)))



full_feattrain_all = sparse.hstack([features_train_all, charfeat_train_all])

full_feattest_all = sparse.hstack([features_test_all, charfeat_test_all])
vectorizer_word = TfidfVectorizer(max_features=1000, analyzer='word', lowercase = True, ngram_range=(1,3), dtype=np.float32)

vectorizer_char = TfidfVectorizer(max_features=40000, lowercase=True, analyzer='char', ngram_range=(3,6),dtype=np.float32)



features_train_all3 = vectorizer_word.fit_transform(df_train['text'].apply(lambda x: np.str_(x)))

features_test_all3 = vectorizer_word.transform(df_test['text'].apply(lambda x: np.str_(x)))



charfeat_train_all3 = vectorizer_char.fit_transform(df_train['text'].apply(lambda x: np.str_(x)))

charfeat_test_all3 = vectorizer_char.transform(df_test['text'].apply(lambda x: np.str_(x)))



full_feattrain_all3 = sparse.hstack([features_train_all3, charfeat_train_all3])

full_feattest_all3 = sparse.hstack([features_test_all3, charfeat_test_all3])
model_lg = OneVsRestClassifier(LogisticRegression(C=20,random_state = 123, n_jobs=5,solver='lbfgs', multi_class='multinomial'))

model_lg.fit(full_feattrain_all, df_train['label'])

pred_lg = model_lg.predict(full_feattrain_all)

pred_lg_test = model_lg.predict(full_feattest_all)
model_sgd = SGDClassifier(loss='modified_huber', penalty='l2', random_state =123, alpha = 0.0001)

model_sgd.fit(full_feattrain_all, df_train['label'])

pred_sgd = model_sgd.predict(full_feattrain_all)

pred_sgd_test = model_sgd.predict(full_feattest_all)
model_base = LogisticRegression(C=23, n_jobs = 5, random_state = 123, solver='lbfgs', multi_class='multinomial', warm_start=True)

model_base.fit(full_feattrain_all3, df_train['label'])

pred_base = model_base.predict(full_feattrain_all3)

pred_base_test = model_base.predict(full_feattest_all3)
df_stack = pd.DataFrame({'pred_lg': pred_lg, 'pred_sgd': pred_sgd, 'pred_base': pred_base,'label': df_train['label']})

print(df_stack.shape)

print(df_stack.head(5))

df_stack = pd.get_dummies(df_stack.drop('label', axis=1))

print(df_stack.shape)
df_stack_test = pd.DataFrame({'pred_lg_test': pred_lg_test, 'pred_sgd_test': pred_sgd_test, 'pred_base': pred_base_test})

print(df_stack_test.shape)

print(df_stack_test.head(15))

df_stack_test = pd.get_dummies(df_stack_test)

print(df_stack_test.shape)
X_train_stack = sparse.hstack([full_feattrain_all, df_stack])

X_test_stack = sparse.hstack([full_feattest_all, df_stack_test])

print(X_train_stack.shape)

print(X_test_stack.shape)
clf = OneVsRestClassifier(LogisticRegression(C=20,random_state = 123, n_jobs=5,solver='lbfgs', multi_class='multinomial'))

clf.fit(X_train_stack, df_train['label'])

y_pred = clf.predict(X_test_stack)
df_test['label'] = np.array(y_pred)

print(df_test.head())

df_test[['label']].to_csv('submission9.csv')