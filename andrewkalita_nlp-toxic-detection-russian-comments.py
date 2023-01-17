import pandas as pd

import numpy as np



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import LinearSVC

from sklearn import model_selection

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.pipeline import Pipeline

from sklearn import metrics



import nltk

from nltk.stem.snowball import SnowballStemmer



import matplotlib.pyplot as plt

import seaborn as sns

import re
nltk.download('averaged_perceptron_tagger_ru')
dfComments = pd.read_csv('../input/russian-language-toxic-comments/labeled.csv')

dfComments.head(10)
dfComments.tail(10)
desc = dfComments.groupby('toxic').describe()



plt.bar('0', desc['comment']['count'][0], label="Non toxical comment", color='green')

plt.bar('1', desc['comment']['count'][1], label="Toxical comment", color='red')

plt.legend()

plt.ylabel('Number of comments')

plt.title('Comment groups')

plt.show()



print('Comment description\n')

print(desc)

print()

print(dfComments.describe())
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))



ax1.hist(dfComments[dfComments['toxic']==0]['comment'].str.len() ,color='green')

ax1.set_title('non toxic')



ax2.hist(dfComments[dfComments['toxic']==1]['comment'].str.len() ,color='red')

ax2.set_title('toxic')



fig.suptitle('Characters in comments')

plt.show()
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))



ax1.hist(dfComments[dfComments['toxic']==0]['comment'].str.split().map(lambda x: len(x)) ,color='green')

ax1.set_title('non toxic')



ax2.hist(dfComments[dfComments['toxic']==1]['comment'].str.split().map(lambda x: len(x)) ,color='red')

ax2.set_title('toxic')



fig.suptitle('Words in comments')

plt.show()
text = np.array(dfComments.comment.values)

target = dfComments.toxic.astype(int).values
def upperCaseRate(string):

    "Returns percentage of uppercase letters in the string"

    return np.array(list(map(str.isupper, string))).mean()
upcaseRate = list(map(upperCaseRate, dfComments.comment.values))
def cleanText(string):

    """This function deletes all symbols except Cyrilic and Base Latin alphabet,

    stopwords, functional parts of speech. Returns string of words stem."""

    # Common cleaning

    string = string.lower()

    string = re.sub(r"http\S+", "", string)

    string = str.replace(string,'Ё','е')

    string = str.replace(string,'ё','е')

    prog = re.compile('[А-Яа-яA-Za-z]+')

    words = prog.findall(string.lower())

    

    # Word Cleaning

    ## Stop Words

    stopwords = nltk.corpus.stopwords.words('russian')

    words = [w for w in words if w not in stopwords]

    ## Cleaning functional POS (Parts of Speech)

    functionalPos = {'CONJ', 'PRCL'}

    words = [w for w, pos in nltk.pos_tag(words, lang='rus') if pos not in functionalPos]

    ## Stemming

    stemmer = SnowballStemmer('russian')

    return ' '.join(list(map(stemmer.stem, words)))
%%time

text = list(map(cleanText, text))
X_train, X_test, y_train, y_test = train_test_split(text, target, test_size=.3, stratify=target, shuffle = True, random_state=0)

print('Dim of train:', len(X_train), '\tTarget rate: {:.2f}%'.format(y_train.mean()))

print("Dim of test:", len(X_test), '\tTarget rate: {:.2f}%'.format(y_test.mean()))
clf_pipeline = Pipeline(

            [("vectorizer", TfidfVectorizer()), # Prod feature: tokenizer=cleanText

            ("classifier", LinearSVC())]

        )



clf_pipeline.fit(X_train, y_train)
cm = metrics.confusion_matrix(y_test, clf_pipeline.predict(X_test))



def plotConfusionMatrix(cm):

    fig = plt.figure(figsize=(7,7))

    sns.heatmap(cm, annot=True, fmt="d")

    plt.title('Confusion Matrix')

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return None



plotConfusionMatrix(cm)
print(metrics.classification_report(y_test, clf_pipeline.predict(X_test)))

f1_base = metrics.f1_score(y_test, clf_pipeline.predict(X_test))
print('\n'.join(clf_pipeline.get_params().keys()))
parameters = {'vectorizer__max_features': (10**3, 10**4),

              'vectorizer__ngram_range': ((1, 2),(2, 3)),

              'classifier__penalty': ('l1','l2'),

              'classifier__C': (range(1,10,2))

             }
%%time

gs_clf = GridSearchCV(clf_pipeline, parameters, scoring='f1', cv = 4, n_jobs=-1)

gs_clf.fit(X_train, y_train)
print(metrics.classification_report(y_test, gs_clf.predict(X_test)))

f1_gsLSVC = metrics.f1_score(y_test, gs_clf.predict(X_test))
parameters = { #'vectorizer__max_features': (10**2, 10**3),

              'vectorizer__ngram_range': [(1, 2),(1, 3)],

              'vectorizer__min_df': [0.,.2,.4,.6,.8,1],

              'classifier__penalty': ('l1','l2'),

              'classifier__C': (range(1,10,2)),

             }
clf_pipeline_LogitReg = Pipeline(

            [("vectorizer", TfidfVectorizer()),

            ("classifier", LogisticRegression())]

        )
def plotROC(y_test, probs, titl=''):

    if titl!='':

        titl = ' ('+titl+')' 

    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)

    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic'+titl)

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    return None
%%time

rndgs_clf_LogitReg = RandomizedSearchCV(clf_pipeline_LogitReg, parameters, scoring='f1', cv = 4, n_jobs=-1)

rndgs_clf_LogitReg.fit(X_train, y_train)



probs = rndgs_clf_LogitReg.predict_proba(X_train)[:,1]

plotROC(y_train, probs, 'Train')



probs = rndgs_clf_LogitReg.predict_proba(X_test)[:,1]

plotROC(y_test, probs, 'Test')
plotConfusionMatrix(metrics.confusion_matrix(y_test, rndgs_clf_LogitReg.predict(X_test)))
print(metrics.classification_report(y_test, rndgs_clf_LogitReg.predict(X_test)))

f1_rndLogR = metrics.f1_score(y_test, rndgs_clf_LogitReg.predict(X_test))
%%time

parameters = {'vectorizer__max_features': (10**2, 10**3),

              'vectorizer__ngram_range': [(1, 2),(1, 3)],

              'vectorizer__min_df': [0.,.2,.4,.6,.8,1],

              'classifier__penalty': ('l1','l2'),

              'classifier__C': (range(1,10,2)),

             }



clf_pipeline_LogitReg = Pipeline(

            [("vectorizer", TfidfVectorizer()),

            ("classifier", LogisticRegression())]

        )



rndgs_clf_LogitReg = RandomizedSearchCV(clf_pipeline_LogitReg, parameters, scoring='f1', cv = 4, n_jobs=-1)

rndgs_clf_LogitReg.fit(X_train, y_train)



probs = rndgs_clf_LogitReg.predict_proba(X_train)[:,1]

plotROC(y_train, probs, 'Train')



probs = rndgs_clf_LogitReg.predict_proba(X_test)[:,1]

plotROC(y_test, probs, 'Test')
print(metrics.classification_report(y_test, rndgs_clf_LogitReg.predict(X_test)))

f1_rndLogR_2 = metrics.f1_score(y_test, rndgs_clf_LogitReg.predict(X_test))
pd.DataFrame([f1_base, f1_gsLSVC, f1_rndLogR, f1_rndLogR_2], index=['BaseLine', 'GS_LSVC', 'rndGS_LogR', 'rndGS_LogR_Adj'], columns=['f1 score'])