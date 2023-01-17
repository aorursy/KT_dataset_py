import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings("ignore")

import re, string

from scipy.special import softmax

from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_union

from scipy.sparse import hstack
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip').fillna(' ')

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip').fillna(' ')
train_text = train['comment_text']

test_text = test['comment_text']

all_text = pd.concat([train_text, test_text])

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
word_vec = TfidfVectorizer(

                    sublinear_tf=True,

                    strip_accents='unicode',

                    analyzer='word',

                    token_pattern=r'\w{1,}',

                    ngram_range=(1, 2),

                    max_features=30000)



char_vec = TfidfVectorizer(

                    sublinear_tf=True,

                    strip_accents='unicode',

                    analyzer='char',

                    ngram_range=(1, 3),

                    max_features=30000)



vec1 = make_union(word_vec, char_vec)

vec1.fit(all_text)

train_features = vec1.transform(train_text)

test_features = vec1.transform(test_text)
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): 

    return re_tok.sub(r' \1 ', s).split()



vec2 = TfidfVectorizer(ngram_range=(1,2), 

                      tokenizer=tokenize,

                      min_df=3, 

                      max_df=0.9, 

                      strip_accents='unicode', 

                      use_idf=1,

                      smooth_idf=1, 

                      sublinear_tf=1)

train_term_doc = vec2.fit_transform(train_text)

test_term_doc = vec2.transform(test_text)
# weighted logistic regression

scores1 = []

sub1 = pd.DataFrame.from_dict({'id': test['id']})

for class_name in class_names:

    train_target = train[class_name]

    classifier = LogisticRegression(C = 0.1, solver='saga', class_weight='balanced')



    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))

    scores1.append(cv_score)

    print('CV score for class {} is {:.4%}'.format(class_name, cv_score))



    classifier.fit(train_features, train_target)

    sub1[class_name] = classifier.predict_proba(test_features)[:, 1]



#print('Total WLR CV score is {:.4%}'.format(np.mean(scores1)))
# nb-svm

train_x = train_term_doc

test_x = test_term_doc



def pr(y_i, y):

    p = train_x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)



def get_mdl(y):

    y = y.values

    r = np.log(pr(1,y) / pr(0,y))

    m = LogisticRegression(C=4, solver='liblinear', dual=True)

    x_nb = train_x.multiply(r)

    return m.fit(x_nb, y), r



scores2 = []

sub2 = pd.DataFrame.from_dict({'id': test['id']})

for i, j in enumerate(class_names):

    m,r = get_mdl(train[j])

    sub2[j] = m.predict_proba(test_x.multiply(r))[:,1]

    y_pred = m.predict_proba(train_x.multiply(r))[:,1]

    print('fit score for class {} is {:.4%}'.format(j, roc_auc_score(train_target, y_pred)))

    scores2.append(roc_auc_score(train_target, y_pred))
sub = sub1.copy()

for i, j in enumerate(class_names):

    pb = np.array([scores1[i], scores2[i]])

    weights = lambda x: x/sum(x)

    w = weights(pb)

    print(w)

    sub[j] = sub1[j]*w[0] + sub2[j]*w[1]
sub.to_csv('submission.csv', index=False)