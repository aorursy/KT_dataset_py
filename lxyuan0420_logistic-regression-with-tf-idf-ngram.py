import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print('train shape:{}, test shape:{}'.format(train.shape,test.shape))
train.head(5)
train.info()
train_copy = train.copy()

labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']

train_copy['positiveCount'] = train_copy[labels].sum(axis=1)

train_copy['positiveCount'].value_counts()
print('Label Cardinality:',train_copy.positiveCount.sum(axis=0)/len(train_copy))
ax = train_copy.positiveCount.value_counts().plot(kind='bar', rot=0);

ax.set_xlabel('Number of labels');

ax.set_ylabel('Number of comments');

ax.set_title('Number of labels each comment has');
vectorizer = TfidfVectorizer(analyzer='word',

                            stop_words='english',

                            ngram_range=(1, 3),

                            max_features=30000,

                            sublinear_tf=True)

X_train = vectorizer.fit_transform(train.comment_text)

X_test = vectorizer.transform(test.comment_text)

Y_train = train[labels]
submission = pd.DataFrame.from_dict({'id': test['id']})



scores = []



for label in labels:

    #build classifier

    LR = LogisticRegression(solver='saga', n_jobs=-1, C=0.5)

    

    #compute cv score

    cv_score = np.mean(cross_val_score(LR, X_train, Y_train[label], cv=3, n_jobs=-1, scoring='roc_auc'))

    scores.append(cv_score)

    print("CV score for class {} is {}".format(label, cv_score))

    

    #re-learn & predict

    LR.fit(X_train, Y_train[label])  

    submission[label] = LR.predict_proba(X_test)[:, 1] #predict

    

print("Average CV scores: {}".format(np.mean(scores)))
submission.to_csv('submission_8-Tfidf_Ngram_LR.csv', index=False)