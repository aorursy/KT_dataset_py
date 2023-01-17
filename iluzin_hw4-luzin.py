# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', low_memory=False)

df_train.set_index('id', inplace=True)

df_train.head()
from sklearn.feature_extraction.text import CountVectorizer



def texts_vectorization(vectorizer, texts_1, texts_2):

    vectorizer.fit(pd.concat([texts_1.fillna(''), texts_2.fillna('')]))

    vect_1 = vectorizer.transform(texts_1.fillna(''))

    vect_2 = vectorizer.transform(texts_2.fillna(''))

    return vect_1, vect_2



vect = CountVectorizer(analyzer='char', ngram_range=(1, 1), min_df=0.001, binary=True)
bag_question1, bag_question2 = texts_vectorization(vect, df_train.question1, df_train.question2)

bag_question1.shape, bag_question2.shape
X = bag_question1 != bag_question2

X.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, df_train.is_duplicate, test_size=0.1)
from sklearn.neural_network import MLPClassifier



clf = MLPClassifier(hidden_layer_sizes=(100, 5), activation='logistic', max_iter=1, warm_start=True, verbose=True)
from sklearn.metrics import log_loss



score = np.inf

while score > 0.58:

    clf.fit(X_train, y_train)

    score = log_loss(y_test, clf.predict_proba(X_test))

    print('score = {}'.format(score))
df_test = pd.read_csv('../input/test.csv', low_memory=False)

df_test.set_index('test_id', inplace=True)

df_test.head()
bag_out_question1 = vect.transform(df_test.question1.fillna(''))

bag_out_question2 = vect.transform(df_test.question2.fillna(''))

bag_out_question1.shape, bag_out_question2.shape
X_out = bag_out_question1 != bag_out_question2

X_out.shape
y_out_proba = clf.predict_proba(X_out)

y_out_proba.shape
df_submission = df_test.drop(df_test.columns, axis='columns')

df_submission['is_duplicate'] = y_out_proba[:, 1]

df_submission.to_csv('submission.csv')