# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')
print('DataFrame Train:')

print(df_train.isnull().any())

print('\n')

print('DataFrame Test:')

print(df_test.isnull().any())
df_test = df_test.fillna('unknown')
col = np.array(df_train.columns)

col = col[2:]

print(col)
print('Dataframe Train:')

for c in col:

    print("The dataframe has '{1}' of comments '{0}' of the total '{2}'.".format(c, 

                                                                                 df_train[c].sum(), 

                                                                                 len(df_train)))
comment_text_all = pd.concat([df_train['comment_text'], df_test['comment_text']],axis=0)
nrow_train = df_train.shape[0]

print(nrow_train)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode')
train_test_comment_text = vectorizer.fit_transform(comment_text_all)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

from sklearn.model_selection import train_test_split
y = df_train[col]

y.head(5)
predict = np.zeros((df_test.shape[0], len(col)))

predict.shape
loss = []

accuracy = []

for num, index in enumerate(col):

    x_train, x_test, y_train, y_test = train_test_split(train_test_comment_text[:nrow_train], y[index])

    print("Fit: {}".format(index))

    model = LogisticRegression(C=5)

    model.fit(x_train, y_train)

    predict[:,num] = model.predict_proba(train_test_comment_text[nrow_train:])[:,1]    

    predict_01 = model.predict_proba(x_test)[:,1]    

    logloss = log_loss(y_test, predict_01)

    print('log loss:', logloss)

    predict_02 = model.predict(x_test)

    acc = np.mean(predict_02 == y_test)

    print('accuracy:', acc)

    loss.append(logloss)

    accuracy.append(acc)

    print('\n')

print('mean column-wise log loss:', np.mean(loss))

print('mean column-wise accuracy:', np.mean(accuracy))
submid = pd.DataFrame({'id': submission["id"]})

submission = pd.concat([submid, pd.DataFrame(predict, columns = col)], axis=1)

submission.to_csv('submission.csv', index=False)