import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv', nrows=10000)

test = pd.read_csv('../input/test.csv')



test.head(10)



# adding preprocessing from this kernel: https://www.kaggle.com/taindow/simple-cudnngru-python-keras

punct_mapping = {"_":" ", "`":" "}

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])    

    for p in punct:

        text = text.replace(p, f' {p} ')     

    return text

train['comment_text'] = train['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

test['comment_text'] = test['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

for col in identity_columns + ['target']:

    train[col] = np.where(train[col] >= 0.5, True, False)
from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(train['comment_text'], 

                                                    train['target'], 

                                                    random_state=0)
from sklearn.feature_extraction.text import TfidfVectorizer





vect = TfidfVectorizer(min_df=3,ngram_range=(1, 2), max_features=30000).fit(X_train)





X_train_vectorized = vect.transform(X_train)

from sklearn.kernel_ridge import KernelRidge

KRR = KernelRidge()#alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



KRR.fit(X_train_vectorized, y_train)



predictions = KRR.predict(vect.transform(X_test))



from sklearn.metrics import roc_auc_score

from sklearn.metrics import mean_squared_error

#print (mean_squared_error(y_test, predictions))

print (roc_auc_score(y_test, predictions))
submission = pd.read_csv('../input/sample_submission.csv', index_col='id')

submission['prediction'] = KRR.predict(vect.transform(test['comment_text']))

submission.to_csv('submission.csv')