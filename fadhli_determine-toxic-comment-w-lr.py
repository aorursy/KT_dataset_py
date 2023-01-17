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
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.head()
train_df.shape
df = pd.concat([train_df['comment_text'], test_df['comment_text']], axis=0)

df = df.fillna("unknown")

nrow_train = train_df.shape[0]
nrow_train
from sklearn.feature_extraction.text import TfidfVectorizer



vect = TfidfVectorizer(stop_words='english').fit(df)

bag = vect.transform(df)
preds
col_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



preds = np.zeros((test_df.shape[0], len(col_names)))
list(enumerate(col_names))
bag[:nrow_train], train_df['toxic']
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss



loss = []



for i, j in enumerate(col_names):

    print('===Fit ' + j)

    model = LogisticRegression()

    model.fit(bag[:nrow_train], train_df[j])

    preds[:,i] = model.predict_proba(bag[nrow_train:])[:,1]

    

    pred_train = model.predict_proba(bag[:nrow_train])[:,1]

    print('log loss:', log_loss(train_df[j], pred_train))

    loss.append(log_loss(train_df[j], pred_train))

    

print('mean column-wise log loss:', np.mean(loss))
submid = pd.DataFrame({'id': test_df["id"]})

submission = pd.concat([submid, pd.DataFrame(preds, columns = col_names)], axis=1)

submission.to_csv('submission.csv', index=False)
pd.DataFrame(preds, columns = col_names)