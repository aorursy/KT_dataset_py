# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import classification_report, accuracy_score

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_json('../input/whats-cooking-kernels-only/train.json')

df.head()
df['ingredient_text'] = df['ingredients'].apply(lambda x: ' '.join([i.replace(' ','_') for i in x]))

df.head()
df_train, df_test = train_test_split(df,stratify=df['cuisine'])
nb = BernoulliNB()

le = LabelEncoder()



#tv = TfidfVectorizer(max_features=500)

cv = CountVectorizer()



X_train = cv.fit_transform(df_train['ingredient_text'])

y_train = le.fit_transform(df_train['cuisine'])



nb.fit(X_train,y_train)

y_train_pred = nb.predict(X_train)



X_test = cv.transform(df_test['ingredient_text'])



df_test['pred'] = le.inverse_transform(nb.predict(X_test))



print(accuracy_score(y_train,y_train_pred))

print(classification_report(df_test['cuisine'],df_test['pred']))
submission = pd.read_json('../input/whats-cooking-kernels-only/test.json')

submission['ingredient_text'] = submission['ingredients'].apply(lambda x: ' '.join([i.replace(' ','_') for i in x]))

submission.head()
X_sub = cv.transform(submission['ingredient_text'])



y_sub = nb.predict(X_sub)



submission['cuisine'] = le.inverse_transform(y_sub)

submission[['id','cuisine']].head()
submission[['id','cuisine']].to_csv('submission.csv',index=False)