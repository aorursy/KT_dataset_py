# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/dma-fall19/yelp_train.csv')

df_test = pd.read_csv('/kaggle/input/dma-fall19/yelp_test.csv')
df_train
df_train.groupby('is_good_rating').count()
majority_train = 159858/(80142 + 159858)

print("Majority benchmark: " + str(majority_train))
from sklearn.linear_model import LogisticRegression as lm

from sklearn.feature_extraction.text import CountVectorizer



vector = CountVectorizer()

Phi_train = vector.fit_transform(df_train['text'])

# Phi_test = vector.transform(df_test['text'])

Y_train = df_train['is_good_rating']



model = lm()

model.fit(Phi_train, Y_train)

training_accuracy = model.score(Phi_train, Y_train)

# test_accuracy = model.score(Phi_test, Y_test)



print('Training Score: ' + str(training_accuracy))
Phi_test = vector.transform(df_test['text'])

model.predict(Phi_test)
submission = pd.DataFrame(index=df_test.review_id)

submission['is_good_rating'] = model.predict(Phi_test)
submission
submission.reset_index().to_csv('submission.csv', index=False)