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

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
df_train = pd.read_csv('/kaggle/input/dma-fall19/yelp_train.csv')

df_test = pd.read_csv('/kaggle/input/dma-fall19/yelp_test.csv')
df_train = df_train.dropna()
x_train = df_train[['cool', 'funny', 'useful', 'user_average_stars',

                    'user_review_count', 'business_latitude', 'business_longitude',

                    'business_review_count', 'business_average_stars']]

x_test = df_test[['cool', 'funny', 'useful', 'user_average_stars',

                    'user_review_count', 'business_latitude', 'business_longitude',

                    'business_review_count', 'business_average_stars']]

y_train = df_train['is_good_rating']
clf = DecisionTreeClassifier()

clf.fit(x_train, y_train)

accuracy_score(clf.predict(x_train), y_train)
clf.predict(x_test)
submission = pd.DataFrame(index=df_test.review_id)

submission['is_good_rating'] = clf.predict(x_test)
submission.reset_index().to_csv('submission.csv', index=False)