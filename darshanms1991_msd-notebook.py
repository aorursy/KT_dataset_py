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
train_df = pd.read_json("../input/train.json")

categories = sorted(train_df['cuisine'].astype("category").unique())

train_df['cuisine'] = train_df['cuisine'].astype("category").cat.codes

train_df['ingredients'] = train_df['ingredients'].apply( lambda x: " ".join(x))
train_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer

train_docs = train_df['ingredients']

train_tf = TfidfVectorizer(lowercase=False, preprocessor=None)

train_tf.fit(train_docs)

train_X= train_tf.transform(train_docs)

train_Y = train_df['cuisine']
from sklearn import svm

clf = svm.SVC(decision_function_shape='ovr')

clf.fit(train_X, train_Y) 

    
test_df = pd.read_json("../input/test.json")

test_df['ingredients'] = test_df['ingredients'].apply( lambda x: " ".join(x))

test_docs = test_df['ingredients']

test_vecs = train_tf.transform(test_docs)

Y = clf.decision_function(test_vecs)
test_df['pred'] = np.argmax(Y, axis=1)

test_df.head()
test_df['cuisine'] = test_df['pred'].apply(lambda x : categories[x])
test_df.to_csv('submission.csv',columns = ['id','cuisine'],index = False)