# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd



train = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')

test = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')

sample_submission = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/sample_submission.csv')
X = train['question_text']

y = train['target'] 
# X_words = [i.split() for i in X]
# unique_words = set()



# for li in X_words:

#     for i in li:

#         if i not in unique_words:

#             unique_words.add(i)
# from nltk import word_tokenize



# X_word_tokenized = list()



# for que in X:

#     X_word_tokenized.append(word_tokenize(que))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=41)
from sklearn.feature_extraction.text import TfidfVectorizer



tf=TfidfVectorizer()

train_tf= tf.fit_transform(X_train)

test_tf= tf.transform(X_test)
# from sklearn.neighbors import KNeighborsClassifier



# model = KNeighborsClassifier(n_neighbors=3)



# # Train the model using the training sets

# model.fit(train_tf,y_train)
# #Predict Output

# predicted_y = model.predict(test_tf)

# model.score(predicted_y,y_test)
# from sklearn.tree import DecisionTreeClassifier



# clf = DecisionTreeClassifier(random_state=0)

# clf.fit(train_tf,y_train)
# predicted_y = clf.predict(test_tf)

# model.score(predicted_y,y_test)
from sklearn.svm import LinearSVC



# Create the LinearSVC model

model = LinearSVC(random_state=1, dual=False)

# Fit the model

model.fit(train_tf, y_train)



# Uncomment and run to see model accuracya

print(f'Model test accuracy: {model.score(test_tf, y_test)*100:.3f}%')
X_val = test['question_text']
val_tf= tf.transform(X_val)
pred = model.predict(val_tf)
sample_submission['prediction'] = pred
sample_submission.to_csv('submission.csv')
sample_submission