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
data = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')
data.head()
sub_data = data.sample(100000)
sub_data.shape
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

sub_data['question_text'] = sub_data['question_text'].str.lower().str.replace('[^a-z\s]', '')
train_x, test_x, train_y, test_y = train_test_split(sub_data[['question_text']],
                                                    sub_data['target'],
                                                   test_size=0.2, random_state=1)
train_x
train_x.shape
# min_df: only those terms whose total frequency is greater than 10 will be picked
import nltk
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['app', 'mobile', 'get', 'would', 'best'])
vectorizer = CountVectorizer(min_df=10, stop_words=stopwords).fit(train_x['question_text'])
train_dtm = vectorizer.transform(train_x['question_text'])
test_dtm = vectorizer.transform(test_x['question_text'])
df_train_dtm = pd.DataFrame(train_dtm.toarray(), columns=vectorizer.get_feature_names())
df_test_dtm = pd.DataFrame(test_dtm.toarray(), columns=vectorizer.get_feature_names())
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(layers.Dense(units=64, input_shape=(df_train_dtm.shape[1],), activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(df_train_dtm, train_y, validation_split=0.2, epochs=5, batch_size=1024, verbose=1)
import matplotlib.pyplot as plt
plt.plot(history.history['val_loss'])
pred_class = model.predict_classes(df_test_dtm.values)
from sklearn.metrics import accuracy_score
accuracy_score(test_y, pred_class)