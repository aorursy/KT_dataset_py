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
import ssl



import pandas as pd



df = pd.read_csv('/kaggle/input/qwerty/train.csv')

df.head(5)



# Importing necessary libraries

import string

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import nltk





lemmatiser = WordNetLemmatizer()





# Defining a module for Text Processing

def text_process(tex):

    # 1. Removal of Punctuation Marks

    nopunct = [char for char in tex if char not in string.punctuation]

    nopunct = ''.join(nopunct)

    # 2. Lemmatisation

    a = ''

    i = 0

    for i in range(len(nopunct.split())):

        b = lemmatiser.lemmatize(nopunct.split()[i], pos="v")

        a = a + b + ' '

    # 3. Removal of Stopwords

    return [word for word in a.split() if word.lower() not

            in stopwords.words('english')]





# Importing necessary libraries

from sklearn.preprocessing import LabelEncoder



y = df['author']

labelencoder = LabelEncoder()

y = labelencoder.fit_transform(y)



# Importing necessary libraries

from PIL import Image

from wordcloud import WordCloud

import matplotlib.pyplot as plt



X = df['text']

wordcloud1 = WordCloud().generate(X[0])  # for EAP

wordcloud2 = WordCloud().generate(X[1])  # for HPL

wordcloud3 = WordCloud().generate(X[3])  # for MWS

print(X[0])

print(df['author'][0])

plt.imshow(wordcloud1, interpolation='bilinear')

plt.show()

print(X[1])

print(df['author'][1])

plt.imshow(wordcloud2, interpolation='bilinear')

plt.show()

print(X[3])

print(df['author'][3])

plt.imshow(wordcloud3, interpolation='bilinear')

plt.show()



# Importing necessary libraries

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



# 80-20 splitting the dataset (80%->Training and 20%->Validation)

X_train, X_test, y_train, y_test = train_test_split(X, y

                                                    , test_size=0.2, random_state=1234)

# defining the bag-of-words transformer on the text-processed corpus # i.e., text_process() declared in II is

# executed...

bow_transformer = CountVectorizer(analyzer=text_process).fit(X_train)

# transforming into Bag-of-Words and hence textual data to numeric..

text_bow_train = bow_transformer.transform(X_train)  # ONLY TRAINING DATA

# transforming into Bag-of-Words and hence textual data to numeric..

text_bow_test = bow_transformer.transform(X_test)  # TEST DATA



# Importing necessary libraries

from sklearn.naive_bayes import MultinomialNB



# instantiating the model with Multinomial Naive Bayes..

model = MultinomialNB()

# training the model...

model = model.fit(text_bow_train, y_train)



model.score(text_bow_train, y_train)

model.score(text_bow_test, y_test)



# Importing necessary libraries

from sklearn.metrics import classification_report



# getting the predictions of the Validation Set...

predictions = model.predict(text_bow_test)

# getting the Precision, Recall, F1-Score

print(classification_report(y_test, predictions))



# Importing necessary libraries

from sklearn.metrics import confusion_matrix

import numpy as np

import itertools

import matplotlib.pyplot as plt





def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0])

            , range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    cm = confusion_matrix(y_test, predictions)

    plt.figure()

    plot_confusion_matrix(cm, classes=[0, 1, 2], normalize=True,

                          title='Confusion Matrix')