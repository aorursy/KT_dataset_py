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
#Function for showing the confusion Matrix

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    See full source and example: 

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
#Function for Cleaning and preprocessing data



def cleanse_data(text_messages):

    message_corpus=[]

    ps = PorterStemmer()

    stopwords_val = stopwords.words('english')

    

    for j in range(0, len(text_messages)):

        text = re.sub('[^a-zA-Z]', ' ', text_messages['review'][j])

        text = text.lower()

        text = text.split()

        text = [ps.stem(word) for word in text if not word in stopwords_val]

        text = ' '.join(text)

        message_corpus.append(text)

    

    return message_corpus
#Importing the libraries that are required for loading, cleaning and pre-processing data

import pandas as pd

import re



from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
#Reading the train file

train_messages = pd.read_csv('/kaggle/input/us-nlp-practicum-2020-a/imdb_train.csv', sep=',')



train_messages.head()
# Cleaning and Pre processing train data

train_corpus = cleanse_data(train_messages)

len(train_corpus)
#Use TF-IDF vectorizer to get the vector that will be input for the algo

from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(ngram_range=(2,3), max_features=30000)

X = cv.fit_transform(train_corpus).toarray()

y=train_messages['sentiment'].values

from sklearn.model_selection import train_test_split

#test size is selected as 20% of whole data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)



#Using Naive Bayes to find the model

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train, y_train)

#Use the model to predict with test set

y_pred=model.predict(X_test)
# to compare y_pred and y_test we will be using confusion matrix



from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

confustion_metrics = confusion_matrix(y_test, y_pred)



print(confustion_metrics)
# Accuracy Score

accuracy_value = accuracy_score(y_test,y_pred)



accuracy_value
# Classification Report

report=classification_report(y_test,y_pred)

print(report)
#Reading the train file

test_messages = pd.read_csv('/kaggle/input/us-nlp-practicum-2020-a/imdb_test.csv', sep=',')



test_messages.head()
# Cleaning and Pre processing train data

test_corpus = cleanse_data(test_messages)

cv = TfidfVectorizer(ngram_range=(2,3), max_features=30000)

X_final = cv.fit_transform(test_corpus).toarray()

y_final_pred=model.predict(X_final)





#Create the file

id =test_messages["id"].values

final_values = pd.DataFrame({'id':id,'sentiment':y_final_pred})

final_values.head()
filename = 'final_submission_AnuTennyson.csv'



final_values.to_csv(filename,index=False)



print('Saved file: ' + filename)