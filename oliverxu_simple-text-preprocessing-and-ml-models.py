# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from nltk.tokenize import word_tokenize

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder

from collections import defaultdict

from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection, naive_bayes, svm

from sklearn.metrics import accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Corpus = pd.read_csv('../input/nlp-getting-started/train.csv')
np.random.seed(500)
# Step - a : Remove blank rows if any.

Corpus['text'].dropna(inplace=True)

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently

Corpus['text'] = [entry.lower() for entry in Corpus['text']]

# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words

Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun

tag_map = defaultdict(lambda : wn.NOUN)

tag_map['J'] = wn.ADJ

tag_map['V'] = wn.VERB

tag_map['R'] = wn.ADV

for index,entry in enumerate(Corpus['text']):

    # Declaring Empty List to store the words that follow the rules for this step

    Final_words = []

    # Initializing WordNetLemmatizer()

    word_Lemmatized = WordNetLemmatizer()

    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.

    for word, tag in pos_tag(entry):

        # Below condition is to check for Stop words and consider only alphabets

        if word not in stopwords.words('english') and word.isalpha():

            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])

            Final_words.append(word_Final)

    # The final processed set of words for each iteration will be stored in 'text_final'

    Corpus.loc[index,'text_final'] = str(Final_words)
Corpus.head()
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['target'],test_size=0.3)
Encoder = LabelEncoder()

Train_Y = Encoder.fit_transform(Train_Y)

Test_Y = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)

Tfidf_vect.fit(Corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)

Test_X_Tfidf = Tfidf_vect.transform(Test_X)
# fit the training dataset on the NB classifier

Naive = naive_bayes.MultinomialNB()

Naive.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset

predictions_NB = Naive.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy

print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
# Classifier - Algorithm - SVM

# fit the training dataset on the classifier

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset

predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy

print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)