import pandas as pd

import numpy as np

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

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns
np.random.seed(500)

Corpus = pd.read_csv('../input/bbc-text.csv',delimiter=',',encoding='latin-1')

Corpus.head()
Corpus.info()
sns.countplot(Corpus.category)

plt.xlabel('Category')

plt.title('CountPlot')
# 1. Removing Blank Spaces

Corpus['text'].dropna(inplace=True)

# 2. Changing all text to lowercase

Corpus['text_original'] = Corpus['text']

Corpus['text'] = [entry.lower() for entry in Corpus['text']]

# 3. Tokenization-In this each entry in the corpus will be broken into set of words

Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]

# 4. Remove Stop words, Non-Numeric and perfoming Word Stemming/Lemmenting.

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun

tag_map = defaultdict(lambda : wn.NOUN)

tag_map['J'] = wn.ADJ

tag_map['V'] = wn.VERB

tag_map['R'] = wn.ADV



Corpus.head()
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
Corpus.drop(['text'], axis=1)

output_path = 'preprocessed_data.csv'

Corpus.to_csv(output_path, index=False)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['category'],test_size=0.3)
Encoder = LabelEncoder()

Train_Y = Encoder.fit_transform(Train_Y)

Test_Y = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)

Tfidf_vect.fit(Corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)

Test_X_Tfidf = Tfidf_vect.transform(Test_X)



print(Tfidf_vect.vocabulary_)
print(Train_X_Tfidf)

# fit the training dataset on the NB classifier

Naive = naive_bayes.MultinomialNB()

Naive.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset

predictions_NB = Naive.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy

print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)









Train_X_Tfidf.shape




print(classification_report(Test_Y, predictions_NB))
from sklearn.ensemble import RandomForestClassifier



from yellowbrick.classifier import ClassPredictionError



# Instantiate the classification model and visualizer

visualizer = ClassPredictionError(

    Naive, classes=Encoder.classes_

)



# Fit the training data to the visualizer

visualizer.fit(Train_X_Tfidf,Train_Y)



# Evaluate the model on the test data

visualizer.score(Test_X_Tfidf, Test_Y)



# Draw visualization

g = visualizer.poof()

# Classifier - Algorithm - SVM

# fit the training dataset on the classifier

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset

predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy

print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print(classification_report(Test_Y,predictions_SVM))