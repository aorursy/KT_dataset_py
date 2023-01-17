import numpy as np

import pandas as pd

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score

from sklearn.model_selection import GridSearchCV

import random

random.seed(100)

import json

from IPython.display import Image

from IPython.core.display import HTML 
df_dataset = pd.read_json('../input/News_Category_Dataset_v2.json', lines = True)

print(df_dataset)
print(len(df_dataset))

print(df_dataset.columns)
df_dataset_small = df_dataset[['category','headline']]

df_dataset_small = df_dataset_small[(df_dataset.category == 'SCIENCE') | (df_dataset.category == 'TECH')]



msk = np.random.rand(len(df_dataset_small)) < 0.5



train = df_dataset_small[msk]

evaluate = df_dataset_small[~msk]
print(len(train), len(evaluate))
import nltk

# nltk.download('punkt')
document = "Tools, e.g., SpaCy and NLTK, help with text preparation. I am looking forward to try it out!"
sentences = nltk.sent_tokenize(document)

for sentence in sentences:

    print(sentence, '\n')
for sentence in sentences:

    words = nltk.word_tokenize(sentence)

    print(words, '\n')
from nltk.stem import PorterStemmer

from nltk.corpus import wordnet

import spacy

nlp = spacy.load("en_core_web_sm")



example = document

doc = nlp(document)



stemmed = [PorterStemmer().stem(word) for word in nltk.word_tokenize(example)]

lemmatized = [token.lemma_ for token in doc]

print('Original:    ', example, '\n')

print('Stemmed:     ', stemmed, '\n')

print('Lemmatized:  ', lemmatized, '\n')
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

print(stop_words)
example = document



list_of_words = [word for word in nltk.word_tokenize(example)]

list_no_stopwords = [word for word in nltk.word_tokenize(example) if not word in stop_words]



print(list_of_words, '\n')

print(list_no_stopwords)
doc = nlp(document)

for chunk in doc.noun_chunks:

    print(chunk.text, '---', chunk.root.text)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



count_vect = CountVectorizer(min_df=2, encoding='latin-1', ngram_range=(1, 3), stop_words='english')

X_train_counts = count_vect.fit_transform(train.headline)

X_evaluate_counts = count_vect.transform(evaluate.headline)



tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_evaluate_tfidf = tfidf_transformer.transform(X_evaluate_counts)



X_train, y_train = X_train_tfidf,train.category

X_test, y_test = X_evaluate_tfidf,evaluate.category
print('Shape:', X_train_tfidf.shape)

print()

print(X_evaluate_tfidf)
# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)
# from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import BernoulliNB



# Choose the type of classifier. 

# clf = MultinomialNB()

clf = BernoulliNB()



# Choose some parameter combinations to try

parameters = {

  'alpha': np.linspace(0.1, 1.5, num=10),

  'fit_prior': [True, False],

  'binarize':[None,0.1]

}



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print('Precision: ',precision_score(y_test, predictions, pos_label='SCIENCE'))

print('Recall:    ',recall_score(y_test, predictions, pos_label='SCIENCE'))

print('Accuracy:  ',accuracy_score(y_test, predictions))
from sklearn.svm import SVC



# Choose the type of classifier. 

clf = SVC()



parameters = {'kernel': ['linear'], # 'sigmoid', 'rbf', 'poly'

              'gamma': [1e-2, 1e-3],

              'C': [1, 10],

              'degree':[1]}#,



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))
from sklearn.ensemble import RandomForestClassifier



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [200], 

              'max_features': ['sqrt'], #'log2', 'auto'

              'criterion': ['entropy'],  # , 'gini'

              'max_depth': [50],

              'min_samples_split': [5],

              'min_samples_leaf': [1]

             }



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))
from sklearn.neighbors import KNeighborsClassifier



# Choose the type of classifier. 

clf = KNeighborsClassifier()



# Choose some parameter combinations to try

parameters = {'n_neighbors':range(10,25),

             'leaf_size':[20, 30]}



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))
from sklearn.neural_network import MLPClassifier



# Choose the type of classifier. 

clf = MLPClassifier()



# Choose some parameter combinations to try

parameters = {

    'hidden_layer_sizes': [(50,50,50)], # , (50,100,50), (100,)

    'activation': ['relu'], # ,'tanh', 'logistic'

    'solver': ['adam'], #, 'sgd','lbfgs'

    'alpha': [0.001], # , 0.0001, 0.05

    'learning_rate': ['adaptive'], #'constant',

    'max_iter': [500],}



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer) #'f1'

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))
Image(url= "https://miro.medium.com/max/3200/1*AzYMZSSmc0Bhv5CJm0atAQ.png")
import spacy

nlp = spacy.load('en_core_web_lg')
doc = nlp('Portugal Germany Spain Lisbon Berlin')

for word1 in doc[:2]:

    for word2 in doc[2:]:

        print(word1.text, word2.text, word1.similarity(word2))
text = "In 2017 there was the first SMART datasprint at NOVA University in Lisbon."



doc = nlp(text)

for word in doc.ents:

    print(word.text, word.label_)
from spacy import displacy

displacy.render(doc, style='ent', jupyter=True)
with open ('../working/news_dataset.txt','w')as fp:

   for line in df_dataset_small[df_dataset.category == 'SCIENCE']['headline'].tolist():

       fp.write(line+"\n")
# ! python3 -m prodigy drop -n 500 animals_news

# ! python3 -m prodigy dataset animals_news "Testset train the algorithm to learn about events in the news"
#! python3 -m prodigy ner.manual animals_news en_core_web_sm "data/news_dataset.txt" --label animal

#! python3 -m prodigy ner.teach animals_news data/en_demonstration "data/news_dataset.txt" --label animal
#! python3 -m prodigy ner.batch-train animals_news en_core_web_sm --output data/en_animals --label animal --eval-split 0.5 --n-iter 6 --batch-size 1
import warnings; warnings.simplefilter('ignore')



nlp = spacy.load('en_core_web_sm')

for number, line in enumerate(df_dataset_small[df_dataset.category == 'SCIENCE']['headline'].tolist()):

    print(displacy.render(nlp(line), style='ent', jupyter=True))

    if number == 15:

        break