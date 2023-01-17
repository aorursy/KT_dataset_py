import nltk



import warnings

warnings.filterwarnings('ignore')
from nltk.corpus import reuters

train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])

test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])
from nltk.stem.porter import PorterStemmer

def tokenize(text):

    tokens = nltk.word_tokenize(text)

    stems = []

    for item in tokens:

        stems.append(PorterStemmer().stem(item))

    return stems
%%time

from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english')



vectorised_train_documents = vectorizer.fit_transform(train_documents)

vectorised_test_documents = vectorizer.transform(test_documents)
from sklearn.preprocessing import MultiLabelBinarizer



mlb = MultiLabelBinarizer()

train_labels = mlb.fit_transform(train_categories)

test_labels = mlb.transform(test_categories)
%%time

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC



classifier = OneVsRestClassifier(LinearSVC())

classifier.fit(vectorised_train_documents, train_labels)
%%time

from sklearn.model_selection import KFold, cross_val_score



kf = KFold(n_splits=10, random_state = 42, shuffle = True)

scores = cross_val_score(classifier, vectorised_train_documents, train_labels, cv = kf)
print('Cross-validation scores:', scores)

print('Cross-validation accuracy: {:.4f} (+/- {:.4f})'.format(scores.mean(), scores.std() * 2))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



predictions = classifier.predict(vectorised_test_documents)



accuracy = accuracy_score(test_labels, predictions)



macro_precision = precision_score(test_labels, predictions, average='macro')

macro_recall = recall_score(test_labels, predictions, average='macro')

macro_f1 = f1_score(test_labels, predictions, average='macro')



micro_precision = precision_score(test_labels, predictions, average='micro')

micro_recall = recall_score(test_labels, predictions, average='micro')

micro_f1 = f1_score(test_labels, predictions, average='micro')



cm = confusion_matrix(test_labels.argmax(axis = 1), predictions.argmax(axis = 1))
print("Accuracy: {:.4f}\nPrecision:\n- Macro: {:.4f}\n- Micro: {:.4f}\nRecall:\n- Macro: {:.4f}\n- Micro: {:.4f}\nF1-measure:\n- Macro: {:.4f}\n- Micro: {:.4f}".format(accuracy, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1))
import matplotlib.pyplot as plt

import seaborn as sb

import pandas as pd



cm_plt = pd.DataFrame(cm[:73])



plt.figure(figsize = (25, 25))

ax = plt.axes()



sb.heatmap(cm_plt, annot=True)



ax.xaxis.set_ticks_position('top')



plt.show()