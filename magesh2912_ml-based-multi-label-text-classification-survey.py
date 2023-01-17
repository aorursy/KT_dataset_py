from nltk.corpus import reuters

import nltk

nltk.download('wordnet')

import pandas as pd

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from yellowbrick.text import TSNEVisualizer

import warnings

warnings.filterwarnings("ignore")
train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])

test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])
print("Number of training documents:", len(train_documents))

print("Number of testing documents:", len(test_documents))
from sklearn.preprocessing import MultiLabelBinarizer



mlb = MultiLabelBinarizer()

train_labels = mlb.fit_transform(train_categories)

test_labels = mlb.transform(test_categories)
trainData = {"content": train_documents}

testData = {"content": test_documents}

trainDf = pd.DataFrame(trainData, columns=["content"])

testDf = pd.DataFrame(testData, columns=["content"])
wordnet_lemmatizer = WordNetLemmatizer()

stemmer = PorterStemmer()

stopwords = set(w.rstrip() for w in open("../input/reuters/reuters/reuters/stopwords"))



def tokenize_lemma_stopwords(text):

    text = text.replace("\n", " ")

    tokens = nltk.tokenize.word_tokenize(text.lower()) # split string into words (tokens)

    tokens = [t for t in tokens if t.isalpha()] # keep strings with only alphabets

    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form

    tokens = [stemmer.stem(t) for t in tokens]

    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful

    tokens = [t for t in tokens if t not in stopwords] # remove stopwords

    cleanedText = " ".join(tokens)

    return cleanedText



def dataCleaning(df):

    data = df.copy()

    data["content"] = data["content"].apply(tokenize_lemma_stopwords)

    return data
cleanedTrainData = dataCleaning(trainDf)

cleanedTestData = dataCleaning(testDf)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics



vectorizer = TfidfVectorizer()

vectorised_train_documents = vectorizer.fit_transform(cleanedTrainData["content"])

vectorised_test_documents = vectorizer.transform(cleanedTestData["content"])
from yellowbrick.text import FreqDistVisualizer

features = vectorizer.get_feature_names()

visualizer = FreqDistVisualizer(features=features, orient='v')

visualizer.fit(vectorised_train_documents)

visualizer.show()
tsne = TSNEVisualizer()

tsne.fit(vectorised_train_documents)

tsne.show()
from yellowbrick.text import UMAPVisualizer

from sklearn.cluster import KMeans



umap = UMAPVisualizer(metric="cosine")

umap.fit(vectorised_train_documents)

umap.show()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss



ModelsPerformance = {}



def metricsReport(modelName, test_labels, predictions):

    accuracy = accuracy_score(test_labels, predictions)



    macro_precision = precision_score(test_labels, predictions, average='macro')

    macro_recall = recall_score(test_labels, predictions, average='macro')

    macro_f1 = f1_score(test_labels, predictions, average='macro')



    micro_precision = precision_score(test_labels, predictions, average='micro')

    micro_recall = recall_score(test_labels, predictions, average='micro')

    micro_f1 = f1_score(test_labels, predictions, average='micro')

    hamLoss = hamming_loss(test_labels, predictions)

    print("------" + modelName + " Model Metrics-----")

    print("Accuracy: {:.4f}\nHamming Loss: {:.4f}\nPrecision:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nRecall:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nF1-measure:\n  - Macro: {:.4f}\n  - Micro: {:.4f}"\

          .format(accuracy, hamLoss, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1))

    ModelsPerformance[modelName] = micro_f1

from sklearn.neighbors import KNeighborsClassifier

from sklearn.multiclass import OneVsRestClassifier



knnClf = KNeighborsClassifier()



knnClf.fit(vectorised_train_documents, train_labels)

knnPredictions = knnClf.predict(vectorised_test_documents)

metricsReport("knn", test_labels, knnPredictions)
from sklearn.tree import DecisionTreeClassifier



dtClassifier = DecisionTreeClassifier()

dtClassifier.fit(vectorised_train_documents, train_labels)

dtPreds = dtClassifier.predict(vectorised_test_documents)

metricsReport("Decision Tree", test_labels, dtPreds)
from sklearn.ensemble import BaggingClassifier



bagClassifier = OneVsRestClassifier(BaggingClassifier(n_jobs=-1))

bagClassifier.fit(vectorised_train_documents, train_labels)

bagPreds = bagClassifier.predict(vectorised_test_documents)

metricsReport("Bagging", test_labels, bagPreds)
from sklearn.ensemble import RandomForestClassifier

rfClassifier = RandomForestClassifier(n_jobs=-1)

rfClassifier.fit(vectorised_train_documents, train_labels)

rfPreds = rfClassifier.predict(vectorised_test_documents)

metricsReport("Random Forest", test_labels, rfPreds)

from sklearn.ensemble import GradientBoostingClassifier



boostClassifier = OneVsRestClassifier(GradientBoostingClassifier())

boostClassifier.fit(vectorised_train_documents, train_labels)

boostPreds = boostClassifier.predict(vectorised_test_documents)

metricsReport("Boosting", test_labels, boostPreds)

from sklearn.naive_bayes import MultinomialNB



nbClassifier = OneVsRestClassifier(MultinomialNB())

nbClassifier.fit(vectorised_train_documents, train_labels)



nbPreds = nbClassifier.predict(vectorised_test_documents)

metricsReport("Multinomial NB", test_labels, nbPreds)
from sklearn.svm import LinearSVC



svmClassifier = OneVsRestClassifier(LinearSVC(), n_jobs=-1)

svmClassifier.fit(vectorised_train_documents, train_labels)



svmPreds = svmClassifier.predict(vectorised_test_documents)

metricsReport("SVC Sq. Hinge Loss", test_labels, svmPreds)
from skmultilearn.problem_transform import LabelPowerset



powerSetSVC = LabelPowerset(LinearSVC())

powerSetSVC.fit(vectorised_train_documents, train_labels)



powerSetSVCPreds = powerSetSVC.predict(vectorised_test_documents)

metricsReport("Power Set SVC", test_labels, powerSetSVCPreds)
print("  Model Name " + " "*10 + "| Micro-F1 Score")

print("-------------------------------------------")

for key, value in ModelsPerformance.items():

    print("  " + key, " "*(20-len(key)) + "|", value)

    print("-------------------------------------------")
from sklearn.metrics import classification_report



print(classification_report(test_labels, svmPreds))