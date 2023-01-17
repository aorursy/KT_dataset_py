import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import nltk

from nltk.stem.snowball import SnowballStemmer



import random

random.seed(1)
data_dir = '../input/data_competition/data_competition' 



relation_users = pd.read_csv(data_dir + "/UserUser.txt", sep="\t", header=None)

relation_users.columns = ["follower", "followed"]



labels_training = pd.read_csv(data_dir + "/labels_training.txt", sep=",")

labels_training.columns = ["news", "label"]



news_users = pd.read_csv(data_dir + "/newsUser.txt", sep="\t", header=None)

news_users.columns = ["news", "user", "times"]
### TRAINING SET ###################################################################################################

texts_train = []

labels_train = []

for i in os.listdir(data_dir + "/news/training/")[0:150] :

    with open(data_dir + "/news/training/"+ i, 'r') as myfile:

        text0=myfile.read().replace('\n', '')

    texts_train.append(text0)

    # get if fake or not

    labels_train.append(int(labels_training[labels_training["news"] == int(i.split('.')[0])]["label"]))

    

    

### TEST SET #######################################################################################################

texts_test = []

labels_test = []

for i in os.listdir(data_dir + "/news/training/")[151:193] :

    with open(data_dir + "/news/training/"+ i, 'r') as myfile:

        text0=myfile.read().replace('\n', '')

    texts_test.append(text0)

    # get if fake or not

    labels_test.append(int(labels_training[labels_training["news"] == int(i.split('.')[0])]["label"]))

    

    

from collections import Counter

print(Counter(labels_train))

print(Counter(labels_test))
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(texts_train, labels_train)



# Performance of NB Classifier

predicted = text_clf.predict(texts_test)

print("Good classification rate: %0.3f" % np.mean(predicted == labels_test))

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_test, predicted))

print("Completeness: %0.3f" % metrics.completeness_score(labels_test, predicted))

print("V-measure: %0.3f" % metrics.v_measure_score(labels_test, predicted))
text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),

                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])



text_clf_svm = text_clf_svm.fit(texts_train, labels_train)

predicted_svm = text_clf_svm.predict(texts_test)

print("Good classification rate: %0.3f" % np.mean(predicted_svm == labels_test))

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_test, predicted_svm))

print("Completeness: %0.3f" % metrics.completeness_score(labels_test, predicted_svm))

print("V-measure: %0.3f" % metrics.v_measure_score(labels_test, predicted_svm))
text_clf_rf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), 

                         ('clf-rf', RandomForestClassifier(n_estimators = 200, min_samples_leaf = 10, n_jobs = -1))])



text_clf_rf = text_clf_rf.fit(texts_train, labels_train)

predicted_rf = text_clf_rf.predict(texts_test)

print("Good classification rate: %0.3f" % np.mean(predicted_rf == labels_test))

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_test, predicted_rf))

print("Completeness: %0.3f" % metrics.completeness_score(labels_test, predicted_rf))

print("V-measure: %0.3f" % metrics.v_measure_score(labels_test, predicted_rf))
text_clf_knn = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), 

                         ('clf-knn', KNeighborsClassifier())])



text_clf_knn = text_clf_knn.fit(texts_train, labels_train)

predicted_knn = text_clf_knn.predict(texts_test)

print("Good classification rate: %0.3f" % np.mean(predicted_knn == labels_test))

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_test, predicted_knn))

print("Completeness: %0.3f" % metrics.completeness_score(labels_test, predicted_knn))

print("V-measure: %0.3f" % metrics.v_measure_score(labels_test, predicted_knn))

text_clf_mlp = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), 

                         ('clf-mlp', MLPClassifier(solver='lbfgs',

                                                   alpha=1e-4,

                                                   hidden_layer_sizes=(5), 

                                                   random_state=1,

                                                   max_iter=500

                                                  ))])



text_clf_mlp = text_clf_mlp.fit(texts_train, labels_train)

predicted_mlp = text_clf_mlp.predict(texts_test)

print("Good classification rate: %0.3f" % np.mean(predicted_mlp == labels_test))

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_test, predicted_mlp))

print("Completeness: %0.3f" % metrics.completeness_score(labels_test, predicted_mlp))

print("V-measure: %0.3f" % metrics.v_measure_score(labels_test, predicted_mlp))
print("MLP : %0.3f" % np.mean(predicted_mlp == labels_test))

print("Random Forest : %0.3f" % np.mean(predicted_rf == labels_test))

print("SVM : %0.3f" % np.mean(predicted_svm == labels_test))

print("Multinomial Naive Bayes : %0.3f" % np.mean(predicted == labels_test))

print("KNN : %0.3f" % np.mean(predicted_knn == labels_test))

texts_all_train = []

labels_all_train = []

for i in os.listdir(data_dir + "/news/training/") :

    with open(data_dir + "/news/training/"+ i, 'r') as myfile:

        text0=myfile.read().replace('\n', '')

    texts_all_train.append(text0)

    # get if fake or not

    labels_all_train.append(int(labels_training[labels_training["news"] == int(i.split('.')[0])]["label"]))



    

texts_valid = []

for i in os.listdir(data_dir + "/news/test/") :

    with open(data_dir + "/news/test/"+ i, 'r') as myfile:

        text0=myfile.read().replace('\n', '')

    texts_valid.append(text0)
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}



gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

gs_clf = gs_clf.fit(texts_all_train, labels_all_train)



print("-----------------Original Features--------------------")

print("Best score: %0.4f" % gs_clf.best_score_)

print("Using the following parameters:")

print(gs_clf.best_params_)
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),

                  'clf-svm__alpha': (1e-1, 1e-2, 1e-3, 1e-4),

                  'clf-svm__n_iter': (2,4,5,7,8,10,15)}



gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)

gs_clf_svm = gs_clf_svm.fit(texts_all_train, labels_all_train)



print("-----------------Original Features--------------------")

print("Best score: %0.4f" % gs_clf_svm.best_score_)

print("Using the following parameters:")

print(gs_clf_svm.best_params_)
parameters_rf = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),

                 'clf-rf__n_estimators': (100, 200, 300,400,500),

                 'clf-rf__min_samples_leaf': (1,2,4,5,10,15)}



gs_clf_rf = GridSearchCV(text_clf_rf, parameters_rf, n_jobs=-1)

gs_clf_rf = gs_clf_rf.fit(texts_all_train, labels_all_train)



print("-----------------Original Features--------------------")

print("Best score: %0.4f" % gs_clf_rf.best_score_)

print("Using the following parameters:")

print(gs_clf_rf.best_params_)
parameters_mlp = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False) ,

                  'clf-mlp__solver': ['lbfgs'], 

                  #'clf-mlp__max_iter': [500,600], 

                  'clf-mlp__alpha': 10.0 ** -np.arange(1, 7), 

                  'clf-mlp__hidden_layer_sizes':np.arange(3, 8) #, 

                  #'clf-mlp__random_state':[0,1,2,3,4,5]

                 }

gs_clf_mlp = GridSearchCV(text_clf_mlp, parameters_mlp, n_jobs=-1)

gs_clf_mlp = gs_clf_mlp.fit(texts_all_train, labels_all_train)



print("-----------------Original Features--------------------")

print("Best score: %0.4f" % gs_clf_mlp.best_score_)

print("Using the following parameters:")

print(gs_clf_mlp.best_params_)
stemmer = SnowballStemmer("english", ignore_stopwords=True)



class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):

        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

    

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')



text_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),

                         #('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=0.001, n_iter=10, random_state=42))])

                         ('clf-mlp', MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=3, 

                                                   random_state=1, max_iter=500))])



text_stemmed = text_stemmed.fit(texts_train, labels_train)

predicted_stemmed = text_stemmed.predict(texts_test)

np.mean(predicted_stemmed == labels_test)
text_best_model = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), 

                         #('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=0.001, n_iter=10, random_state=42))])

                         ('clf-mlp', MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=3, 

                                                   random_state=1, max_iter=500))])



text_best_model = text_best_model.fit(texts_all_train, labels_all_train)

predicted_final = text_best_model.predict(texts_valid)
#####################################################################################

l = os.listdir(data_dir + "/news/test/")[0:47]

id_doc = [int(i.split('.')[0]) for i in l]

res_only_text_mining = pd.DataFrame(

    {'doc': id_doc,

     'class': predicted_final

    })

res_only_text_mining.sort_values(['doc'], ascending=[True])