import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.matutils import Sparse2Corpus
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, confusion_matrix
import os
print(os.listdir("../input"))
data = pd.read_csv("../input/spam.csv", encoding="latin-1")
data.head()
data.describe()
data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
data.rename(columns=dict(v1="Class", v2="Message"), inplace=True)
limit = int(0.9*len(data))
train = data.loc[:limit]
test = data.loc[limit:]
vectorizer = CountVectorizer(stop_words="english", 
                             ngram_range=(1, 3))
term_document_matrix = vectorizer.fit_transform(train.Message)
corpus = Sparse2Corpus(term_document_matrix, documents_columns=False)
id2word = {value: key for key, value in vectorizer.vocabulary_.items()}
model = LdaModel(corpus=corpus, 
                 id2word=id2word, 
                 random_state=0)

def transform_messages_to_topic_vectors(data, vectorizer, model):
    num_topics = model.num_topics
    message_topics = pd.DataFrame(index=map(str, range(num_topics)))    
    for index, message in data.Message.iteritems():
        message_transformed = (vectorizer.transform([message]))
        message_corpus = (Sparse2Corpus(message_transformed, documents_columns=False))
        topics_framework = dict.fromkeys(map(str, range(0, num_topics)))
        topics_specific = dict(list(model[message_corpus])[0])
        for key in topics_specific:
            topics_framework[str(key)] = topics_specific[key]
        message_topics[index] = pd.Series(topics_framework)
    message_topics = message_topics.T.fillna(0)
    data = data.join(message_topics)
    data.drop("Message", axis=1, inplace=True)
    data = pd.get_dummies(data, drop_first=True)
    data.rename(columns=dict(Class_spam="Spam"), inplace=True)
    return data

train = transform_messages_to_topic_vectors(train, vectorizer, model)
test = transform_messages_to_topic_vectors(test, vectorizer, model)
train.head()
topics = list(train.columns)
topics.remove("Spam")
ax = sns.countplot(train[topics].astype(bool).sum(axis=1), color="C0")
ax.set_xlabel("Number of Exhibited Topics")
ax.set_ylabel("Number of Messages")
fig = plt.gcf()
fig.set_size_inches(16, 4)
ax = train[topics].astype(bool).sum(axis=0).plot.bar(color="C0")
ax.set_xlabel("Topic")
ax.set_ylabel("Number of Messages")
fig = plt.gcf()
fig.set_size_inches(24, 8)
print("The most prevalent ngrams in the most prevalent topic are:\n- {}".format('\n- '.join([id2word[term[0]] for term in model.get_topic_terms(50)])))
X_train = train.drop("Spam", axis=1)
Y_train = train.Spam
X_test = test.drop("Spam", axis=1)
Y_test = test.Spam
sns.countplot(Y_train, color="C0")
print("Number of spam messages in the test set: {}"\
      .format(test["Spam"].value_counts()[1]))
scaler = StandardScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train)
X_test[X_test.columns] = scaler.transform(X_test)
%%capture --no-stdout

param_grid = dict(C=np.logspace(-3, 3, 7))
best_C_linear = GridSearchCV(LinearSVC(class_weight="balanced", random_state=0), 
                             param_grid, 
                             cv=5).fit(X_train, Y_train)\
                                  .best_params_["C"]
print("The best value for the tuning parameter 'C' is {}.".format(best_C_linear))
svm_linear = LinearSVC(C=best_C_linear, 
                       random_state=0, 
                       max_iter=1e6).fit(X_train, Y_train)
priors = list(train.Spam.value_counts().div(train.Spam.value_counts().sum()))
nb = GaussianNB(priors).fit(X_train, Y_train)
%%capture --no-stdout

param_grid["gamma"] = ["auto", "scale"]
best_svm_rbf_params = GridSearchCV(SVC(class_weight="balanced", 
                                       random_state=0, 
                                       max_iter=1000), 
                                   param_grid, 
                                   cv=5).fit(X_train, Y_train)\
                                        .best_params_
print("The best value for the tuning parameter 'C' is {}."\
      .format(best_svm_rbf_params["C"]))
print("The best value for 'gamma' is {}.".format(best_svm_rbf_params["gamma"]))
svm_rbf = SVC(C=best_svm_rbf_params["C"], 
              gamma=best_svm_rbf_params["gamma"], 
              random_state=0, 
              max_iter=-1).fit(X_train, Y_train)
classifiers = [("svm_linear", svm_linear), 
               ("nb", nb), 
               ("svm_rbf", svm_rbf)]
ensemble = VotingClassifier(classifiers).fit(X_train, Y_train)
svm_linear_score = svm_linear.score(X_test, Y_test)
nb_score = nb.score(X_test, Y_test)
svm_rbf_score = svm_rbf.score(X_test, Y_test)
ensemble_score = ensemble.score(X_test, Y_test)

print("The linear SVM has a test accuracy score of {:.3f}.".format(svm_linear_score))
print("The Gaussian naive Bayes classifer has a test accuracy score of {:.3f}."\
      .format(nb_score))
print("The SVM with an RBF kernel has a test accuracy score of {:.3f}."\
      .format(svm_rbf_score))
print("The ensemble voting classifier has a test accuracy score of {:.3f}."\
      .format(ensemble_score))
dummy_constant = DummyClassifier("constant", 
                                 random_state=0, 
                                 constant=0).fit(X_train, Y_train)
dummy_constant_score = dummy_constant.score(X_test, Y_test)

print("The dummy constant classifier has a test accuracy score of {:.3f}."\
      .format(dummy_constant_score))
dummy_constant_preds = dummy_constant.predict(X_test)
svm_linear_preds = svm_linear.predict(X_test)
nb_preds = nb.predict(X_test)
svm_rbf_preds = svm_rbf.predict(X_test)
ensemble_preds = ensemble.predict(X_test)

dummy_constant_score = average_precision_score(Y_test, dummy_constant_preds)
svm_linear_score = average_precision_score(Y_test, svm_linear_preds)
nb_score = average_precision_score(Y_test, nb_preds)
svm_rbf_score = average_precision_score(Y_test, svm_rbf_preds)
ensemble_score = average_precision_score(Y_test, ensemble_preds)

print("The dummy constant classifier has a test average precision score of {:.3f}."\
      .format(dummy_constant_score))
print("The linear SVM has a test average precision score of {:.3f}.".format(svm_linear_score))
print("The Gaussian naive Bayes classifer has a test average precision score of {:.3f}."\
      .format(nb_score))
print("The SVM with an RBF kernel has a test average precision score of {:.3f}."\
      .format(svm_rbf_score))
print("The ensemble voting classifier has a test average precision score of {:.3f}."\
      .format(ensemble_score))
confusion = pd.DataFrame(confusion_matrix(Y_test, ensemble_preds))
confusion = confusion.div(confusion.sum().sum())
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
ax = sns.heatmap(confusion, vmin=0, vmax=1, annot=True, fmt=".0%")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.set_ticks((0, .25, .5, .75, 1))
ax.collections[0].colorbar.set_ticklabels(("0%", "25%", "50%", "75%", "100%"))