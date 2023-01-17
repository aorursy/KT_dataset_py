import itertools

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import log_loss

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer



%matplotlib inline
train = pd.read_csv("../input/train.csv")
train.shape
train.head(5)
train.tail(5)
# Check for null values

train.isnull().values.any()
# Check for any empty string values

for text_id, text, author in zip(train.id, train.text, train.author):

    if text == '' or author == '':

        print(text_id)
num_total = len(train)

num_eap = len(train.loc[train.author == 'EAP'])

num_hpl = len(train.loc[train.author == 'HPL'])

num_mws = len(train.loc[train.author == 'MWS'])



fig, ax = plt.subplots()

eap, hpl, mws = plt.bar(np.arange(1, 4), [(num_eap/num_total)*100, (num_hpl/num_total)*100, (num_mws/num_total)*100])

ax.set_xticks(np.arange(1, 4))

ax.set_xticklabels(['Edgar Allan Poe', 'H.P. Lovecraft', 'Mary Shelley'])

ax.set_ylim([0, 60])

ax.set_ylabel('Percentage of Entries', fontsize=12)

ax.set_xlabel('Author Name', fontsize=12)

ax.set_title('Percentage of Text Entries by Author')

plt.show()
feature_col_names = ['text']

predicted_class_name = ['author']

x = train[feature_col_names].values

y = train[predicted_class_name].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)



# Flatten the arrays for later use

x_train = x_train.ravel()

x_test = x_test.ravel()

y_train = y_train.ravel()

y_test = y_test.ravel()
word_vectorizer = TfidfVectorizer()

x_train_word = word_vectorizer.fit_transform(x_train)

x_test_word = word_vectorizer.transform(x_test)



word_clf = MultinomialNB()

word_clf.fit(x_train_word, y_train)



print("Accuracy of word_clf: {0}".format(accuracy_score(y_test, word_clf.predict(x_test_word))))

print("log_loss of word_clf: {0}".format(log_loss(y_test, word_clf.predict_proba(x_test_word))))
word_clf = MultinomialNB(alpha=.5)

word_clf.fit(x_train_word, y_train)



print("Accuracy of word_clf with alpha=.5: {0}".format(accuracy_score(y_test, word_clf.predict(x_test_word))))

print("log_loss of word_clf with alpha=.5: {0}".format(log_loss(y_test, word_clf.predict_proba(x_test_word))))



word_clf = MultinomialNB(alpha=.1)

word_clf.fit(x_train_word, y_train)



print("Accuracy of word_clf with alpha=.1: {0}".format(accuracy_score(y_test, word_clf.predict(x_test_word))))

print("log_loss of word_clf with alpha=.1: {0}".format(log_loss(y_test, word_clf.predict_proba(x_test_word))))
pipeline = Pipeline(steps=[('tfidf', TfidfVectorizer()), ('mnb', MultinomialNB())])

    

parameters = {'tfidf__ngram_range': [(1,1), (1,2), (1,3)],

              'mnb__alpha': [.1, .05, .005]}



# For more updates on progress change verbose to something like 10

grid = GridSearchCV(pipeline, parameters, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)



grid.fit(x_train, y_train)
print('Best score: {0}'.format(grid.best_score_))

print('Best parameters: {0}'.format(grid.best_params_))
word_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

x_train_word = word_vectorizer.fit_transform(x_train)

x_test_word = word_vectorizer.transform(x_test)



word_clf = MultinomialNB(alpha=.05)

word_clf.fit(x_train_word, y_train)



word_clf_pred_proba = word_clf.predict_proba(x_test_word)



print("Accuracy of word_clf: {0}".format(accuracy_score(y_test, word_clf.predict(x_test_word))))

print("log_loss of word_clf: {0}".format(log_loss(y_test, word_clf_pred_proba)))
char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 9))

x_train_char = char_vectorizer.fit_transform(x_train)

x_test_char = char_vectorizer.transform(x_test)



char_clf = MultinomialNB(alpha=.005)

char_clf.fit(x_train_char, y_train)



char_clf_pred_proba = char_clf.predict_proba(x_test_char)



print("Accuracy of char_clf: {0}".format(accuracy_score(y_test, char_clf.predict(x_test_char))))

print("log_loss of char_clf: {0}".format(log_loss(y_test, char_clf_pred_proba)))
# From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    #else:

        #print('Confusion matrix, without normalization')



    #print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
word_conf_mat = confusion_matrix(y_test, word_clf.predict(x_test_word))

plot_confusion_matrix(word_conf_mat, classes=['EAP', 'HPL', 'MWS'], title='Author Confusion Matrix - Word Classifier')
char_conf_mat = confusion_matrix(y_test, char_clf.predict(x_test_char))

plot_confusion_matrix(char_conf_mat, classes=['EAP', 'HPL', 'MWS'], title='Author Confusion Matrix - Character Classifier')
combined_proba = []

combined_predictions = []

for w, c in zip(word_clf_pred_proba, char_clf_pred_proba):

    prob = (w[0]+c[0])/2, (w[1]+c[1])/2, (w[2]+c[2])/2

    combined_proba.append(prob)

    if prob[0] > prob[1] and prob[0] > prob[2]:

        combined_predictions.append('EAP')

    elif prob[1] > prob[2]:

        combined_predictions.append('HPL')

    else:

        combined_predictions.append('MWS')

        

print("Accuracy of combined classifiers: {0}".format(accuracy_score(y_test, combined_predictions)))

print("log_loss of combined classifiers: {0}".format(log_loss(y_test, combined_proba)))
combined_conf_mat = confusion_matrix(y_test, combined_predictions)

plot_confusion_matrix(combined_conf_mat, classes=['EAP', 'HPL', 'MWS'], title='Author Confusion Matrix - Combined Classifiers')