# libs for analysis and visualising

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import re



# libs for transforming and pre-processing our dataset

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, StratifiedKFold

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



# import models to form sentiment analysers

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RandomizedSearchCV
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
PATH = "/kaggle/input/nlp-getting-started/"

train_df = pd.read_csv(f'{PATH}train.csv', low_memory=False)
train_df.shape
train_df.head()
train_df['target'].value_counts().plot.bar()

plt.title("Training data")

plt.show()
def clean_text(text):

    """ Cleaning function to remove unwanted features using regular expressions """

    # remove HTML

    cleaned_text = re.sub('<[^>]*>', '', text.lower())

    

    # remove punctuation and symbols

    cleaned_text = re.sub('[\W]+', ' ', cleaned_text)

    

    # remove urls

    cleaned_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', cleaned_text)

    

    # remove emojis

    emojis = re.compile("["

                        u"\U0001F600-\U0001F64F"  # emojis

                        u"\U0001F300-\U0001F5FF"  # symbols / pictographs

                        u"\U0001F680-\U0001F6FF"  # transport / map symbols

                        u"\U0001F1E0-\U0001F1FF"  # iOS flags

                        u"\U00002702-\U000027B0"

                        u"\U000024C2-\U0001F251]+", flags=re.UNICODE)

    cleaned_text = emojis.sub(r'', cleaned_text)

    

    return cleaned_text





def split_and_stem(text, lemmatize_text=False):

    """ Form tokenised stemmed text using a list comp and return """

    if lemmatize_text:

        tokenised = [lemmatizer.lemmatize(word) for word in text.split()]

    else:

        tokenised = [porter.stem(word) for word in text.split()]

    return tokenised





def remove_stopwords(text):

    """ Remove stopwords from the text after split and stemming """

    words = [word for word in split_and_stem(text, lemmatize_text=True) if word not in sw]

    # remove 1 letter words

    words = [word for word in words if len(word) > 1]

    new_text = " ".join(words)

    return new_text





def preprocess_text(text):

    """ Preprocess text through cleaning, stemming and stop-word removal """

    cleaned = clean_text(text)

    tokenised = remove_stopwords(cleaned)

    return tokenised
porter = PorterStemmer()

lemmatizer = WordNetLemmatizer()



# stop words - add additionals

sw = stopwords.words('english')

sw.append('http')

sw.append('co')

sw.append('û_')





# clean title, summary, and text data

train_df.loc[:, 'cleaned text'] = train_df['text'].apply(preprocess_text)





# form X and y data, with integer mapping for sentiment labels

X = train_df['cleaned text'].values

y = train_df['target'].values



# obtain training and validation splits - 80% training data, 20% val data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=0)
print("Shapes of our data: \nX_train: {0}\ny_train: {1}\nX_val: {2}\ny_val: {3} ".format(X_train.shape,

                                                                                           y_train.shape,

                                                                                           X_val.shape,

                                                                                           y_val.shape))
# use a bi-gram model for our vectoriser

vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2))



# fit to training data and transform

X_train_vec = vectorizer.fit_transform(X_train)

X_val_vec = vectorizer.transform(X_val)
def multi_model_cross_validation(clf_tuple_list, X, y, K_folds=10, score_type='accuracy', random_seed=0):

    """ Find cross validation scores, and print and return results """

    

    model_names, model_scores = [], []

    

    for name, model in clf_list:

        k_fold = StratifiedKFold(n_splits=K_folds, shuffle=True, random_state=random_seed)

        cross_val_results = cross_val_score(model, X, y, cv=k_fold, scoring=score_type, n_jobs=-1)

        model_names.append(name)

        model_scores.append(cross_val_results)

        print("{0:<40} {1:.5f} +/- {2:.5f}".format(name, cross_val_results.mean(), cross_val_results.std()))

        

    return model_names, model_scores





def boxplot_comparison(model_names, model_scores, figsize=(12, 6), score_type="Accuracy",

                       title="Sentiment Analysis Classification Comparison"):

    """ Boxplot comparison of a range of models using Seaborn and matplotlib """

    

    fig = plt.figure(figsize=figsize)

    fig.suptitle(title, fontsize=18)

    ax = fig.add_subplot(111)

    sns.boxplot(x=model_names, y=model_scores)

    ax.set_xticklabels(model_names)

    ax.set_xlabel("Model", fontsize=16) 

    ax.set_ylabel("Model Score ({})".format(score_type), fontsize=16)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)

    plt.show()

    return



# list of classifiers to compare - use some additional models this time

clf_list = [("Perceptron", Perceptron(eta0=0.1)),

            ("Logistic Regression", LogisticRegression(C=10.0)),

            ("Support Vector Machine", SVC(kernel='linear', C=1.0)),

            ("Decision Tree", DecisionTreeClassifier()),

            ("Random Forrest", RandomForestClassifier(n_estimators=10)),

            ("Multinomial Naive Bayes", MultinomialNB()),

            ("Ridge Classifier", RidgeClassifier())]

            #("Gradient Boosting", GradientBoostingClassifier()),

            #("Multi-Layer Perceptron", MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 2), max_iter=100))]





# calculate cross-validation scores and print / plot for each model accordingly

model_names, model_scores = multi_model_cross_validation(clf_list, X_train_vec, y_train)

boxplot_comparison(model_names, model_scores)
def test_set_performances(clf_tuple_list, X_train, y_train, X_test, 

                          y_test, score_type='accuracy', print_results=True):

    """ Find test set accuracy and F1 Score performance for all classifiers 

        and return """

    

    model_names, model_accuracies, model_f1 = [], [], []

    

    if print_results:

        print("{0:<30} {1:<10} {2:<10} \n{3}".format("Model", "Accuracy", 

                                                     "F1-Score", "-"*50))

    

    # fit each model to training data and form predictions

    for name, model in clf_list:

        

        # fit on training, predict on test

        model.fit(X_train, y_train)

        y_preds = model.predict(X_test)

        

        # find accuracy and f1 (macro) scores

        accuracy = accuracy_score(y_test, y_preds)

        test_f1 = f1_score(y_test, y_preds, average='macro')

        

        # append model results

        model_names.append(name)

        model_accuracies.append(accuracy)

        model_f1.append(test_f1)

        

        if print_results:

            print("{0:<30} {1:<10.5f} {2:<10.5f}".format(name, accuracy, test_f1))

            

    return model_names, model_accuracies, model_f1





# obtain accuracy and f1 metrics and print for each model

model_names, test_acc, test_f1 = test_set_performances(clf_list, X_train_vec, y_train, X_val_vec, y_val)
# barplot of model accuracies

sns.set(style="darkgrid")

sns.barplot(model_names, test_acc, alpha=0.9)

plt.title('Test set accuracy', weight='bold')

plt.ylabel('Accuracy', fontsize=12, weight='bold')

plt.xticks(rotation=90)

plt.ylim(0.0, 1.0)

plt.show()



# barplot of model f1 scores

sns.set(style="darkgrid")

sns.barplot(model_names, test_f1, alpha=0.9)

plt.title('Test set F1 Score', weight='bold')

plt.ylabel('F1 Score (Macro)', fontsize=12, weight='bold')

plt.xticks(rotation=90)

plt.ylim(0.0, 1.0)

plt.show()
def plot_confusion_matrix(true_y, pred_y, title='Confusion Matrix', figsize=(8,6)):

    """ Custom function for plotting a confusion matrix for predicted results """

    conf_matrix = confusion_matrix(true_y, pred_y)

    conf_df = pd.DataFrame(conf_matrix, columns=np.unique(true_y), index = np.unique(true_y))

    conf_df.index.name = 'Actual'

    conf_df.columns.name = 'Predicted'

    plt.figure(figsize = figsize)

    plt.title(title)

    sns.set(font_scale=1.4) # label size

    sns.heatmap(conf_df, cmap="Blues", annot=True,annot_kws={"size": 16}) # font size

    plt.show()

    return
# create a log reg classifier and predict using test set

svc_clf = SVC(kernel='linear', C=1.0)

svc_clf.fit(X_train_vec, y_train)

predictions = svc_clf.predict(X_val_vec)



# print performance statistics

print("Samples incorrectly classified: {0} out of {1}.".format((y_val != predictions).sum(),

                                                                len(y_val)))



print("Logistic Regression classifier accuracy: {0:.2f}%".format(accuracy_score(predictions, y_val)*100.0))



# plot a confusion matrix of our results

plot_confusion_matrix(y_val, predictions, 

                      title="SVC Confusion Matrix", figsize=(5,5))



# print recall, precision and f1 score results

print(classification_report(y_val, predictions))
# form X and y data, with integer mapping for sentiment labels

X = train_df['cleaned text'].values

y = train_df['target'].values
# use a bi-gram model for our vectoriser

vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2))



# fit to training data and transform

X_vec = vectorizer.fit_transform(X)
# create final svc classifier

svc_clf = SVC(kernel='linear', C=1.0)

svc_clf.fit(X_vec, y)
test_df = pd.read_csv(f'{PATH}test.csv', low_memory=False)
# clean test data and vectorise using tfidf

test_df.loc[:, 'cleaned text'] = test_df['text'].apply(preprocess_text)

X_test = test_df['cleaned text'].values

X_test_vec = vectorizer.transform(X_test)
# obtain predictions using svc model

test_preds = svc_clf.predict(X_test_vec)

test_df['target'] = test_preds
submission = test_df.loc[:, ['id', 'target']]

submission.head()
submission.to_csv('submission.csv', index=False)