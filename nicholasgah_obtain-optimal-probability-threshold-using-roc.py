import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

import nltk

from nltk.corpus import stopwords

from bs4 import BeautifulSoup

import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip", sep="\t")

df.head()
print("Unique Values of Sentiment are: {}".format(", ".join(list(map(str,df["Sentiment"].unique())))))
X = df["Phrase"].tolist()

Y = df["Sentiment"].apply(lambda i: 0 if i <= 2 else 1)
Y.value_counts()
lemmatizer = WordNetLemmatizer()

def proc_text(messy): #input is a single string

    first = BeautifulSoup(messy, "lxml").get_text() #gets text without tags or markup, remove html

    second = re.sub("[^a-zA-Z]"," ",first) #obtain only letters

    third = second.lower().split() #obtains a list of words in lower case

    fourth = set([lemmatizer.lemmatize(str(x)) for x in third]) #lemmatizing

    stops = set(stopwords.words("english")) #faster to search through a set than a list

    almost = [w for w in fourth if not w in stops] #remove stop words

    final = " ".join(almost)

    return final
X = [proc_text(i) for i in X]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=100, test_size=0.2, stratify=Y)

print("Training Set has {} Positive Labels and {} Negative Labels".format(sum(y_train), len(y_train) - sum(y_train)))

print("Test Set has {} Positive Labels and {} Negative Labels".format(sum(y_test), len(y_test) - sum(y_test)))
pos_weights = (len(y_train) - sum(y_train)) / (sum(y_train)) 

pipeline_tf = Pipeline([

    ('tfidf', TfidfVectorizer()),

    ('classifier', DecisionTreeClassifier(random_state=100, class_weight={0: 1, 1: pos_weights}))

])
pipeline_tf.fit(X_train, y_train)
predictions = pipeline_tf.predict(X_test)

predicted_proba = pipeline_tf.predict_proba(X_test)
print("Accuracy Score Before Thresholding: {}".format(accuracy_score(y_test, predictions)))

print("Precision Score Before Thresholding: {}".format(precision_score(y_test, predictions)))

print("Recall Score Before Thresholding: {}".format(recall_score(y_test, predictions)))

print("F1 Score Before Thresholding: {}".format(f1_score(y_test, predictions)))

print("ROC AUC Score: {}".format(roc_auc_score(y_test, predicted_proba[:, -1])))
y_actual = pd.Series(y_test, name='Actual')

y_predict_tf = pd.Series(predictions, name='Predicted')

df_confusion = pd.crosstab(y_actual, y_predict_tf, rownames=['Actual'], colnames=['Predicted'], margins=True)

print(df_confusion)
false_pos_rate, true_pos_rate, proba = roc_curve(y_test, predicted_proba[:, -1])

plt.figure()

plt.plot([0,1], [0,1], linestyle="--") # plot random curve

plt.plot(false_pos_rate, true_pos_rate, marker=".", label=f"AUC = {roc_auc_score(y_test, predicted_proba[:, -1])}")

plt.title("ROC Curve")

plt.ylabel("True Positive Rate")

plt.xlabel("False Positive Rate")

plt.legend(loc="lower right")
optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]

roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in predicted_proba[:, -1]]
print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(y_test, predictions), accuracy_score(y_test, roc_predictions)))

print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(y_test, predictions), precision_score(y_test, roc_predictions)))

print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(y_test, predictions), recall_score(y_test, roc_predictions)))

print("F1 Score Before and After Thresholding: {}, {}".format(f1_score(y_test, predictions), f1_score(y_test, roc_predictions)))
y_actual = pd.Series(y_test, name='Actual')

y_predict_tf = pd.Series(roc_predictions, name='Predicted')

df_confusion = pd.crosstab(y_actual, y_predict_tf, rownames=['Actual'], colnames=['Predicted'], margins=True)

print (df_confusion)