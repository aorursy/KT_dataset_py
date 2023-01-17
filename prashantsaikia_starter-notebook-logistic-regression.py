# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the training data

df = pd.read_csv('../input/nlp-getting-started/train.csv')

df.head(10)
# Pre-processing

import re

from nltk.stem.wordnet import WordNetLemmatizer



stop_words = ['in', 'of', 'at', 'a', 'the']



def pre_process(text):

    

    # lowercase

    text=str(text).lower()



    # remove numbers

    text=re.sub(r'[0-9]+', '', text)

    

    #remove tags

    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    

    # correct some misspellings and/or replace some text with others that might be easier to handle

    text=text.replace('do not', "don't")

    

    # remove special characters except spaces, apostrophes and dots

    text=re.sub(r"[^a-zA-Z0-9.']+", ' ', text)

    

    # remove stopwords

    text=[word for word in text.split(' ') if word not in stop_words]

    

    # lemmatize

    lmtzr = WordNetLemmatizer()

    text = ' '.join((lmtzr.lemmatize(i)) for i in text)

    

    return text



df.text = df.text.apply(pre_process)

df.head(10)
# Analyse if real or not



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import pandas as pd



def analyse_realness(df):



    # Take 70% of the data as training data, and the rest as validation data; experiment with the %age to be allocated to each

    train = df.text[0:int(0.7*len(df))]

    val = df.text[int(0.7*len(df))+1:]

    train_target = df.target[0:int(0.7*len(df))]

    val_target = df.target[int(0.7*len(df))+1:]



    # Vectorize the text; i.e., convert the text into numerical form in order to be able to do the analysis

    stop_words = ['in', 'of', 'at', 'a', 'the']

    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words) # Taking a window size of upto 3 words

    ngram_vectorizer.fit(train)

    X_train = ngram_vectorizer.transform(train)

    X_val = ngram_vectorizer.transform(val)



    # Train the model

    model = LogisticRegression() # play around with the parameters in Logisticregression() to find the optimal parameters

    model.fit(X_train, train_target)

    

    # Predictions on the validation set

    val_preds = model.predict(X_val)



    # Check model accuracy on the validation data

    val_acc = accuracy_score(val_target, val_preds)



    return val_acc, X_val, val_target, val_preds, model, ngram_vectorizer



val_acc, X_val, val_target, val_preds, model, ngram_vectorizer = analyse_realness(df)

print('Validation accuracy: {0:.2f}%'.format(100*val_acc))
# Evaluate the model



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

import time



def model_evaluation(X_test, test_target, target_names, model, figsize=(4,3)):

    conf_mat = metrics.confusion_matrix(test_target, model.predict(X_test))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)

    plt.ylabel('Actual')

    plt.xlabel('Predicted')

    plt.title('Confusion matrix (Test data)')

    plt.gcf().subplots_adjust(bottom=0.15) # To avoid xlabel from being cut off

    

labels = df.target.unique()

labels = [str(i) for i in labels]

model_evaluation(X_test=X_val, test_target=val_target, target_names=labels, model=model)
metrics.classification_report(val_target, model.predict(X_val), target_names=labels)
# Plotting ROC curve



fpr, tpr, threshold = metrics.roc_curve(val_target, val_preds)

roc_auc = metrics.auc(fpr, tpr)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
# Predicting on the hitherto unseen test dataset for submission

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')



# Pre-process the new dataset to clean it and make it suitable for analysis

test_df.text = test_df.text.apply(pre_process)



# Convert the text into numbers for analysis with the same vectorizer as the one used for the original dataset (to preserve the dimensions)

features = ngram_vectorizer.transform(test_df.text).toarray()



# Predict the targets and save them to the target column in the submission df

submission.target = model.predict(features)



submission.to_csv('submission.csv', index=False)