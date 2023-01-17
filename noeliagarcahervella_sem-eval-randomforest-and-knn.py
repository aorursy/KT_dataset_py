# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords 

from unicodedata import normalize

from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from sklearn.base import clone

from sklearn.preprocessing import label_binarize

from scipy import interp

from sklearn.metrics import roc_curve, auc 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def bag_of_words(train, test):



    tfidf = TfidfVectorizer()

    

    train_bag = tfidf.fit_transform(train)

    test_bag = tfidf.transform(test)

    

    return tfidf, train_bag, test_bag
def train_predict_model(classifier, train_tweet, train_sentiment, test_tweet, test_sentiment):

    # build model    

    classifier.fit(train_tweet, train_sentiment)

    # predict using model

    predictions = classifier.predict(test_tweet) 

    return predictions 
def get_metrics(true_labels, predicted_labels):

    

    parameters = []

    

    parameters.append(np.round(metrics.accuracy_score(true_labels,predicted_labels),4))

    parameters.append(np.round(metrics.precision_score(true_labels,predicted_labels,average='weighted'),4))

    parameters.append(np.round(metrics.recall_score(true_labels,predicted_labels,average='weighted'),4))

    parameters.append(np.round(metrics.f1_score(true_labels,predicted_labels,average='weighted'),4) )



    print('Accuracy:', parameters[0])

    print('Precision:', parameters[1])

    print('Recall:', parameters[2])

    print('F1 Score:', parameters[3])

    

    return parameters

    

    
def display_classification_report(true_labels, predicted_labels, classes=[1,0]):



    report = metrics.classification_report(true_labels, 

                                           predicted_labels, 

                                           classes) 

    print(report)

    return report
def display_confusion_matrix(true_labels, predicted_labels, classes=[1,0]):



    total_classes = len(classes)

    level_labels = [total_classes*[0], list(range(total_classes))]



    cm = metrics.confusion_matrix(true_labels,predicted_labels, classes)

    cm_frame = pd.DataFrame(cm, pd.MultiIndex([['Predicted:'], classes], level_labels), 

                            pd.MultiIndex([['Actual:'], classes], level_labels)) 

    print(cm_frame)
def display_model_performance_metrics(true_labels, predicted_labels, classes=[1,0]):

    

    print('Model Performance metrics:')

    print('-'*30)

    parameters = get_metrics(true_labels, predicted_labels)

    print('\nModel Classification report:')

    print('-'*30)

    report = display_classification_report(true_labels, predicted_labels, classes)

    print('\nPrediction Confusion Matrix:')

    print('-'*30)

    display_confusion_matrix(true_labels, predicted_labels, classes)

    

    return parameters, report
def create_dataframes():

    

    csv_row = ['Sentiment-R','Sentiment-SVM','Sentiment-NV','Sentiment-KN','Yelp-R','Yelp-SVM','Yelp-NV','Yelp-KN','Sem-R','Sem-SVM','Sem-NV','Sem-KN']

    df1 = pd.DataFrame(columns=('precision', 'recall', 'f1-score'), index = csv_row)

    df2 = pd.DataFrame(columns=('precision-neg','recall-neg', 'f1-score-neg', 'precision-pos', 'recall-pos', 'f1-score-pos', 'precision-neu', 'recall-neu', 'f1-score-neu'), index = csv_row)

    df3 = pd.DataFrame(columns=('precision', 'recall', 'f1-score'), index = csv_row)

    df4 = pd.DataFrame(columns=('precision-macro', 'recall-macro', 'f1-score-macro', 'precision-weighted', 'recall-weighted', 'f1-score-weighted'), index = csv_row)

    df5 = pd.DataFrame(columns=('Accuracy', 'Precision', 'Recall', 'F1 Score'), index = csv_row)

    

    return df1, df2, df3, df4
def fill_out_dataframes(name, report_split, df2, df3, df4, df5, init_param, df6):

    i=4



    parameters = []

    precision = []

    recall = []

    score = []

    support = []

    sentiments_level = []



    while( i < len(report_split)-1):

        

        if report_split[i] == "macro" or report_split[i] == "weighted":

            param = report_split[i] + " " + report_split[i+1] 

            parameters.append(param)

            precision.append(report_split[i+2])

            recall.append(report_split[i+3])

            score.append(report_split[i+4])

            support.append(report_split[i+5])

            i = i + 6

        else :

            parameters.append(report_split[i])

            if report_split[i] == "accuracy": 

                precision.append(-1)

                recall.append(-1)

                score.append(report_split[i+1])

                support.append(report_split[i+2])

                i = i+3

            else: 

                precision.append(report_split[i+1])

                recall.append(report_split[i+2])

                score.append(report_split[i+3])

                support.append(report_split[i+4])

                i = i+5

                

    j = 0 

    while( parameters[j] != 'accuracy' and j < len(parameters)):

        sentiments_level.append(parameters[j])

        j = j + 1

                

                

    df1 = pd.DataFrame({'precision': precision, 'recall': recall, 'f1-score': score, 'support': support}, index=parameters)

    

    

    

    datos = [df1.loc['macro avg', ['precision']]['precision'],df1.loc['macro avg', ['recall']]['recall'],df1.loc['macro avg', ['f1-score']]['f1-score']]

    df2.loc[name] = datos

    

    m = 0

    datos1 = []

    while(m < 3):

        

        if (len(sentiments_level) == 2 and m == 2):

            datos1.append(-1)

            datos1.append(-1)

            datos1.append(-1)

        else : 

            datos1.append(df1.loc[sentiments_level[m], ['precision']]['precision'])

            datos1.append(df1.loc[sentiments_level[m], ['recall']]['recall'])

            datos1.append(df1.loc[sentiments_level[m], ['f1-score']]['f1-score'])

        

        m = m + 1

            

    

    df3.loc[name] = datos1

    

    datos2 = [df1.loc['weighted avg', ['precision']]['precision'],df1.loc['weighted avg', ['recall']]['recall'],df1.loc['weighted avg', ['f1-score']]['f1-score']]

    df4.loc[name] = datos2

    

    datos3 = [df1.loc['macro avg', ['precision']]['precision'],df1.loc['macro avg', ['recall']]['recall'],df1.loc['macro avg', ['f1-score']]['f1-score'],df1.loc['weighted avg', ['precision']]['precision'],df1.loc['weighted avg', ['recall']]['recall'],df1.loc['weighted avg', ['f1-score']]['f1-score']]

    df5.loc[name] = datos3

    

    df6.loc[name] = init_param

    
def create_csv(df1,df2,df3,df4,df5):

    df1.to_csv('macro-avg.csv')

    df2.to_csv('values-data.csv')

    df3.to_csv('weighted-avg.csv')

    df4.to_csv('macro-weighted-avg.csv')

    df5.to_csv('metrics.csv')

  
def delete_neutral(sentiment, data):

    data_cleaned = data.loc[data['Sentiment'] != sentiment, ['Sentiment', 'Tweet']]

    return data_cleaned
data_train = pd.read_csv("/kaggle/input/semeval/semevalTrain.csv")

data_train
train = delete_neutral(0, data_train)

train
data_test = pd.read_csv("/kaggle/input/semeval/semevalTest.csv")

data_test
test = delete_neutral(0, data_test)

test
tfidf, train_bag, test_bag = bag_of_words(train['Tweet'].values.astype(str), test['Tweet'].values.astype(str))
rfc = RandomForestClassifier(n_jobs=-1)
rfc_tfidf_predictions = train_predict_model(rfc,train_bag, train['Sentiment'],

                                    test_bag, test['Sentiment'])
parameters1, report1 = display_model_performance_metrics(test['Sentiment'], rfc_tfidf_predictions,[-1,1])
neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
neigh_tfidf_predictions = train_predict_model(neigh,train_bag, train['Sentiment'],

                                    test_bag, test['Sentiment'])
parameters2, report2 = display_model_performance_metrics(test['Sentiment'], neigh_tfidf_predictions,[-1,1])
gnb = GaussianNB()
gnb_tfidf_predictions = train_predict_model(gnb,train_bag.toarray(), train['Sentiment'], test_bag.toarray(), test['Sentiment'])
parameters3, report3 = display_model_performance_metrics(test['Sentiment'], gnb_tfidf_predictions,[-1,1])
clf = svm.SVC()
clf_tfidf_predictions = train_predict_model(clf,train_bag, train['Sentiment'], test_bag, test['Sentiment'])
parameters4,report4 = display_model_performance_metrics(test['Sentiment'], clf_tfidf_predictions,[-1,1])
df1 = pd.read_csv("/kaggle/input/results/macro-avg.csv", encoding='ISO-8859-1', engine='python', index_col = 0)

df2 = pd.read_csv("/kaggle/input/results/values-data.csv", encoding='ISO-8859-1', engine='python', index_col = 0)

df3 = pd.read_csv("/kaggle/input/results/weighted-avg.csv", encoding='ISO-8859-1', engine='python', index_col = 0)

df4 = pd.read_csv("/kaggle/input/results/macro-weighted-avg.csv", encoding='ISO-8859-1', engine='python', index_col = 0)

df5 = pd.read_csv("/kaggle/input/results/metrics.csv", encoding='ISO-8859-1', engine='python', index_col = 0)
report1

report1_split = report1.split()

fill_out_dataframes('Sem-R', report1_split, df1, df2, df3, df4, parameters1, df5)
report2

report2_split = report2.split()

fill_out_dataframes('Sem-KN', report2_split, df1, df2, df3, df4, parameters2, df5)
report3

report3_split = report3.split()

fill_out_dataframes('Sem-NB', report3_split, df1, df2, df3, df4, parameters3, df5)
report4

report4_split = report4.split()

fill_out_dataframes('Sem-SVM', report4_split, df1, df2, df3, df4, parameters4, df5)
create_csv(df1, df2, df3, df4, df5)
df1
df2
df3
df4
df5