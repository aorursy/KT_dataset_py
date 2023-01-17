# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os, sys

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pwd
!ls ../
!ls ../input/amazon-fine-food-reviews/
!pip install jupyterthemes
import pandas as pd

import numpy as np

import sqlite3

import re



import seaborn as sns

from matplotlib import pyplot as plt

from wordcloud import WordCloud, STOPWORDS

from jupyterthemes import jtplot
# from sklearn.preprocessing import StandardScaler



from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score

from sklearn.model_selection import train_test_split



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.dummy import DummyClassifier



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from xgboost import XGBClassifier



import nltk

from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk import ngrams



from string import punctuation

from itertools import chain



# from imblearn.over_sampling import SMOTE

# from imblearn.under_sampling import RandomUnderSampler
# set plot rc parameters



# jtplot.style(grid=False)

plt.rcParams['figure.facecolor'] = 'white'

plt.rcParams['axes.facecolor'] = '#464646'

#plt.rcParams['axes.edgecolor'] = '#FFFFFF'

plt.rcParams['figure.figsize'] = 10, 7

plt.rcParams['text.color'] = '#666666'

plt.rcParams['axes.labelcolor'] = '#666666'

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['axes.titlesize'] = 16

plt.rcParams['xtick.color'] = '#666666'

plt.rcParams['xtick.labelsize'] = 14

plt.rcParams['ytick.color'] = '#666666'

plt.rcParams['ytick.labelsize'] = 14



# plt.rcParams['font.size'] = 16



sns.color_palette('dark')

%matplotlib inline
# establish connection with database



conn = sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite')
# sneak peek of data



pd.read_sql_query('''SELECT * FROM Reviews LIMIT 10''', conn)
# read data into data frame

# drop data points where score is 3 neutral

# score > 3 are positive reviews and score < 3 are negative



data_df = pd.read_sql_query('''

SELECT

    HelpfulnessNumerator as helpful,

    HelpfulnessDenominator,

    Score,

    Text

FROM

    Reviews

WHERE

    Score != 3''', conn)
data_df.shape
# drop duplicate data points



data_df.drop_duplicates(inplace=True)

data_df.shape
data_df.head()
data_df['Sentiment'] = data_df['Score'].apply(lambda x: 1 if x>3 else 0)
# remove non alphabetic characters



def remove_non_alphabet(sentence):

    pattern = re.compile(r'[^a-z]+')

    sentence = sentence.lower()

    sentence = pattern.sub(' ', sentence).strip()

    

    return sentence



data_df['Clean_text'] = data_df['Text'].apply(lambda x: remove_non_alphabet(x))
# remove stopwords and stemming



def remove_stop_words(sentence):

    # Tokenize

    word_list = word_tokenize(sentence)

    # stop words

    stopwords_list = set(stopwords.words('english'))

    # remove stop words

    word_list = [word for word in word_list if word not in stopwords_list]

    # stemming

    ps  = PorterStemmer()

    word_list = [ps.stem(word) for word in word_list]

    # list to sentence

    sentence = ' '.join(word_list)

    

    return sentence



data_df['Clean_text'] = data_df['Clean_text'].apply(lambda x: remove_stop_words(x))
data_df.head()
# vectorize data 



# bag of words

bow = CountVectorizer()

X_bow = bow.fit_transform(data_df['Clean_text'])

# Xtest_bow = bow.fit_transform(test_df['Clean_text'])



# tfidf 

tfidf = TfidfVectorizer()

X_tfidf = tfidf.fit_transform(data_df['Clean_text'])

# Xtest_tfidf = tfidf.fit_transform(test_df['Clean_text'])



# ytrain and ytest

# Ytrain = train_df['Sentiment']

# Ytest = test_df['Sentiment']
Xtrain_bow, Xtest_bow, Ytrain, Ytest = train_test_split(X_bow, data_df['Sentiment'], test_size=0.25, random_state=12)

Xtrain_tfidf, Xtest_tfidf, Ytrain, Ytest = train_test_split(X_tfidf, data_df['Sentiment'], test_size=0.25, random_state=12)
# plot word cloud function



def plot_wordcloud(sentences, title):

    # create word cloud

    wordcloud = WordCloud(background_color='black',

                          max_words=200).generate(str(sentences))

    # plt params

    fig = plt.figure(figsize=[15,15])

    plt.axis('off')

    plt.suptitle(title, fontsize=18)

    plt.subplots_adjust(top=1.4)

    plt.imshow(wordcloud)

    plt.show()

    

    return
# plot word cloud for training data with positive examples

plot_wordcloud(data_df[data_df['Sentiment'] == 1]['Clean_text'], 'data points with positive sentiment')
# plot word cloud for training data with negative examples

plot_wordcloud(data_df[data_df['Sentiment'] == 0]['Clean_text'], 'data points with negative sentiment')
fig = plt.figure(figsize=[6,8])

plt.suptitle('Sentiment Distribution', fontsize=18)

ax = sns.countplot(data=data_df,x='Sentiment')

ax.set_xticklabels(['negative', 'positive'])

plt.show()
# Function to print model performance summary statistics



def performance_summary(model, Xtrain, Xtest, Ytrain, Ytest):

    

    Ytrain_pred = model.predict(Xtrain)

    Ytest_pred = model.predict(Xtest)



    # model performance

    # accuracy score

    print('Training Accuracy:\n', accuracy_score(Ytrain, Ytrain_pred))

    print('\n')

    print('Test Accuracy:\n', accuracy_score(Ytest, Ytest_pred))

    print('\n')

    # classification report

    print('Classification Report training:\n', classification_report(Ytrain,Ytrain_pred))

    print('\n')

    print('Classification Report test:\n', classification_report(Ytest,Ytest_pred))

    

    return
# Function to plot Confusion matrix



def plot_confusion_matrix(model, Xtrain, Xtest, Ytrain, Ytest):

    

    Ytrain_pred = model.predict(Xtrain)

    Ytest_pred = model.predict(Xtest)



    # confusion matrix

    fig, axs = plt.subplots(1,2,

                            figsize=[15,5])

    axs = axs.flatten()

    

    axs[0].title.set_text('Training data')

    # axs[0].set_xlabel('Predicted label')

    # axs[0].set_ylabel('True label')

    axs[1].title.set_text('Test data')

    # axs[1].set_xlabel('Predicted label')

    # axs[1].set_ylabel('True label')

    

    fig.text(0.26, 0.01, 'Predicted label', ha='center', size=14)

    fig.text(0.69, 0.01, 'Predicted label', ha='center', size=14)

    fig.text(0.08, 0.5, 'True label', va='center', rotation='vertical', size=14)

    fig.text(0.5, 0.5, 'True label', va='center', rotation='vertical', size=14)

    

    sns.heatmap(confusion_matrix(Ytrain,Ytrain_pred),

                    annot=True,

                    xticklabels=['negative', 'positive'],

                    yticklabels=['negative', 'positive'],

                    fmt="d",

                    ax=axs[0])

    

    sns.heatmap(confusion_matrix(Ytest,Ytest_pred),

                    annot=True,

                    xticklabels=['negative', 'positive'],

                    yticklabels=['negative', 'positive'],

                    fmt="d",

                    ax=axs[1])

    plt.show()

    

    return
# Function to plot ROC



def plot_roc(model, Xtrain, Xtest, Ytrain, Ytest):

    # ROC curve and area under ROC curve



    # get FPR and TPR for training and test data

    Ytrain_pred_proba = model.predict_proba(Xtrain)

    fpr_train, tpr_train, thresholds_train = roc_curve(Ytrain, Ytrain_pred_proba[:,1])

    # tpr fpr are swapped 

    roc_auc_train = auc(fpr_train, tpr_train)

    Ytest_pred_proba = model.predict_proba(Xtest)

    fpr_test, tpr_test, thresholds_test = roc_curve(Ytest, Ytest_pred_proba[:,1])

    # tpr fpr are swapped

    roc_auc_test = auc(fpr_test, tpr_test)



    # print area under roc curve

    print ('AUC_ROC train:\t', roc_auc_train)

    print ('AUC_ROC test:\t', roc_auc_test)



    # plot auc roc

    fig, axs = plt.subplots(1,2,

                            figsize=[15,5],

                            sharex=False,

                            sharey=False)

    

    # training data

    axs[0].set_title('Receiver Operating Characteristic trainning')

    axs[0].plot(fpr_train,

                tpr_train,

                sns.xkcd_rgb['greenish cyan'],

                label='AUC = %0.2f'% roc_auc_train)

    axs[0].legend(loc='lower right', facecolor='#232323')

    

    axs[0].plot([0,1],[0,1],

                ls='--',

                c=sns.xkcd_rgb['red pink'])

    

    axs[0].set_xlim([-0.01,1.01])

    axs[0].set_ylim([-0.01,1.01])

    axs[0].set_ylabel('True Positive Rate')

    axs[0].set_xlabel('False Positive Rate')

    

    # test data

    axs[1].set_title('Receiver Operating Characteristic testing')

    axs[1].plot(fpr_test,

                tpr_test,

                sns.xkcd_rgb['greenish cyan'],

                label='AUC = %0.2f'% roc_auc_test)

    axs[1].legend(loc='lower right', facecolor='#232323')

    

    axs[1].plot([0,1],[0,1],

                ls='--',

                c=sns.xkcd_rgb['red pink'])

    

    axs[1].set_xlim([0.0,1.0])

    axs[1].set_ylim([0.0,1.0])

    axs[1].set_ylabel('True Positive Rate')

    axs[1].set_xlabel('False Positive Rate')



    plt.show()

    

    return
lr_clf = GridSearchCV(LogisticRegression(),

                     cv=5,

                     param_grid={'C':[1e-2,1e-1,1,10,100]},

                     scoring='accuracy')
lr_clf.fit(Xtrain_bow, Ytrain)
lr_clf.best_params_, lr_clf.best_score_
# train logistic regression model for bag of words

logreg_bow = LogisticRegression(C=1)

logreg_bow.fit(Xtrain_bow, Ytrain)



# train logistic regression model for tfidf

logreg_tfidf = LogisticRegression(C=10)

logreg_tfidf.fit(Xtrain_tfidf, Ytrain)
# model performance

print('*'*25+'Bag of Words'+'*'*25)

performance_summary(logreg_bow, Xtrain_bow, Xtest_bow, Ytrain, Ytest)

print('\n\n\n')

print('*'*28+'TF-IDF'+'*'*28)

performance_summary(logreg_tfidf, Xtrain_tfidf, Xtest_tfidf, Ytrain, Ytest)
# pot confusion matrix and roc curve

# Bag of words

print('*'*25+'Bag of words'+'*'*25)

plot_confusion_matrix(logreg_bow, Xtrain_bow, Xtest_bow, Ytrain, Ytest)

plot_roc(logreg_bow, Xtrain_bow, Xtest_bow, Ytrain, Ytest)



# Tf idf

print('*'*28+'TF-IDF'+'*'*28)

plot_confusion_matrix(logreg_tfidf, Xtrain_tfidf, Xtest_tfidf, Ytrain, Ytest)

plot_roc(logreg_tfidf, Xtrain_tfidf, Xtest_tfidf, Ytrain, Ytest)