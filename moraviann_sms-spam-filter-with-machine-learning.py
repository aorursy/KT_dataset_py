import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re





df = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding='latin-1', usecols = [0, 1])



df.rename(columns = {'v1':'label','v2':'text'},inplace=True)



df['class'] = df.label.map({'ham':0, 'spam':1})



df['copy'] = df.text



df.info()

df.head(10)
import nltk

from nltk.corpus import stopwords



porter = nltk.PorterStemmer() #"distribute", "distributing", "distributor" or "distribution".

stop_words = nltk.corpus.stopwords.words('english')



def clean_text(string):

    message = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', string)

    message = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', #Replace URLs with 'httpaddr'

                     message)

    message = re.sub(r'£|\$', 'money', message) #Replace money symbols with 'moneysymb'

    message = re.sub(

        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', #Replace phone numbers with 'phonenumbr'

        'phonenumbr', message)

    message = re.sub(r'\d+(\.\d+)?', 'numbr', message)  #Replace numbers with 'numbr'

    message = re.sub(r'[^\w\d\s]', ' ', message)

    message = re.sub(r'\s+', ' ', message)

    message = re.sub(r'^\s+|\s+?$', '', message.lower())

    return ' '.join(

    porter.stem(term) 

    for term in message.split()

    if term not in set(stop_words)

    )
clean_text("going to vacation!!! 5734 I have ,.$ £")



textCopy = df['text'].copy()

textCopy = textCopy.apply(clean_text)

df["copy"] = textCopy

df.head(5)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer





vectorizer = CountVectorizer() 



vectorizer.fit(textCopy)



tf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))



ngrams = vectorizer.fit_transform(textCopy).toarray()
from collections import Counter



spam_df = df[df['class'] == 1] #create sub-dataframe of spam text

ham_df = df[df['class'] == 0] #sub-dataframe of ham text

spam_df['copy'] = spam_df['copy'].map(clean_text)

ham_df['copy'] = ham_df['copy'].map(clean_text)

spam_df['new_column'] = spam_df['copy'].apply(lambda x: Counter(x.split(' ')))

forspam=Counter(" ".join(spam_df['copy']).split()).most_common(10)

forham=Counter(" ".join(ham_df['copy']).split()).most_common(10)



spamfeat=[]

for i in range(len(forspam)):

    spamcounter=forspam[i][0]

    spamfeat.append(spamcounter)



spamnumber=[]

for i in range(len(forspam)):

    spamcounter=forspam[i][1]

    spamnumber.append(spamcounter)



hamnumber=[]

for i in range(len(forham)):

    spamcounter=forham[i][1]

    hamnumber.append(spamcounter)

    

hamfeat=[]

for i in range(len(forham)):

    spamcounter=forham[i][0]

    hamfeat.append(spamcounter)
import seaborn as sns



fig, (ax,ax1) = plt.subplots(1,2,figsize = (25, 8))

sns.barplot(x = spamfeat, y=spamnumber, ax = ax)

ax.set_ylabel('count', fontsize = 15)

ax.set_xlabel('word',fontsize = 15)

ax.tick_params(labelsize=12)

ax.set_title('spam top 10 words', fontsize = 15)



sns.barplot(x = hamfeat, y = hamnumber, ax = ax1)

ax1.set_ylabel('count', fontsize = 15)

ax1.set_xlabel('word',fontsize = 15)

ax1.tick_params(labelsize=15)

ax1.set_title('ham top 10 words', fontsize = 15)
from wordcloud import WordCloud



forS=" ".join(spam_df['copy'])

forH=" ".join(ham_df['copy'])

spam_word_cloud = WordCloud(width = 600, height = 400, background_color = 'white').generate(forS)

ham_word_cloud = WordCloud(width = 600, height = 400,background_color = 'white').generate(forH)



fig, (ax, ax2) = plt.subplots(1,2, figsize = (18,8))

ax.imshow(spam_word_cloud)

ax.axis('off')

ax.set_title('spam word cloud', fontsize = 20)

ax2.imshow(ham_word_cloud)

ax2.axis('off')

ax2.set_title('ham word cloud', fontsize = 20)

plt.show()
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split, KFold, cross_val_score



n_folds = 5

def f1_cv(model):

    kf = KFold(n_folds, shuffle = True, random_state = 29).get_n_splits(ngrams)

    f1 = cross_val_score(model, ngrams, df["class"], scoring = 'f1', cv = kf )

    return (f1)



from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression





x_train, x_test, y_train, y_test = train_test_split(ngrams, df['class'].values, test_size=.40)



clfs = {

    'Decision_tree': DecisionTreeClassifier(),

    'gradient_descent': SGDClassifier(),

    'Naive_bayes': GaussianNB(),

    'Logistic_Regression': LogisticRegression()

}



for clf_name in clfs.keys():

    print("Training",clf_name,"classifier")

    clf = clfs[clf_name]

    clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)

    print(classification_report(y_test, y_predict))

    print()


