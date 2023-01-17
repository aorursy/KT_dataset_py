import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os



# os.listdir('/kaggle/input/bbc-full-text-document-classification')

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    count = 0

    for filename in filenames:

        if count > 5:

            break

        print(os.path.join(dirname, filename))

        count += 1
# check if both files are same

!diff '/kaggle/input/bbc-full-text-document-classification/bbc-fulltext (document classification)/bbc/tech/128.txt' '/kaggle/input/bbc-full-text-document-classification/bbc/tech/128.txt'
data_dir = '/kaggle/input/bbc-full-text-document-classification/bbc'

os.listdir(data_dir)
!cat {data_dir}/README.TXT
from collections import defaultdict

frame = defaultdict(list)



for dirname, _, filenames in os.walk(data_dir):

    for filename in filenames:

        frame['category'].append(os.path.basename(dirname))

        

        name = os.path.splitext(filename)[0]

        frame['document_id'].append(name)



        path = os.path.join(dirname, filename)

        # throwed UnicodeDecodeError without encoding

        # Googled "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3" - https://stackoverflow.com/a/55391198/7445772

        with open(path ,'r', encoding= 'unicode_escape') as file:

            frame['text'].append(file.read())

df = pd.DataFrame.from_dict(frame)
df.head(3)
df['category'].value_counts()

# bbc is data_dir with just readme file, we can drop it and proceed
df.drop(0, axis=0, inplace=True)
len(df['document_id'].unique())
# seems like incremental numbers for document names 

sorted_ids = sorted(df['document_id'].unique())

print(sorted_ids[:10], sorted_ids[-10:])
import random



num = 5

sample = random.sample(range(df.text.shape[0]), num)



for idx in sample:

    print('*'*30)

    values = df.iloc[idx]

    print('Document ID : ', values['document_id'])

    print('Category : ', values['category'])

    print('Text : \n'+'-'*7)

    print(values['text'])

    print('='*36)
# https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/

text = df["text"].str.split("\n", n = 1, expand = True) 



df["title"] =  text[0]

df['story'] = text[1]
df.head()
for idx in sample[:2]:

    print('*'*30)

    values = df.iloc[idx]

    print('Document ID : ', values['document_id'])

    print('Category : ', values['category'])

    print('Title : \n'+'-'*9)

    print(values['title'])

    print('\nStory : \n'+'-'*9)

    print(values['story'])

    print('='*36)
sns.countplot(df.category)

plt.title('Number of documents in each category')

plt.show()
# Googled "How to calculate number of words in a string in DataFrame" - https://stackoverflow.com/a/37483537/4084039



word_dict = dict(df['title'].str.split().apply(len).value_counts())



idx = np.arange(len(word_dict))

plt.figure(figsize=(20,5))

p1 = plt.bar(idx, list(word_dict.values()))



plt.ylabel('Number of docments')

plt.xlabel('Number of words in document title')

plt.title('Words for each title of the document')

plt.xticks(idx, list(word_dict.keys()))

plt.show()
cat_titles_word_count = defaultdict(list)

for category in df.category.unique():

    val = df[df['category']==category]['title'].str.split().apply(len).values

    cat_titles_word_count[category]=val
# distribution of titles across categories

plt.boxplot(cat_titles_word_count.values())

keys = cat_titles_word_count.keys()

plt.xticks([i+1 for i in range(len(keys))], keys)

plt.ylabel('Words in document title')

plt.grid()

plt.show()
# distribution of words in title

plt.figure(figsize=(10,3))

for key, value in cat_titles_word_count.items():

    sns.kdeplot(value,label=key, bw=0.6)

plt.legend()

plt.title('Distribution of words in title')

plt.show()
category_story_word_count = defaultdict(list)

for category in df.category.unique():

    val = df[df['category']==category]['story'].str.split().apply(len).values

    category_story_word_count[category]=val
# distribution of stories across categories

plt.boxplot(category_story_word_count.values())

plt.title('Distribution of words in stories across categories')

keys = category_story_word_count.keys()

plt.xticks([i+1 for i in range(len(keys))], keys)

plt.ylabel('Words in stories')

plt.grid()

plt.show()
# distribution of words in story

fig, axes = plt.subplots(2, 3, figsize=(15,8), sharey=True)

ax = axes.flatten()

plt.suptitle('Distribution of words in story')



for idx, (key, value) in enumerate(category_story_word_count.items()):

    sns.kdeplot(value,label=key, bw=0.6, ax=ax[idx])

plt.legend()

plt.show()
# printing random titles

samples = random.sample(range(len(df.title)), 10)

for idx in samples:

    print(df.title[idx])

    print('-'*36)
# stop words

import nltk

from nltk.corpus import stopwords

stopwords = stopwords.words('english')

print(stopwords)
import re



def clean_text(text):

    # decontraction : https://stackoverflow.com/a/47091490/7445772

    # specific

    text = re.sub(r"won't", "will not", text)

    text = re.sub(r"can\'t", "can not", text)

    # general

    text = re.sub(r"n\'t", " not", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'s", " is", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'t", " not", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'m", " am", text)



    # remove line breaks \r \n \t remove from string 

    text = text.replace('\\r', ' ')

    text = text.replace('\\"', ' ')

    text = text.replace('\\t', ' ')

    text = text.replace('\\n', ' ')



    # remove stopwords

    text = ' '.join(word for word in text.split() if word not in stopwords)



    # remove special chars

    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    text = text.lower()

    return text
from tqdm import tqdm
processed_titles = []

for title in tqdm(df['title'].values):

    processed_title = clean_text(title)

    processed_titles.append(processed_title)
# titles after processing

for idx in samples:

    print(processed_titles[idx])

    print('-'*36)
for idx in samples[:2]:

    print(df.story[idx])

    print('-'*36)
processed_stories = []

for story in tqdm(df['story'].values):

    processed_story = clean_text(story)

    processed_stories.append(processed_story)
for i in range(5):

    print(df.category.values[i])

    print(processed_titles[i])

    print(processed_stories[i])

    print('-'*100)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelEncoder
# considering only the words which appeared in at least 5 document titles.

vectorizer = CountVectorizer(min_df=5)

title_bow = vectorizer.fit_transform(processed_titles)

print("Shape after one hot encodig ",title_bow.shape)
vectorizer = TfidfVectorizer(min_df=5)

title_tfidf = vectorizer.fit_transform(processed_titles)

print("Shape after one hot encodig ",title_tfidf.shape)
vectorizer = TfidfVectorizer(analyzer = "word", ngram_range =(1, 2),max_features = 750, min_df=5)

title_tfidf_ngram = vectorizer.fit_transform(processed_titles)

print("Shape after one hot encodig ",title_tfidf_ngram.shape)

# considering only the words which appeared in at least 10 document.

vectorizer = CountVectorizer(min_df=10)

story_bow = vectorizer.fit_transform(processed_stories)

print("Shape after one hot encodig ",story_bow.shape)
vectorizer = TfidfVectorizer(min_df=10)

story_tfidf = vectorizer.fit_transform(processed_stories)

print("Shape after one hot encodig ",story_tfidf.shape)
vectorizer = TfidfVectorizer(analyzer = "word", ngram_range =(1, 4),max_features = 7500, min_df=10)

story_tfidf_ngram = vectorizer.fit_transform(processed_stories)

print("Shape after one hot encodig ",story_tfidf_ngram.shape)

vectorizer = LabelEncoder()

category_onehot = vectorizer.fit_transform(df['category'].values)

# print(vectorizer.get_feature_names())



print("Shape after one hot encoding : ",category_onehot.shape)
from scipy.sparse import hstack
X_bow = hstack((title_bow, story_bow))

X_tfidf =  hstack((title_tfidf, story_tfidf))

X_ngram = hstack((title_tfidf_ngram, story_tfidf_ngram))

print(X_bow.shape, X_tfidf.shape, X_ngram.shape)
print(type(category_onehot), type(X_bow))
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn import linear_model, naive_bayes, svm
def show_confusion_matrix(prediction, y_test):

    # https://stackoverflow.com/a/48018785/7445772 

    labels = ['tech', 'sport', 'business', 'entertainment', 'politics']

    cm = confusion_matrix(y_test, prediction, idx)

    print(cm)

    ax= plt.subplot()

    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

    # labels, title and ticks

    ax.set_xlabel('Predicted labels');

    ax.set_ylabel('True labels'); 

    sax.set_title('Confusion Matrix'); 

    ax.xaxis.set_ticklabels(labels, rotation=90); 

    ax.yaxis.set_ticklabels(labels[::-1], rotation=90);

    plt.title('Confusion matrix of the classifier')

    plt.show()

X = X_bow.toarray()

y = category_onehot

print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state=9)



model = linear_model.LogisticRegression()

model.fit(X_train,y_train)

LR_prediction = model.predict(X_test)



model = naive_bayes.MultinomialNB()

model.fit(X_train,y_train)

NB_prediction = model.predict(X_test)



model = svm.SVC()

model.fit(X_train,y_train)

SVM_prediction = model.predict(X_test)
print('Accuracy with BOW vectors \n'+'-'*15)

print(f'Using Logistic regression : {accuracy_score(LR_prediction, y_test)}')

print(f'Using Naive Bayes : {accuracy_score(NB_prediction, y_test)}')

print(f'Using Support Vector Machines : {accuracy_score(SVM_prediction, y_test)}')

show_confusion_matrix(LR_prediction, y_test)

show_confusion_matrix(NB_prediction, y_test)

show_confusion_matrix(SVM_prediction, y_test)
X = X_tfidf.toarray()

y = category_onehot

print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state=9)



model = linear_model.LogisticRegression()

model.fit(X_train,y_train)

LR_prediction = model.predict(X_test)



model = naive_bayes.MultinomialNB()

model.fit(X_train,y_train)

NB_prediction = model.predict(X_test)



model = svm.SVC()

model.fit(X_train,y_train)

SVM_prediction = model.predict(X_test)
print('Accuracy with TF IDF vectors \n'+'-'*15)

print(f'Using Logistic regression : {accuracy_score(LR_prediction, y_test)}')

print(f'Using Naive Bayes : {accuracy_score(NB_prediction, y_test)}')

print(f'Using Support Vector Machines : {accuracy_score(SVM_prediction, y_test)}')

show_confusion_matrix(LR_prediction, y_test)

show_confusion_matrix(NB_prediction, y_test)

show_confusion_matrix(SVM_prediction, y_test)
X = X_ngram.toarray()

y = category_onehot

print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state=9)



model = linear_model.LogisticRegression()

model.fit(X_train,y_train)

LR_prediction = model.predict(X_test)



model = naive_bayes.MultinomialNB()

model.fit(X_train,y_train)

NB_prediction = model.predict(X_test)



model = svm.SVC()

model.fit(X_train,y_train)

SVM_prediction = model.predict(X_test)
print('Accuracy with TF-IDF n-gram vectors \n'+'-'*15)

print(f'Using Logistic regression : {accuracy_score(LR_prediction, y_test)}')

print(f'Using Naive Bayes : {accuracy_score(NB_prediction, y_test)}')

print(f'Using Support Vector Machines : {accuracy_score(SVM_prediction, y_test)}')

show_confusion_matrix(LR_prediction, y_test)

show_confusion_matrix(NB_prediction, y_test)

show_confusion_matrix(SVM_prediction, y_test)
#final model



X = X_tfidf.toarray()

y = category_onehot

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2)



final_model = linear_model.LogisticRegression()

final_model.fit(X_train,y_train)

LR_prediction = final_model.predict(X_test)

accuracy_score(LR_prediction, y_test)
tech_289 = '''Britain's first mobile phone call was made across the Vodafone network on 1 January 1985 by veteran comedian Ernie Wise. In the 20 years since that day, mobile phones have become an integral part of modern life and now almost 90% of Britons own a handset. Mobiles have become so popular that many people use their handset as their only phone and rarely use a landline.



The first ever call over a portable phone was made in 1973 in New York but it took 10 years for the first commercial mobile service to be launched. The UK was not far behind the rest of the world in setting up networks in 1985 that let people make calls while they walked. The first call was made from St Katherine's dock to Vodafone's head office in Newbury which at the time was over a curry house. For the first nine days of 1985 Vodafone was the only firm with a mobile network in the UK. Then on 10 January Cellnet (now O2) launched its service. Mike Caudwell, spokesman for Vodafone, said that when phones were launched they were the size of a briefcase, cost about Â£2,000 and had a battery life of little more than 20 minutes.



"Despite that they were hugely popular in the mid-80s," he said. "They became a yuppy must-have and a status symbol among young wealthy business folk." This was also despite the fact that the phones used analogue radio signals to communicate which made them very easy to eavesdrop on. He said it took Vodafone almost nine years to rack up its first million customers but only 18 months to get the second million. "It's very easy to forget that in 1983 when we put the bid document in we were forecasting that the total market would be two million people," he said. "Cellnet was forecasting half that." Now Vodafone has 14m customers in the UK alone. Cellnet and Vodafone were the only mobile phone operators in the UK until 1993 when One2One (now T-Mobile) was launched. Orange had its UK launch in 1994. Both newcomers operated digital mobile networks and now all operators use this technology. The analogue spectrum for the old phones has been retired. Called Global System for Mobiles (GSM) this is now the most widely used phone technology on the planet and is used to help more than 1.2 billion people make calls. Mr Caudwell said the advent of digital technology also helped to introduce all those things, such as text messaging and roaming that have made mobiles so popular.

'''
