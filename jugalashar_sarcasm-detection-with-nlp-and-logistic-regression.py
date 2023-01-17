#Loading all the necessary libraries needed

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sn

import os
#Reading the Headlines from a Json file and checking if data is balanced or not. It is sort of balanced

df= pd.read_json("../input/Sarcasm_Headlines_Dataset.json", lines=True)

df.head()

df.info()

df['is_sarcastic'].value_counts()
#display the data distribution by a barplot

labels = ['Sarcastic', 'Not Sarcastic']

count_sarcastic = len(df[df['is_sarcastic']==1])

count_notSarcastic = len(df[df['is_sarcastic']==0])

values =[count_sarcastic, count_notSarcastic]

y_pos = [0,1]

bars = plt.bar(x=y_pos, height=values, width=.15, color='y')

for bar in bars:

    yval = bar.get_height()

    plt.xticks(y_pos, labels)

    plt.text(bar.get_x(), yval + .1, yval)
#splitting data into train and test sets

train_X, test_X, train_y, test_y = train_test_split(df['headline'], df['is_sarcastic'], test_size=0.3)
#exploratory data analysis

#creating a wordcloud for aesthetics

from wordcloud import WordCloud, STOPWORDS



wordcloud = WordCloud(background_color='black', stopwords = STOPWORDS,

                max_words = 100, max_font_size = 100, 

                random_state = 15, width=800, height=400)



plt.figure(figsize=(16, 12))

wordcloud.generate(str(df.loc[df['is_sarcastic'] == 1, 'headline']))

plt.imshow(wordcloud)
#Punctuations dont help in our analysis, thus striping them

import string

from string import digits, punctuation



head_cleaned = []

for head in df['headline']:

    clean = head.translate(str.maketrans('', '', punctuation))

    clean= clean.translate(str.maketrans('', '', digits))

    head_cleaned.append(clean)

    

#comparing the original vs new texts

print('Original texts :')

print(df['headline'][24])

print('\nAfter cleansed :')

print(head_cleaned[24])
#Tokenization

head_tokens =[]

for head in head_cleaned:

    head_tokens.append(head.split())

    

index=50

print('Before tokenization :')

print(head_cleaned[index])

print('\nAfter tokenization :')

print(head_tokens[index])
#Lemmatization

import nltk

nltk.download('wordnet')

from nltk.corpus import wordnet

nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer



def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "ADV": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN)



lemmatizer= WordNetLemmatizer()



head_lemma = []

for tokens in head_tokens:

    lemm=[lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]

    head_lemma.append(lemm)
#Training the model and building a data pipeline of TFIDF and Logistic regression



tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)

# multinomial logistic regression a.k.a softmax classifier

logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', 

                           random_state=17, verbose=1)

tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf), 

                                 ('logit', logit)])



tfidf_logit_pipeline.fit(train_X, train_y)



valid_pred = tfidf_logit_pipeline.predict(test_X)



accuracy_score(test_y, valid_pred)
#Explaining the model by a confusion matrix



confusion_matrix = pd.crosstab(test_y, valid_pred, rownames=['Actual'], colnames=['Predicted'])

#print(confusion_matrix)

sn.heatmap(confusion_matrix, annot=True)
#Displaying the most important words from these headlines

import eli5

eli5.show_weights(estimator=tfidf_logit_pipeline.named_steps['logit'],

                  vec=tfidf_logit_pipeline.named_steps['tf_idf'])