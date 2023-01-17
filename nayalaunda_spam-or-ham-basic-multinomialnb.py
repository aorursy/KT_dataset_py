
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import  WordCloud 

df = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding = 'ISO-8859-1',usecols =['v1','v2'])

df.columns = ['label','data']

df['b_label'] = df['label'].map({'ham':0 , 'spam':1 })
y = df.iloc[:,-1]

count_vec = CountVectorizer(decode_error = 'ignore')
x = count_vec.fit_transform(df['data'])

xtrain , xtest , ytrain ,ytest = train_test_split(x,y,test_size = 0.3)

clf = MultinomialNB()
clf.fit(xtrain,ytrain)
print('train_score:',clf.score(xtrain,ytrain))
print('test_score:',clf.score(xtest,ytest))

#visualize data

def vis(label):
    words = ''
    for msg in df[df['label']==label]['data']:
        msg = msg.lower()
        words+= msg + ''
    wordcloud = WordCloud(width = 600,height = 400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

vis('spam')
vis('ham')

df['prediction'] = clf.predict(x)

sneaky_spam = df[(df['prediction']==0) & (df['b_label']==1)]['data']
for msg in sneaky_spam:
    print(msg)
    
not_actually_spam = df[(df['prediction']==1) & (df['b_label']==0)]['data']
for msg in sneaky_spam:
    print(msg)