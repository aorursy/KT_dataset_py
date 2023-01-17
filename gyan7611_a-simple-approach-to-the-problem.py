import pandas as pd
df_train = pd.read_csv('../input/train.csv')



df_test = pd.read_csv('../input/test.csv')
#Let's see the first few rows of the training set

df_train.head()
#Let's check the no. of classes

df_train['author'].unique()
'''

Let's check if the classes are in comparable proportion just to be sure if it 

is a case of an unbalanced problem

'''

import matplotlib.pyplot as plt

import seaborn as sns

counts = df_train['author'].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(counts.index, counts.values, alpha=0.8)

plt.ylabel('Frequency')

plt.xlabel('Author')

plt.show()
#We will use nltk for getting the list of stopwords

import nltk

from nltk.corpus import stopwords

st = stopwords.words('english')
#Let's get our stopword free text in both train and test

df_train['text'] = df_train['text'].map(lambda x : ' '.join([token for token in x.split() if token.lower() not in st]))

df_test['text'] = df_test['text'].map(lambda x : ' '.join([token for token in x.split() if token.lower() not in st]))




#Importing the count vectorizer

from sklearn.feature_extraction.text import CountVectorizer



#Initializing the vectorizer

count_vect = CountVectorizer()



#Transforming the words to vectors

X_train_counts = count_vect.fit_transform(df_train['text'])
#Let's use Naive Bayes for the first predictions

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
#Defining x and y for the model

x = X_train_counts

y = df_train['author']
#Fitting the model

clf.fit(x,y)
#Transforming the test set words to vectors

x_test = df_test['text']

x_test = count_vect.transform(x_test)
#Getting the probabilities for all the classes

probs = clf.predict_proba(x_test)
#Getting different variables for different columns

EAP = [x[0] for x in probs]

HPL = [x[1] for x in probs]

MWS = [x[2] for x in probs]
#Creating the columns

df_test['EAP'] = EAP

df_test['HPL'] = HPL

df_test['MWS'] = MWS
#Structuring the output as the sample submission

df_test = df_test[['id','EAP','HPL','MWS']]

df_test.to_csv('f21.csv',index=False)
#Writing to disk

df_test.to_csv('output.csv',index=False)