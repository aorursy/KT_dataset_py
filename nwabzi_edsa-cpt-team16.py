#from google.colab import drive

#drive.mount('/content/gdrive')
import nltk

import string

from nltk.corpus import stopwords

from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#train  =  pd.read_csv('/content/gdrive/My Drive/Explore/train.csv' )

#test = pd.read_csv('/content/gdrive/My Drive/Explore/test.csv' )



train  =  pd.read_csv('train.csv' )

test = pd.read_csv('test.csv' )
train.head(3)
train.isnull().sum()
train.shape
train.iloc[1,1].split('|||')[:10]


print('We have : ',len(train.iloc[1,1].split('||')), ' posts in each row.')
train['type'].unique()
total = train.groupby(['type']).count()*50

total #### show the total dataframe


plt.figure(figsize = (12,6))



plt.bar(np.array(total.index), height = total['posts'],)

plt.xlabel('Personality types', size = 14)

plt.ylabel('Number of posts available', size = 14)

plt.title('Total posts for each personality type')


train['mind'] = train['type'].apply(lambda x: x[0] == 'E').astype('int')

train['energy'] = train['type'].apply(lambda x: x[1] == 'N').astype('int')

train['nature'] = train['type'].apply(lambda x: x[2] == 'T').astype('int')

train['tactics'] = train['type'].apply(lambda x: x[3] == 'J').astype('int')
train.head()
messages=pd.concat([train,test],join='inner')
messages.info()
pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

subs_url = r'url-web'

messages['posts'] = messages['posts'].replace(to_replace = pattern_url, value = subs_url, regex = True)
messages.head()
def remove_punctuation(text):

    '''a function for removing punctuation'''

    import string

    # replacing the punctuations with no space, 

    # which in effect deletes the punctuation marks 

    translator = str.maketrans('', '', string.punctuation)

    # return the text stripped of punctuation marks

    return text.translate(translator)
messages['posts'] = messages['posts'].apply(remove_punctuation)
messages.head()
#### let's confirm if we correctly got stopwords

stopwords.words('english')[0:10] 
sw = stopwords.words('english')
def remove_stopwords(text):

    '''a function for removing the stopword'''

    # removing the stop words and lowercasing the selected words

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    # joining the list of words with space separator

    return " ".join(text)
messages['posts'] = messages['posts'].apply(remove_stopwords)
messages.head(3)
#### let's take the words to their root source

stemmer = SnowballStemmer("english")



def stemming(text):    

    '''a function which stems each word in the given text'''

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text) 
messages['posts'] = messages['posts'].apply(stemming)
#count_vectorizer = CountVectorizer()
tfid_vectorizer = TfidfVectorizer("english")


#print(np.mean(train.mind==1))

#print(np.mean(train.energy==1))

#print(np.mean(train.nature==1))

#print(np.mean(train.tactics==1))




msg_train1, msg_test1, label_train1, label_test1 =train_test_split(messages['posts'].iloc[0:6506], train['mind'])

msg_train2, msg_test2, label_train2, label_test2 =train_test_split(messages['posts'].iloc[0:6506], train['energy'])

msg_train3, msg_test3, label_train3, label_test3 =train_test_split(messages['posts'].iloc[0:6506], train['nature'])

msg_train4, msg_test4, label_train4, label_test4 =train_test_split(messages['posts'].iloc[0:6506], train['tactics'])
#from sklearn.pipeline import Pipeline



#pipeline = Pipeline([

 #   ('bow', CountVectorizer(ngram_range=(1,3),min_df=2,max_df=1.0)),#,binary=True)),  # strings to token integer counts

  #  ('tfidf', TfidfTransformer()),

   # ('model',LogisticRegression(solver='lbfgs',multi_class='ovr',C=5,class_weight='balanced') ),# integer counts to weighted TF-IDF scores,# train on TF-IDF vectors w/ Naive Bayes classifier

#])
#### fit the train for the mind



pipeline.fit(msg_train1,label_train1)

predictions1 = pipeline.predict(msg_test1)

y_prob=pipeline.predict_proba(msg_test1)

y_thresh = np.where(y_prob[:,1] > 0.77, 1, 0)



#### print the results for mind



print(accuracy_score(predictions1,label_test1)*100)

print(log_loss(predictions1,label_test1))

pipeline.fit(messages['posts'].iloc[0:6506],train['mind'])

predictions1=pipeline.predict(messages['posts'].iloc[6506:])

y_thresh1=pd.Series(predictions1)

y_thresh1[:10]
#### fit for the energy trait



pipeline.fit(msg_train2,label_train2)

predictions2 = pipeline.predict(msg_test2)

y_prob=pipeline.predict_proba(msg_test2)

y_thresh2 = np.where(y_prob[:,1] > 0.14, 1, 0)

#### print the Results for energy

print(accuracy_score(predictions2,label_test2)*100)

print(log_loss(predictions2,label_test2))

pipeline.fit(messages['posts'].iloc[0:6506],train['energy'])

predictions2=pipeline.predict(messages['posts'].iloc[6506:])

y_thresh2=pd.Series(predictions2)

y_thresh2[:10]
#### fit for the nature trait



pipeline.fit(msg_train3,label_train3)

predictions3 = pipeline.predict(msg_test3)

y_prob=pipeline.predict_proba(msg_test3)

y_thresh3 = np.where(y_prob[:,1] > 0.54, 1, 0)
#### print the Results for energy

print(accuracy_score(predictions3,label_test3)*100)

print(log_loss(predictions3,label_test3))

pipeline.fit(messages['posts'].iloc[0:6506],train['nature'])

predictions3=pipeline.predict(messages['posts'].iloc[6506:])

y_thresh3=pd.Series(predictions3)

y_thresh3[:10]


#### fit for the tactics trait



pipeline.fit(msg_train4,label_train4)

predictions4 = pipeline.predict(msg_test4)

y_prob=pipeline.predict_proba(msg_test4)

y_thresh4 = np.where(y_prob[:,1] > 0.60, 1, 0)
#### print the Results for nature

print(accuracy_score(predictions4,label_test4)*100)

print(log_loss(predictions4,label_test4))

pipeline.fit(messages['posts'].iloc[0:6506],train['tactics'])

predictions4=pipeline.predict(messages['posts'].iloc[6506:])

y_thresh4=pd.Series(predictions4)

y_thresh4[:10]



#### Save our results for kaggle



values=pd.concat([y_thresh1,y_thresh2,y_thresh3,y_thresh4],axis=1)

values.index=test.index

values=values.rename(columns={0:'mind',1:'energy',2:'nature',3:'tactics'})

values.index=test.id

values.head()

values.info()

values.to_csv('/content/gdrive/My Drive/Explore/Final_Output.csv')






