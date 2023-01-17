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
path='/kaggle/input/quora-insincere-questions-classification/train.csv'

train = pd.read_csv(path,nrows=1000)

train.head()
#target=0 - can be asked in public forum

#target=1 - insincere questions

#qid is just ID - we can ignore
## converting every character to lower case

docs=train['question_text'].str.lower()

docs.sample(20)

##remove non-alphabets

docs= docs.str.replace('[^a-z ]','') # retaining alphabets, spaces and remove evrything else
## remove commonly used words



import nltk #natural language tool kit

stopwords=nltk.corpus.stopwords.words('english')

stemmer=nltk.stem.PorterStemmer()

def clean_sentance(doc):

    words=doc.split(' ')

    words_clean=[stemmer.stem(word) for word in words if word not in stopwords] #removing the common words

# and stemming all the words

    return ' '.join(words_clean) #because sklearn expects one string

    

docs=docs.apply(clean_sentance)
#docs.apply(lambda v: v.split(' ')).head() - another way
#Example



# stemmer=nltk.stem.PorterStemmer()

# stemmer.stem('organization')
# Document term matrix- is also called as sparsity matrix because maximum will be 0 in the matrix
#Term Frequency - Inverse Document Frequency (TF-IDF)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



dtm_vectorizer=CountVectorizer()



train_x,validate_x,train_y,validate_y=train_test_split(docs,train['target'],test_size=0.2,random_state=1)



dtm_vectorizer.fit(train_x) #it only identifies unique words - only fit



#do not use fit and transform together



#always first fit and then transform on training, testing etc

#or the columns order will change in the test data





dtm_train=dtm_vectorizer.transform(train_x) # we are tranforming here on train data

#it will compress the matrix, ignore 0's and only give number - compressed output and will give it as input to model

dtm_validate=dtm_vectorizer.transform(validate_x) # we are tranforming here on test data

dtm_train #2405 unique values in training matrix



#800x2405 - total number of elements

#out of whihc 4822 non 0 values we have

# most of the are 0's. Hence, sparse metrix
df_dtm_train=pd.DataFrame(dtm_train.toarray(),columns=dtm_vectorizer.get_feature_names(),index=train_x.index)

df_dtm_train
df_dtm_train.sum().sort_values(ascending=False).head(20).plot.bar()
from sklearn.naive_bayes import MultinomialNB

model=MultinomialNB().fit(dtm_train,train_y)

train_y_pred=model.predict(dtm_validate)



from sklearn.metrics import accuracy_score,f1_score

print(accuracy_score(validate_y,train_y_pred))

print(f1_score(validate_y,train_y_pred))
from nltk.sentiment import SentimentIntensityAnalyzer

sentiment_analyzer=SentimentIntensityAnalyzer()

sentiment_analyzer.polarity_scores('i like india')