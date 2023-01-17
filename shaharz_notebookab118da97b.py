"""

@author: Shahar Zuler

"""

import pandas as pd     

import re

from nltk.corpus import stopwords 

import random

from sklearn.externals import joblib





def review_to_words( raw_review ):

    ''' preproceccing the sentences'''

    # Remove non-letters:        

    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 

    # Convert to lower case, split into individual words

    words = letters_only.lower().split()                             

    # convert the stop words to a set

    stops = set(stopwords.words("english"))                  

    # Remove stop words

    meaningful_words = [w for w in words if not w in stops]  

    # Join the words back into one string separated by space, nd return the result.

    return( " ".join( meaningful_words ))   





model = joblib.load('../input/SVMmodel2.pkl') 

vectorizer = joblib.load('../input/vectorizer2.pkl')



# reding the CSV file containing all tweets from the test set:

train=pd.DataFrame.from_csv('../input/ObTr_app.csv', encoding='cp1252')  



# choose a random example: 

num=random.randint(0,len(train)-1) 

X=train['data_tweets'][num]



#you can rewrite X manually to ***any tweet you like***. for example:

#X='All across America people chose to get involved, get engaged and stand up. Each of us can make a difference, and all of us ought to try. So go keep changing the world in 2018.'

print('The tweet: ')

print(X)



# preprocess the tweet:

clean_review = review_to_words(X)



# vectorize the tweet and make it an array:

data_feature = vectorizer.transform([clean_review])

data_feature = data_feature.toarray()





print('please type T if you guess Trump and O if you guess Obama')

your_guess = str(input()).lower()

# predict

pred =model.predict(data_feature)[0]

truth = train['data_labels'][num]



# some semanthics:

flag = 1

if your_guess == 't':

    guess = 0

elif your_guess == 'o':

    guess = 1

else:

    print('Please retype only T or O above.')

    flag = 0



defined_guess = 'Trump' if your_guess == 't' else 'Obama'

defined_pred = 'Trump' if pred == 0 else 'Obama'

defined_truth = 'Trump' if truth == 0 else 'Obama'



if flag == 1:

    print('Your guess was: ' + defined_guess)

    print ('My guess was: ' + defined_pred)

    print('And the truth is :' + defined_truth)

    
