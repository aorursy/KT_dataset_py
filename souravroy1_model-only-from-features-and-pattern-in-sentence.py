



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input/train.csv"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # to do mathametical operation

import string #for text pre-processing

from nltk.corpus import stopwords #for removing stopwords

import re #Regular expression operations

import xgboost as xgb #For predicting the values

from sklearn.model_selection import KFold #for cross validations(CV)

from sklearn import metrics #for getting CV score

from collections import Counter #counting of words in the texts

import operator

from nltk import ngrams

import nltk # major package for language processing

from nltk import word_tokenize # for toconizing

import matplotlib.pyplot as plt
training_df = pd.read_csv("../input/train.csv")
training_df.head()
training_author_df=training_df.groupby('author',as_index=False).count()

training_author_df
# taking in the first field

text_string=training_df.iloc[0]['text']

text_string
string.punctuation

def remove_punctuation_from_string(string1):

    string1=string1.lower() # changing to lower case

    translation_table=dict.fromkeys(map(ord,string.punctuation),' ')

    string2=string1.translate(translation_table)

    return string2

print('After processing')

test_string=remove_punctuation_from_string(text_string)

test_string
def remove_stopwords_from_string(string1):

    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*') #compiling all stopwords.

    string2 = pattern.sub('', string1) #replacing the occurrences of stopwords in string1

    return string2



print('After processing')

test_string = remove_stopwords_from_string(test_string)

test_string
training_df['text_backup']=training_df['text']
# usiing apply to remove the unwanted words.

training_df['text']=training_df['text'].apply(lambda x: remove_punctuation_from_string(x))

training_df['text']=training_df['text'].apply(lambda x: remove_stopwords_from_string(x))
# now I have cleaned the data. Its time for processing the data and create features from it.

#Feature 1 : Finding total words in the sentance

training_df['feature1']=training_df['text_backup'].apply(lambda x: len(str(x).split()))
#Feature 2 : Counting no of charecter in a variable

training_df['feature2']=training_df['text_backup'].apply(lambda x: len(str(x)))
#Feature 3 : Avg leangth of words used in the sentance.

training_df['feature3']=training_df['feature2']/training_df['feature1']
# Feature 4: Count total stop words  in a sentence.

stop_words=set(stopwords.words('english'))

training_df['Feature4']=training_df['text_backup'].apply(lambda x: len([w for w in str(x).lower().split() 

                                                                       if w in stop_words ]))
#finding the words that are used the most 

all_text_without_sw= ''

for i in training_df.itertuples():

    all_text_without_sw = all_text_without_sw +str(i.text)

    #getting count of each word

    counts=Counter(re.findall(r"[\w']+", all_text_without_sw))

    #deleting from counts

    del counts["'"]

    # getting top 50 words

    sorted_x=dict(sorted(counts.items(), key=operator.itemgetter(1), reverse = True)[:50])

    

# Feature 5 : The count of top words

    

training_df['Feature5']= training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() 

                                                                     if w in sorted_x]))
# Feature 6 : least used words

reverted_x=dict(sorted(counts.items(), key=operator.itemgetter(1))[:1000])



training_df['Feature6']=training_df['text'].apply(lambda x: len ([w for w in str(x).lower().split() 

                                                                  if w in reverted_x]))
# Feature 7 : Find the total no of puntuation

training_df['Feature7']= training_df['text_backup'].apply(lambda x: len([w for w in str(x) 

                                                                         if w in string.punctuation]))
#Feature-8: Count of UPPER case words.



training_df['Feature8']=training_df['text'].apply(lambda x: 

                                                  len([w for w in str(x).replace('I','i')

                                                                 .replace('A','a').split() if w.isupper()==True])) 
#Feature-9: Count of Title case words



training_df['Feature9']= training_df['text'].apply(lambda x: len([w for w in str(x).replace('I','i')

                                                                  .replace('A','a').split() if w.istitle==True]))
starting_words = sorted(list(map(lambda word : word[:2],filter(lambda word :

                                                               len(word) > 3,all_text_without_sw.split()))))

sw_counts = Counter(starting_words)

top_30_sw = dict(sorted(sw_counts.items(), key=operator.itemgetter(1),reverse=True)[:30])







#Feature-10: Count of (Most words start with)

training_df['Feature_10'] = training_df['text'].apply(lambda x: 

                                                      len([w for w in str(x).lower().split()

                                                           if w[:2] in top_30_sw and w not in stop_words]) )
#Feature-11: Count of (Most words end with)

ending_words = sorted(list(map(lambda word : word[-2:],filter(lambda word : len(word) > 3,all_text_without_sw.split()))))

ew_counts = Counter(ending_words)

top_30_ew = dict(sorted(sw_counts.items(), key=operator.itemgetter(1),reverse=True)[:30])

training_df['Feature_11'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() 

                                                                     if w[:2] in top_30_ew and w not in stop_words]) )
di = {'EAP': 0,'HPL':1, 'MWS':2}

training_df=training_df.replace({"author": di})

testing_df=testing_df.replace({"author": di})
y=training_df['author']

X=training_df.drop(['author'],1)

y=pd.get_dummies(y)
y.head()
from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y, test_size=0.33, random_state=42)
#converting to matrix

Xtrain=Xtrain.values

ytrain=ytrain.values

Xtest=Xtest.values

ytest=ytest.values
import keras

from keras.models import Sequential

from keras.layers import Dense
classifier=Sequential() 

training_df.shape
classifier.add(Dense(units=10, kernel_initializer="uniform",activation='relu',input_dim=11))

classifier.add(Dense(units=8, kernel_initializer="uniform", activation='softmax'))

classifier.add(Dense(units=6, kernel_initializer="uniform", activation='relu'))

classifier.add(Dense(units=3, activation='softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(Xtrain,ytrain,batch_size=5, epochs=10)
from sklearn.ensemble import RandomForestClassifier

ry=training_df['author']

rX=training_df.drop(['author'],1)

rXtrain,rXtest,rytrain,rytest=train_test_split(rX,ry, test_size=0.33, random_state=42)
clf=RandomForestClassifier()

clf.fit(rXtrain,rytrain)

from sklearn.model_selection import cross_val_score

print(cross_val_score(clf, rXtrain, rytrain))