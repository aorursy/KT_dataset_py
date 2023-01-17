import warnings

warnings.filterwarnings("ignore")



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm



import re

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer
stop_words = set(stopwords.words('english'))

stop_words
stemer = SnowballStemmer('english')

print(stemer.stem('Tasty'))
final = pd.read_csv('../input/preprocessed-amazon-fine-food-reviews/preprocessed_reviews.csv')
final
final = final.drop(['preprocessed_reviews','preprocessed_reviews_summary'],axis = 1)
final[1:1000]
final['Score'].value_counts()
final['Text'][27]
def cleanhtml(sntce):

    cleanr = re.compile('<,*?>')

    clntxt = re.sub(cleanr,' ',sntce)

    return clntxt

def cleanpunc(sntce):

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sntce)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    return  cleaned

final_string=[]

for i, sent in enumerate(tqdm(final['Text'].values)):

    filtered_sentence=[]

    #print(sent);

    sent=cleanhtml(sent) # remove HTMl tags

    for w in sent.split():

        for cleaned_words in cleanpunc(w).split():

            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    

                if(cleaned_words.lower() not in stop_words):

                    s=(stemer.stem(cleaned_words.lower())).encode('utf8')

                    filtered_sentence.append(s)

    str1 = b" ".join(filtered_sentence) #final string of cleaned words

    final_string.append(str1)



final['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review 

final['CleanedText']=final['CleanedText'].str.decode("utf-8")
final
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(final['CleanedText'],final['Score'],test_size = 0.3, shuffle = False)
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer() #in scikit-learn
bigram_x_train = count_vect.fit_transform(x_train.astype('U'))

bigram_x_test = count_vect.transform(x_test.astype('U'))

print("the type of count vectorizer - x test",type(bigram_x_test))

print("the shape of out text BOW bigram vectorizer - x test",bigram_x_test.get_shape())

print("="*50)

print("the type of count vectorizer - x train",type(bigram_x_train))

print("the shape of out text BOW bigram vectorizer - x train",bigram_x_train.get_shape())
print(bigram_x_test[346])
count_vect.get_feature_names()[4000]
from sklearn.naive_bayes import MultinomialNB

MultiNB = MultinomialNB(alpha = 10)

#fitting Multinomial naive bayes classifier with train data

MultiNB.fit(bigram_x_train,y_train)

y_predict = MultiNB.predict(bigram_x_test)
from sklearn import metrics

print('model accuracy = '+ str(metrics.accuracy_score(y_test,y_predict)))
features = pd.DataFrame(data = MultiNB.feature_log_prob_.T,index=count_vect.get_feature_names(), columns=["0","1"])
display_features = features.sort_values(by='0', ascending=False)[200:210]

print(display_features)
import seaborn as sns

tn, fp, fn, tp = metrics.confusion_matrix(y_test,(y_predict)).ravel()

ax = sns.heatmap([[fn,tn],[fp,tp]],yticklabels=["Actual 0","Actual 1"],\

                 xticklabels=["Predicted 0","Predicted 1"],annot = True,fmt='d')

ax.set_title('bag of words')