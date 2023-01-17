import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import sqlite3

import nltk 

import string



#from sklearn.feature_extraction.text import TfidTransformer

#from sklearn.feature_extraction.text import TfidVectorizer

#from sklearn.feature_extraction.text import CountVectorizer



from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve

from sklearn.metrics import accuracy_score

from nltk.stem.porter import PorterStemmer
con = sqlite3.connect("../input/amazon-fine-food-reviews/database.sqlite")
#selecting those reviews where score is not equal to 3

filt_data = pd.read_sql_query("select * from Reviews where score != 3 ", con)



def partition(x):

    if x < 3:

        return 'negative'

    return 'positive'



actual_score = filt_data['Score']

posneg = actual_score.map(partition)

filt_data['Score'] = posneg
filt_data.head(10)
#sorting data according to ProductId

sorted_data = filt_data.sort_values('ProductId' , axis = 0 , ascending = True)
#deduplication of data

final = sorted_data.drop_duplicates(subset = {"UserId" , "ProfileName" , "Text" , "Time" } , keep = 'first' , inplace = False)

final.shape
#finding how much % data still remains

(final['Id'].size * 1.0) / (filt_data['Id'].size * 1.0) * 100
# we observe that Helpfullness numerator should alwyas be greater than helpfullness denominator

# the products or tuples that d not follw this rule need to be removed



final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]

final.shape
plt.figure(figsize = (10,5))

sns.countplot(final['Score'] , palette = 'gist_rainbow')

plt.xlabel("Reviews")

plt.ylabel("No. of Reviews")

plt.show()



print(final['Score'].value_counts())
import re

#nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer





from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score , precision_score , recall_score

from nltk.metrics.scores import (precision , recall , f_measure)
# Removing stop words , stemming and lemmetization



#set of stop words

stop = set(stopwords.words('english'))

words_to_keep = set(('not'))

stop -= words_to_keep



#initialising snowball stemmer

sno = nltk.stem.SnowballStemmer('english')





#removing html tags

def cleanhtml(sentence):    #function cleans word of html tags

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr , ' ', sentence)

    return cleantext



#removing punctuation or special characters

def cleanpunc(sentence):    #function cleans words with these symbols

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    return cleaned



# text processing code

i = 0

str1 = ' '

final_string = []

all_positive_words = []

all_negative_words = []

w = ''



for sent in final['Text'].values:

    filtered_sentence = []

    sent = cleanhtml(sent)

    for s in sent.split():

        for cleaned_words in cleanpunc(s).split():

            if((cleaned_words.isalpha()) & (len(cleaned_words) > 2)):

                if(cleaned_words.lower() not in stop):

                    w = (sno.stem(cleaned_words.lower())).encode('utf8')

                    filtered_sentence.append(w)

                    if(final['Score'].values)[i] == 'positive':

                        all_positive_words.append(w)

                    if(final['Score'].values)[i] == 'negative':

                        all_negative_words.append(w)

                else:

                    continue

            else:

                continue

                

    str1 = b" ".join(filtered_sentence)



    final_string.append(str1)

    i += 1
#adding an extra column

final['CleanedText'] = pd.Series(final_string)

final['CleanedText'] = final['CleanedText'].str.decode("utf-8")
final.head(10)
final.isnull().sum()
#removing all the rows with null reviews after pre processing

cleaned_final = final

cleaned_final.dropna()
print(final.shape)

print(cleaned_final.shape)

cleaned_final.isnull().sum()
cleaned_final['CleanedText'].replace('', np.nan, inplace=True)

cleaned_final.dropna(subset=['CleanedText'], inplace=True)

cleaned_final.shape
#sorting the data according to time in asc

timesortdata = cleaned_final.sort_values('Time' , axis = 0 , ascending = True , inplace = False , kind = 'quicksort' , na_position = 'last')

x = cleaned_final['CleanedText'].values

y = cleaned_final['Score']



X_train ,X_test ,Y_train ,Y_test = train_test_split(x ,y ,test_size = 0.3, random_state = 0)
countVector = CountVectorizer(min_df = 1000)

xTrainVector = countVector.fit_transform(X_train)

xTestVector = countVector.transform(X_test)



stdScale = StandardScaler(with_mean = False)

xTrVecStd = stdScale.fit_transform(xTrainVector)

xTsVecStd = stdScale.transform(xTestVector)
Depths = [3,4,5,6,7,8,9,10,11,12,13,14,15]

paramGrid = {'max_depth':Depths}



model = GridSearchCV(DecisionTreeClassifier() , paramGrid , scoring = 'accuracy' , cv = 3 , n_jobs = -1 , pre_dispatch = 2)

model.fit(xTrVecStd,Y_train)



print("Accuracy : ", model.score(xTsVecStd , Y_test))



#cross validation

cv = [1-i for i in model.cv_results_['mean_test_score']]



#optimum depth

opt_depth = model.best_estimator_.max_depth

print("Optimal depth : ", opt_depth)



#Decision Tree Classifier with optimal depth

DT = DecisionTreeClassifier(max_depth = opt_depth)

DT.fit(xTrVecStd , Y_train)

predictions = DT.predict(xTsVecStd)
precision = precision_score(Y_test, predictions, pos_label = 'positive')

print("Precision = " , precision)



recall = recall_score(Y_test, predictions , pos_label = 'positive')

print("\nRecall = ", recall)



f1 = f1_score(Y_test, predictions , pos_label = 'positive')

print("\nf1 Score = ", f1)
