# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from nltk.corpus import stopwords 
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 

from sklearn.naive_bayes import MultinomialNB

train_data_csv_name="../input/data_new.xlsx"
df = pd.read_excel(train_data_csv_name,usecols=[0],header=None)
df_y_positive= pd.read_excel(train_data_csv_name,usecols=[1],header=None)
df_y_negative= pd.read_excel(train_data_csv_name,usecols=[2],header=None)
df_y_bad= pd.read_excel(train_data_csv_name,usecols=[3],header=None)

def getString(dataframe):
    list=[]
    for s in dataframe.values:
        a = s.tolist()
        list.append(str(a))
        
    return list

cv = CountVectorizer()
X_train_counts = cv.fit_transform(getString(df))
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf_positive = MultinomialNB().fit(X_train_tfidf, df_y_positive)
clf_negative = MultinomialNB().fit(X_train_tfidf, df_y_negative)
clf_bad = MultinomialNB().fit(X_train_tfidf, df_y_bad)



def text_process(mess):
    mess = [char for char in mess if char not in string.punctuation]
    mess = ''.join(mess)
    return [word for word in mess.split() if word.lower() not in stopwords.words('english') ]

s= input("enter any string...")
#text= ''.join(s)
#text=text_process(s.split())
X_counts = cv.transform(text_process(s))
X_tfidf = tfidf_transformer.transform(X_counts)
    
positive_predicted = clf_positive.predict(X_tfidf)
positive_count=0
for x in positive_predicted:
    if x==1:
        positive_count=positive_count+1
            
negative_predicted = clf_negative.predict(X_tfidf)
negative_count=0
for x in negative_predicted:
    if x==1:
        negative_count=negative_count+1

bad_predicted = clf_bad.predict(X_tfidf)
bad_count=0
for x in bad_predicted:
    if x==1:
        bad_count=bad_count+1
        

if(bad_count>0):
    res= "Bad %s"%(s)
else:
    if(positive_count-negative_count)>0:
        res="Positive %s"%(s)
    elif(negative_count-positive_count)>0:
        res= "Negative %s"%(s)
    else:
        res="Neutral %s"%(s)


print(res)
