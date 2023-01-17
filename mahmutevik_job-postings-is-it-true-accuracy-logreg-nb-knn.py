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
# Kullanılacak kütüphaneler.
import pandas as pd
import re
import nltk 
from nltk.corpus import stopwords
import nltk as nlp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import nltk as nlp
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# csv formatındaki datamızı df değişkenine atıyoruz
df = pd.read_csv("/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv",encoding = "latin1")
df.head()
df.describe()
print(df.columns)
del df["salary_range"]
del df["job_id"]

# datadaki metinleri tek bir text verisi haline getiriyoruz.
df.fillna(" ",inplace = True)
df['text']=df['title']+" "+df['location']+" "+df['department']+" "+df['company_profile']+" "+df['description']+" "+df['requirements']+" "+df['benefits']+" "+df['employment_type']+" " +df['required_education']+" "+df['industry']+" "+df['function'] 
df.head()
# datada gereksiz bilgileri siliyoruz.
del df['title']
del df['location']
del df['department']
del df['company_profile']
del df['description']
del df['requirements']
del df['benefits']
del df['employment_type']
del df['required_experience']
del df['required_education']
del df['industry']
del df['function']

df.head()
# Cleaning Data:
text_list = []
for text in df.text:
    text= re.sub("[^a-zA-Z]"," ",text) #a-z A-Z aralığı dışındaki tüm ifadeleri boşluk ile değiştiriyoruz
    text = text.lower()   # buyuk harftan kucuk harfe çeviriyoruz
    text = nltk.word_tokenize(text) # textdeki her bir kelimeyi ayırıyoruz
    lemma = nlp.WordNetLemmatizer() 
    text = [lemma.lemmatize(word) for word in text] # for döngüsü ile textin içindeki her kelimeyi köklerine ayırıyoruz
    text = " ".join(text)  # tek tek ayırdığımız kelimeleri aralarına boşluk koyarak tekrar birleştiriyoruz
    text_list.append(text) # oluşturduğumuz tüm textleri bir listenin içine topluyoruz  
# Bag of Words:
 
max_features = 150    # textde en çok kullandılan 150 kelime.
# stop_words ile ingilizce harici kelimeleri siliyoruz.
count_vectorizer = CountVectorizer(max_features=max_features, stop_words = "english") 

# metodumuzu textler üzerinde fit ediyoruz ve sonucu bir liste haline getiriyoruz.   
sparce_matrix = count_vectorizer.fit_transform(text_list).toarray() 

print("en sık kullanılan {} kelimeler :{}".format(max_features,count_vectorizer.get_feature_names()))
x = sparce_matrix
y = df.iloc[:,3].values 
# datamızı train ve test olarak 0.1 oranıyla ayırdık.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state = 42)
# Logistic Regression:
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("logreg accuracy {}".format(lr.score(x_test,y_test)))


# Naive Bayes
nb = GaussianNB()
nb.fit(x_train,y_train)

# prediction 
y_pred = nb.predict(x_test)

print("nb accuracy:",nb.score(y_pred.reshape(-1,1),y_test))

# KNN 
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} knn score: {} ".format(3,knn.score(x_test,y_test)))

score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


