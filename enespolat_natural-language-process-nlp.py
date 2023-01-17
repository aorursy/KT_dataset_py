# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/gender-classifier-DFE-791531.csv",encoding="latin1")
data=pd.concat([data.gender,data.description],axis=1)
data.shape
data.dropna(axis=0,inplace=True)
data.shape
data.gender=[1 if each =="female" else 0 for each in data.gender]
data.head(15)
#cleaning data
#regular expression RE
import re

first_description=data.description[4]
description=re.sub("[^a-zA-Z]"," ",first_description)
description=description.lower()
description


#stopwords (irrelavent words) gereksiz kelimeler the and
import nltk #natural language tool kit
#nltk.download("stopwords")
from nltk.corpus import stopwords

#split all words
#description=description.split()#sadece bosluk arar
#split yerine tokenizer kullanabiliriz 
description=nltk.word_tokenize(description) #shouldn't-> should n't
#gereksiz kelimeleri çıkar
description=[word for word in description if not word in set(stopwords.words("english"))]
#kelime kökleri bulma lematazation
import nltk as nlp
lema= nlp.WordNetLemmatizer()
description=[lema.lemmatize(word)for word in description]
description
description=" ".join(description)
description
description_list=[]
for description in data.description:
    description=re.sub("[^a-zA-Z]"," ",description)
    description=description.lower()
    description=nltk.word_tokenize(description)
    #description=[word for word in description if not word in set(stopwords.words("english"))]
    lema= nlp.WordNetLemmatizer()
    description=[lema.lemmatize(word)for word in description]
    description=" ".join(description)
    description_list.append(description)
description_list
    
#bag of words için
from sklearn.feature_extraction.text import CountVectorizer
max_features=5000
count_vectorizer=CountVectorizer(max_features=max_features,stop_words="english")

sparse_matrix=count_vectorizer.fit_transform(description_list).toarray()
print("en sık kullanılan {} kelimeler {}".format(max_features,count_vectorizer.get_feature_names()))


#
y=data.iloc[:,0].values# male or females classes
x=sparse_matrix
#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.1,random_state=42)
#naive bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
#predict
y_pred=nb.predict(x_test)
print("accuracy ",nb.score(y_pred.reshape(-1,1),y_test))


