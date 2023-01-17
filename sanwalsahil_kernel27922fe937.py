# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)



# Any results you write to the current directory are saved as output.
import seaborn as sns

import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv('../input/bbc-fulltext-and-category/bbc-text.csv')
data.head()
data['category'].value_counts()
type(data['category'])
sns.catplot(x = 'category', kind='count',data=data)
data.describe()
data.index
#loading stop words list

stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

print(stop_words)
def preprocess(text):

    text = text.lower()

    text = re.sub('[^a-zA-Z ]','',text)

    # removing stop words

    wordsList = text.split()

    newWordsList = []

    for word in wordsList:

        if word  not in stop_words: # remove stop words

            word = stemmer.stem(word) #using porter stemmer

            word = lemmatizer.lemmatize(word)

            newWordsList.append(word)

            

    return " ".join(newWordsList)
sampleText = data['text'][35]

sampleText
smplePre = preprocess(sampleText)

smplePre
x = data['text'].apply(lambda x:preprocess(x))
x
tv = TfidfVectorizer()

x_tf= tv.fit_transform(x)
le = LabelEncoder()

y = le.fit_transform(data['category'])
#new_y = y.reshape(-1,1)

#pd.DataFrame(new_y)[0].value_counts()
#enc = OneHotEncoder(categories='auto')

#y = enc.fit_transform(new_y).toarray()

pd.DataFrame(y)
from sklearn.model_selection import train_test_split



x_train,x_test, y_train,y_test = train_test_split(x_tf,y,test_size=0.2)
model = MultinomialNB()

model.fit(x_train,y_train)
model.score(x_test,y_test)
sample_data = "ROnaldo scored a wonder full goal as brazil wins the world cup final and take home the cup"

preData = preprocess(sample_data)

finalSample = tv.transform([preData])
result =model.predict(finalSample)

resClass = le.inverse_transform(result)
resClass
import pickle

pkl_Filename = "classModel"



with open(pkl_Filename, 'wb') as file:

    pickle.dump(model,file)



from IPython.display import FileLink

FileLink(pkl_Filename)