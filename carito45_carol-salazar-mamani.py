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
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from functools import reduce 
import numpy as np

# definicion de corpus
texts = [['i', 'have', 'a', 'cat'], 
        ['he', 'have', 'a', 'dog'], 
        ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]

dictionary = list(enumerate(set(list(reduce(lambda x, y: x + y, texts)))))
print(dictionary)
def vectorize(text): 
    vector = np.zeros(len(dictionary)) 
    for i, word in dictionary: 
        num = 0 
        for w in text: 
            if w == word: 
                num += 1 
        if num: 
            vector[i] = num 
    return vector

for t in texts: 
    print(vectorize(t))
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(1,1))
vect.fit_transform(['i have no cows','no, i have cows']).toarray()
vect.vocabulary_
vect = CountVectorizer(ngram_range=(1,2))
vect.fit_transform(['i have no cows','no, i have cows']).toarray()
vect.vocabulary_
from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(3,3), analyzer='char_wb')

n1, n2, n3, n4 = vect.fit_transform(['andersen', 'petersen', 'petrov', 'smith']).toarray()


euclidean(n1, n2), euclidean(n2, n3), euclidean(n3, n4)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
data = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
data.head(n=10)
data= pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"Tipo", "v2":"Letras"})
data.head()
data.info()
count1=Counter(" ".join(data[data['Tipo']=='ham']["Letras"]).split()).most_common(30)
df1=pd.DataFrame.from_dict(count1)
print(df1.head())
df1 = df1.rename(columns={0:"palabras non-spam", 1 :"count"})
count2 = Counter(" ".join(data[data['Tipo']=='spam']["Letras"]).split()).most_common(30)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0:"palabras spam", 1:"count_"})
df1.plot.bar(legend="False")
y_pos=np.arange(len(df1["palabras non-spam"]))
plt.xticks(y_pos,df1["palabras non-spam"])
plt.title('Palabras frecuentes en mensajes no-spam')
plt.xlabel('Palabras')
plt.ylabel('Numero')
plt.show()
df2.plot.bar(legend= False , color='red')
y_pos=np.arange(len(df2["palabras spam"]))
plt.xticks(y_pos,df2["palabras spam"])
plt.title('Palabras frecuentes en mensajes spam')
plt.xlabel('Palabras')
plt.ylabel('Numero')
plt.show()
data['Tipo'].value_counts()
data['Tipo'].value_counts().plot.bar()
data["Tipo"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()
data['Tipo'].replace('spam', 0, inplace = True)
data['Tipo'].replace('ham', 1, inplace = True)

# checking the values of the labels now
data['Tipo'].value_counts()
from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'gray', width = 1000, height = 1000, max_words = 50).generate(str(data['Letras']))

plt.rcParams['figure.figsize'] = (10, 10)
plt.title('Most Common words in the dataset', fontsize = 20)
plt.axis('off')
plt.imshow(wordcloud)
spam = ' '.join(text for text in data['Letras'][data['Tipo'] == 0])

wordcloud = WordCloud(background_color ='white', max_words = 50, height = 1000, width = 1000).generate(spam)

plt.rcParams['figure.figsize'] = (10, 10)
plt.axis('off')
plt.title('Most Common Words in Spam Messages', fontsize = 20)
plt.imshow(wordcloud)
ham = ' '.join(text for text in data['Letras'][data['Tipo'] == 1])

wordcloud = WordCloud(background_color = 'gray', max_words = 50, height = 1000, width = 1000).generate(ham)

plt.rcParams['figure.figsize'] = (10, 10)
plt.axis('off')
plt.title('Most Common Words in Ham Messages', fontsize = 20)
plt.imshow(wordcloud)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
feat= feature_extraction.text.CountVectorizer(stop_words = 'english',max_features=100)
X= feat.fit_transform(data["Letras"])
np.shape(X)
data['Tipo'].replace('spam', 0, inplace = True)
data['Tipo'].replace('ham', 1, inplace = True)
data.head()
X_train,X_test,y_train,y_test = model_selection.train_test_split(X, data["Tipo"], test_size=0.2, random_state=42)
print([np.shape(X_train),np.shape(X_test)])
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.svm import SVC
# usamos Support Vector Machine de :https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
import string
string.punctuation
from nltk.corpus import stopwords
stopwords.words("english")[100:110]
from sklearn import svm
svc = svm.SVC()
svc.fit(x_train, y_train)
score_train = svc.score(X_train, y_train)
score_test = svc.score(X_test, y_test)
SVC()
# para validar debe usar una matriz de confusión usando el siguiente código:
matr_confusion_test = metrics.confusion_matrix(y_test, svc.predict(x_test))
pd.DataFrame(data = matr_confusion_test, columns = ['Prediccion spam', 'Prediccion no-spam'],
            index = ['Real spam', 'Real no-spam'])