# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import StringIO

df=pd.read_csv('/kaggle/input/email-spam-classification/emails.csv')
df.isnull().sum()
df=df.iloc[:,0:2]
df.shape
df.sample(3)
df.isnull().sum()
df.dropna(subset=['spam'],inplace=True)
df.sample(3)
df.sample()['text'].values
#naive-Bayes  -----> Predictor
#text preprocessing



#1. converting to lower case





def toLowerCase(text):

    return text.lower()





df['text']=df['text'].apply(toLowerCase)
def removespecchracter(text):

    x=""

    for i in text:

        if i.isalnum():

            x=x+i

        else:

            x=x+' '

    return x
df['text']=df['text'].apply(removespecchracter)

from nltk.corpus import stopwords 
stop_words = stopwords.words('english')

#using list_comprehension





df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

        

    
df['text'][0]


from nltk.stem.porter import PorterStemmer



ps=PorterStemmer()



df['text'] = df['text'].apply(lambda x: [ps.stem(y) for y in x])

    

    
def joinList(text):

    return ''.join(text)



df['text']=df['text'].apply(joinList)
df.sample(3)
from sklearn.feature_extraction.text import CountVectorizer





cv=CountVectorizer(max_features=10000)



X=cv.fit_transform(df['text']).toarray()
X.shape
y=df.iloc[:,-1].values
y
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)





from sklearn.naive_bayes import MultinomialNB

clf=MultinomialNB()

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))
import pickle



pickle.dump(cv,open('vectorizer.pkl','wb+'))
pickle.dump(clf,open('main_model.pkl','wb+'))