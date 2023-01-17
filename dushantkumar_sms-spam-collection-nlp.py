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
import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt
data = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin1')

data= data.iloc[:,[0,1]]

data.columns = ["label", "message"]
data.head()
data.shape
data.describe(include='all')
data['label'].value_counts()
data.isnull().sum()
pd.set_option('display.max_colwidth',150)

data.head()
import string

s=string.punctuation
# def Remove_pun(txt):

#     text_rem= "".join([c for c in txt if c not in s])

#     return text_rem

# data['msg_clean']= data['message'].apply(lambda x: Remove_pun(x))
data.head()
# import re



# def token(txt):

#     token = re.split('\W+',txt)

#     return token

# data['Msg_clean_token'] = data['msg_clean'].apply(lambda x: token(x.lower()))

# data.head()   
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus= []

for i in range(len(data)):

    MSG=re.sub('[^a-zA-Z]',' ',data['message'][i]) 

    MSG=MSG.lower()

    MSG=MSG.split() #not a-z and A-Z only non letter will be replace by space

    ps= PorterStemmer()

    all_stopwords= stopwords.words('english')

    MSG= [ps.stem(word) for word in MSG if not word in set(all_stopwords)]

    MSG= ' '.join(MSG)

    corpus.append(MSG)
corpus
from sklearn.feature_extraction.text import CountVectorizer

cv= CountVectorizer()



X= cv.fit_transform(corpus).toarray()

Y=data.iloc[:,0].values
from sklearn.preprocessing import LabelEncoder

LE= LabelEncoder()

Y=LE.fit_transform(Y)
#split the data



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.22, random_state = 0)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)

print("Accuracy score {}".format(acc_logreg))

cm=confusion_matrix(y_test,y_pred)

print("Confusion metrics  {}".format(cm))
# # Gaussian Naive Bayes

# from sklearn.naive_bayes import GaussianNB

# gaussian = GaussianNB()

# gaussian.fit(x_train, y_train)

# y_pred = gaussian.predict(x_test)

# cm=confusion_matrix(y_test,y_pred)

# print("Confusion metrics  {}".format(cm))

# print(accuracy_score(y_test,y_pred))
new_msg = 'Hey Vick, Lets go for dinner tonight'

new_msg = re.sub('[^a-zA-Z]', ' ', new_msg)

new_msg = new_msg.lower()

new_msg = new_msg.split()

ps = PorterStemmer()

all_stopwords = stopwords.words('english')

all_stopwords.remove('not')

new_msg = [ps.stem(word) for word in new_msg if not word in set(all_stopwords)]

new_msg = ' '.join(new_msg)

new_corpus = [new_msg]

new_X_test = cv.transform(new_corpus).toarray()

new_y_pred = logreg.predict(new_X_test)

print(new_y_pred)