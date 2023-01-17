# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expression libary.
import nltk # Natural Language toolkit
nltk.download("stopwords")  #downloading stopwords
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
nltk.download('wordnet')
import nltk as nlp


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv",sep=",")
df.head()
df.info()
df.isnull().sum() # looking for null values
review_list=[]

for review in df.Review:
    review=re.sub("[^a-zA-z]"," ",review) # if expression in the sentence is not a word then this code change them to space
    review=review.lower() # turns all word in the sentence into lowercase.
    review=nltk.word_tokenize(review) # splits the words that are in the sentence from each other.
    lemma=nlp.WordNetLemmatizer()
    review=[lemma.lemmatize(word) for word in review] # this code finds the root of the word for a word in the sentence and change them to their root form.
    review=" ".join(review)
    review_list.append(review) # store sentences in list
    

from sklearn.feature_extraction.text import CountVectorizer #Bag of Words

max_features=500 # "number" most common(used) words in reviews

count_vectorizer=CountVectorizer(max_features=max_features,stop_words="english") # stop words will be dropped by stopwords command

sparce_matrix=count_vectorizer.fit_transform(review_list).toarray()# this code will create matrix that consist of 0 and 1.


sparce_matrix.shape 
sparce_matrix
print("Top {} the most used word by reviewers: {}".format(max_features,count_vectorizer.get_feature_names()))
data=pd.DataFrame(count_vectorizer.get_feature_names(),columns=["Words"])
data.head()
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
plt.subplots(figsize=(12,12))
wordcloud=WordCloud(background_color="white",width=1024,height=768).generate(" ".join(data.Words[100:]))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
df.Rating.value_counts()
X=sparce_matrix
y=df.Rating
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,log_loss,precision_score
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score,roc_curve


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("x_train",X_train.shape)
print("x_test",X_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)
from sklearn.svm import SVC
lgbm_model=LGBMClassifier()

lgbm_model.fit(X_train,y_train)
y_pred=lgbm_model.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print("Precision:",precision_score(y_test, y_pred,average="micro"))
xgb=XGBClassifier()
xgb_model=xgb.fit(X_train,y_train)
y_pred=xgb_model.predict(X_test)


print("Accuracy:",accuracy_score(y_test, y_pred))
print("Precision:",precision_score(y_test, y_pred,average="micro"))
from sklearn.naive_bayes import BernoulliNB

nb=GaussianNB()
nb2=BernoulliNB()

nb_model=nb.fit(X_train,y_train)
nb2_model=nb2.fit(X_train,y_train)
y_pred=nb_model.predict(X_test)
y_pred2=nb2_model.predict(X_test)


print("Accuracy:",accuracy_score(y_test, y_pred))
print("Precision:",precision_score(y_test, y_pred,average="micro"))
print("**************************************************************")
print("Accuracy_NB2:",accuracy_score(y_test, y_pred2))
print("Precision_NB2:",precision_score(y_test, y_pred2,average="micro"))
from sklearn.ensemble import RandomForestClassifier


rf_model=RandomForestClassifier(random_state=42)
rf_model.fit(X_train,y_train)
y_pred=rf_model.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))
print("Precision:",precision_score(y_test, y_pred,average="micro"))
