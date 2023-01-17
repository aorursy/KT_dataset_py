import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import nltk
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import f1_score,accuracy_score
from xgboost import XGBClassifier
import os
print(os.listdir("../input"))

data=pd.read_excel('../input/train.xlsx')
data.head()
data['news_text']=data['news_text'].str.lower().str.replace('[^0-9a-z ]','')
data['news_text'][0]
#docs=' '.join(data['news_text'].str)
stemmer=PorterStemmer()
stopwords=nltk.corpus.stopwords.words('english')
def clean_data(text):
    words=nltk.word_tokenize(text)
    words=[stemmer.stem(word) for word in words if word not in stopwords]
    return " ".join(words)
new_clean_data=pd.DataFrame()
for i in range(data.shape[0]):
    try:
        new_data={'news_text':str(clean_data(data.iloc[i]['news_text'])),
                  'id':data.iloc[i]['id'],
                  'type':data.iloc[i]['type']}
    except:continue
    new_clean_data=new_clean_data.append(new_data,ignore_index=True)
new_clean_data=new_clean_data.dropna()
new_clean_data.head()
x=new_clean_data.drop('type',axis=1)
y=new_clean_data['type']
train_x,test_x,train_y,test_y=train_test_split(x['news_text'],y,test_size=0.3,random_state=100)
# vectorizer=CountVectorizer()
# cv_train_x=vectorizer.fit_transform(train_x)
# cv_test_x=vectorizer.transform(test_x)

# vectorizer=TfidfVectorizer()
# tf_train_x=vectorizer.fit_transform(train_x)
# tf_test_x=vectorizer.transform(test_x)

# vectorizer=HashingVectorizer()
# h_train_x=vectorizer.fit_transform(train_x)
# h_test_x=vectorizer.transform(test_x)

# dt_model=DecisionTreeClassifier()
# dt_model.fit(cv_train_x,train_y)
# pred=dt_model.predict(cv_test_x)
# print("accuracy is: %0.3f"%accuracy_score(pred,test_y))
# print('f1_score is: %0.3f'%f1_score(pred,test_y, average='weighted'))
def vector_variable(vector,x1,x2):
    vec=vector()
    return vec.fit_transform(x1), vec.transform(x2)
def get_score(model,x1,y1,x2,y2):
    import warnings
    warnings.filterwarnings("ignore")
    model_name=model
    model=model()
    model.fit(x1,y1)
    pred=model.predict(x2)
    (print('f1_score for ',model_name,' is: %0.3f'%f1_score(pred,y2, average='weighted')))  
cv_train_x,cv_test_x=vector_variable(CountVectorizer,train_x,test_x)
get_score(LogisticRegression,cv_train_x,train_y,cv_test_x,test_y)
get_score(MultinomialNB,cv_train_x,train_y,cv_test_x,test_y)
get_score(PassiveAggressiveClassifier,cv_train_x,train_y,cv_test_x,test_y)
get_score(DecisionTreeClassifier,cv_train_x,train_y,cv_test_x,test_y)
get_score(RandomForestClassifier,cv_train_x,train_y,cv_test_x,test_y)
get_score(AdaBoostClassifier,cv_train_x,train_y,cv_test_x,test_y)
get_score(XGBClassifier,cv_train_x,train_y,cv_test_x,test_y)
tf_train_x,tf_test_x=vector_variable(TfidfVectorizer,train_x,test_x)
get_score(LogisticRegression,tf_train_x,train_y,tf_test_x,test_y)
get_score(MultinomialNB,tf_train_x,train_y,tf_test_x,test_y)
get_score(PassiveAggressiveClassifier,tf_train_x,train_y,tf_test_x,test_y)
get_score(DecisionTreeClassifier,tf_train_x,train_y,tf_test_x,test_y)
get_score(RandomForestClassifier,tf_train_x,train_y,tf_test_x,test_y)
get_score(AdaBoostClassifier,tf_train_x,train_y,tf_test_x,test_y)
get_score(XGBClassifier,tf_train_x,train_y,tf_test_x,test_y)
h_train_x,h_test_x=vector_variable(HashingVectorizer,train_x,test_x)
get_score(LogisticRegression,h_train_x,train_y,h_test_x,test_y)
#get_score(MultinomialNB,h_train_x,train_y,h_test_x,test_y)
get_score(PassiveAggressiveClassifier,h_train_x,train_y,h_test_x,test_y)
get_score(DecisionTreeClassifier,h_train_x,train_y,h_test_x,test_y)
get_score(RandomForestClassifier,h_train_x,train_y,h_test_x,test_y)
get_score(AdaBoostClassifier,h_train_x,train_y,h_test_x,test_y)
get_score(XGBClassifier,h_train_x,train_y,h_test_x,test_y)
pa_model=PassiveAggressiveClassifier()
pa_model.fit(tf_train_x,train_y)
pred=pa_model.predict(tf_test_x)
update_article=pd.DataFrame({'Articles':test_x,'Predicted_type':pred})
update_article.to_csv('Predicted Result.csv',index=False)
update_article.head()








