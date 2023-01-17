# importing the libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)
spam = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding='ISO-8859-1')



print(spam.shape)

spam.head()
#drop unusless columns & renaming the rest

spam=spam.iloc[:,:2]

print(spam.shape)

spam.columns=['label','sms']

spam.head()
spam.info()
#check for null

spam.isnull().sum()
spam.describe()
spam.groupby('label').describe()
spam['len']=spam.sms.apply(len)

spam.head()
f,ax=plt.subplots(1,2,figsize=(20,10))





sns.distplot(spam[spam['label']=='ham']['len'], ax=ax[0]);

ax[0].set_title('Ham SMS')



sns.distplot(spam[spam['label']=='spam']['len'], ax=ax[1],color='r');

ax[1].set_title('Spam SMS')
#looks like lenght is a key feature for spam detection!



spam.groupby('label')['len'].describe().round(2)
#lets see if there is wording difference in a first view:

from wordcloud import WordCloud

import matplotlib.pyplot as plt



def word_cloud(label):

    words = ''

    for msg in spam[spam['label'] == label]['sms']:

        words += msg + ' '

    wordcloud = WordCloud(width=600, height=400).generate(words)

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.title(label)

    plt.show()
word_cloud('spam')
word_cloud('ham')
spam['binary_labels']=spam['label'].map({'ham':0,'spam':1})

spam.head()
import spacy

nlp = spacy.load("en_core_web_sm")
def sms_process(sms_text,stem=False):

    text=nlp(sms_text)

    d=[]

    for token in text:

        if not token.is_stop and not token.pos_=='PUNCT':

            if stem==True:

                token=token.lemma_.lower()

            else:

                token=token.lower_

            d.append(token)

    return ' '.join(d)



    
sms_process('HELLO. i am running the fuck .',stem=True)
sms_process('HELLO. i am running the fuck .')
spam['sms_feat']=spam['sms'].apply(sms_process)

spam['sms_feat_lem']=spam['sms'].apply(lambda x: sms_process(x,stem=True))



spam.head()
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer



tf_idf=TfidfVectorizer(decode_error='ignore')



count_vector=CountVectorizer(decode_error='ignore')
#TF-IDF

X_tfidf= tf_idf.fit_transform(spam['sms_feat'])

X_tfidf_lem= tf_idf.fit_transform(spam['sms_feat_lem'])
#COUNT VECT

X_count=count_vector.fit_transform(spam['sms_feat'])

X_count_lem=count_vector.fit_transform(spam['sms_feat_lem'])

Y=spam['binary_labels'].values

Y
from sklearn.model_selection import train_test_split



X_train_tf,X_test_tf,y_train_tf,y_test_tf = train_test_split(X_tfidf,Y,test_size=0.2,random_state=42)



X_train_tf_lem,X_test_tf_lem,y_train_tf_lem,y_test_tf_lem = train_test_split(X_tfidf_lem,Y,test_size=0.2,random_state=42)
X_train_c,X_test_c,y_train_c,y_test_c = train_test_split(X_count,Y,test_size=0.2,random_state=42)



X_train_c_lem,X_test_c_lem,y_train_c_lem,y_test_c_lem = train_test_split(X_count_lem,Y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



#first trying with default settings and same random seed



alg= [SVC(random_state=42,gamma='scale'),

      LogisticRegression(random_state=42),

      MultinomialNB(),

      DecisionTreeClassifier(random_state=42),

      KNeighborsClassifier(),

      RandomForestClassifier(random_state=42)]



from sklearn.metrics import accuracy_score



def train_alg(algorithm,X_train_df,X_test_df,Y_train_df,Y_test_df):

    algorithm.fit(X_train_df,Y_train_df)

    y_pred=algorithm.predict(X_test_df)

    return accuracy_score(y_pred,Y_test_df)

    
alg_name=[]

acc_scores=[]



for item in alg:

    alg_name.append(item.__class__.__name__)

    acc_scores.append(train_alg(item,X_train_tf,X_test_tf,y_train_tf,y_test_tf))

    df_tfidf=pd.DataFrame(index=alg_name,data=acc_scores,columns=['acc_scores'])
alg_name=[]

acc_scores2=[]



for item in alg:

    alg_name.append(item.__class__.__name__)

    acc_scores2.append(train_alg(item,X_train_tf_lem,X_test_tf_lem,y_train_tf_lem,y_test_tf_lem))

    df_tfidf_lem=pd.DataFrame(index=alg_name,data=acc_scores2,columns=['acc_scores2'])
alg_name=[]

acc_scores3=[]



for item in alg:

    alg_name.append(item.__class__.__name__)

    acc_scores3.append(train_alg(item,X_train_c,X_test_c,y_train_c,y_test_c))

    df_count=pd.DataFrame(index=alg_name,data=acc_scores3,columns=['acc_scores3'])
alg_name=[]

acc_scores4=[]



for item in alg:

    alg_name.append(item.__class__.__name__)

    acc_scores4.append(train_alg(item,X_train_c_lem,X_test_c_lem,y_train_c_lem,y_test_c_lem))

    df_count_lem=pd.DataFrame(index=alg_name,data=acc_scores4,columns=['acc_scores4'])
scores=pd.concat([df_tfidf,df_tfidf_lem,df_count,df_count_lem],axis=1)

scores
scores.max().sort_values(ascending=False)
scores.acc_scores4.sort_values(ascending=False)
from sklearn.model_selection import GridSearchCV



parameters={'kernel':['linear','rbf','sigmoid'],

           'C': [0.001,0.01,0.1,1,10,20,50,100]

           }
from datetime import datetime



grid_search=GridSearchCV(SVC(random_state=42,gamma='scale'),parameters,cv=5,verbose=True)



t0=datetime.now()

grid_search.fit(X_train_c_lem,y_train_c_lem)

print('duration:',datetime.now()-t0)
print(grid_search.best_params_)
grid_search.best_estimator_
from sklearn.metrics import classification_report



svc_f=SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',

    kernel='linear', max_iter=-1, probability=False, random_state=42,

    shrinking=True, tol=0.001, verbose=False)



svc_f.fit(X_train_c_lem,y_train_c_lem)

y_svc_pred=svc_f.predict(X_test_c_lem)

print('Accuracy:',accuracy_score(y_svc_pred,y_test_c_lem))

print(classification_report(y_svc_pred,y_test_c_lem))
from sklearn.metrics import confusion_matrix



print(confusion_matrix(y_svc_pred,y_test_c_lem))
param_nb={'alpha':[0.001,0.01,0.1,0.2,1,2,3,4,5,6,7,10,100]}



grid_search_nb=GridSearchCV(MultinomialNB(),param_nb,cv=5,verbose=True)



t0=datetime.now()

grid_search_nb.fit(X_train_c,y_train_c)

print('duration:',datetime.now()-t0)



print(grid_search_nb.best_params_)



grid_search_nb.best_estimator_
MNB=MultinomialNB(alpha=3, class_prior=None, fit_prior=True)



MNB.fit(X_train_c,y_train_c)

y_mnb_pred=MNB.predict(X_test_c)



print('Accuracy:',accuracy_score(y_mnb_pred,y_test_c))

print(classification_report(y_mnb_pred,y_test_c))
print(confusion_matrix(y_mnb_pred,y_test_c))