def Text_Length(Data):
    
    Word_lengths=[]

    for i in Data:
        
        word=len(nltk.word_tokenize(i))
        Word_lengths.insert(len(Word_lengths),word)
        
    return(Word_lengths)
def Wordify(Data):
    
    Words=""
    
    for i in Data:
    
         token=nltk.word_tokenize(i)
    
         for j in token:
        
             if(j.lower not in stop_words):
                
                     Words+=j.lower()
                    
    return(Words)

def Featurize(Data):
    
    Feature=[]

    for i in Data:
    
       if(i=="spam"):
        
           Feature.insert(len(Feature),1)
        
       else:
        
           Feature.insert(len(Feature),0)
            
    return(Feature)

#Necessary Imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #for developing visualizations
import nltk #for general text processing
import matplotlib.pyplot as plt #for developing visualizations 


#Random Sample of the data

Data=pd.read_csv("../input/spam.csv",encoding = "ISO-8859-1")
Data.sample(n=10).head(n=5)


#Distribution of Class of Dataset

print("Length of Dataset: ",len(Data))
print("Spam: ",len(Data[Data['v1']=="spam"]))
print("Ham: ",len(Data[Data['v1']=="ham"]))


Data[Data['v1']=="ham"][0:5]['v2']


Data[Data['v1']=="spam"][0:5]['v2']

from wordcloud import WordCloud #to generate word clouds 
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
Spam=Data[Data['v1']=="spam"]['v2']
Ham=Data[Data['v1']=="ham"]['v2']
spam_wordcloud=WordCloud(width=600,height=400).generate(Wordify(Spam))
ham_wordcloud=WordCloud(width=600,height=400).generate(Wordify(Ham))
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
Spam_Lengths=pd.Series(Text_Length(Spam),name="Spam Lengths")
Ham_Lengths=pd.Series(Text_Length(Ham),name="Ham Lengths")
fig,ax=plt.subplots(1,2,figsize=(15,6))
sns.distplot(Spam_Lengths,ax=ax[0])
sns.distplot(Ham_Lengths,ax=ax[1])
fig.show()
Ham_Lengths.describe()
Spam_Lengths.describe()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


BiGram=CountVectorizer(ngram_range=(1, 2))
X_BOW=BiGram.fit_transform(Data['v2'])


tfidf=TfidfTransformer()
X_T=tfidf.fit_transform(X_BOW)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_T,Featurize(Data['v1']), test_size=0.4, random_state=0)

#Necessary imports for machine learning algorithms

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

gNB=GaussianNB()
gNB.fit(X_train.toarray(),y_train)
accuracy_score(gNB.predict(X_test.toarray()),y_test)
mNB=MultinomialNB()
mNB.fit(X_train.toarray(),y_train)
accuracy_score(mNB.predict(X_test.toarray()),y_test)
rbfsvm=SVC(kernel="rbf")
rbfsvm.fit(X_train.toarray(),y_train)
accuracy_score(rbfsvm.predict(X_test.toarray()),y_test)
linearsvm=SVC(kernel="linear")
linearsvm.fit(X_train.toarray(),y_train)
linearsvm_results=linearsvm.predict(X_test.toarray())
accuracy_score(linearsvm_results,y_test)
Logit=linear_model.LogisticRegression()
Logit.fit(X_train.toarray(),y_train)
accuracy_score(Logit.predict(X_test.toarray()),y_test)
sns.heatmap(confusion_matrix(linearsvm_results,y_test),annot=True)
print(classification_report(y_test,linearsvm_results))
