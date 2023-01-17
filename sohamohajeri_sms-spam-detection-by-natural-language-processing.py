import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.tokenize import WhitespaceTokenizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer   
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
df_sms = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',delimiter=',',encoding='latin-1')
df_sms.head()
df_sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)
df_sms.columns=['label','message']
df_sms.head()
df_sms.describe()
df_sms.groupby('label').describe()
df_sms['length']=df_sms['message'].apply(lambda x: len(x))
df_sms.head()
plt.figure(figsize=(8,6))
plt.hist(x='length', bins=150, data=df_sms,edgecolor='black')
plt.title('Distribution of the Length of Messages', fontsize=15)
plt.xlabel('Message Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()
df_sms['length'].max()
df_sms['length'].describe()
df_sms[df_sms['length']==910]['message']
plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
plt.hist(x='length', bins=60, data=df_sms[df_sms['label']=='ham'],edgecolor='black', color='m')
plt.title('Distribution of the Length of Ham Messages', fontsize=15)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Message Length', fontsize=12)

plt.subplot(1,2,2)
plt.hist(x='length', bins=60, data=df_sms[df_sms['label']=='spam'],edgecolor='black', color='teal')
plt.title('Distribution of the Length of Spam Messages', fontsize=15)
plt.ylabel('')
plt.yticks([])
plt.xlabel('Message Length', fontsize=12)

plt.show()
ps=PorterStemmer()
wst= WhitespaceTokenizer() 

##### 1. Converting text to lower case
def lower_func (x):
    return x.lower()


##### 2. Removing Numbers from the text corpus
def remove_number_func (x): 
    new=""
    for a in x:
        if a.isdigit()==False:
            new=new+a
    return new


##### 3. Removing punctuation 
def remove_punc_func(x):
    new=''
    for a in x:
        if a not in string.punctuation:
            new=new+a
    return new

##### 4. Removing special characters
def remove_spec_char_func(x):
    new=''
    for a in x:
        if (a.isalnum()==True) or (a==' '):
            new=new+a
    return(new)

##### 5. Removing english stopwords
def remove_stopwords(x):
    new=[]
    for a in x.split():
        if a not in stopwords.words('english'):
            new.append(a)
    return " ".join(new)

##### 6. Stemming words to root words
def stem_func(x):
    wordlist = word_tokenize(x)
    psstem = [ps.stem(a) for a in wordlist]
    return ' '.join(psstem)

##### 7. Removing extra whitespaces 
def remove_whitespace_func(x):
    return(wst.tokenize(x))

def compose(f, g):
    return lambda x: f(g(x))

final=compose(compose(compose(compose(compose(compose(remove_whitespace_func,stem_func),remove_stopwords),remove_spec_char_func),remove_punc_func),remove_number_func),lower_func)
X=df_sms['message']
y=df_sms['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline = Pipeline([
    ('bow', CountVectorizer()),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', MultinomialNB()), 
])
pipeline.fit(X_train, y_train)
prediction1=pipeline.predict(X_test)
print(classification_report(y_test,prediction1))
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=final)),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', MultinomialNB()), 
])
pipeline.fit(X_train, y_train)
prediction2=pipeline.predict(X_test)
print(classification_report(y_test,prediction2))
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer='word', stop_words='english')),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', MultinomialNB()), 
])
pipeline.fit(X_train, y_train)
prediction3=pipeline.predict(X_test)
print(classification_report(y_test,prediction3))
pipeline = Pipeline([
    ('bow', CountVectorizer()),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', PassiveAggressiveClassifier()), 
])
pipeline.fit(X_train, y_train)
prediction4=pipeline.predict(X_test)
print(classification_report(y_test,prediction4))
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=final)),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', PassiveAggressiveClassifier()), 
])
pipeline.fit(X_train, y_train)
prediction5=pipeline.predict(X_test)
print(classification_report(y_test,prediction5))
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer='word', stop_words='english')),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', PassiveAggressiveClassifier()), 
])
pipeline.fit(X_train, y_train)
prediction6=pipeline.predict(X_test)
print(classification_report(y_test,prediction6))
pipeline = Pipeline([
    ('bow', CountVectorizer()),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', RandomForestClassifier()), 
])
pipeline.fit(X_train, y_train)
prediction7=pipeline.predict(X_test)
print(classification_report(y_test,prediction7))
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=final)),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', RandomForestClassifier()), 
])
pipeline.fit(X_train, y_train)
prediction8=pipeline.predict(X_test)
print(classification_report(y_test,prediction8))
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer='word', stop_words='english')),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', RandomForestClassifier()), 
])
pipeline.fit(X_train, y_train)
prediction9=pipeline.predict(X_test)
print(classification_report(y_test,prediction9))
print('MultinomialNB Classifier without text pre-processing', accuracy_score(y_test,prediction1))
print('MultinomialNB Classifier with text pre-processing: ', accuracy_score(y_test,prediction2))
print('MultinomialNB Classifier with only removing stop words', accuracy_score(y_test,prediction3))
print('\n')
print('Passive Aggressive Classifier without text pre-processing: ', accuracy_score(y_test,prediction4))
print('Passive Aggressive Classifier with text pre-processing: ', accuracy_score(y_test,prediction5))
print('Passive Aggressive Classifier with only removing stop words: ', accuracy_score(y_test,prediction6))
print('\n')
print('Random Forest Classifier without text pre-processing: ', accuracy_score(y_test,prediction7))
print('Random Forest Classifier with text pre-processing: ', accuracy_score(y_test,prediction8))
print('Random Forest Classifier with only removing stop words: ', accuracy_score(y_test,prediction9))