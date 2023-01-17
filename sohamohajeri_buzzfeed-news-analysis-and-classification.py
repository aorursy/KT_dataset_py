import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import string

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk import word_tokenize

from nltk.tokenize import WhitespaceTokenizer 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
df_real=pd.read_csv('../input/fakenewsnet/BuzzFeed_real_news_content.csv')
df_real.shape
df_fake=pd.read_csv('../input/fakenewsnet/BuzzFeed_fake_news_content.csv')
df_fake.shape
df=pd.concat([df_real,df_fake],axis=0)
df.shape
df['news_type']=df['id'].apply(lambda x: x.split('_')[0])
df.head(2)
df.shape
df.info()
df.describe()
df.drop(['id','url', 'top_img','authors','publish_date','canonical_link','meta_data'],axis=1, inplace=True)
df.isnull().sum()
(df.isnull().sum())/(df.shape[0])*100
df['contain_movies']=df['movies'].apply(lambda x: 0 if str(x)=='nan' else 1)
df['contain_images']=df['images'].apply(lambda x: 0 if str(x)=='nan' else 1)
df.drop(['movies','images'],axis=1,inplace=True)
df.head(2)
real_order=df[df['news_type']=='Real']['source'].value_counts().sort_values(ascending=False).index
plt.figure(figsize=(10,6))

sns.countplot(y='source', data=df[df['news_type']=='Real'],order=real_order,palette='summer')

plt.xlabel('Count',fontsize=12)

plt.ylabel('Source',fontsize=12)

plt.title('Sources of Real News',fontsize=15)

plt.show()
fake_order=df[df['news_type']=='Fake']['source'].value_counts().sort_values(ascending=False).index
plt.figure(figsize=(10,6))

sns.countplot(y='source',data=df[df['news_type']=='Fake'],order=fake_order,palette='autumn')

plt.xlabel('Count',fontsize=12)

plt.ylabel('Source',fontsize=12)

plt.title('Sources of Fake News',fontsize=20)

plt.show()
new=[]

for x in df[df['news_type']=='Fake']['source'].unique():

    if x in df[df['news_type']=='Real']['source'].unique():

        new.append(x)

print(new)
df['common']=df['source'].apply(lambda x: x if x in new else 0)
df1=df[df['common']!=0]
plt.figure(figsize=(10,6))

sns.countplot(y='common',data=df1,hue='news_type',palette='viridis')

plt.xlabel('Count',fontsize=12)

plt.ylabel('Source',fontsize=12)

plt.legend(loc='best', title='News Type',fontsize=10)

plt.title('Common Sources of Real and Fake News',fontsize=20)

plt.show()
df.head(2)
plt.figure(figsize=(10,6))

sns.countplot(x='contain_movies', data=df, hue='news_type', palette='PuBuGn_r')

plt.xlabel('Movies Linked to News',fontsize=12)

plt.ylabel('Count',fontsize=12)

plt.legend(loc='best', title='News Type',fontsize=10)

plt.title('Number of Different News Type Versus Linked Movies',fontsize=18)

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(x='contain_images', data=df, hue='news_type', palette='PuBuGn_r')

plt.xlabel('Images Linked to News',fontsize=12)

plt.ylabel('Count',fontsize=12)

plt.legend(loc='upper left', title='News Type',fontsize=10)

plt.title('Number of Different News Type Versus Linked Images',fontsize=18)

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
df_fake=df[df['news_type']=='Fake']
cv1 = CountVectorizer(analyzer=final)

cv1.fit(df_fake['title'])

bow1=cv1.transform(df_fake['title'])
pd.DataFrame(bow1.todense()).shape
new1=[]

for x in range(0,459):

    new1.append(cv1.get_feature_names()[x])
matrix1=pd.DataFrame(bow1.todense(),columns=new1)
sm1=[]

for x in new1:

    sm1.append(matrix1[x].sum())
trans1=matrix1.transpose()
trans1['sum']=sm1
top1=trans1.sort_values(by='sum', ascending=False).head(20)
df_real=df[df['news_type']=='Real']
cv2 = CountVectorizer(analyzer=final)

cv2.fit(df_real['title'])

bow2=cv2.transform(df_real['title'])
pd.DataFrame(bow2.todense()).shape
new2=[]

for x in range(0,436):

    new2.append(cv2.get_feature_names()[x])
matrix2=pd.DataFrame(bow2.todense(),columns=new2)
sm2=[]

for x in new2:

    sm2.append(matrix2[x].sum())
trans2=matrix2.transpose()
trans2['sum']=sm2
top2=trans2.sort_values(by='sum', ascending=False).head(20)
top1.drop(list(range(0,91)),axis=1,inplace=True)
top1['type']=['Fake']*20
top2.drop(list(range(0,91)),axis=1,inplace=True)
top2['type']=['Real']*20
conc1=pd.concat([top1,top2])
plt.figure(figsize=(12,10))

sns.barplot(y=conc1.index,x='sum',data=conc1,hue='type',palette='viridis')

plt.xticks(rotation=90)

plt.xlabel('Term Frequency of Words',fontsize=12)

plt.ylabel('Top Words in Titles',fontsize=12)

plt.legend(title='News Type',fontsize=12)

plt.title('Frequency of Words in the Title of News',fontsize=20)

plt.show()
cv3 = CountVectorizer(analyzer=final)

cv3.fit(df_fake['text'])

bow3=cv3.transform(df_fake['text'])
pd.DataFrame(bow3.todense()).shape
new3=[]

for x in range(0,4958):

    new3.append(cv3.get_feature_names()[x])
matrix3=pd.DataFrame(bow3.todense(),columns=new3)
sm3=[]

for x in new3:

    sm3.append(matrix3[x].sum())
trans3=matrix3.transpose()
trans3['sum']=sm3
top3=trans3.sort_values(by='sum', ascending=False).head(30)
cv4 = CountVectorizer(analyzer=final)

cv4.fit(df_real['text'])

bow4=cv4.transform(df_real['text'])
pd.DataFrame(bow4.todense()).shape
new4=[]

for x in range(0,6529):

    new4.append(cv4.get_feature_names()[x])
matrix4=pd.DataFrame(bow4.todense(),columns=new4)
sm4=[]

for x in new4:

    sm4.append(matrix4[x].sum())
trans4=matrix4.transpose()
trans4['sum']=sm4
top4=trans4.sort_values(by='sum', ascending=False).head(30)
top3.drop(list(range(0,91)),axis=1,inplace=True)
top3['type']=['Fake']*30
top4.drop(list(range(0,91)),axis=1,inplace=True)
top4['type']=['Real']*30
conc2=pd.concat([top3,top4])
plt.figure(figsize=(12,10))

sns.barplot(y=conc2.index,x='sum',data=conc2,hue='type',palette='viridis')

plt.xticks(rotation=90)

plt.xlabel('Term Frequency of Words',fontsize=12)

plt.ylabel('Top Words in Texts',fontsize=12)

plt.legend(title='News Type',fontsize=12,loc='lower right')

plt.title('Frequency of Words in the Text of News',fontsize=20)

plt.show()
df['title_length']=df['title'].apply(lambda x: len(x))
plt.figure(figsize=(10,6))

sns.kdeplot(df[df['news_type']=='Real']['title_length'])

sns.kdeplot(df[df['news_type']=='Fake']['title_length'])

plt.xlabel('Title Length',fontsize=12)

plt.ylabel('Density',fontsize=12)

plt.legend(title='News Type',fontsize=10,labels=['Real','Fake'])

plt.title('Distribuiton of Title Length for Real and Fake News',fontsize=15)

plt.show()
X1=df['text']

y1=df['news_type']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
pp=Pipeline([

    ('bow',CountVectorizer(analyzer=final)),

    ('tfidf',TfidfTransformer()),

    ('classifier',RandomForestClassifier())

    ])
pp.fit(X1_train,y1_train)
predictions1=pp.predict(X1_test)
print(confusion_matrix(y1_test, predictions1))

print('\n')

print(classification_report(y1_test, predictions1))
pp=Pipeline([

    ('bow',CountVectorizer()),

    ('tfidf',TfidfTransformer()),

    ('classifier',RandomForestClassifier())

    ])
pp.fit(X1_train,y1_train)
predictions2=pp.predict(X1_test)
print(confusion_matrix(y1_test, predictions2))

print('\n')

print(classification_report(y1_test, predictions2))
pp=Pipeline([

    ('bow',CountVectorizer(analyzer=final)),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB())

    ])
pp.fit(X1_train,y1_train)
predictions3=pp.predict(X1_test)
print(confusion_matrix(y1_test, predictions3))

print('\n')

print(classification_report(y1_test, predictions3))
pp=Pipeline([

    ('bow',CountVectorizer()),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB())

    ])
pp.fit(X1_train,y1_train)
predictions4=pp.predict(X1_test)
print(confusion_matrix(y1_test, predictions4))

print('\n')

print(classification_report(y1_test, predictions4))
pp=Pipeline([

    ('bow',CountVectorizer(analyzer=final)),

    ('tfidf',TfidfTransformer()),

    ('classifier',PassiveAggressiveClassifier())

    ])
pp.fit(X1_train,y1_train)
predictions5=pp.predict(X1_test)
print(confusion_matrix(y1_test, predictions5))

print('\n')

print(classification_report(y1_test, predictions5))
pp=Pipeline([

    ('bow',CountVectorizer()),

    ('tfidf',TfidfTransformer()),

    ('classifier',PassiveAggressiveClassifier())

    ])
pp.fit(X1_train,y1_train)
predictions6=pp.predict(X1_test)
print(confusion_matrix(y1_test, predictions6))

print('\n')

print(classification_report(y1_test, predictions6))
X2=df['title']

y2=df['news_type']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)
pp=Pipeline([

    ('bow',CountVectorizer(analyzer=final)),

    ('tfidf',TfidfTransformer()),

    ('classifier',RandomForestClassifier())

    ])
pp.fit(X2_train,y2_train)
predictions7=pp.predict(X2_test)
print(confusion_matrix(y2_test, predictions7))

print('\n')

print(classification_report(y2_test, predictions7))
pp=Pipeline([

    ('bow',CountVectorizer()),

    ('tfidf',TfidfTransformer()),

    ('classifier',RandomForestClassifier())

    ])
pp.fit(X2_train,y2_train)
predictions8=pp.predict(X2_test)
print(confusion_matrix(y2_test, predictions8))

print('\n')

print(classification_report(y2_test, predictions8))
pp=Pipeline([

    ('bow',CountVectorizer(analyzer=final)),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB())

    ])
pp.fit(X2_train,y2_train)
predictions9=pp.predict(X2_test)
print(confusion_matrix(y2_test, predictions9))

print('\n')

print(classification_report(y2_test, predictions9))
pp=Pipeline([

    ('bow',CountVectorizer()),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB())

    ])
pp.fit(X2_train,y2_train)
predictions10=pp.predict(X2_test)
print(confusion_matrix(y2_test, predictions10))

print('\n')

print(classification_report(y2_test, predictions10))
pp=Pipeline([

    ('bow',CountVectorizer(analyzer=final)),

    ('tfidf',TfidfTransformer()),

    ('classifier',PassiveAggressiveClassifier())

    ])
pp.fit(X2_train,y2_train)
predictions11=pp.predict(X2_test)
print(confusion_matrix(y2_test, predictions11))

print('\n')

print(classification_report(y2_test, predictions11))
pp=Pipeline([

    ('bow',CountVectorizer()),

    ('tfidf',TfidfTransformer()),

    ('classifier',PassiveAggressiveClassifier())

    ])
pp.fit(X2_train,y2_train)
predictions12=pp.predict(X2_test)
print(confusion_matrix(y2_test, predictions12))

print('\n')

print(classification_report(y2_test, predictions12))
df['title_text']=df['title']+': ' +df['text']
X3=df['title_text']

y3=df['news_type']
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=42)
pp=Pipeline([

    ('bow',CountVectorizer(analyzer=final)),

    ('tfidf',TfidfTransformer()),

    ('classifier',RandomForestClassifier())

    ])
pp.fit(X3_train,y3_train)
predictions13=pp.predict(X3_test)
print(confusion_matrix(y3_test, predictions13))

print('\n')

print(classification_report(y3_test, predictions13))
pp=Pipeline([

    ('bow',CountVectorizer()),

    ('tfidf',TfidfTransformer()),

    ('classifier',RandomForestClassifier())

    ])
pp.fit(X3_train,y3_train)
predictions14=pp.predict(X3_test)
print(confusion_matrix(y3_test, predictions14))

print('\n')

print(classification_report(y3_test, predictions14))
pp=Pipeline([

    ('bow',CountVectorizer(analyzer=final)),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB())

    ])
pp.fit(X3_train,y3_train)
predictions15=pp.predict(X3_test)
print(confusion_matrix(y3_test, predictions15))

print('\n')

print(classification_report(y3_test, predictions15))
pp=Pipeline([

    ('bow',CountVectorizer()),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB())

    ])
pp.fit(X3_train,y3_train)
predictions16=pp.predict(X3_test)
print(confusion_matrix(y3_test, predictions16))

print('\n')

print(classification_report(y3_test, predictions16))
pp=Pipeline([

    ('bow',CountVectorizer(analyzer=final)),

    ('tfidf',TfidfTransformer()),

    ('classifier',PassiveAggressiveClassifier())

    ])
pp.fit(X3_train,y3_train)
predictions17=pp.predict(X3_test)
print(confusion_matrix(y3_test, predictions17))

print('\n')

print(classification_report(y3_test, predictions17))
pp=Pipeline([

    ('bow',CountVectorizer()),

    ('tfidf',TfidfTransformer()),

    ('classifier',PassiveAggressiveClassifier())

    ])
pp.fit(X3_train,y3_train)
predictions18=pp.predict(X3_test)
print(confusion_matrix(y3_test, predictions18))

print('\n')

print(classification_report(y3_test, predictions18))
print('Text_Random Forest Classifier_With Text Preprocessing: ', round(100*accuracy_score(y1_test,predictions1)))

print('Text_Random Forest Classifier_Without Text Preprocessing: ', round(100*accuracy_score(y1_test,predictions2)))

print('Text_Naive Bayes Classifier_With Text Preprocessing: ', round(100*accuracy_score(y1_test,predictions3)))

print('Text_Naive Bayes Classifier_Without Text Preprocessing: ', round(100*accuracy_score(y1_test,predictions4)))

print('Text_Passive Aggressive Classifier_With Text Preprocessing: ', round(100*accuracy_score(y1_test,predictions5)))

print('Text_Passive Aggressive Classifier_Without Text Preprocessing: ', round(100*accuracy_score(y1_test,predictions6)))

print('\n')

print('Title_Random Forest Classifier_With Text Preprocessing: ', round(100*accuracy_score(y2_test,predictions7)))

print('Title_Random Forest Classifier_Without Text Preprocessing: ', round(100*accuracy_score(y2_test,predictions8)))

print('Title_Naive Bayes Classifier_With Text Preprocessing: ', round(100*accuracy_score(y2_test,predictions9)))

print('Title_Naive Bayes Classifier_Without Text Preprocessing: ', round(100*accuracy_score(y2_test,predictions10)))

print('Title_Passive Aggressive Classifier_With Text Preprocessing: ', round(100*accuracy_score(y2_test,predictions11)))

print('Title_Passive Aggressive Classifier_Without Text Preprocessing: ', round(100*accuracy_score(y2_test,predictions12)))

print('\n')

print('Text&Title_Random Forest Classifier_With Text Preprocessing: ', round(100*accuracy_score(y3_test,predictions13)))

print('Text&Title_Random Forest Classifier_Without Text Preprocessing: ', round(100*accuracy_score(y3_test,predictions14)))

print('Text&Title_Naive Bayes Classifier_With Text Preprocessing: ', round(100*accuracy_score(y3_test,predictions15)))

print('Text&Title_Naive Bayes Classifier_Without Text Preprocessing: ', round(100*accuracy_score(y3_test,predictions16)))

print('Text&Title_Passive Aggressive Classifier_With Text Preprocessing: ', round(100*accuracy_score(y3_test,predictions17)))

print('Text&Title_Passive Aggressive Classifier_Without Text Preprocessing: ', round(100*accuracy_score(y3_test,predictions18)))