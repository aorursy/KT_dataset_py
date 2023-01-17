import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



#NLTK functioncs

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.corpus import wordnet



stop = set(stopwords.words('english'))   #_____________ import stop words



# timing function

import time   

start = time.clock() #_________________ measure efficiency timing





# read data

train = pd.read_csv('../input/train.csv',encoding='utf8')[:25000]  #_______________________ open data files

test = pd.read_csv('../input/test.csv',encoding='utf8')[:25000]  #_______________________ open data files

print(train.head(3))

train.fillna(value='leeg',inplace=True)

#train=train.dropna(axis=0, how='any')  #clean empty rows that give trouble

end = time.clock()

print('open:',end-start)
def cleantxt(q1,q2):

    #print(q1,q2)

    q1words = nltk.word_tokenize(q1)

    q2words = nltk.word_tokenize(q2)

    equq1 = [w for w in q1words if w in q2words]

    #difq1 = [stemmer.stem(w.decode("utf8")) for w in q1words if w not in q2words]  # stemming sometimes simplifies things

    #difq2 = [stemmer.stem(w.decode("utf8")) for w in q2words if w not in q1words ]

    difq1 = [w for w in q1words if w not in q2words] 

    difq2 = [w for w in q2words if w not in q1words ]

    #print(difq1,difq2)

    #wsq1=[wordnet.synsets(w.decode("utf8")) for w in difq1]

    #wsq2=[wordnet.synsets(w.decode("utf8")) for w in difq2]    #synsetting seems to find only stemmable words...

    netto=list(set(difq1+difq2))

    return q1words,q2words,difq1,difq2,equq1

    



q1=[]

q2=[]

di1=[]

di2=[]

eq=[]

for xi in range(len(train)):

    q1words,q2words,difq1,difq2,equq1=cleantxt(train.iloc[xi].question1,train.iloc[xi].question2)

    q1.append(q1words)

    q2.append(q2words)

    di1.append(difq1)

    di2.append(difq2)

    eq.append(equq1)



train['q1']=q1

train['q2']=q2

train['di1']=di1

train['di2']=di2

train['eq']=eq



q1=[]

q2=[]

di1=[]

di2=[]

eq=[]

for xi in range(len(test)):

    q1words,q2words,difq1,difq2,equq1=cleantxt(test.iloc[xi].question1,test.iloc[xi].question2)

    q1.append(q1words)

    q2.append(q2words)

    di1.append(difq1)

    di2.append(difq2)

    eq.append(equq1)



test['q1']=q1

test['q2']=q2

test['di1']=di1

test['di2']=di2

test['eq']=eq



print(train.head())

    

end = time.clock()

print('cleaned:',end-start)
# Lets redo it but splitted... and use the existing vocabulary

from nltk.tokenize import word_tokenize



s1=train['q1'].map(lambda x: ' '.join(x),na_action=None)

s2=train['q2'].map(lambda x: ' '.join(x),na_action=None)

sd3=train['di1'].map(lambda x: ' '.join(x),na_action=None)                  

sd4=train['di2'].map(lambda x: ' '.join(x),na_action=None)  

se5=train['eq'].map(lambda x: ' '.join(x),na_action=None)  

st1=test['q1'].map(lambda x: ' '.join(x),na_action=None)

st2=test['q2'].map(lambda x: ' '.join(x),na_action=None)

temp=s1.append(s2)

temp=temp.append(st1)

temp=temp.append(st2)

count_vectorizer = CountVectorizer(ngram_range=(1, 2),min_df=0.00001)

count_vectorizer.fit(temp)  #Learn vocabulary and idf, return term-document matrix.



count1_vectorizer = CountVectorizer(ngram_range=(1, 2),vocabulary=count_vectorizer.vocabulary_)

count1_vectorizer.fit_transform(s1)  #Learn vocabulary and idf, return term-document matrix.

freq1_term_matrix = count_vectorizer.transform(s1)

count2_vectorizer = CountVectorizer(ngram_range=(1, 2),vocabulary=count_vectorizer.vocabulary_)

count2_vectorizer.fit_transform(s2)

#s2>s1

freq2_term_matrix = count_vectorizer.transform(s2) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

count3_vectorizer = CountVectorizer(ngram_range=(1, 2),vocabulary=count_vectorizer.vocabulary_)

count3_vectorizer.fit_transform(sd3)

freq3_term_matrix = count_vectorizer.transform(sd3) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

count4_vectorizer = CountVectorizer(ngram_range=(1, 2),vocabulary=count_vectorizer.vocabulary_)

count4_vectorizer.fit_transform(sd4)

freq4_term_matrix = count_vectorizer.transform(sd4) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

count5_vectorizer = CountVectorizer(ngram_range=(1, 2),vocabulary=count_vectorizer.vocabulary_)

count5_vectorizer.fit_transform(se5)

freq5_term_matrix = count_vectorizer.transform(se5) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

#se5>s1



tfidf1 = TfidfTransformer(norm="l2")

tf1_idf_matrix = tfidf1.fit_transform(freq1_term_matrix)

tfidf2 = TfidfTransformer(norm="l2")

tf2_idf_matrix = tfidf2.fit_transform(freq2_term_matrix)

tfidf3 = TfidfTransformer(norm="l2")

tf3_idf_matrix = tfidf3.fit_transform(freq3_term_matrix)

tfidf4 = TfidfTransformer(norm="l2")

tf4_idf_matrix = tfidf4.fit_transform(freq4_term_matrix)

tfidf5 = TfidfTransformer(norm="l2")

tf5_idf_matrix = tfidf5.fit_transform(freq5_term_matrix)





print('Questions1 x Words', tf1_idf_matrix.shape)

print('Questions2 x Words', tf2_idf_matrix.shape)

#print('Differenc1 x Words', tf3_idf_matrix.shape)

print('Differenc2 x Words', tf4_idf_matrix.shape)

print('Equality x Words', tf5_idf_matrix.shape)



#als je similariteit wilt zien...

#print('Q similarity',tf1_idf_matrix[:10].dot(tf2_idf_matrix[:10].T) )



print('example first 10 questions similarity')

corr1=tf1_idf_matrix[:30].dot(tf2_idf_matrix[:30].T).diagonal().round(1)

print(corr1)

print('example equality 1 - eq ')

corr2=tf1_idf_matrix[:30].dot(tf5_idf_matrix[:30].T).diagonal().round(1)

print(corr2)

print('example equality 2 - eq')

corr3=tf2_idf_matrix[:30].dot(tf5_idf_matrix[:30].T).diagonal().round(1)

print(corr3)

print('example difference 1 - dif1 ')

corr4=tf1_idf_matrix[:30].dot(tf3_idf_matrix[:30].T).diagonal().round(1)

print(corr4)

print('example difference 2 - dif2')

corr5=tf2_idf_matrix[:30].dot(tf4_idf_matrix[:30].T).diagonal().round(1)

print(corr5)

print('example difference 1 - dif2')

corr6=tf1_idf_matrix[:30].dot(tf4_idf_matrix[:30].T).diagonal().round(1)

print(corr6)

print('example difference 2 - dif1')

corr7=tf2_idf_matrix[:30].dot(tf3_idf_matrix[:30].T).diagonal().round(1)

print(corr7)



ctrain=pd.DataFrame([])

ctrain['q12']=tf1_idf_matrix[:].dot(tf2_idf_matrix[:].T).diagonal().round(2)

ctrain['q1eq']=tf1_idf_matrix[:].dot(tf5_idf_matrix[:].T).diagonal().round(2)

ctrain['q2eq']=tf2_idf_matrix[:].dot(tf5_idf_matrix[:].T).diagonal().round(2)

ctrain['q1d1']=tf1_idf_matrix[:].dot(tf3_idf_matrix[:].T).diagonal().round(2)

ctrain['q2d2']=tf2_idf_matrix[:].dot(tf4_idf_matrix[:].T).diagonal().round(2)

ctrain['q1d2']=tf1_idf_matrix[:].dot(tf4_idf_matrix[:].T).diagonal().round(2)

ctrain['q2d1']=tf2_idf_matrix[:].dot(tf3_idf_matrix[:].T).diagonal().round(2)



    

print(ctrain.head())



end = time.clock()

print('train tfidf:',end-start)
#the same for the test

from nltk.tokenize import word_tokenize



s1=test['q1'].map(lambda x: ' '.join(x),na_action=None)

s2=test['q2'].map(lambda x: ' '.join(x),na_action=None)

sd3=test['di1'].map(lambda x: ' '.join(x),na_action=None)                  

sd4=test['di2'].map(lambda x: ' '.join(x),na_action=None)  

se5=test['eq'].map(lambda x: ' '.join(x),na_action=None)  



count_vectorizer = CountVectorizer(ngram_range=(2, 2),min_df=0.00001)

count_vectorizer.fit(temp)  #Learn vocabulary and idf, return term-document matrix.





count1_vectorizer = CountVectorizer(ngram_range=(2, 2),vocabulary=count_vectorizer.vocabulary_)

count1_vectorizer.fit_transform(s1)  #Learn vocabulary and idf, return term-document matrix.

freq1_term_matrix = count_vectorizer.transform(s1)

count2_vectorizer = CountVectorizer(ngram_range=(2, 2),vocabulary=count_vectorizer.vocabulary_)

count2_vectorizer.fit_transform(s2)

#s2>s1

freq2_term_matrix = count_vectorizer.transform(s2) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

count3_vectorizer = CountVectorizer(ngram_range=(2, 2),vocabulary=count_vectorizer.vocabulary_)

count3_vectorizer.fit_transform(sd3)

freq3_term_matrix = count_vectorizer.transform(sd3) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

count4_vectorizer = CountVectorizer(ngram_range=(2, 2),vocabulary=count_vectorizer.vocabulary_)

count4_vectorizer.fit_transform(sd4)

freq4_term_matrix = count_vectorizer.transform(sd4) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

count5_vectorizer = CountVectorizer(ngram_range=(2, 2),vocabulary=count_vectorizer.vocabulary_)

count5_vectorizer.fit_transform(se5)

freq5_term_matrix = count_vectorizer.transform(se5) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

#se5>s1



tfidf1 = TfidfTransformer(norm="l2")

tf1_idf_matrix = tfidf1.fit_transform(freq1_term_matrix)

tfidf2 = TfidfTransformer(norm="l2")

tf2_idf_matrix = tfidf2.fit_transform(freq2_term_matrix)

tfidf3 = TfidfTransformer(norm="l2")

tf3_idf_matrix = tfidf3.fit_transform(freq3_term_matrix)

tfidf4 = TfidfTransformer(norm="l2")

tf4_idf_matrix = tfidf4.fit_transform(freq4_term_matrix)

tfidf5 = TfidfTransformer(norm="l2")

tf5_idf_matrix = tfidf5.fit_transform(freq5_term_matrix)





print('Questions1 x Words', tf1_idf_matrix.shape)

print('Questions2 x Words', tf2_idf_matrix.shape)

#print('Differenc1 x Words', tf3_idf_matrix.shape)

print('Differenc2 x Words', tf4_idf_matrix.shape)

print('Equality x Words', tf5_idf_matrix.shape)



#als je similariteit wilt zien...

#print('Q similarity',tf1_idf_matrix[:10].dot(tf2_idf_matrix[:10].T) )



print('example first 10 questions similarity')

corr1=tf1_idf_matrix[:30].dot(tf2_idf_matrix[:30].T).diagonal().round(1)

print(corr1)

print('example equality 1 - eq ')

corr2=tf1_idf_matrix[:30].dot(tf5_idf_matrix[:30].T).diagonal().round(1)

print(corr2)

print('example equality 2 - eq')

corr3=tf2_idf_matrix[:30].dot(tf5_idf_matrix[:30].T).diagonal().round(1)

print(corr3)

print('example difference 1 - dif1 ')

corr4=tf1_idf_matrix[:30].dot(tf3_idf_matrix[:30].T).diagonal().round(1)

print(corr4)

print('example difference 2 - dif2')

corr5=tf2_idf_matrix[:30].dot(tf4_idf_matrix[:30].T).diagonal().round(1)

print(corr5)

print('example difference 1 - dif2')

corr6=tf1_idf_matrix[:30].dot(tf4_idf_matrix[:30].T).diagonal().round(1)

print(corr6)

print('example difference 2 - dif1')

corr7=tf2_idf_matrix[:30].dot(tf3_idf_matrix[:30].T).diagonal().round(1)

print(corr7)



ttrain=pd.DataFrame([])

ttrain['q12']=tf1_idf_matrix[:].dot(tf2_idf_matrix[:].T).diagonal().round(1)

ttrain['q1eq']=tf1_idf_matrix[:].dot(tf5_idf_matrix[:].T).diagonal().round(1)

ttrain['q2eq']=tf2_idf_matrix[:].dot(tf5_idf_matrix[:].T).diagonal().round(1)

ttrain['q1d1']=tf1_idf_matrix[:].dot(tf3_idf_matrix[:].T).diagonal().round(1)

ttrain['q2d2']=tf2_idf_matrix[:].dot(tf4_idf_matrix[:].T).diagonal().round(1)

ttrain['q1d2']=tf1_idf_matrix[:].dot(tf4_idf_matrix[:].T).diagonal().round(1)

ttrain['q2d1']=tf2_idf_matrix[:].dot(tf3_idf_matrix[:].T).diagonal().round(1)



print(ttrain.head())

end = time.clock()

print('test tfidf:',end-start)
import xgboost as xgb



gbm = xgb.XGBClassifier(silent=False).fit(ctrain,train['is_duplicate'])

predictor = gbm.predict(ttrain)

print(gbm)

print('model prediction first 20')

for xi in range (0,40):

    print(predictor[xi],test.iloc[xi].question2)



end = time.clock()

print('XGB model trained:',end-start)
submit=[]

co1=[]

co2=[]

co3=[]

co4=[]

co5=[]

co6=[]

co7=[]

batch=1000

for xi in range(0,len(train),batch):

    if xi+batch>len(train):

        batch=len(train)-xi

    corr1=tf1_idf_matrix[xi:xi+batch].dot(tf2_idf_matrix[xi:xi+batch].T).diagonal().round(2)

    corr2=tf1_idf_matrix[xi:xi+batch].dot(tf5_idf_matrix[xi:xi+batch].T).diagonal().round(2)

    corr3=tf2_idf_matrix[xi:xi+batch].dot(tf5_idf_matrix[xi:xi+batch].T).diagonal().round(2)

    corr4=tf1_idf_matrix[xi:xi+batch].dot(tf3_idf_matrix[xi:xi+batch].T).diagonal().round(2)

    corr5=tf2_idf_matrix[xi:xi+batch].dot(tf4_idf_matrix[xi:xi+batch].T).diagonal().round(2)

    corr6=tf1_idf_matrix[xi:xi+batch].dot(tf4_idf_matrix[xi:xi+batch].T).diagonal().round(2)

    corr7=tf2_idf_matrix[xi:xi+batch].dot(tf3_idf_matrix[xi:xi+batch].T).diagonal().round(2)    

    co1.extend(corr1)

    co2.extend(corr2)    

    co3.extend(corr3)

    co4.extend(corr4)

    co5.extend(corr5)    

    co6.extend(corr6)

    co7.extend(corr7)    

    #submit.extend(corr6/corr3)

    submit.extend(corr1+(corr6/corr3-0.3))

train['len']= train['eq'].map(lambda x: len(x))

train['dif']=train['di2'].map(lambda x:len(x))



end = time.clock()

print('estimate if similarity of q1-diff2 is bigger then equality of q2 with common stem:',end-start) 
import numpy as np

similXY=pd.DataFrame([])

similXY['q12']=co1

similXY['q1e12']=co2

similXY['q2e12']=co3

similXY['q1d1']=co4

similXY['q2d2']=co5

similXY['q1d2']=co6

similXY['q2d1']=co7

similXY['isDUP']=(np.asarray(co1)+np.asarray(co2)+np.asarray(co3))/3

similXY['isNOT']=(np.asarray(co4)+np.asarray(co5))/2

similXY['is_duplicate']=similXY['isDUP']>similXY['isNOT']

similXY['is_duplicate']=similXY['is_duplicate']*1



similXY.fillna(value=0)

print(similXY.head(30))
import seaborn as sns

import numpy as np









sns.set(style="white", color_codes=True)

similsample=similXY.sample(n=300)

sns.pairplot(similsample, hue="is_duplicate", size=3)
submiss=pd.DataFrame(abs(similXY['isNOT']-1))

submiss.fillna(value=0,inplace=True)

submiss.to_csv('TfIdf_submission.csv')

print(submiss.describe())



# DUP/NOTDUP = 11

# pure DUP   = 0.8233



end = time.clock()

print('cleaned:',end-start)