import time

start = time.clock()



#open data

import pandas as pd

import numpy as np

import nltk

import codecs



SampleSize=100000

datas = pd.read_csv('../input/train.csv') #

#datas=datas[datas['is_duplicate'] == 1]

#datas=datas.sample(SampleSize)

datas=datas[0:SampleSize]

datas = datas.fillna('leeg')

print(datas.head())



def cleantxt(x):    # aangeven sentence

    x = x.lower()

    # Removing non ASCII chars

    x = x.replace(r'[^\x00-\x7f]',r' ')

    # Pad punctuation with spaces on both sides

    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:

        x = x.replace(char, ' ' + char + ' ')

    return x



datas['question1']=datas['question1'].map(cleantxt)

datas['question2']=datas['question2'].map(cleantxt)



end = time.clock()

print('open:',end-start)
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer

token_dict = {}

stemmer = PorterStemmer()



def canary(row):

    shared = [w for w in row[3] if w not in row[4]]

    canary=(0.5*len(shared)/len(row[3]) + 0.5*len(shared)/len(row[4]))

    return canary



def stem_tokens(tokens, stemmer):

    stemmed = []

    for item in tokens:

        stemmed.append(stemmer.stem(item))

    return stemmed



def tokenize(text):

    tokens = nltk.word_tokenize(text)

    stems = stem_tokens(tokens, stemmer)

    return stems



def equ_words(q1words,q2words):

    return [w for w in q1words if w in q2words]



q1=datas['question1']

q2=datas['question2']



def same(row):

    q1words = str(row[3]).lower().split()

    q2words = str(row[4]).lower().split()

    equq1 = [w for w in q1words if w in q2words]

    return ' '.join(equq1)





def diff1(row):

    q1words = str(row[3]).lower().split()

    q2words = str(row[4]).lower().split()

    equq1 = [w for w in q1words if w not in q2words]

    return ' '.join(equq1)





def diff2(row):

    q1words = str(row[3]).lower().split()

    q2words = str(row[4]).lower().split()

    equq1 = [w for w in q2words if w not in q1words]

    

    return ' '.join(equq1)



print(datas.head())

datas['same_words'] = datas.apply(same, axis=1, raw=True)

datas['diff_word1'] = datas.apply(diff1, axis=1, raw=True)

datas['diff_word2'] = datas.apply(diff2, axis=1, raw=True)

datas['canary'] = datas.apply(canary,axis=1,raw=True)



q3=datas['same_words']

q4=datas['diff_word1']

q5=datas['diff_word2']

end = time.clock()

print('same/diff words:',end-start)

print(datas.head())

print(q1.shape,q2.shape)

qt=q1.append(q2)

qt=qt.append(q3)

qt=qt.append(q4)

qt=qt.append(q5)

print(qt)

print(qt.shape)





tfidf = TfidfVectorizer(stop_words='english')

tfs = tfidf.fit_transform(qt)

end = time.clock()

print('tfidf finished timing:',end-start)

print(tfidf)
#similarity=tfs.dot(tfs.T)



corr_q12=[]

corr_q1e=[]

corr_q1d1=[]

corr_q1d2=[]

corr_d1d2=[]

for xi in range (0,SampleSize):

    v1=tfs[xi]

    v2=tfs[SampleSize+xi]

    simQ12=v1.dot(v2.T)

    v3=tfs[2*SampleSize+xi]

    simQ1eq=v1.dot(v3.T)

    v4=tfs[3*SampleSize+xi]

    simQ1d1=v1.dot(v4.T)

    v5=tfs[4*SampleSize+xi]

    simQ1d2=v1.dot(v5.T)

    simd1d2=v4.dot(v5.T)

    



    if xi/1000==round(xi/1000):

        #print(datas.iloc[xi],similarity[xi,xi],similarity[SampleSize+xi,xi],similarity[2*SampleSize+xi,xi],similarity[3*SampleSize+xi,xi],similarity[4*SampleSize+xi,xi],similarity[4*SampleSize+xi,3*SampleSize+xi])

        #print(similarity[xi,xi],similarity[SampleSize+xi,xi],similarity[2*SampleSize+xi,xi],similarity[3*SampleSize+xi,xi],similarity[4*SampleSize+xi,xi],similarity[4*SampleSize+xi,3*SampleSize+xi])

        end = time.clock()

        print('correlate time/1000:',end-start)

    corr_q12.append(simQ12.toarray()[0,0])

    corr_q1e.append(simQ1eq.toarray()[0,0])

    corr_q1d1.append(-simQ1d1.toarray()[0,0])

    corr_q1d2.append(-simQ1d2.toarray()[0,0])

    corr_d1d2.append(simd1d2.toarray()[0,0])

 

end = time.clock()

print('corr time finished:',end-start)

#print(corr_d1d2)
#grafiek voor de tfidf compleet 

import math 

import seaborn as sns

import math

import matplotlib.pyplot as plt



anov=pd.DataFrame(datas['is_duplicate'])

anov=anov.set_index([[i for i in range(0,SampleSize)]])

anov2=pd.DataFrame(corr_q12)

anov3=pd.DataFrame(corr_q1e)

anov4=pd.DataFrame(corr_q1d1)

anov5=pd.DataFrame(corr_q1d2)

anov6=pd.DataFrame(corr_d1d2)

anov7=anov2+anov3+anov4

anov8=datas['canary']

print(anov7)

anov = pd.concat([anov,anov2 ],axis=1)

anov = pd.concat([anov,anov3 ],axis=1)

anov = pd.concat([anov,anov4 ],axis=1)

anov = pd.concat([anov,anov5 ],axis=1)

anov = pd.concat([anov,anov6 ],axis=1)

anov = pd.concat([anov,anov7 ],axis=1)

anov = pd.concat([anov,anov8],axis=1)

anov.columns=['is_duplicate','corr_q12','corr_q1e','corr_q1d1','corr_q1d2','corr_d1d2','ampli','canary']

anov = anov.fillna(1)

print(anov)

anov=pd.DataFrame(anov)

#, index=datas.index,columns=[['is_duplicate'],['predict']])



sns.set(style="white", color_codes=True)

#anov.fillna(value=0,inplace=True)

#sns.jointplot(x="prob", y="_True", data=similXY, size=5)

sns.pairplot(anov, hue="is_duplicate", size=3)

#sns.pairplot(similXY.drop("Id", axis=1), hue="is_duplicate", size=2, diag_kind="kde")



plt.savefig('quiqai.png') 

plt.show()



end = time.clock()

print('graph finished:',end-start)
from sklearn.cross_validation import train_test_split

similXY=anov

y=anov['is_duplicate']

#Xpr1_=datas['wopr1'].str[1:-1].str.split(',',expand=True)

#Xpr1_.fillna(0)

#print(Xpr1_)

X_train, X_test, y_train, y_test = train_test_split(anov[['corr_q12','corr_q1e','corr_q1d1','corr_q1d2','corr_d1d2','ampli','canary']], y, test_size=.3, random_state=0)

#print(Xwp_voca)

print('There are {} samples in the training set and {} samples in the test set'.format(

X_train.shape[0], X_test.shape[0]))

print()

# scaling the dataset

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)

end = time.clock()

print('graph finished:',end-start)



print('After standardizing our features, the first 5 rows of our data now look like this:\n')

X_train_print = pd.DataFrame(X_train_std)

X_train_print.head(5)

from sklearn.svm import SVC



svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

svm.fit(X_train_std, y_train)

print('________The accuracy SVM classifier')

print(' {:.2f} out of 1 training data is '.format(svm.score(X_train_std, y_train)))

print('  {:.2f} out of 1 test data is'.format(svm.score(X_test_std, y_test)))

end = time.clock()

print('graph finished:',end-start)



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski')

knn.fit(X_train_std, y_train)

print('________The accuracy of the knn classifier')

print(' {:.2f} out of 1 on training data'.format(knn.score(X_train_std, y_train)))

print(' {:.2f} out of 1 on test data'.format(knn.score(X_test_std, y_test)))



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=2)

model.fit(X_train_std,y_train)

print('________The Accuracy RandomForest')

print('{:.2f} on training'.format(model.score(X_train_std, y_train)))

print('{:.2f} on test'.format(model.score(X_test_std, y_test)))



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_std,y_train)

print('________Logistic Regress')

print('{:.2f} on training'.format(model.score(X_train_std, y_train)))

print('{:.2f} on test'.format(model.score(X_test_std, y_test)))

print('coeff:',model.coef_ )

print('intercept:',model.intercept_ )





from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='gini',max_depth=4,presort=True)

model.fit(X_train_std,y_train)

print('________Decision Tree')

end = time.clock()

print('graph finished:',end-start)



print('{:.2f} on training'.format(model.score(X_train_std, y_train)))

print('{:.2f} on test'.format(model.score(X_test_std, y_test)))