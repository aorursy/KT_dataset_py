import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



#del train,test

train=pd.read_csv("../input/sydi.csv",names=['sid','scui','symptom','did','dcui','dicpc','disease','d1','dp','acute'])

#test=pd.read_csv("../input/test.csv")

train["teller"]=1
train2=train.groupby(['sid','did']).sum().reset_index()

train3=train2.pivot(index='did', columns='sid', values='teller')

train3=train3.fillna(0)
symp=train.groupby(['sid','symptom']).sum().reset_index()

dis=train.groupby(['did','disease']).sum().reset_index()

train3=pd.DataFrame(train3.values,index=dis['disease'],columns=symp['symptom'] )
symp[symp['symptom']=='Sneezing'],symp[symp['symptom']=='Fever']
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.decomposition import TruncatedSVD



vectorizer = CountVectorizer()

#X = vectorizer.fit_transform(train3.symps.fillna(''))

svd = TruncatedSVD(n_components=100)

u_=svd.fit_transform(train3)

v_=svd.fit_transform(train3.T)

svd.explained_variance_ratio_.sum()
u_.shape,v_.shape
from sklearn.metrics.pairwise import cosine_similarity

usim=cosine_similarity(u_)

usim[1].shape

vsim=cosine_similarity(v_)

usim.shape,vsim.shape
#disease similarities

# 223 ear swelling

train3['d1']=usim[223]

train3.sort_values('d1')[-10:]



symp.iloc[54],symp.iloc[210],symp.iloc[222],len(symp)
v_.shape,v_,v_[1]
train3['d1']=cosine_similarity(u_,v_[[4]].reshape(1,-1))



train3.sort_values('d1')[-10:]





Xsim=cosine_similarity(u_/svd.singular_values_,(v_[4]+v_[5]).reshape(1,-1))

train3['d1']=pd.DataFrame( Xsim ).values

train3.sort_values('d1').iloc[-10:,-1:]
#FOUT



v_[4]*v_[5]

Xsim=cosine_similarity(u_/svd.singular_values_,(v_[4]*v_[5]).reshape(1,-1))



train3['d1']=pd.DataFrame( Xsim ).values

train3.sort_values('d1')[-10:]
v_.shape
Xsim=cosine_similarity(u_/svd.singular_values_,(v_[4]).reshape(1,-1) )

Xsim2=cosine_similarity(u_/svd.singular_values_,(v_[5]).reshape(1,-1) )

print(Xsim*Xsim2)

train3['d1']=pd.DataFrame( Xsim*Xsim2 ).values

train3.sort_values('d1').iloc[-10:,-1:]
#np.linalg.inv( np.diag(svd.singular_values_) ) == (np.diag(1/svd.singular_value.shapes_))

temp=np.dot( (np.dot(u_,np.diag(1/svd.singular_values_)) ),v_[4] )

train3['d1']=[x for x in temp]

train3.sort_values('d1').iloc[-10:,-1:]
#np.linalg.pinv(svd.singular_values_*v_)

symp
q=np.zeros( (len(v_), 1) )

q[4]=1

q[5]=1

qu_=np.dot(q.T,(v_/svd.singular_values_))

train3['d1']=cosine_similarity(u_,qu_)

train3.sort_values('d1')[-10:]
qu_=np.dot(q.T,np.linalg.pinv(v_/svd.singular_values_).T)

train3['d1']=cosine_similarity(u_,qu_)

train3.sort_values('d1')[-10:]
u_.shape,train.shape,qu_.shape,cosine_similarity(qu_,u_).shape
#qu_=np.dot(q.T,(v_/svd.singular_values_))

#print(qu_)



#brute force original matrix similarity OK but limited

train3['d1']=cosine_similarity(train3.iloc[:,:-1],q.T)

train3.sort_values('d1')[-10:]
#svd reconstructed brute matrix similarity

train3['d1']=cosine_similarity(np.dot(u_,v_.T),q.T)

train3.sort_values('d1')[-10:]