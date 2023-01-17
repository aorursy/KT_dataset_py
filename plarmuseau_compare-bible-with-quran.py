import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

bible= pd.read_csv('../input/bibleverses/bible_data_set.csv') 
quran= pd.read_csv('../input/the-holy-quran/en.yusufali.csv')
quran2=pd.DataFrame(quran.groupby(['Surah'])['Text'].agg(lambda col: ' '.join(col)) ) #.agg(lambda col: ''.join(str(col))) )
quran2=quran2.reset_index()
bible=pd.DataFrame( bible.groupby(['book','chapter'])['text'].agg(lambda col: ' '.join(col)) )
bible=bible.reset_index()
bible

def mapping6(data1,veld1,klas2,data2,veld2,k): #,data3,veld3,k):
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer    
    #from scipy.sparse.linalg import svds    
    from sklearn.decomposition import TruncatedSVD
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron,SGDClassifier
    #data1 the unknown start data
    #data2 the database you want to link with and classify with
    print(data1.shape,veld1,klas2,data2.shape,veld2,k)
    kleur=['b']*len(data1)+['r']*len(data2)
    #vectorize

    vect = TfidfVectorizer(ngram_range=(1,1),stop_words='english',norm='l2',strip_accents='ascii') #,min_df=0.000125125) 
    vectdata2=vect.fit_transform(data2[veld2]).toarray()
    vectdata1=vect.transform(data1[veld1]).toarray()  #.append(data3[veld3])
    #print('vect',vect.fit_transform(data1[veld1].append(data2[veld2])))
    
    #svd
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
    U123=svd.fit_transform(np.concatenate((vectdata1,vectdata2), axis=0))
    pd.DataFrame(U123[:,:2]).plot.scatter(x=0,y=1,c=kleur)
    Xr=svd.inverse_transform(U123)
    #U123,s123,V123=svds(vect.fit_transform(data1[veld1].append(data2[veld2])),k=k) #.append(data3[veld3])

    print("datasvd",U123.shape)

    #temp=np.concatenate( (U123[len(data1):len(data1)+len(data2)]*s123[:k]   , dwm[len(data1):len(data1)+len(data2)]), axis=1 )
    temp=Xr[len(data1):len(data1)+len(data2)] #*s123[:k]
    U2=pd.DataFrame( temp  , index= data2.index)

    
    temp=Xr[:len(data1)] #*s123[:k]
    U1=pd.DataFrame( temp , index=data1.index )
    
    et = ExtraTreesClassifier(n_estimators=25, max_depth=300, min_samples_split=5, min_samples_leaf=1, random_state=None, min_impurity_decrease=1e-7)
    # TRAINING
    #et = SGDClassifier(n_jobs=4,max_iter=100)
    model = OneVsRestClassifier(et)
    #temp=pd.DataFrame( dwm[ len(data1):len(data1)+len(data2) ] )
    temp=U2.T #.T.append(temp.T)   #U2  append word tfidf vector
    #temp=temp.append( pd.DataFrame( list(data2['doc_vector'])).T )   #append gensim
    print('U2',temp.shape)
    model.fit(temp.T,data2[klas2])
    print( (model.predict(temp.T)==data2[klas2]).mean()*100 ) 
    
    #PREDICTING
    #temp=pd.DataFrame(dwm[ :len(data1)] )
    temp=U1.T #.T.append(temp.T)
    #temp=temp.append( pd.DataFrame( list(data1['doc_vector'])).T )   #append gensim
    data1[klas2]=model.predict(temp.T)
    
    return data1
mapping6(quran2,'Text','book',bible,'text',50)