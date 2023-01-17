import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


''' group sum
'''

def grouping(df,dtest,grby):
    agg_col =  df.select_dtypes(include='number').columns.values
    tot=df.append(test,ignore_index=True)
    print('stats on',agg_col)
    for gci in grby:
    
        group = tot.groupby(gci)
        # group count, mean, max, min
        gCount = group.size()
        gCount=pd.DataFrame(gCount,columns=[gci+'_count'])
        gMean = group.mean().rename(columns=lambda s: gci+'_avg.' + s)
        #print(gMean)
        gMax = group[agg_col].max().rename(columns=lambda s: gci+'_max.' + s)
        gMin = group[agg_col].min().rename(columns=lambda s: gci+'_min.' + s)
        grbij=pd.concat([gCount, gMean, gMax, gMin], axis=1)
        df=df.merge( grbij,how='left',left_on=gci,right_index=True)
        dtest=dtest.merge( grbij,how='left',left_on=gci,right_index=True)

    return df,dtest




def groupvarwithtarget(df,dtest,target):
    agg_col = [xi for xi in df.select_dtypes(include='number').columns.values if xi not in target]
    tot=df #.append(test,ignore_index=True)
    print('stats on',agg_col)
    for gci in agg_col:
        if len( tot[gci].unique())<500:
            group = tot[target+[gci]].groupby(gci)
            # group count, mean, max, min
            gCount = group.size()
            gCount=pd.DataFrame(gCount,columns=[gci+'_count'])
            gMean = group.mean().rename(columns=lambda s: gci+'_avg.' + s)
            #print(gMean)
            gMax = group.quantile(0.75).rename(columns=lambda s: gci+'_max.' + s)
            gMin = group.min().rename(columns=lambda s: gci+'_min.' + s)
            grbij=pd.concat([gCount, gMean, gMax, gMin], axis=1)
            df=df.merge( grbij,how='left',left_on=gci,right_index=True)
            dtest=dtest.merge( grbij,how='left',left_on=gci,right_index=True)

    return df,dtest


def concatvartotxt(df,dtest,target):
    agg_col = [xi for xi in df.select_dtypes(include='number').columns.values if xi not in target]
    train_txt =df.Description.fillna(' ')+' '+df.Name.fillna(' ')+' '
    test_txt=dtest.Description.fillna(' ')+' '+dtest.Name.fillna(' ')+' '
    print('stats on',agg_col)
    for gci in agg_col:
        if len( df[gci].unique())<500:
            train_txt+=df[gci].fillna(0).map(str)+gci+' '
            test_txt+=dtest[gci].fillna(0).map(str)+gci+' '
    print('concat variables',df.shape,dtest.shape,len(train_txt),len(test_txt))
    return train_txt,test_txt

train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')

import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import datasets


from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

def klassif(Xtrain,Xtes,y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB 
    from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from mlxtend.classifier import StackingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, train_test_split
    
    print('klassify',Xtrain.shape,Xtes.shape,y.shape)

    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = MLPClassifier(alpha=1,max_iter=400 ) #RandomForestClassifier(random_state=1)
    clf3 = SVC(kernel="linear", C=0.025,max_iter=400) #GaussianNB()
    clf4 =ExtraTreesClassifier()
    lr = LogisticRegression(solver='lbfgs',multi_class='auto')
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3,clf4], meta_classifier=lr)

    label = ['KNN', 'MLP', 'SVC', 'Xtree','Stacking Classifier']
    clf_list = [clf1, clf2, clf3, clf4,sclf]
    
    grid = itertools.product([0,1],repeat=2)

    #y=traind.AdoptionSpeed.values
    clf_cv_mean = []
    clf_cv_std = []
    for clf, label, grd in zip(clf_list, label,grid):
        
        scores = cross_val_score(clf, Xtr[:len(y)], y, cv=3, scoring='accuracy')
        print ( "Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label) )
        
        clf.fit(Xtrain[:len(y)], y)
    return pd.DataFrame( clf.predict(Xtes),index=Xtes.index)

#train = pd.read_csv('../input/train/train.csv')
#test = pd.read_csv('../input/test/test.csv')

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD


def textmatrix(train1,test1,ncomp):
    
    #train_desc,test_desc=concatvartotxt(train1,test1,['AdoptionSpeed','demand'])
    train_desc,test_desc=concatvartotxt(train1,test1,['Type','AdoptionSpeed'])
    #print(train_desc[0])

    tfv = TfidfVectorizer(min_df=3,  max_features=50000,
            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1, 1), use_idf=1, smooth_idf=1, sublinear_tf=1,
            stop_words = 'english')
    #tfv= CountVectorizer()

    # Fit TFIDF
    #tfv.fit(train_desc)
    X =  tfv.fit_transform(train_desc.append(test_desc))
    #svd
    svd = TruncatedSVD(n_components=ncomp)
    Xtr = svd.fit_transform(X)
    print(svd.explained_variance_ratio_.sum())
    print(svd.explained_variance_ratio_)
    
    Xtra = pd.DataFrame(Xtr[:len(train1)], columns=['svd_{}'.format(i) for i in range(ncomp)])
    Xtes = pd.DataFrame(Xtr[len(train1):], columns=['svd_{}'.format(i) for i in range(ncomp)],index=test1.index)
    print('tfidf-svd',train1.shape,Xtra.shape,test1.shape,Xtes.shape)
    return Xtra,Xtes


def sentimentcatch(train,test,train_id,test_id):
    import json

    doc_sent_mag = []
    doc_sent_score = []
    nf_count = 0
    for pet in train_id:
        try:
            with open('../input/train_sentiment/' + pet + '.json', 'r') as f:
                sentiment = json.load(f)
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            nf_count += 1
            doc_sent_mag.append(-1)
            doc_sent_score.append(-1)

    train.loc[:, 'doc_sent_mag'] = doc_sent_mag
    train.loc[:, 'doc_sent_score'] = doc_sent_score

    doc_sent_mag = []
    doc_sent_score = []
    nf_count = 0
    for pet in test_id:
        try:
            with open('../input/test_sentiment/' + pet + '.json', 'r') as f:
                sentiment = json.load(f)
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            nf_count += 1
            doc_sent_mag.append(-1)
            doc_sent_score.append(-1)

    test.loc[:, 'doc_sent_mag'] = doc_sent_mag
    test.loc[:, 'doc_sent_score'] = doc_sent_score
    return train,test

trainc=train[train['Type']==2]
traind=train[train['Type']==1]
testc=test[test['Type']==2]
testd=test[test['Type']==1]
print(trainc.shape,testc.shape,traind.shape,testd.shape)
Xtr,Xtest=textmatrix(train,test,300)
pred=klassif(Xtr,Xtest,train.Type.values)
print ('Predicition accuracy cat dog txt',(pred[0]==test.Type).mean() )
pred['PetID']=test['PetID']

print(numcol)
print( train[train['Name'].map(lambda x: str(x)[:4])=='Mama']['Name'] )
Xtr1,Xtest1=sentimentcatch(train,test,train.PetID,test.PetID)
numcol = [xi for xi in test.select_dtypes(include='number').columns.values if xi not in ['Type']]
pred=klassif(Xtr1[numcol],Xtest1[numcol],train.Type.values)
print ('Predicition accuracy cat dog txt',(pred[0]==test.Type).mean() )

numcol = [xi for xi in test.select_dtypes(include='number').columns.values if xi not in ['Type']]
pred=klassif(train[numcol],test[numcol],train.Type.values)
print ('Predicition accuracy cat dog features',(pred[0]==test.Type).mean() )
pred=klassif(train[['Breed1','Breed2']],test[['Breed1','Breed2']],train.Type.values)
print ('Predicition accuracy cat dog features',(pred[0]==test.Type).mean() )


train[['Breed1','Type','Quantity']].groupby(['Breed1','Type']).count()
# test is ill in the same bed
test[['Breed1','Type','Quantity']].groupby(['Breed1','Type']).count()
breed = pd.read_csv('../input/breed_labels.csv')

trainbr=train.merge(breed,how='left',left_on='Breed1',right_on='BreedID',suffixes=('','_br'))
trainbr[trainbr.Type*1!=trainbr.Type_br].sort_values('Breed1')
