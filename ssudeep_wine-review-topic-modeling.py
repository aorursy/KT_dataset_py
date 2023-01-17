import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
wdat1 = pd.read_csv('../input/winemag-data_first150k.csv',delimiter=',',index_col=0,quotechar='"')
wdat2 = pd.read_csv('../input/winemag-data-130k-v2.csv',delimiter=',',index_col=0,quotechar='"')
print(wdat1.shape)
wdat1.head()
print(wdat2.shape)
wdat2.head()
# encode as unicode
wdat1['country']=wdat1['country'].values.astype('U')
wdat1['description']=wdat1['description'].values.astype('U')
wdat1['variety']=wdat1['variety'].values.astype('U')
wdat1['price']=wdat1['price'].fillna(0.0)
# dataset 2
wdat2['country']=wdat2['country'].values.astype('U')
wdat2['description']=wdat2['description'].values.astype('U')
wdat2['variety']=wdat2['variety'].values.astype('U')
wdat2['taster_name']=wdat2['taster_name'].values.astype('U')
wdat2['price']=wdat2['price'].fillna(0.0)
# convert to dict
desMap = {}
for idx,row in wdat1.iterrows():
    desMap[row['description']] = { 'country':row['country'],'points':row['points'],'variety':row['variety'],'price':row['price'],'taster':'nan'}
for idx,row in wdat2.iterrows():
    try:
        desMap[row['description']]['taster'] = row['taster_name']
        if desMap[row['description']]['price'] == 0 and row['price']>0:
            desMap[row['description']]['price'] = row['price']
    except KeyError:
        tasterName = 'nan' if len(str(row['taster_name']))<=4 else row['taster_name']
        desMap[row['description']] = { 'country':row['country'],'points':row['points'],'variety':row['variety'],'price':row['price'],'taster':tasterName }
print(len(desMap))
from collections import Counter
tasters = list()
country = list()
points  = list()
for des,dat in desMap.items():
    tasters.append(dat['taster'])
    country.append(dat['country'])
    points.append(dat['points'])
Counter(tasters)
Counter(country)
Counter(points)
wdes = list(desMap.keys())
wdes[:5]
import nltk
from nltk.stem import WordNetLemmatizer
wtokens = list()
for i,d in enumerate(wdes):
    wtokens.append(nltk.pos_tag(nltk.tokenize.word_tokenize(d)))
    if i%10000==0:
        print(i)
wtokens[:5]
lemmatizer = WordNetLemmatizer()
vset = set(['VB','VBD','VBG','VBN','VBP','VBZ'])
nset = set(['NN','NNS','NNP','NNPS'])
advset = set(['RB','RBR','RBS'])
adjset = set(['JJ','JJR','JJS'])
wmatch = re.compile('^\w{2,}.*$', re.IGNORECASE)
wlemmas = list()
for wt in wtokens:
    wtlist = list()
    for tk in wt:
        if tk[1] in vset:
            wtlist.append(lemmatizer.lemmatize(tk[0].lower(), 'v'))
        elif tk[1] in nset:
            wtlist.append(lemmatizer.lemmatize(tk[0].lower(), 'n'))
        elif tk[1] in advset:
            wtlist.append(lemmatizer.lemmatize(tk[0].lower(), 'r'))
        elif tk[1] in adjset:
            wtlist.append(lemmatizer.lemmatize(tk[0].lower(), 'a'))
        elif re.match(wmatch,string=tk[1]):
            wtlist.append(tk[0].lower())
    wlemmas.append(" ".join(wtlist))
wlemmas[:5]
from sklearn.feature_extraction.text import TfidfVectorizer
tf1 = TfidfVectorizer(analyzer='word',stop_words='english',min_df=0.002,max_df=0.95,ngram_range=(1,5),lowercase=True)
desTfIdf = tf1.fit_transform(wlemmas)
print(desTfIdf.shape)
print(tf1.get_feature_names())
ntokens = [ len(s.split(' ')) for s in tf1.get_feature_names() ]
print(Counter(ntokens))
from sklearn.decomposition import NMF
def get_topics(H,feature_names,W,docs,kw='NMF',n_topWords=25,n_topDocs=25):
    '''
    Given an H matrix, words mapped, a W matrix, mapped documents, number of top words and number of 
    top documents, return a dictionary with n_topWords and n_topDocs
    source: https://towardsdatascience.com/improving-the-interpretation-of-topic-models-87fd2ee3847d 
    '''
    topicMap = {}
    for topicInd,topic in enumerate(H):
        keywords = [feature_names[i] for i in topic.argsort()[:-n_topWords-1:-1] ]
        descriptions = list()
        topDocInd = np.argsort( W[:,topicInd] )[::-1][0:n_topDocs]
        for docInd in topDocInd:
            descriptions.append(docs[docInd])
        topicMap[kw+'.'+str(topicInd)] = {'kw':keywords,'desc':descriptions}
    return(topicMap)

def get_info(desMap,deslist):
    infoMap = {}
    variety = list()
    points = list()
    country = list()
    taster = list()
    price = list()
    for des in deslist:
        if des in desMap:
            variety.append(desMap[des]['variety'])
            points.append(desMap[des]['points'])
            price.append(desMap[des]['price'])
            country.append(desMap[des]['country'])
            taster.append(desMap[des]['taster'])
    infoMap['variety']= Counter(variety)
    infoMap['points']= points
    infoMap['price']= price
    infoMap['country']= Counter(country)
    infoMap['taster']= Counter(taster)
    return(infoMap)
        
nmf1 = NMF(max_iter=750,n_components=25, random_state=42, alpha=.05, l1_ratio=0.5,solver='cd',init='nndsvda',shuffle=True)
nmf1.fit(desTfIdf)
nmfW1 = nmf1.transform(desTfIdf)
print(nmfW1.shape)
print (nmf1.components_.shape)
descTopicsNmf = get_topics(nmf1.components_,tf1.get_feature_names(),nmfW1,wdes,'NMF',100,100)
print('**** Keywords ******')
print(descTopicsNmf['NMF.0']['kw'][:25])
print('**** Descriptions ******')
print(descTopicsNmf['NMF.0']['desc'][:5])
topic0Info = get_info(desMap,descTopicsNmf['NMF.0']['desc'])
print('**** Stats ******')
print('   variety   ')
print(topic0Info['variety'])
print('   country   ')
print(topic0Info['country'])
print('   tasters   ')
print(topic0Info['taster'])
print('   points   ')
print(pd.Series(topic0Info['points']).describe())
print('   price   ')
print(pd.Series(topic0Info['price']).describe())
print('**** Keywords ******')
print(descTopicsNmf['NMF.1']['kw'][:25])
print('**** Descriptions ******')
print(descTopicsNmf['NMF.1']['desc'][:5])
topic1Info = get_info(desMap,descTopicsNmf['NMF.1']['desc'])
print('**** Stats ******')
print('   variety   ')
print(topic1Info['variety'])
print('   country   ')
print(topic1Info['country'])
print('   tasters   ')
print(topic1Info['taster'])
print('   points   ')
print(pd.Series(topic1Info['points']).describe())
print('   price   ')
print(pd.Series(topic1Info['price']).describe())
def get_topic_similarity(topicDict,ntopics=100):
    '''
    Calculate Jaccard index based on ton n topics in wine description
    '''
    simMat = np.ones((len(topicDict),len(topicDict)),dtype=np.float)
    for i,k in enumerate(topicDict.keys()):
        kwords = set(topicDict[k]['kw'][:ntopics])
        for j,l in enumerate(topicDict.keys()):
            if j>i: # 
                lwords = set(topicDict[l]['kw'][:ntopics])
                JI = len(kwords & lwords)/float(len(kwords | lwords))
                simMat[j,i] = JI
                simMat[i,j] = JI
    return(pd.DataFrame(simMat,index=list(topicDict.keys()),columns=list(topicDict.keys())))

def get_variety_similarity(topicDict,desmap,nvariety=0):
    '''
    Calculate Jaccard index based on top N (in terms of count) 
    wine varieties in wine description topics
    '''
    varietySim = np.ones((len(topicDict),len(topicDict)),dtype=np.float)
    for i,k in enumerate(topicDict.keys()):
        variety1 = list()
        for desc in topicDict[k]['desc']:
            variety1.append(desmap[desc]['variety'])
        if nvariety == 0:
            varTopN1 = set(variety1)
        else:
            varTup1 = sorted(list(Counter(variety1).items()),key=lambda vit: vit[1],reverse=True)
            varTopN1 = set([v[0] for v in varTup1[:nvariety]])
        for j,l in  enumerate(topicDict.keys()):
            if j>i:
                variety2 = list()
                for desc in topicDict[l]['desc']:
                    variety2.append(desmap[desc]['variety'])
                if nvariety == 0:
                    varTopN2 = set(variety2)
                else:
                    varTup2 = sorted(list(Counter(variety2).items()),key=lambda vit: vit[1],reverse=True)
                    varTopN2 = set([v[0] for v in varTup2[:nvariety]])
                JI = len(varTopN1 & varTopN2)/float(len(varTopN1 | varTopN2))
                varietySim[i,j] = JI
                varietySim[j,i] = JI
    return((pd.DataFrame(varietySim,index=list(topicDict.keys()),columns=list(topicDict.keys()))))
topicSimilarityNmf = get_topic_similarity(descTopicsNmf)
sns.clustermap(topicSimilarityNmf,square=True,figsize=(15,15),cmap=sns.light_palette((0.231,0.349,0.596),n_colors=11))
print(topicSimilarityNmf.apply(np.sort,axis=1).iloc[:,-2])
varietySimilarityNmf = get_variety_similarity(descTopicsNmf,desMap,0)
sns.clustermap(varietySimilarityNmf,square=True,figsize=(15,15),cmap=sns.light_palette((0.447,0.184,0.2156),n_colors=11))
print(varietySimilarityNmf.apply(np.sort,axis=1).iloc[:,-2])
similarTopics = pd.DataFrame({'topicSimilarity':topicSimilarityNmf.columns[topicSimilarityNmf.values.argsort()[:,-2]],'varietySimilarity':varietySimilarityNmf.columns[varietySimilarityNmf.values.argsort()[:,-2]]},index=topicSimilarityNmf.index)
similarTopics
varietySimilarityNmf10 = get_variety_similarity(descTopicsNmf,desMap,10)
similarTopics10 = pd.DataFrame({'topicSimilarity':topicSimilarityNmf.columns[topicSimilarityNmf.values.argsort()[:,-2]],'varietySimilarity10':varietySimilarityNmf10.columns[varietySimilarityNmf10.values.argsort()[:,-2]]},index=topicSimilarityNmf.index)
similarTopics10