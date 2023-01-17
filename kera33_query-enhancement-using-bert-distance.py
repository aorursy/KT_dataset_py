# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
##import torch, wordnet and BERT

import torch

from nltk.corpus import wordnet as wn

from transformers import *

from scipy.spatial.distance import cdist

import gc

import re

import itertools

#import display libraries

from IPython.display import HTML, display

from ipywidgets import interact, Layout, HBox, VBox, Box

import ipywidgets as widgets

from IPython.display import clear_output
##aggregate original files into various indexed data frames, done locally before uploading to kaggle

import os

import json

from copy import deepcopy

from tqdm.notebook import tqdm

import networkx as nx

import ast

from nltk.corpus import stopwords



def format_name(author):

    middle_name = " ".join(author['middle'])  

    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])



def format_affiliation(affiliation):

    text = []

    location = affiliation.get('location')

    if location:

        text.extend(list(affiliation['location'].values()))

    

    institution = affiliation.get('institution')

    if institution:

        text = [institution] + text

    return ", ".join(text)



def format_authors(authors, with_affiliation=False):

    name_ls = []  

    for author in authors:

        name = format_name(author)

        if with_affiliation:

            affiliation = format_affiliation(author['affiliation'])

            if affiliation:

                name_ls.append(f"{name} ({affiliation})")

            else:

                name_ls.append(name)

        else:

            name_ls.append(name)  

    return ", ".join(name_ls)



def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}    

    for section, text in texts:

        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    

    return body



def format_bib(bibs):

    if type(bibs) == dict:

        bibs = list(bibs.values())

    bibs = deepcopy(bibs)

    formatted = []   

    for bib in bibs:

        bib['authors'] = format_authors(

            bib['authors'], 

            with_affiliation=False

        )

        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]

        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)



def load_files(dirname):

    filenames = os.listdir(dirname)

    raw_files = []

    for filename in tqdm(filenames):

        filename = dirname + filename

        file = json.load(open(filename, 'rb'))

        raw_files.append(file)   

    return raw_files



def generate_clean_df(all_files):

    cleaned_files = []    

    for file in tqdm(all_files):

        features = [

            file['paper_id'],

            file['metadata']['title'],

            format_authors(file['metadata']['authors']),

            format_authors(file['metadata']['authors'], 

                           with_affiliation=True),

            format_body(file['abstract']),

            format_body(file['body_text']),

            format_bib(file['bib_entries']),

            file['metadata']['authors'],

            file['bib_entries']

        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()    

    return clean_df



pmc_dir = r'c:/kag/2020-04-01/custom_license/custom_license/'

pmc_files = load_files(pmc_dir)

pmc_df = generate_clean_df(pmc_files)



comm_dir = r'c:/kag/2020-04-01/comm_use_subset/comm_use_subset/'

comm_files = load_files(comm_dir)

comm_df = generate_clean_df(comm_files)



noncomm_dir = r'c:/kag/2020-04-01/noncomm_use_subset/noncomm_use_subset/'

noncomm_files = load_files(noncomm_dir)

noncomm_df = generate_clean_df(noncomm_files)



bio_dir = r'c:/kag/2020-04-01/biorxiv_medrxiv/biorxiv_medrxiv/'

bio_files = load_files(bio_dir)

bio_df = generate_clean_df(bio_files)



df = pd.concat([pmc_df,comm_df,noncomm_df, bio_df], axis=0)



df2 = pd.read_csv(r'c:/kag/2020-04-01/metadata.csv')

df = pd.merge(df2, df, left_on='sha', right_on='paper_id', how='left')

df.drop(['abstract_x', 'authors_x'], axis=1, inplace=True)

#intermediary save

#df.to_csv('alldata2.csv', index=False)



##compute page rank out of raw bibliography column to see the most relevant papers

df = df.fillna("")

#create a list of tuples of citation nodes

dec = []

i=0

while i<len(df):

    print(i)

    bf = df.loc[i,'raw_bibliography']

    if len(bf)>10:

        #infer the dictionary contained in raw_bibliography

        bf = ast.literal_eval(bf)

        j=0

        while j < len(bf.keys()):

            bfk = list(bf.keys())[j]

            try:

                #see if there is a link in the data frame based on doi key

                bfdoi = bf[bfk]['other_ids']['DOI'][0]

                n = df.loc[df['doi']==bfdoi,'pid']

            except:

                n = []

            if len(n)>0:

                ##if there is a link to a document in our df then append a tuple of nodes linked

                dec.append((i,n.values[0]))

            j=j +1

    i=i+1

#save list as a df

dec = pd.DataFrame(dec)

dec.columns=['n1','n2']

dec['w']=1

#build a graph out of tuples

graph = nx.from_pandas_edgelist(dec, 'n1', 'n2')

#compute page rank of nodes

pr = nx.pagerank(graph, alpha=0.85)

#build a df out of it

pt = pr.items()

res = pd.DataFrame([x[1] for x in enumerate(pt)])

res.columns = ['pid','pr']

#divide by the three largest and cap all to it (to reduce outlier impact of first place)

maxpd = res['pr'].sort_values(ascending=False).head(3).values[0]

res.loc[res['pr']>maxpd,'pr']=maxpd

res['pr'] = res['pr']/maxpd

#intermediary save

#res.to_csv('pagerank.csv')



##add page rank to the initial dataset

df['pid'] = df.index

df = pd.merge(df, res, on='pid')

##save the final used dataset

#df.to_csv('alldataset.csv', encoding='cp852')



##find the voabulary used in the papers

sw = set(stopwords.words('english'))

##create a long sentence list out of the whole df

sents = []        

for i in range(len(df)):

    print (i)

    sents.extend(df.loc[int(i),'text'].split(' '))  

    sents.extend(df.loc[int(i),'titley'].split(' '))

    sents.extend(df.loc[int(i),'titlex'].split(' '))

    sents.extend(df.loc[int(i),'abstract'].split(' '))

##find unique words

vs = pd.DataFrame(list(set(sents)))

vs.columns = ['word']

#eliminate stop words

vs = vs.loc[~vs['word'].isin(sw),:]

##using BERT encode and decode words into tokens

vs['token'] = vs['word'].apply(lambda x: tokenizer.encode(x)[1:-1])

vs['dec'] = vs['token'].apply(lambda x: tokenizer.decode(x))

##eliminate unkowns and wrong reconstruction

vs = vs.loc[vs['dec']==vs['word'],:]

numbers = ['1','2','3','4','5','6','7','8','9','0']

##eliminate words starting with a number

vs = vs.loc[~vs['word'].str.slice(0,1).isin(numbers),:]

##eliminate special characters

vs = vs.loc[~vs['word'].isin(puncts),:]    

##count word appearance in papers

vs.loc[:,'ct']=0

df = df.fillna('')

i=0

for i in range(len(df)):

    print (i)

    docw = df.loc[i,'text'].split()

    vs.loc[vs['word'].isin(docw), 'ct'] = vs.loc[vs['word'].isin(docw),'ct']+1

##eliminate words that appear too rare (1 instance) or too often (>10600out of 45000)

vs = vs.loc[((vs['ct']>=2) & (vs['ct']<=10600)),:]

#intermediate save

#vs.to_csv('vocabtoolarge3.csv')



vs = vs.sort_values(by='word')

vs = vs.reset_index(inplace=False, drop=True)

vs.index = vs['word']

##initialize paper id and in-paper frequency where each word appears

vs['docs'] = ""

vs['docf'] = ""

allw = set(vs['word'])

##

i=0    

while i<len(df):

    print (i)

    title = str(df.loc[i,'titley'])

    if len(title)==0:

        title = str(df.loc[i,'titlex'])

    ##look in title+abstarct+text

    docw = (title+str(df.loc[i,'abstract'])+str(df.loc[i,'text'])).split()

    if len(docw)>0:

        dp = pd.DataFrame(docw)

        dp.columns = ['c']

        dfp = pd.DataFrame(dp.groupby('c')['c'].count())

        dfp.loc[:,'word']= dfp.index.values

        dfp = dfp.loc[dfp['word'].isin(allw)]

        dfp = dfp.sort_values(by='word')

        vs.loc[vs['word'].isin(docw), 'docs'] = vs.loc[vs['word'].isin(docw),'docs']+str(i)+","

        vs.loc[vs['word'].isin(docw), 'docf'] = vs.loc[vs['word'].isin(docw),'docf']+dfp.loc[:,'c'].astype(str)+","

    i=i+1



    

#save the final indexed frequency used    

#vs.to_csv('vocabindexeddocs4.csv')



##load BERT to index embeddings of the vocabulary

config =  BertConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=False)

model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", config = config)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

vs = vs.fillna("")

## tokenise each word

vs['token'] = vs['word'].apply(lambda x: tokenizer.encode(x)[1:-1])

vs['len'] = vs['token'].str.len()

## keep only tokenisable words

vs = vs.loc[vs['len']>0,:]

vs.loc[:,'embed']=""

##create placeholder for the 768 embedding vector for each word

vse = pd.DataFrame(np.zeros((len(vs), 768)))

i=0

while i<len(vs):

    #print (i)

    ##create embedding

    vse.loc[i,:] = embed_text(vs.loc[i,'token'], model)

    i = i +1

##add word and token columns

vse['word'] = vs['word']

vse['token'] = vs['token']

##save the indexed embeddings to be used

#vse.to_csv('vocabembeds6.csv')



# upload the indexed sources (see above)



# covid synonyms frequencies by docs

cfd = pd.read_csv('/kaggle/input/covidfreq/covidfreqdocs-v3.csv')

# word frequencies by docs

vs = pd.read_csv('/kaggle/input/vocabindexed/vocabindexeddocs4.csv', usecols = ['word','ct','docs','docf'])

# wordnet synsets to be used for query extension

sdf = pd.read_csv('/kaggle/input/covidsyn/synssets2.csv')

# upload bert embeddings

vse = pd.read_csv('/kaggle/input/embeds/vocabembeds6.csv')

vse.drop(['Unnamed: 0'],axis=1, inplace=True)

# upload the preprocessed papers (see above)



##get the papers

df = pd.read_csv('/kaggle/input/cord19/alldataset.csv', encoding='cp852')

df.drop_duplicates(subset=['pid'],inplace=True)

df.drop_duplicates(subset=['url'],inplace=True)

df['year'] = df['publish_time'].str.slice(0,4)

df['year'] = df['year'].fillna("2020")

#df.loc[df['year'].str.len()<2,'year']

df['year'] = df['year'].astype(int)

##compute flag before covid-19 

df['month']= 12

df['befcovid']=0

df.loc[df['publish_time'].str.len()==10,'month'] = df.loc[df['publish_time'].str.len()==10,'publish_time'].str.slice(5,7).astype(int)

df.loc[~df['publish_time'].str.slice(0,4).isin(['2019','2020']),'befcovid']=1

df.loc[((df['publish_time'].str.slice(0,4).isin(['2019'])) & (df['month']<11)) ,'befcovid']=1
#helper functions and list



puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]



def embed_text(tokens, model):

    

    outs = model(torch.tensor(tokens).unsqueeze(0))[0][0].detach()

    outs = np.array(outs)

    outs = outs.mean(axis=0)

    return outs



#    w - brute synonims (plurals, singulars)

#    morph

#    from sysnet to exapnd one word 

#    from bert tokens and letter match percentage

#    from custom dictionaries for covid



def bertnn(emb, maxdistbnn):

    wsn = pd.DataFrame(vse.loc[:'word'])

    wsn.loc[:,'dist'] = cdist(emb.reshape(1,768),vse.loc[:,vse.columns[0:768]]).T

    wsn = wsn.sort_values(by='dist')     

    return wsn.loc[wsn['dist']<maxdistbnn,'word'].values[:10]

    

    



def fyndsyn(word, vs, sdf, maxdistbnn=10, maxdist=7*1e-1):

    if word=='covid':

        ws = ['covid','wuhan','covid19','covinfected','sars2']

    else:

        tok = tokenizer.encode(word)[1:-1]

        emb = embed_text(tok, model)

        ws = []

        wsc = []

        wsc.extend(sdf.loc[sdf['w']==word,'s'].values)

        wsc.extend(sdf.loc[sdf['s']==word,'w'].values)

        word2 = wn.morphy(word)

    #    print (word2)

        wsc.extend(sdf.loc[sdf['w']==word2,'s'].values)

        wsc.extend(sdf.loc[sdf['s']==word2,'w'].values)

        wsc = list(set(wsc))

        word3 = word+'s'

        word5 = word+'es'

        word6 = word+'ses'

        word4 = word[:-1]

        word7 = word[:-2]

        wsc.extend([word3,word4, word5, word6, word7])

        if word2!=None:

            wsc.append(word2)

        wsc = [w for w in wsc if w in vs['word'].values] 

        wsc = list(set(wsc))       

        if len(wsc)>0:

    #        print (len(wsc))

            wsc = pd.DataFrame(wsc)

            wsc.columns = ['word']

            wsn = pd.DataFrame(vse.loc[vse['word'].isin(wsc['word'].values),'word'])

            wsn.columns = ['word']

            wsn['dist'] = cdist(emb.reshape(1,768),vse.loc[vse['word'].isin(wsc['word'].values),vse.columns[0:768]]).T

            wsc = wsn.copy()

    #        wsc = wsc.sort_values(by='dist')

            ws.extend(wsc.loc[wsc['dist']<maxdist,'word'].values)

        ws.append(word)

        ws.extend(bertnn(emb, maxdistbnn))

        if word4 in ws and word4 in vs['word'].values:

                wsc = []

                wsc.extend(sdf.loc[sdf['w']==word4,'s'].values)

                wsc.extend(sdf.loc[sdf['s']==word4,'w'].values)

                if len(wsc)>0:

                    wsc = pd.DataFrame(wsc)

                    wsc.columns = ['word']

                    wsn = pd.DataFrame(vse.loc[vse['word'].isin(wsc['word'].values),'word'])

                    wsn.columns = ['word']

                    wsn['dist'] = cdist(emb.reshape(1,768),vse.loc[vse['word'].isin(wsc['word'].values),vse.columns[0:768]]).T

                    wsc = wsn.copy()

                    ws.extend(wsc.loc[wsc['dist']<maxdist,'word'].values)

        if word7 in ws and word7 in vs['word'].values:

                wsc = []

                wsc.extend(sdf.loc[sdf['w']==word7,'s'].values)

                wsc.extend(sdf.loc[sdf['s']==word7,'w'].values)

                if len(wsc)>0:

                    wsc = pd.DataFrame(wsc)

                    wsc.columns = ['word']

                    wsn = pd.DataFrame(vse.loc[vse['word'].isin(wsc['word'].values),'word'])

                    wsn.columns = ['word']

                    wsn['dist'] = cdist(emb.reshape(1,768),vse.loc[vse['word'].isin(wsc['word'].values),vse.columns[0:768]]).T

                    wsc = wsn.copy()

                    ws.extend(wsc.loc[wsc['dist']<maxdist,'word'].values)

        ws = list(set(ws))

        ws = [w for w in ws if w in vs['word'].values] 

    return ws



def snippet(df, wqs, wq3, doc, maxlen=500, maxpen=0):



    i=0

    adf = {}

    while i<len(wqs):

        wq = wqs[i]

        inds = []

        j=0

        while j < len(wq3.loc[wq3['word_y']==wq,'word_x']):

            wqj = wq3.loc[wq3['word_y']==wq,'word_x'].values[j]

            new = [x.start() for x in re.finditer(wqj, df.loc[df['pid']==doc,'text'].values[0])]

            inds.extend(new)

#            print (len(inds))

            j= j + 1

        inds = list(set(inds))

        inds.sort()

        if len(inds)>0:

            adf[wq] = inds 

        i = i + 1

    wqs = list(adf.keys())

    list1 = np.array(range(len(adf)))

    list1 = 1 + list1

    r=0

    pcombs = []

    for r in range(len(adf)):

        combs = [x for x in itertools.combinations(list1, r+2)]

        pcombs.extend(combs)

    if len(pcombs)==0:

        pcombs.extend([1])

    res = pd.DataFrame(pcombs)

    ##generated a list of all possible combinations of only words found from two to all found

    res.columns = [str(i) for i in list(range(len(adf)))]

    res = res.fillna(0)

    res['start']=0

    res['stop']=0

    r=-1

    extreme = 1e7 ##artificial limit beyond size of any document to help with good solutions

    p=0

    while p <len(pcombs) and len(wqs)>1:

        pv = pcombs[p]

        r=r+1

        ##solutions matrix has 2**len(pv)-1 entries because at each step we compute the right and left additions to existing solutions so power of 2

        sols = np.zeros((2**(len(pv)-1),2))

        sols2 = np.zeros((2**(len(pv)-1),2))

        v1 = np.array(adf[wqs[pv[0]-1]])

        v2 = np.array(adf[wqs[pv[1]-1]])



        ##the two vectors of index to substract from each other

        v1 = np.repeat(v1, len(v2), axis=0).reshape(len(v1),len(v2))

        ##repeat to allign for broadcast difference

        vd=v1-v2

        ##the min of positive differences is the closest to the right

        vd = vd.flatten()

        ## right closest

        vd[vd<0]=extreme

        ##artificially push negatives to very far positives in order to keep only interesting positives for further min

        if min(vd)>0 and min(vd)<extreme:

            m = np.argmin(vd)    

            m1 = m//len(v2)    ##index of 1, necessary since we flattened the vector to recall the correct row

            m2 = m-m1*len(v2)  ##index of 2

            ##first solution right way first choice

            sols[0] = [v2[m2], v1[m1,0] ]

        ##the max of negative differences is the closest to the left

        ## left closest

        vd=v1-v2

        ##the min of positive differences is the closest to the right

        vd = vd.flatten()

        vd[vd>0]=-extreme

         ##artificially push positives to very far negatives in order to keep only interesting negatives for further max

        if max(vd)<0 and max(vd)>-extreme:

            m = np.argmax(vd)    

            m1 = m//len(v2)    ##index of 1, necessary since we flattened the vector to recall the correct row

            m2 = m-m1*len(v2)  ##index of 2 , 

            ##start bottom half with left way first choice

            sols[1] = [ v1[m1,0], v2[m2] ]

        

        ##explore the next vectors of indices

        i=2

        nk=0

        while i< len(pv):

            nk=0

            k=0

            while k<2**(i-1):

                ## if we have a current solution and we compare to a new vector of indexes for a new word, we are looking for the closest to the avg of current interval see line geometry

                avg = (sols[k,0]+sols[k,1])/2

                if avg>0:   

                    vk = np.array(adf[wqs[pv[i]-1]])

                    ##right

                    vd = vk-avg 

                    vd[vd<0]=extreme

                    if min(vd)>0 and min(vd)<extreme:

                        m = np.argmin(vd)

                        sols2[nk] = [min(sols[k,0],vk[m]), max(sols[k,1],vk[m])]

                    else:

                        sols2[nk] = [0, 0]##invalidates solutions as not keeping pace

                    nk = nk+1

                    ##left 

                    vd = vk-avg

                    vd[vd>0]=-extreme

                    if max(vd)<0 and max(vd)>-extreme:

                        m = np.argmax(vd)

                        sols2[nk] = [min(sols[k,0],vk[m]), max(sols[k,1],vk[m])]

                    else:

                        sols2[nk] = [0, 0]##invalidates solutions as not keeping pace

                    nk = nk+1              

                k= k + 1

            ##copy temp solution to good solution

            sols = sols2

            i= i + 1

                 

        sdf = pd.DataFrame(sols)     

        sdf.columns = ['s','e']

        ##delete no solutions (0,0)

        sdf = sdf.loc[sdf['s']!=0,:]

        sdf.reset_index(inplace=True, drop=True)

        if len(sdf)>0:

            sdf['len'] = sdf.iloc[:,1]-sdf.iloc[:,0]

            spos = sdf['len'].idxmin()

            ##chooose the winner with the smallest len

            res.loc[r,['start','stop']]=sdf.iloc[spos,:2].values

        p=p+1

    

    if len(wqs)>1:

        res = res.astype(int)

        res = res.loc[res['start']!=0,:]

        res['len'] = res['stop']- res['start']

#        res = res.loc[res['len']<maxlen,:]

        res['penalty']=0

            

        i=len(wqs)*(len(wqs)+1)*10/2

        #    penalty = word rank * 10 (biggest penalty to rarest word)

        j=0

        res['penalty']= i

        while j<len(wqs):

            res['penalty']=res['penalty']-(len(wqs)+1-res.iloc[:,j])*10*(res.iloc[:,j]!=0).astype(int)

            j = j + 1

        res = res.loc[res['penalty']<=maxpen,:]

        res = res.sort_values(by=['len','start'])

        res.loc[res['len']>maxlen,'len']= maxlen

        res.reset_index(inplace=True, drop=True)

    else:

        res.loc[0,'start'] = min(adf[wqs[0]])

        res.loc[0,'len'] = 20

        res.loc[0,'penalty'] = 0 

    ##create output for snippet

    out = list(res.loc[0,['start','len', 'penalty']].astype(int).values)

    ##add missing words to output

    out.extend([list(set(wq3['word_y'])-set(wqs))])

    ##add query extended words to output

    exq = {}

    for w in wqs:

        listw = list(wq3.loc[wq3['word_y']==w,'word_x'].values) 

        if len(listw)>1:

            exq[w] = list(wq3.loc[wq3['word_y']==w,'word_x'].values) 

    out.extend([exq])

    return out
def showtext(st, lent, df, doc):

    lt = df.loc[df['pid']==doc,'text'].values[0][max(st-1000,0):st].rfind('.')

    rt = df.loc[df['pid']==doc,'text'].values[0][st+lent+max(0,200-lent):st+lent+max(0,200-lent)+1000].find(".")

    return df.loc[df['pid']==doc,'text'].values[0][max(st-1000,0)+lt+2:st+lent+max(0,200-lent)+rt+1]



#main function



##first expand the query then aggregate the findings back to original words

## then organise per doc and sort by penalty according to missing words in the target docs

def query(Q):

    ##replace symbols in Q

    for i in puncts:

        Q = Q.replace(i," ")

    wq = Q.split(" ")

    ##transform into covid synonyms

    wq2 = []

    i=0

    while i < len(wq)-1:

        if (wq[i]+"-"+wq[i+1]) in cfd.iloc[:,0].values:

            wq2.append("covid")

            i = i + 1

            print(i)

        else:

            wq2.append(wq[i])

            if i == len(wq)-2:

                wq2.append(wq[i+1])

        i= i + 1 

    if i>0:

        wq = wq2

    ##sort by rarity

    wqr =  vs.loc[vs['word'].isin(wq),['word','ct']]   

    wqr = wqr.sort_values(by='ct')

    wq = wqr['word'].values

    ##enrich the query

    i=0

    wq3 = []

    for w in wq:

        wq2 = fyndsyn(w, vs, sdf)

        wq2 = pd.DataFrame(wq2)

        wq2.columns=['syns']

        wq2['word'] = w

        if i==0:

            wq3 = wq2.copy()

        wq3 = pd.concat([wq3,wq2], axis=0)

        i=i+1

    if len(wq3)>0:

        wq3.drop_duplicates(inplace=True)

        ##recover the frequency of all forms of words in docs

        wqf = pd.DataFrame(vs.loc[vs['word'].isin(wq3.syns.values),['ct','word']])

        wq3 = pd.merge(wqf,wq3,how='left',left_on='word',right_on='syns')

        wq3['docs'] = wq3['word_x'].apply(lambda x: vs.loc[vs['word']==x,'docs'].values)

        wq3['docf'] = wq3['word_x'].apply(lambda x: vs.loc[vs['word']==x,'docf'].values)

        wq4 = pd.DataFrame(wq3.groupby(by='word_y')['ct'].sum())

        wq4['docs']=""

        ##agregate and distinct per initial word the doc frequency 

        for w in wq4.index:

            wq5 = wq3.loc[wq3['word_y']==w,['docs','docf']]

            if len(wq5)>0:

                wq5.reset_index(inplace=True, drop=True)

                wq6 = pd.DataFrame(wq5.loc[0,'docs'][0].split(','))

                wq6.columns = ['doc']

                wq6 = wq6.iloc[:-1,:]

                wq6.loc[:,'doc'] = wq6.loc[:,'doc'].astype(int)

                wq6['freq'] = wq5.loc[0,'docf'][0].split(',')[:-1]

                wq6.loc[:,'freq'] = wq6.loc[:,'freq'].astype(int)

                j=1

                while j<len(wq5):

                    wq7 = pd.DataFrame(wq5.loc[j,'docs'][0].split(','))

                    wq7.columns = ['doc']

                    wq7 = wq7.iloc[:-1,:]

                    wq7['freq'] = wq5.loc[j,'docf'][0].split(',')[:-1]

                    wq7.loc[:,'freq'] = wq7.loc[:,'freq'].astype(int)

                    wq6 = pd.concat([wq6,wq7], axis=0)

                    j = j + 1

                    del wq7

                wq6.reset_index(inplace=True, drop=False)

    

    

                wq8 = pd.DataFrame(wq6.groupby(by='doc')['freq'].sum())

                wq8.reset_index(inplace=True)

                wq8['word'] = w

                if w == wq4.index[0]:

                    wq9 = wq8.copy()

                else:

                    wq9 = pd.concat([wq9,wq8], axis=0)

                del wq8

            del wq5, wq6

        gc.collect()

        

        

    #    build the final output

        adf = pd.DataFrame(np.zeros((wq9.doc.nunique(),1)))

        adfcols = ['doc']

        adf.columns = adfcols

        adf['doc'] = wq9.loc[:,'doc'].unique()

        ##arange data by doc rows and init words as cols

        for w in wq:

            adf = pd.merge(adf, wq9.loc[wq9['word']==w,['doc','freq']], on='doc', how='left')

            adf.loc[:,w]= adf.loc[:,'freq'].fillna(0)

            adf.drop('freq', axis=1, inplace=True)

    

        ##order by presence of all/rare words

        adf['penalty']=0

        

        i=len(wq)*10

    #    penalty = word rank * 10 (biggest penalty to rarest word)

        for w in wq:

            adf['penalty']=adf['penalty']-i*(adf.loc[:,w]==0).astype(int)

            i= i-10

        adf = adf.sort_values(by='penalty', ascending=False)

        adf.reset_index(inplace=True, drop=True)

        adf['doc'] = adf['doc'].astype(int)

        adf = pd.merge(adf, df[['pid','pr', 'url', 'publish_time', 'befcovid', 'title_orig', 'journal', 'text','year']], how='left', left_on='doc', right_on='pid')

        adf.drop_duplicates(subset=['doc'],inplace=True)

        adf = adf.loc[adf['text'].str.len()>0,:]

        ##hard filter for covid papers only after 2019

        if 'covid' in wq:

            adf = adf.loc[adf['befcovid']==0,:]

        adf = adf.fillna("")

        adf = adf.sort_values(by=['penalty','pr'], ascending = [False,False])

        adf.reset_index(inplace=True, drop=True)

        adf = adf.iloc[:20,:]

        wq4 = wq4.sort_values(by='ct')

        wq3 = wq3[['word_y','word_x']]

        wq4 = wq4.index.values

    else:

        adf, wq3, wq4 = [], [], []

    return adf, wq3, wq4
##recompute doc frequencies for covid word

cfd['docs'] = cfd['docs'].str.slice(0,-1)

cfd['docf'] = cfd['docf'].str.slice(0,-1)

newds = vs.loc[vs['word']=='covid','docs'].values

newdf = vs.loc[vs['word']=='covid','docf'].values

wq6 = pd.DataFrame(newds[:-1])

wq6.columns = ['doc']

wq6['freq'] = newdf[:-1]

wq6.loc[:,'freq'] = wq6.loc[:,'freq'].astype(int)

i=0

while i<len(cfd):

    wq7 = pd.DataFrame(cfd.loc[i,'docs'].split(','))

    wq7.columns = ['doc']

    wq7['freq'] = cfd.loc[i,'docf'].split(',')

    wq7.loc[:,'freq'] = wq7.loc[:,'freq'].astype(int)

    wq6 = pd.concat([wq6,wq7], axis=0)

    i = i + 1

    

wq8 = wq6.groupby(by='doc')['freq'].sum()

wq8.columns = ['freq']

newds = wq8.index.astype(int).values

newdf = wq8.astype(int).values

nds = ""

for i in newds:

    nds = nds+str(i)+","

ndf = ""

for i in newdf:

    ndf = ndf+str(i)+","

vs.loc[vs['word']=='covid','docs'] = nds

vs.loc[vs['word']=='covid','docf'] = ndf

vs.loc[vs['word']=='covid','ct'] = 1 ##make it artificially very rare to raise the importance in search
##load BERT model "deepset/covidbertbase" to try

config =  BertConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True)

model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", config = config)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

##play with synonyme function 

w = 'chloroquine'

ws = fyndsyn(w, vs, sdf) #findsynonyms(w, e, vs, ve)

print (ws)
Q = 'clinical trial chloroquine'

newadf, wq3, wqs = query(Q)

doc = newadf.loc[0,'doc']

st, lent, pen, mis, exq = snippet(df, wqs, wq3, doc, 1500, (len(wqs)-1)*10)

print (showtext(st,lent,df, doc))
text = widgets.Text(

        value='',

        placeholder='e.g. clinical trial',

        description='Look for:',

        disabled=False,

        layout=Layout(width='30%', height='50px')

    )

display(text)

def showtable(text, origdf, wq3, wqs, covw=False, ys=2010, ye=2020, kv=10):

    ##if only nov checked we filter out by the existing flag

    if covw:

        ndaf = origdf.loc[origdf['befcovid']==0,:]

    else:

        ndaf = origdf

    ##filter by year range

    ndaf = ndaf.loc[ndaf['year']>=int(ys),:]

    ndaf = ndaf.loc[ndaf['year']<=int(ye),:]

    #make the list of papers

    title_to_id = ndaf.set_index('show')['doc'].to_dict()

    #create the objects

    yearW = widgets.IntRangeSlider(min=1950, max=2020, value=[ys, ye], description='Year Range', continuous_update=False, layout=Layout(width='40%'))

    covidW = widgets.Checkbox(value=covw,description='Only after Nov 2020',disabled=False, indent=False, layout=Layout(width='20%'))

    kWidget = widgets.IntSlider(value=10, description='k', max=50, min=1, layout=Layout(width='20%'))

    bulletW = widgets.Select(options=title_to_id.keys(), layout=Layout(width='90%', height='200px'), description='Title:')

    #function when paper clicked

    def main_function(bullet, k=5, year_range=[1950, 2020], only_covid19=False):

        doc = ndaf.loc[ndaf['show']==bullet,'doc'].values[0]

        #find relevant snippet

        st, lent, pen, mis, exq = snippet(ndaf, wqs, wq3, doc, 1500, (len(wqs)-1)*10)

        #show url, page rank, missing words, snippet and query extension

        h = '<br/>'.join(['<a href="' + l + '" target="_blank">'+ n + '</a>' +' (PageRank: ' + "{:.2f}".format(s) + ')' \

                          for l, n, s in ndaf.loc[ndaf['show']==bullet,['url','show', 'pr']].values])

        h = h + ' missing words: '

        for m in mis:

            h = h + m

        if len(mis)==0:

            h = h + " None"

        h = h + '<br/>'

        h = h + '<br/>'

        h = h + showtext(st, lent, ndaf, doc)

        h = h + '<br/>'

        h = h + '<br/>'

        h = h + 'query was extended: ' + str(exq) + '<br/>'

        display(HTML(h))

    #filter by nov 2020 flag

    def on_check_change(change):

        clear_output()

        out3 = showtable(text, origdf, wq3, wqs, covw=covidW.value)

        display(text)

        display(out3)

    #filter by new year range

    def on_year_change(change):

        clear_output()

        out3 = showtable(text, origdf, wq3, wqs, covw=covidW.value, ys = change['new'][0], ye = change['new'][1])

        display(text)

        display(out3)

    yearW.observe(on_year_change, names='value')

    covidW.observe(on_check_change, names='value')

    widget = widgets.interactive(main_function, bullet=bulletW, k=kWidget, year_range=yearW, only_covid19=covidW)

    controls = VBox([Box(children=[widget.children[:-1][1], widget.children[:-1][2], widget.children[:-1][3]], layout=Layout(justify_content='space-around')), widget.children[:-1][0]])

    output = widget.children[-1]

    output2 = VBox([controls, output])

    return output2



def showanswer(text):

    clear_output()

    Q = text.value    

    ##if nothing inserted

    if len(Q)==0:

        out3 = "No search words"

        display(text)

    else:

        newadf, wq3, wqs = query(Q)

        if len(newadf)>0:

            newadf['show']= newadf['title_orig']+", "+newadf['publish_time']+", "+newadf['journal']

        display(text)

        if len(newadf)>0:

            out3 = showtable(text, newadf, wq3, wqs)

        else:

            out3 = "No papers found"

        

    display(out3)

    



    

newtext = text.on_submit(showanswer)

newadf['show']= newadf['title_orig']+", "+newadf['publish_time']+", "+newadf['journal']

out3 = showtable(text, newadf, wq3, wqs)

display(out3)