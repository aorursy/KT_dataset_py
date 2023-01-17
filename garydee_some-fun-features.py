import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import nltk

from difflib import SequenceMatcher

import regex as re

from time import time

import Levenshtein

import nltk

from sklearn import preprocessing

%matplotlib inline
# Data Import

t0 = time()

alldata = pd.read_csv('../input/questions.csv')

alldata.set_value(169290, 'question1', '111?')

alldata.set_value(105796,'question2','')

alldata.set_value(201871,'question2','')

data = alldata.head(20000).copy()

print(time() - t0)
def normalized_word_share(row):

    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))

    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    

    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))



def str_jaccard(row):

    s1 = set(row['question1'].split(" "))

    s2 = set(row['question2'].split(" "))

    return float(len(s1 & s2)) / len(s1 | s2)



def str_levenshtein(row):

    s1 = row['question1']

    s2 = row['question2']

    return Levenshtein.ratio(s1, s2)



t0 = time()

min_max_scaler = preprocessing.MinMaxScaler()

#Remove the punctuation with regular expressions.

#data['question1'] = data['question1'].apply(lambda x: re.sub("\p{P}+","", x))

#data['question2'] = data['question2'].apply(lambda x: re.sub("\p{P}+", "", x))



data['question1len'] = data['question1'].apply(lambda x: len(x.split()))

data['question2len'] = data['question2'].apply(lambda x: len(x.split()))

data['SentSimilarity'] = data.apply(lambda row: SequenceMatcher(None,row['question1'], row['question2']).ratio(), axis=1)



data['StringDiff'] = data.apply(lambda row: ' '.join(list(set(row['question1'].split()) - set(row['question2'].split()))), axis=1)

data['SenLenDiff'] = data.apply(lambda row: abs(row['question1len']-row['question2len']), axis=1)

data['CharLenDiff'] = data.apply(lambda row: abs(len(row['question1'])-len(row['question2'])), axis=1)





data['WordShare'] = data.apply(normalized_word_share, axis=1)

data['WordShare'] = min_max_scaler.fit_transform(data['WordShare'].values.reshape(-1,1))

data['Jacard'] = data.apply(str_jaccard, axis=1)

data['Levenshtein'] = data.apply(str_levenshtein, axis=1)



data[['Levenshtein','WordShare','Jacard','SentSimilarity','question1','question2']].head()
from gensim.models import doc2vec

from collections import namedtuple



def wordlist(row):

    doc = []

    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')

    s1 = row['question1'].lower().split(" ")

    s2 = row['question2'].lower().split(" ")

    doc.append(analyzedDocument(s1,[1]))

    doc.append(analyzedDocument(s2,[2]))

    model = doc2vec.Doc2Vec(doc, size = 100, window = 300, min_count = 1, workers = 4)

    return model.docvecs[0],model.docvecs[1]



data = data.merge(

                   data.apply(lambda row: pd.Series(wordlist(row)),axis=1)

                  ,left_index=True

                  ,right_index=True

                 )



cols = data.columns.values

cols[-2] = 'vecq1'

cols[-1] = 'vecq2'

data.columns = cols
data['EucDist'] = data.apply(lambda x: np.linalg.norm(x['vecq1']-x['vecq2']),axis=1)

min_max_scaler = preprocessing.MinMaxScaler()

data['EucDistScaled'] = min_max_scaler.fit_transform(data['EucDist'].values.reshape(-1,1))
#I cannot take credit for this fine piece of work... Stack Overflow to the rescue!

#https://stackoverflow.com/questions/4576077/python-split-text-on-sentences

import re



caps = "([A-Z])".encode('utf-8')

prefixes = "(Mr|St|Mrs|Ms|Dr)[.]".encode('utf-8')

suffixes = "(Inc|Ltd|Jr|Sr|Co)".encode('utf-8')

starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)".encode('utf-8')

acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)".encode('utf-8')

websites = "[.](com|net|org|io|gov)".encode('utf-8')

digits = "([0-9])".encode('utf-8')



def split_into_sentences(text):

    if text[-1] != "?".encode('utf-8'): text = text+"?".encode('utf-8')

    text = " ".encode('utf-8') + text + "  ".encode('utf-8')

    text = text.replace("\n".encode('utf-8')," ".encode('utf-8'))

    text = re.sub(prefixes,"\\1<prd>".encode('utf-8'),text)

    text = re.sub(websites,"<prd>\\1".encode('utf-8'),text)

    if "Ph.D".encode('utf-8') in text: text = text.replace("Ph.D.".encode('utf-8'),"Ph<prd>D<prd>".encode('utf-8'))

    if "No.".encode('utf-8') in text: text = text.replace("No.".encode('utf-8'),"No<prd>".encode('utf-8'))

    if "i.e.".encode('utf-8') in text: text = text.replace("i.e.".encode('utf-8'),"i<prd>e<prd>".encode('utf-8'))

    if "i.e".encode('utf-8') in text: text = text.replace("i.e".encode('utf-8'),"i<prd>e".encode('utf-8'))

    if "e.g.".encode('utf-8') in text: text = text.replace("e.g.".encode('utf-8'),"e<prd>g<prd>".encode('utf-8'))

    if "cf.".encode('utf-8') in text: text = text.replace("cf.".encode('utf-8'),"cf<prd>".encode('utf-8'))

    if "et al.".encode('utf-8') in text: text = text.replace("et al.".encode('utf-8'),"et al<prd>".encode('utf-8'))

    text = re.sub("\s".encode('utf-8') + caps + "[.] ".encode('utf-8')," \\1<prd> ".encode('utf-8'),text)

    text = re.sub(acronyms+" ".encode('utf-8')+starters,"\\1<stop> \\2".encode('utf-8'),text)

    text = re.sub(caps + "[.]".encode('utf-8') + caps + "[.]".encode('utf-8') + caps + "[.]".encode('utf-8'),"\\1<prd>\\2<prd>\\3<prd>".encode('utf-8'),text)

    text = re.sub(caps + "[.]".encode('utf-8') + caps + "[.]".encode('utf-8'),"\\1<prd>\\2<prd>".encode('utf-8'),text)

    text = re.sub(" ".encode('utf-8')+suffixes+"[.] ".encode('utf-8')+starters," \\1<stop> \\2".encode('utf-8'),text)

    text = re.sub(" ".encode('utf-8')+suffixes+"[.]".encode('utf-8')," \\1<prd>".encode('utf-8'),text)

    text = re.sub(" ".encode('utf-8') + caps + "[.]".encode('utf-8')," \\1<prd>".encode('utf-8'),text)

    text = re.sub(digits + "[.]".encode('utf-8') + digits,"\\1<prd>\\2".encode('utf-8'),text)

    text = re.sub("[,.]\s.".encode('utf-8') + digits,"\\1<prd>".encode('utf-8'),text)

    text = re.sub("[,.]".encode('utf-8') + digits,"\\1<prd>".encode('utf-8'),text)

    if "”".encode('utf-8') in text: text = text.replace(".”".encode('utf-8'),"”.".encode('utf-8'))

    if "\"".encode('utf-8') in text: text = text.replace(".\"".encode('utf-8'),"\".".encode('utf-8'))

    if "!".encode('utf-8') in text: text = text.replace("!\"".encode('utf-8'),"\"!".encode('utf-8'))

    if "?".encode('utf-8') in text: text = text.replace("?\"".encode('utf-8'),"\"?".encode('utf-8'))

    if "...".encode('utf-8') in text: text = text.replace("...".encode('utf-8'),".".encode('utf-8'))

    if "..".encode('utf-8') in text: text = text.replace("..".encode('utf-8'),".".encode('utf-8'))

    text = text.replace(".".encode('utf-8'),".<stop>".encode('utf-8')) 

    text = text.replace("?".encode('utf-8'),"?<stop>".encode('utf-8'))

    text = text.replace("!".encode('utf-8'),"!<stop>".encode('utf-8'))

    text = text.replace("<prd>".encode('utf-8'),".".encode('utf-8'))

    sentences = text.split("<stop>".encode('utf-8'))

    sentences = sentences[:-1]

    sentences = [s.strip() for s in sentences]

    return sentences



data['question1Sent'] = data.apply(lambda x: split_into_sentences(x['question1'].encode('utf-8')),axis=1)

data['q1Sentences'] = data.apply(lambda x: len(x['question1Sent']),axis=1)

data['question2Sent'] = data.apply(lambda x: split_into_sentences(x['question2'].encode('utf-8')),axis=1)

data['q2Sentences'] = data.apply(lambda x: len(x['question2Sent']),axis=1)



data['SenDiff'] = abs(data['q1Sentences'] - data['q2Sentences'])

data['SenDiff'] = min_max_scaler.fit_transform(data['SenDiff'].values.reshape(-1,1))
from bokeh.plotting import figure, output_file, show

from bokeh.io import  output_notebook

from bokeh.models import HoverTool, BoxSelectTool,BoxZoomTool, ResetTool



def probbin(df,var,targ,b):

    bins = np.linspace(df[var].min(), df[var].max(), b)

    groups = df[[var,targ]].groupby(np.digitize(df[var], bins))

    return groups.mean()



b = 10

variables = [

             'WordShare'

            ,'Jacard'

            ,'Levenshtein'

            ,'SentSimilarity'

            ,'EucDistScaled'

            ,'SenDiff'

            ]

target = 'is_duplicate'

linecl = ['blue','pink','orange','green','black','red']



tools = [ HoverTool(),BoxZoomTool(), ResetTool()] 

output_notebook()

p = figure(plot_width=800, plot_height=400,tools = tools)

for idx,v in enumerate(variables):

    var1=probbin(data,v,target,b)

    p.circle(var1[v],var1[target],line_width=2,line_color=linecl[idx])

    p.line(var1[v],var1[target],line_width=2,line_color=linecl[idx],legend = v)

p.legend.location = "top_left"

p.xaxis.axis_label = "Variable"

p.yaxis.axis_label = "Binned Probability (" + str(b) + " bins)"

show(p)
d = figure(plot_width=800, plot_height=400)

d.circle(data['q1Sentences'],data['q2Sentences'],line_width=2)

d.xaxis.axis_label = "question 1 number of sentences"

d.yaxis.axis_label = "question 2 number of sentences"

show(d)
#Many sentences in this question... although the counts seem to be off by 1.

print(data[['question2']][data['q2Sentences']==13].values)
#Parts of speech tags from nltk - nltk.help.upenn_tagset()

#Parts of speech determined using the Penn Treebank tag set.

tags = [ 

         "CC","CD","DT","EX","FW","IN","JJ","JJR","JJS"

        ,"LS","MD","NN","NNP","NNPS","NNS","PDT","POS"

        ,"PRP","PRP$","RB","RBR","RBS","RP","SYM","TO"

        ,"UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT"

        ,"WP","WP$","WRB"

        ]



#Take the difference in words between the two strings

data['StringDiffStruct'] = data.apply(lambda row: ' '.join(list(set(row['question1'].split()) - set(row['question2'].split()))), axis=1)



#Apply the tagging functions from nltk

data['StringDiffStruct'] = data.apply(lambda row: nltk.pos_tag(nltk.word_tokenize(row['StringDiffStruct'])),axis=1)



#Make the list of tuples into a list

data['StringDiffStruct'] = data['StringDiffStruct'].apply(lambda l: ' '.join([item for sublist in l for item in sublist]))



#Make some features.

for t in tags:

    data[t] = data['StringDiffStruct'].apply(lambda x: len(re.findall('\\b'+t+'\\b', x)))
from sklearn.ensemble import RandomForestClassifier

from bokeh.charts import Bar

y = data['is_duplicate']

x = data[tags]



clf = RandomForestClassifier()

clf.fit(x,y)

imps = clf.feature_importances_

plot = {

        'partsofspeech': tags

       ,'importance': imps

       }

b = Bar(plot,values='importance',label='partsofspeech',legend=False)

show(b)