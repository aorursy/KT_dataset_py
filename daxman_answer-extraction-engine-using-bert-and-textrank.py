#!pip install summa

!pip install cdqa

#!pip install vaderSentiment
!pip install summa
import numpy as np

import pandas as pd

import os

import json

import glob

import sys

import urllib



import pickle

import gc

import json

import re 

import random



import os.path

from os import path



# load TextRank summarizer

from summa.summarizer import summarize

from summa.keywords import keywords



# necessary for cdQA

from ast import literal_eval



from cdqa.utils.filters import filter_paragraphs

from cdqa.utils.download import download_model, download_bnpp_data

from cdqa.pipeline.cdqa_sklearn import QAPipeline

from cdqa.utils.converters import generate_squad_examples



#vaderSentiment

#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, "../")



global parse_all



root_path = '/kaggle/input/CORD-19-research-challenge'

corpus_frac = 0.1 #fraction of corpus to use

use_distilled = False # use BERT or DistilBERT 

makeNew = False



if (makeNew):

    load_prior = False #

    load_all = True

    just_meta = True

    save_load = True

    parse_all = False

    exit_after_load = True

else:

    load_prior = True #

    load_all = False

    just_meta = False

    save_load = False

    parse_all = False

    exit_after_load = False
# Get all the files saved into a list and then iterate over them like below to extract relevant information

# hold this information in a dataframe and then move forward from there. 
global url_link

global ans_df

global current_task

url_link={}



# Just set up a quick blank dataframe to hold all these medical papers. 

corona_features = {"doc_id": [], "source": [], "title": [],

                  "abstract": [], "text_body": []}

corona_df = pd.DataFrame.from_dict(corona_features)



#links indexed by title

linkdict={}



#our output frame

ans_features = {"query": [], "answers": [], "keys": [],"summary": [],"context":[],"task":[]}

ans_df = pd.DataFrame.from_dict(ans_features)



gc.collect()
# Cool so dataframe now set up, lets grab all the json file names. 



# For this we can use the very handy glob library



json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)

json_filenames = random.sample(json_filenames,int( len(json_filenames)*corpus_frac))



#see https://gist.github.com/sebleier/554280  for stoplists

stoplist1=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

stoplist2=["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"]

stopSet1 = set(stoplist1)

stopSet2 = set(stoplist2)

#support functions

def clean_dataset(text):

    if (not (isinstance(text, str)) ): return text

    text=re.sub("[\[].*?[\]]", "", text)#remove in-text citation

    text=re.sub(r'^https?:\/\/.*[\r\n]*', '',text, flags=re.MULTILINE)#remove hyperlink

    text=re.sub(r'^a1111111111 a1111111111 a1111111111 a1111111111 a1111111111.*[\r\n]*',' ',text)#have no idea what is a11111.. is, but I remove it now

    text=re.sub(' +', ' ',text ) #remove extra space, but was not able to remove all, see examples in the following cells

    text=re.sub(r's/ ( *)/\1/g','',text)

    

    return text





# Now we just iterate over the files and populate the data frame. 

def return_corona_df(json_filenames, df, source,linkdict):

    lim=100000

    cnt=0

    global url_link

    global parse_all

    

    for file_name in json_filenames:

        cnt+=1

        if (cnt>lim):break

        if ((cnt % 1000) ==0):

            print ("Load JSON {}".format(cnt))

        row = {"doc_id": None, "source": None, "title": None,"authors": None,

              "abstract": None, "text_body": None, "paragraphs":[],"bibliography": None}



        with open(file_name) as json_data:

            data = json.load(json_data)



            row['doc_id'] = data['paper_id']

            row['title'] = data['metadata']['title']

            

            lowTitle = row['title'].lower()

            linkdict[lowTitle]="0000"

            

            authors = ", ".join([author['first'] + " " + author['last'] \

                                 for author in data['metadata']['authors'] if data['metadata']['authors']])

            row['authors'] = authors

            bibliography = "\n ".join([bib['title'] + "," + bib['venue'] + "," + str(bib['year']) \

                                      for bib in data['bib_entries'].values()])

            row['bibliography'] = bibliography

            

            #find any DOI enties

            for bib in data['bib_entries'].values():

                bib_title_low=bib['title'].lower()

               # bib_data[lowTitle] = bib

                if ('other_ids' in bib):

                    ids = bib['other_ids']

                    if('DOI' in ids):

                        dois = ids['DOI']

                        for doi in dois:

                            linkdict[bib_title_low]=doi

                            #print ("{} -> {}".format(lowTitle,doi))

                    

            # Now need all of abstract. Put it all in 

            # a list then use str.join() to split it

            # into paragraphs. 



            abstract_list = [data['abstract'][x]['text'] for x in range(len(data['abstract']) - 1)]

            abstract = "\n ".join(abstract_list)



            row['abstract'] = abstract



            # And lastly the body of the text. For some reason I am getting an index error

            # In one of the Json files, so rather than have it wrapped in a lovely list

            # comprehension I've had to use a for loop like a neanderthal. 

            

            # Needless to say this bug will be revisited and conquered. 

            row['paragraphs']=abstract_list

            

            body_list = []

            for _ in range(len(data['body_text'])):

                try:

                    body_list.append(data['body_text'][_]['text'])

                    row['paragraphs'].append(data['body_text'][_]['text'])

                except:

                    pass



            body = "\n ".join(body_list)

            

            row['text_body'] = body

            

            # Augment the paragraphs with Textrank summaries

            extra_list=[]

            summary_threshold=2048

            #if (len(body)>summary_threshold):

            #    extra_list.append("TR1: " + summarize(body, ratio=0.1))

            #    extra_list.append("TR2: " + summarize(body, ratio=0.3))

            if (len(abstract)>summary_threshold):                

                extra_list.append("TR3: " + summarize(abstract, ratio=0.3))

            for subtext in row['paragraphs']:

                if (len(subtext)>summary_threshold):

                    extra_list.append("TR4: " + summarize(subtext, ratio=0.3))

            for subtext in extra_list:

                row['paragraphs'].append(subtext)

                

       

            #define links

            searchTitle = row['title']

            searchTitle = re.sub(r'\W+',' ', searchTitle)

            if (len(searchTitle)>160):

                p =searchTitle.find(' ',128)

                if (p>0):

                    searchTitle = searchTitle[0:p]

            qdict={'q': "!ducky filetype:pdf "+searchTitle}

            if (len(body_list)==0):

                #not body text -> assume no free pdf on web

                qdict={'q': "!ducky "+searchTitle}

            url_link[lowTitle]="https://duckduckgo.com/?"+urllib.parse.urlencode(qdict)



            # Now just add to the dataframe. 

            

            if source == 'b':

                row['source'] = "biorxiv_medrxiv"

            elif source == "c":

                row['source'] = "common_use_sub"

            elif source == "n":

                row['source'] = "non_common_use"

            elif source == "p":

                row['source'] = "pmc_custom_license"

                

            if (not(parse_all)):

                del row['source']

                del row['authors']

                del row['abstract']

                del row['text_body']

                del row['bibliography']



            df = df.append(row, ignore_index=True)

            

    return df

    

def return_append_metadata_df(df,linkdict):

    global url_link

    global parse_all

    # load the meta data from the CSV file using 3 columns (abstract, title, authors),

    meta_df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','abstract','authors','doi','full_text_file'])

    #drop duplicates

    meta_df=meta_df.drop_duplicates()

    #drop NANs 

    meta_df=meta_df.dropna()

    # convert abstracts to lowercase

    #df["abstract"] = df["abstract"].str.lower()   

    lim=100000

    cnt=0

    for index, row in meta_df.iterrows():

        cnt+=1

        if (cnt>lim):break

        if ((cnt % 1000) ==0):

            print ("Load Metadata {}".format(cnt))

        

        new_row = {"doc_id": None, "source": None, "title": None,"authors": None,

              "abstract": None, "text_body": None, "paragraphs":[],"bibliography": None}

        new_row['paragraphs'].append( row['abstract'])

        new_row['title'] = row['title']

        new_row['authors']=row['authors']

        new_row['abstract']=row['abstract']

        new_row['text_body']=row['abstract']

        new_row['doc_id']=row['doi']

        new_row['source']=row['full_text_file']

        

        if (not(parse_all)):

                del new_row['source']

                del new_row['authors']

                del new_row['abstract']

                del new_row['text_body']

                del new_row['bibliography'] 

                

        linkdict[row['title'].lower()]=row['doi']

        url_link[row['title'].lower()]='https://doi.org/'+row['doi']

        df = df.append(new_row,ignore_index=True)



    return df



def filterDF(source_df,regex):

    # save the processing and space

    #if (regex == "(.*)"):

    #    return source_df

    start_mem = source_df.memory_usage().sum() / 1024**2

    print('Memory usage of source_df is {:.2f} MB'.format(start_mem))

    

    filtered_features = { "title":[],"paragraphs":[]}

    filtered_df = pd.DataFrame.from_dict(filtered_features)

    memcount =0

    mb = 1024*1024

    memlimit = 128 * mb

    for index, row in source_df.iterrows():

        keep=False

        plist=[]

        for p in row['paragraphs']:

            

            valid = re.search(regex,p)

            keep = keep or valid

            if (valid):

                plen = len(p)

                if (memcount+plen < memlimit):

                    plist.append(p)

                    memcount += plen

                    

        if (keep):

            filtered_row = {}

            filtered_row['title']=row['title']

            filtered_row['paragraphs']=plist #row['paragraphs']

            filtered_df = filtered_df.append(filtered_row,ignore_index=True )

    

    print("memcount = {}".format(memcount))

    end_mem = filtered_df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return filtered_df

    

def predict2(

        cdqa_pipeline,

        queryPre: str = None,

        query: str = None,

        n_predictions: int = None,

        retriever_score_weight: float = 0.35,

        return_all_preds: bool = False,

    ):



    best_idx_scores = cdqa_pipeline.retriever.predict(queryPre)



    squad_examples = generate_squad_examples(

        question=query,

        best_idx_scores=best_idx_scores,

        metadata=cdqa_pipeline.metadata,

        retrieve_by_doc=cdqa_pipeline.retrieve_by_doc,

        )

    examples, features = cdqa_pipeline.processor_predict.fit_transform(X=squad_examples)

    prediction = cdqa_pipeline.reader.predict(

        X=(examples, features),

        n_predictions=n_predictions,

        retriever_score_weight=retriever_score_weight,

        return_all_preds=return_all_preds,

        )

    return prediction

from summa import summarizer,keywords

from summa.summarizer import summarize



import networkx as nx

%matplotlib inline

import matplotlib.pyplot as plt

import re

import textwrap 

def word_wrap(value,n):

   # wrapper = textwrap.TextWrapper(width=n) 

    wrapped = textwrap.wrap(text=value,width=n) 

    return wrapped



def showNetwork(text):

    nx_graph = nx.DiGraph()

    w_nodes=set()



    #print(keywords.keywords(text,scores=True))

    keyScore={}

    sentScore={}

    linked=set()

    keyPhraseSet = set()

    #labels={}

    toks = text.split(' ')

    tlen = int( len(toks) * 0.5 )

    

    # get a set of keywords

    ks = keywords.keywords(text,scores=True,ratio=0.8)

    for (key,score) in ks:

        keyScore[key]=score

        w_nodes.add(key)

        #nx_graph.add_node(key, name=key)

        #labels[key]=key



    #nx_graph.add_node("root", name="root")

    #nx_graph.add_node("term", name="term")



    # Get a sentence summary

    ss = summarize(text,words=128,scores=True,split=True)

    

    #count how many times a keyword appears in each summary sentence

    keyCount={}

    for (key,score) in ks:

        keyCount[key]=0

        # ignore stopwords

        if (key in stopSet1):

            continue

        #count appearences in each senntence

        for (skey,score) in ss:

            search_key = key.lower().replace(" ",".*")

            if (re.search(search_key,skey.lower()) ):

                keyCount[key]+=1

    

    # find the set of words            

    tokSet =set()            

    for (skey,sscore) in ss:

        tkey=re.sub('[\W+]+',' ',skey)

        toks = tkey.split(' ')

        for t in toks:

            # ignore stopwords

            if (t in stopSet1):

                continue

            #generate new entries

            if (len(t)>3) and (not(t in keyCount)):

                keyCount[t]=0

                tokSet.add(t)

                

    #count how many times a raw toke appears in the summary sentences

    for key in tokSet:

        for (skey,score) in ss:

            search_key = key.lower().replace(" ",".*")

            if (re.search(search_key,skey.lower()) ):

                keyCount[key]+=1

        

    #build a graph of the sentences and the keywords/tokens

    # that occur in two or more sentences

    for (skey,sscore) in ss:

        sentScore[skey]=sscore

        w_nodes.add(skey)

        nx_graph.add_node(skey, name=skey)

        #nx_graph.add_edge(skey, "root", name="") 

        #labels[skey]=skey

        

        #keyphrase to sentences links

        for (key,kscore) in ks:

            if (keyCount[key]>1) and (len(key)>2):

                keyPhraseSet.add(key)

                search_key = key.lower().replace(" ",".*")

                if (re.search(search_key,skey.lower()) ) :

                    nx_graph.add_node(key, 

                                      name="{0}".format(key,kscore)

                                     , node_color='#00b4d9')

                    nx_graph.add_edge(key, skey, name="",weight=1)

                    linked.add(key)

                    

        #tokens to sentence links

        for key in tokSet:

            if (keyCount[key]>1) and (len(key)>2):

                search_key = key.lower().replace(" ",".*")

                if (re.search(search_key,skey.lower()) ) :

                    nx_graph.add_node(key, name="{0}".format(key,kscore))

                    nx_graph.add_edge(key, skey, name="",weight=0.1)

                    linked.add(key)

    

    #tokens to keyphrases

    for (key,kscore) in ks:

        for tkey in tokSet:

            if ( (tkey.lower() in key.lower()) 

                 and (tkey in linked) and (key in linked) ):

                    nx_graph.add_edge(tkey, key, name="",weight=0.9)



    #print(summarize(text, words=50))

    kgp = keywords.get_graph(text)

    sgp = summarizer.get_graph(text)



    kn_score ={}

    for (s_name,o_name) in kgp.edges():

        kn_score[s_name]=0



    for (s_name,o_name) in kgp.edges():

        for (key,score) in ks:

            if (score> kn_score[s_name]) and (s_name in key): 

                kn_score[s_name]=score

   #add colors

    ncolors = []

    for node in nx_graph:

        if node in keyPhraseSet:

            ncolors.append("red")

        else:

            if node in tokSet:

                ncolors.append("lightgreen")

            else:

                ncolors.append("lightblue")

    ecolors = []

    for u,v,d in nx_graph.edges(data=True):   

        if u in keyPhraseSet:

            ecolors.append("red")

        else:

            if u in tokSet:

                ecolors.append("lightgreen")

            else:

                ecolors.append("lightblue")

   #plot the graph

    fig = plt.figure(figsize=(20, 20))

    #fig = plt.figure(figsize=(20, 20))

    #ax = fig.add_subplot(111)

   # select a style

    #_pos = nx.kamada_kawai_layout(nx_graph)

    _pos = nx.spring_layout(nx_graph

                            ,k = 0.6

                            ,iterations = 100

                            ,threshold = 0.005)



   #draw the raw nodes and edges

    _ = nx.draw_networkx_nodes(nx_graph, pos=_pos,

                               node_color=ncolors,alpha=0.5)

    _ = nx.draw_networkx_edges(nx_graph, pos=_pos,

                               edge_color=ecolors,alpha=0.6)

    

    #generate new word wrapped labels for the nodes

    lablesN={}

    names = nx.get_node_attributes(nx_graph, 'name')

    for n in names:

        p = _pos[n]

        t=names[n]

        lablesN[n]="\n".join(word_wrap(t,32))

        _pos[n] = p

    y=0

    for (k,p) in lablesN.items():

        if (not(k in _pos)):

            y+=0.1

            _pos[k]=(0,y)

            #print ("lab:{} -> {}".format(k,_pos[k]))

            

    #draw the lables of the nodes and edges

   # _ = nx.draw_networkx_labels(nx_graph, 

   #                             pos=_pos, 

   #                             fontsize=9,

   #                             font_color ="white",

   #                             labels=lablesN)

    

    #draw node labels one at a time for individual color

    for node in nx_graph:

        if node in keyPhraseSet:

            nx.draw_networkx_labels(nx_graph, 

                                pos=_pos, 

                                font_size=12,

                                font_color ="cyan",

                                font_weight="black",

                                labels={node:lablesN[node]})

        else:

            if node in tokSet:

                nx.draw_networkx_labels(nx_graph, 

                    pos=_pos, 

                    font_size=9,

                    font_color ="lightgreen",

                    alpha = 0.8,

                    labels={node:lablesN[node]})

            else:

                nx.draw_networkx_labels(nx_graph, 

                    pos=_pos, 

                    font_size=12,

                    font_color ="white",

                    labels={node:lablesN[node]})

    

    #_ = nx.draw_networkx_labels(nx_graph, pos=_pos, fontsize=8)



    names = nx.get_edge_attributes(nx_graph, 'name')



    _ = nx.draw_networkx_edge_labels(nx_graph,

                                     pos=_pos, 

                                     edge_labels=names, 

                                     font_color ="cyan",

                                     fontsize=8)

    #fig.set_facecolor("#00000F")

    plt.gca().set_facecolor("#00000F")

    # SHOW THE PLOT

    plt.show()
text = """



 Multivariable regression analysis showed that elevated high sensitivity troponin I (OR 2.68, 95%CI 1.31-5.49, P=0.007), neutrophils (OR 1.14, 95%CI 1.01-1.28, P=0.033) and depressed oxygen saturation (OR 0.94, 95%CI 0.89-0.99, P=0.027) on admission were associated with rapid death of patients with COVID-19.

 Elevated high sensitivity troponin, neutrophils and depressed oxygen saturation predicted the rapid death of patients. 

 Compared with patients without pneumonia, those with pneumonia were 15 years older and had a higher rate of hypertension, higher frequencies of having a fever and cough, and higher levels of interleukin-6 (14.61 vs. 8.06pg/mL, P=0.040), B lymphocyte proportion (13.0% vs.10.0%, P=0.024), low account (<190/μL) of CD8+ T cells (33.3% vs. 0, P=0.019). 

 For example, a large observational report 2 including 1099 patients with confirmed COVID-19 infection indicated that in 173 with severe disease there existed the comorbidities of hypertension (23·7%), diabetes mellitus (16·2%), coronary heart diseases (5·8%), and cerebrovascular disease (2·3%). 

  Even though COVID-19 is highly contagious , control measures have proven to be very effective. 

 Meanwhile, numbers of patients with COVID-19 infection had chronic comorbidities , mainly hypertension, diabetes and cardiovascular disease, which is similar to MERS-COV population. 

 

 

 """

showNetwork(text)
if (just_meta):

    corona_df = return_append_metadata_df( corona_df,linkdict)

    if (load_all):

        corona_df = return_corona_df(json_filenames, corona_df, 'b',linkdict)

else:

    if (load_prior):

        #load prior CSV 

        #corona_df= pd.read_csv('/kaggle/working/kaggle_covid-19_open_csv_format.csv')

        #load prior pkl

        if path.exists('/kaggle/input/covid-19-corpus-pickle-factory/kaggle_covid-19_pickle.pkl'):

            print("Loading pickled KB from /kaggle/input/covid-19-corpus-pickle-factory")

            corona_df= pd.read_pickle('/kaggle/input/covid-19-corpus-pickle-factory/kaggle_covid-19_pickle.pkl')

            #restore the url_link info dictionary

            with open('/kaggle/input/covid-19-corpus-pickle-factory/url_links.pkl','rb') as handle:

                url_link = pickle.load(handle)

        else:

            print("Loading pickled KB from /kaggle/working")

            corona_df= pd.read_pickle('/kaggle/working/kaggle_covid-19_pickle.pkl')

            #restore the url_link info dictionary

            with open('/kaggle/working/url_links.pkl','rb') as handle:

                url_link = pickle.load(handle)

    else:

        #generate

        corona_df = return_corona_df(json_filenames, corona_df, 'b',linkdict)



if (save_load): 

        # save

        corona_out = corona_df.to_csv('kaggle_covid-19_open_csv_format.csv')

        corona_pkl = corona_df.to_pickle('kaggle_covid-19_pickle.pkl')

         # Store (serialize) the url_link dictionary

        with open('url_links.pkl', 'wb') as handle:

            pickle.dump(url_link, handle, protocol=pickle.HIGHEST_PROTOCOL) 

            

#if (exit_after_load):

#    exit(keep_kernel=True)

    #raise SystemExit("Exit after load")
corona_df.shape

len(url_link)
#url_link

!pip install cdqa
if (use_distilled):

    # Downloading pre-trained DistilBERT fine-tuned on SQuAD 1.1

    download_model('distilbert-squad_1.1', dir='./models')

else:

    #download the model

    download_model(model='bert-squad_1.1', dir='./models')

gc.collect()
#subsample full corpus to fit in memory

#corona_df=corona_df.sample(frac = corpus_frac)

corona_df.head(5)

len(corona_df)
#find duplicate values. (remove also files that do not have a title)



try:

    dfdrop= corona_df[corona_df['title'].duplicated() == True]

    #dfdrop.head()

    corona_df= corona_df.drop(dfdrop.index)

    dfdrop2 = corona_df[corona_df.astype(str)['paragraphs'] == '[]']

    corona_df= corona_df.drop(dfdrop2.index)

except:

    gc.collect()



    

#corona_df['text_body'] =corona_df['text_body'].apply(clean_dataset)

#corona_df['title']     =corona_df['title'].apply(clean_dataset)

#corona_df['abstract']  =corona_df['abstract'].apply(clean_dataset)

#corona_df=corona_df[['doc_id','title','abstract','text_body','paragraphs']]

corona_df=corona_df[['doc_id','title','paragraphs']]



corona_df.head()
corona_df.mask(corona_df.eq('None')).dropna()

corona_df = corona_df.replace(to_replace='None', value=np.nan).dropna()

corona_df = corona_df.reset_index(drop=True)





corona_df.head(5)
#load the data into df

corona_df = filter_paragraphs(corona_df)
corona_df.head(5)

if (use_distilled):

    # Loading QAPipeline with CPU version of DistilBERT Reader pretrained on SQuAD 1.1

    cdqa_pipeline = QAPipeline(reader='models/distilbert_qa.joblib')

else:

    # Loading QAPipeline with CPU version of BERT Reader pretrained on SQuAD 1.1

    cdqa_pipeline = QAPipeline(reader='models/bert_qa.joblib')

    
cdqa_pipeline
# Fitting the retriever to the list of documents in the dataframe

sub_kb = corona_df.sample(frac=corpus_frac)

try:

    #cdqa_pipeline.fit_retriever(df=corona_df)

    cdqa_pipeline.fit_retriever(df=sub_kb)

except:

    print("Unexpected error:", sys.exc_info()[0])    
# Sending a question to the pipeline and getting prediction

query_list=[ 'What presents a emerging threat to global health?'

              ,'When did COVID-19 appear?'

              ,'Where did COVID-19 first appear?'

              ,'What was the original source of COVID-19?'

              ,'What was the original host of COVID-19?'



              ,'What is an effective treatment for COVID-19?'

              ,'What is the motality rate of COVID-19?'

              , 'How does COVID-19 respond to the presence of copper?'

              , 'How are ACE2 receptors affected?'

              , 'What patients are most susceptible to COVID-19?'

              , 'How is COVID-19 outbreak similar to the 1918 pandemic?'

              , 'How many people will die globally of coronavirus?'

              , 'How is COVID-19 similar to MERS or SARS?'

              , 'How does COVID-19 differ from MERS and SARS?'

              , 'What is known about coronavirus transmission, incubation, and environmental stability?'

              , 'What do we know about natural history, transmission, and diagnostics for COVID-19?'

              , 'What have we learned about coronavirus infection prevention and control?'

              ,'What is the range of incubation periods for COVID-19 in humans?'

              ,'What is the prevalence of asymptomatic shedding and transmission of coronavirus?'

              ,'How does Seasonality affect transmission rate of coronavirus?'

              ,'How long does the virus persist on surfaces of different materials like copper, stainless steel, plastic?'

              ,'Coronavirus shedding from infected persons?'

              ,'What is the effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings?'

              ,'What is the effectiveness of social distancing in reducing coronavirus transmission rate?'

              ,'What is the best method to detect COVID-19 in asymptomatic patients?'

              ,'What are COVID-19 risk factors?'

              ,'What populations are suceptible to coronavirus?'

              ,'What is the risk of fatality from COVID-19 among symptomatic hospitalized patients?'

              ,'What co-infections present special risk with COVID-19?'

              ,'What is the incubation period of COVID-19?'

              ,'What is the serial interval of COVID-19?'

             ,'What viral inhibitors are being examined for coronavirus and COVID-19?'

              ,'Are there any non pharmaceutical interventions for COVID-19?'

              ,'What is the compilance rate with bans on mass gatherings?'

              ,'What will the economic impant of the COVID-19 pandemic be?'

              ,'How is AI being used to monitor and evaluate real-time health care delivery?'

              ,'Would chloroquine phosphate effective be effective against coronavirus?'

              ,'How does ritonavir act as an anti-viral?'

              ,'How does chloroquine act as an anti-viral?'

              ,'What is the most effective anti-viral against coronavirus COVID-19?'

              ,'When will the COVID-19 pandemic end?'

              ,'How long will the COVID-19 pandemic last?'

              ,'What is the survival rate for COVID-19 infections?'

              ,'What is the survival rate for COVID-19 for those over 65 years of age?'

            ,'Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups'

            ,'Does smoking or pre-existing pulmonary disease increase risk of COVID-19?'

            ,'Are neonates and pregnant women at greater risk of COVID-19?'

            ,'What is the severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups?'

            ,'Does rise in pollution increase risk of COVID-19?'

            ,'Are there public health mitigation measures that could be effective for control of COVID-19?'

              ]
query_list=[ 

            'SARS-CoV-2 covid coronavirus infected cases disease | What are common COVID-19 symptoms?'



            ,'disease pandemic increasing resistance | What presents a emerging threat to global health?'

            ,'SARS-CoV-2 covid coronavirus infected cases disease | When did COVID-19 appear?'

            ,'SARS-CoV-2 covid coronavirus infected cases disease | Where did COVID-19 first appear?'

            ,'SARS-CoV-2 covid coronavirus infected cases disease | What was the original source of COVID-19?'

            ,'SARS-CoV-2 covid coronavirus infected cases disease | What was the original source of SARS-COV2?'

            ,'SARS-CoV-2 covid coronavirus infected cases disease | What was the original host of COVID-19?'

            ,'SARS-CoV-2 covid coronavirus infected cases disease | What animal did SARS-COV2 come from?'

            ,'What is the effective reproductive number of COVID-19?'

            ,'What is the basic reproductive number (r0) of COVID-19?'

            ,'What is the incubation period in days for COVID-19? '

            ,'What is the mean or average COVID-19 incubation period in days?'

            ,'What is the range of COVID-19 incubation periods in days?'

            ,'SARS-CoV-2 covid coronavirus drug therapy treatment study trials efficacy | What is an effective treatment for COVID-19?'

            ,'SARS-CoV-2 death fatality | What is the motality rate of COVID-19?'

            ,'SARS-CoV-2 death motality | What is the fatality rate of COVID-19?'

            ,'SARS-CoV-2 covid coronavirus cell bind receptor infect membrane express cause ACE-2| What is the role of ACE2 receptors in COVID-19?'

            ,'SARS-CoV-2 covid coronavirus case severe infection treatment | What patients are most susceptible to COVID-19?'

            ,'SARS-CoV-2 coronavirus epidemic | How is COVID-19 outbreak similar to the 1918 pandemic?'

            ,'SARS-CoV-2 compared suggest | How is COVID-19 similar to MERS or SARS?'

            ,'SARS-CoV-2 covid coronavirus infect infected range patients | What is the range of incubation periods for COVID-19 in humans?'

            ,'SARS-CoV-2 covid-19 viral infection excreted | What is the rate or frequency of asymptomatic shedding and transmission of coronavirus?'

            ,'SARS-CoV-2 covid-19 coronavirus seasonal season weather summer winter spring fall hot warm cold peak time| How does Seasonality affect transmission rate of coronavirus?'

            ,'SARS-CoV-2 covid-19 coronavirus survive surface food contamination infectious | How long does the coronavirus persist on surfaces of different materials like copper, stainless steel, plastic?'

            ,'SARS-CoV-2 covid-19 coronavirus survive surface food contamination infectious plastic metal steel copper paper cardboard aluminum polystyrene glass | How long does the coronavirus persist on surfaces?'

            ,'SARS-CoV-2 covid-19 coronavirus survive deactivate inactivation disinfection decontaminate nutralize surfaces sterilize| What is an effective disinfectant for coronavirus?'

            ,'SARS-CoV-2 covid-19 coronavirus drug therapy | What is the most effective anti-viral against coronavirus COVID-19?'

            ,'SARS-CoV-2 coronavirus smoke lungs respritory COPD| Does smoking or pre-existing pulmonary disease increase risk of COVID-19?'

            ,'SARS-CoV-2 coronavirus smoke | Does smoking increase risk of COVID-19?'

            ,'SARS-CoV-2 coronavirus lungs respritory COPD| Does pre-existing pulmonary disease increase risk of COVID-19?'

            ,'SARS-CoV-2 coronavirus  hypertension | How does high-blood pressure affect COVID-19?'

            ,'SARS-CoV-2 coronavirus | How does diabetes affect COVID-19?'

            ,'SARS-CoV-2 coronavirus factors | What increases the risk of COVID-19?'

            ,'SARS-CoV-2 coronavirus factors | What are common risk factors for COVID-19 patients?'

            ,'SARS-CoV-2 babies coronavirus maternal | Are neonates and pregnant women at greater risk of COVID-19?'



]	
!pip install summa
doIt = False

if (doIt):

    #The query processor and collector

    ans_dict={} 

    key_dict={}

    sum_dict={}   

    ans_entry={}

    ans_lists={}

    num_predictions=5

    showIt =True

    rel_threshold = 0.7



    for query in query_list:

      ans_dict[query]=[]

      ans_entry[query]=[]

      #predictions = cdqa_pipeline.predict(query,n_predictions=num_predictions,retriever_score_weight=0.35)

      readQuery = query

      context=""

      if ('|' in query):

        queryp = query.split('|')

        readQuery = queryp[1] # the NL query

        context = queryp[0] #extra context keywords



      predictions= predict2(cdqa_pipeline,queryPre=query,query=readQuery,

                            n_predictions=num_predictions,

                            retriever_score_weight=0.45)

      ptext=""

      if (showIt):

          print("--------------------")



      max_score =predictions[0][3]



      ans_lists[query]=[]

      for predict in predictions:

        rel_score = predict[3]/max_score

        if (rel_score < rel_threshold): continue



        if (showIt):

            print('context: {}'.format(query))

            print('query: {}'.format(readQuery))

            print('answer: {}'.format(predict[0]))

            print('score: {}'.format(predict[3]))

            print('rscore: {}\n'.format(rel_score))

            print('title: {}'.format(predict[1]))

            print('paragraph: {}\n'.format(predict[2]))

        ans_dict[query].append(predict[0])

        ptext += "{}\n".format(predict[2])



        #generate a clickable link

        searchTitle = predict[1]

        searchTitle = re.sub(r'\W+',' ', searchTitle)

        if (len(searchTitle)>160):

            p =searchTitle.find(' ',128)

            if (p>0):

                searchTitle = searchTitle[0:p]

        qdict={'q': "!ducky filetype:pdf "+searchTitle}

        if (len(predict[2])==0):

            qdict={'q': "!ducky "+searchTitle}

        link="https://duckduckgo.com/?"+urllib.parse.urlencode(qdict)

        linkkey = predict[1].lower()

        if (linkkey in url_link) and (len(url_link[linkkey])<160):

            link = url_link[linkkey]





        relevant_paragraph = predict[2]

        relevant_paragraph = relevant_paragraph.replace(predict[0],'<strong style="color: red;"><em> {} </em></strong>'.format(predict[0]))

        qual = "r={1:8.2} a={0:8.3}".format(predict[3],rel_score)

        ans=('<b>'+predict[0]+' ('+ qual+')</b> -  <a href="'+link+'" target="_blank"><i>'+predict[1]+'</i></a>')

        ans += '<br> paragraph: {}\n'.format(relevant_paragraph)

        ans_entry[query].append(ans)

        ans_lists[query].append(predict[0])





      numChar=1024

      target_ratio = numChar / len(ptext)

      if (target_ratio > 0.5):

        target_ration = 0.5



      sum = summarize(ptext, ratio=target_ratio)

      keyterms = keywords(ptext,ratio=0.1).replace("\n"," , ")

      key_dict[query]=keyterms

      sum_dict[query]=sum

      if (showIt):

        print('keywords: {}\n'.format(keyterms))

        print('summary: {}\n'.format(sum))



      print("DONE")



import functools

from IPython.core.display import display, HTML



def subSetEngine(subsetRegex, query_list,base_df,

                 rel_threshold=0.7,

                 include_paragraphs=True,

                 focus_on_sentence=True):

    global ans_df

    

    if (use_distilled):

        # Loading QAPipeline with CPU version of DistilBERT Reader pretrained on SQuAD 1.1

        cdqa_pipeline_KB = QAPipeline(reader='models/distilbert_qa.joblib')

    else:

        # Loading QAPipeline with CPU version of BERT Reader pretrained on SQuAD 1.1

        cdqa_pipeline_KB = QAPipeline(reader='models/bert_qa.joblib')



    # Fitting the retriever to the list of documents in the dataframe

    print("filtering df ...")

    sub_kb = filterDF(base_df,subsetRegex)

    kbfrac = len(sub_kb)/len(base_df)

    



    if (len(sub_kb)==0): return

    print("building KB ...")

    try:

        #cdqa_pipeline.fit_retriever(df=corona_df)

        cdqa_pipeline_KB.fit_retriever(df=sub_kb)

    except:

        print("Unexpected error:", sys.exc_info()[0])  

        return

    

    display(HTML("<H1> SubDB {0} = {1} entries ( {2:8.3}%)</H1>"

                 .format(subsetRegex,len(sub_kb),100*kbfrac)))    

    #The query processor and collector

    ans_dict={} 

    key_dict={}

    sum_dict={}   

    ans_entry={}

    ans_lists={}

    src_dict={}

    ptext_dict={}

    stext_dict={}

    num_predictions=5

    showIt =False

    



    for query in query_list:

      ans_dict[query]=[]

      ans_entry[query]=[]

      #predictions = cdqa_pipeline_KB.predict(query,n_predictions=num_predictions,retriever_score_weight=0.35)

      

      readQuery = query

      context=""

      if ('|' in query):

        queryp = query.split('|')

        readQuery = queryp[1] # the NL query

        context = queryp[0] #extra context keywords



      predictions= predict2(cdqa_pipeline_KB,queryPre=subsetRegex+" "+query,query=readQuery,

                            n_predictions=num_predictions,

                            retriever_score_weight=0.45)

      ptext=""

      stext=""

      if (showIt):

          print("--------------------")



      max_score =predictions[0][3]



      ans_lists[query]=[]

      for predict in predictions:

        rel_score = predict[3]/max_score

        if (rel_score < rel_threshold): continue



        if (showIt):

            print('context: {}'.format(query))

            print('query: {}'.format(readQuery))

            print('answer: {}'.format(predict[0]))

            print('score: {}'.format(predict[3]))

            print('rscore: {}\n'.format(rel_score))

            print('title: {}'.format(predict[1]))

            print('paragraph: {}\n'.format(predict[2]))

        ans_dict[query].append(predict[0])

        ptext += "{}\n".format(predict[2])



        #generate a clickable link

        searchTitle = predict[1]

        searchTitle = re.sub(r'\W+',' ', searchTitle)

        if (len(searchTitle)>160):

            p =searchTitle.find(' ',128)

            if (p>0):

                searchTitle = searchTitle[0:p]

        qdict={'q': "!ducky filetype:pdf "+searchTitle}

        if (len(predict[2])==0):

            qdict={'q': "!ducky "+searchTitle}

        link="https://duckduckgo.com/?"+urllib.parse.urlencode(qdict)

        linkkey = predict[1].lower()

        if (linkkey in url_link) and (len(url_link[linkkey])<160):

            link = url_link[linkkey]





        relevant_paragraph = predict[2]

        relevant_paragraph = relevant_paragraph.replace(predict[0],'<strong style="color: red;"><em> {} </em></strong>'.format(predict[0]))

        qual = "r={1:8.2} a={0:8.3}".format(predict[3],rel_score)

        ans=('<b>'+predict[0]+' ('+ qual+')</b> -  <a href="'+link+'" target="_blank"><i>'+predict[1]+'</i></a>')

        if (include_paragraphs):

            ans += '<br> paragraph: {}\n'.format(relevant_paragraph)

        if (focus_on_sentence):

            rpar = predict[2]

            sents =  re.split(r'(?<=[^A-Z].[.?!]) +(?=[A-Z])', rpar)

            for sent in sents:

                if (predict[0] in sent):

                    stext += "{}\n".format(sent)

                    sent = sent.replace(predict[0],'<strong style="color: red;"><em> {} </em></strong>'.format(predict[0]))

                    ans += '<br> sentence: {}\n'.format(sent)    

                

        ans_entry[query].append(ans)

        ans_lists[query].append(predict[0])





      numChar=512

      target_ratio = numChar / len(ptext)

      if (focus_on_sentence):

          target_ratio = numChar / len(stext)  

      if (target_ratio > 0.5):

        target_ration = 0.5



      if (focus_on_sentence):

        #print("Summarizing sentences:{}".format(stext))

        sum = summarize(stext, ratio=target_ratio)

      else:

        sum = summarize(ptext, ratio=target_ratio)

        

      keyterms = keywords.keywords(ptext,ratio=0.1).replace("\n"," , ")

      key_dict[query]=keyterms

      sum_dict[query]=sum

      if (showIt):

        print('keywords: {}\n'.format(keyterms))

        print('summary: {}\n'.format(sum))

        

      ptext_dict[query] = ptext

      stext_dict[query] = stext

        

      if (focus_on_sentence):

        src_dict[query]=stext

     #   showNetwork(stext)

      else:

        src_dict[query]=ptext

     #   showNetwork(ptext)

    

    # Append to answer frame

      row = {

                "task":None,

                "context":None,

                "query": None,

                "answers": None,

                "keys": None,

                "summary": None 

          }

      row["context"]=context

      row["task"]=current_task

      row["query"]=readQuery

      row["answers"] = " , ".join(ans_dict[query])

      row["keys"]=key_dict[query]

      row["summary"]=sum_dict[query]



      ans_df= ans_df.append(row, ignore_index=True)



      print("DONE :{}".format(query))

    



    #display(HTML('<table>'))

    #display(HTML('<tr><th></th></tr>'))

    for query in query_list:

        

        #display(HTML('<tr>'))

        #display(HTML('<td>'))

        showNetwork(ptext_dict[query])

        #display(HTML('</td>'))

        #display(HTML('</tr>'))

        

        #display(HTML('<tr>'))

        #display(HTML('<td>'))

        showNetwork(stext_dict[query])

        #display(HTML('</td>'))

        #display(HTML('</tr>'))

        

        display(HTML('<table>'))

        display(HTML('<tr>'))

        display(HTML('<td>'))

        readQuery = query

        context=""

        if ('|' in query):

            queryp = query.split('|')

            context = queryp[0] #extra context keywords

            readQuery = queryp[1] # the NL query



        #print('query: {}\n'.format(query))

        #print('    keys: {}\n'.format(key_dict[query]))

        #print('    sum: {}\n'.format(sum_dict[query]))

        display(HTML('<h1>Query: '+readQuery+'</h1>'))

        if (len(context)>0):

            display(HTML('<h2>Context: '+context+'</h2>'))



        display(HTML('<h2>Extracted Keywords: </h2>'+key_dict[query]))

        display(HTML('<h2>Answers: </h2>'))

        key_list = key_dict[query].split(" , ")

        key_list.sort(reverse=True,key=len)





        for a in ans_entry[query]:

            for k in key_list:

                if (len(k)<3):continue

                par = "paragraph:"

                if (par in a):

                    indx = a.index(par) + len(par)

                    pre = a[0:indx]

                    post = a[indx:]

                    a =  pre+ post.replace(k+' ','<em style="color: DarkBlue;">{}</em> '.format(k))



            #print('    ans: {}\n'.format(a))

            display(HTML(a))



        summ = sum_dict[query]

        for a in ans_lists[query]:

            summ =  summ.replace(a,'<strong style="color: red;"><em>{}</em></strong>'.format(a))



        for k in key_list:

            if (len(k)<3):continue

            summ =  summ.replace(k+' ','<em style="color: DarkBlue;">{}</em> '.format(k))



        #display(HTML('<h2>Keywords: </h2>'+ " , ".join(key_list) ))

        display(HTML('<h2>Summary: </h2> '+summ+'</h2>'))

        display(HTML('<hr color="red" size="8" align="center" noshade/>'))

        display(HTML('</td>'))

        display(HTML('</tr>'))

        

        display(HTML('</table>'))  

        

    #display(HTML('</table>'))    


        
doIt =False

cp="(death|fatal|mortal).*(rate)"#"(.*)"

threshold = 0.5

querySet=[]

showPara=False

showSentence=True

if (doIt):

    print("COVID-19 Corpus Answer Extraction engine.")

    print("Use 'context:<regex> to focus on a subset of entries'")

    print("Use 'query:<question> to add a question to a batch'")

    print("Use 'run' to generate a report")

    print("And of course 'help'")

    flag=True

    while(flag==True):

        user_response = input(">")

        user_input = user_response

        user_response=user_response.lower().strip()

        if (":" in user_response):

            args = user_response.split(':')

            if (user_response.startswith('context:')):

                cp = args[1]

                print("  setting context pattern to:{}".format(cp))

            if (user_response.startswith('threshold:')):

                threshold = float(args[1])

                print("  setting threshold to:{}".format(threshold))

            if (user_response.startswith('query:')):

                query=args[1]

                querySet.append(query)

                print("  adding '{}' to query set.".format(query))

            if (user_response.startswith('paragraphs:')):

                val=args[1]

                showPara = ((val=="true") or (val=="on"))

                print("  setting paragraph display :{}".format(showPara))

            if (user_response.startswith('sentence:')):

                val=args[1]

                showSentence = ((val=="true") or (val=="on"))

                print("  setting sentence display :{}".format(showSentence))

                

        else:

            if (user_response == "run"):

                print("Running query set:")

                print("  Context = '{}'".format(cp))

                print("  Threshold = {}".format(threshold))

                print("  ShowParagraphs = {}".format(showPara))

                print("  showSentence = {}".format(showSentence))

                print("  Queries:")

                for q in querySet:

                    print("      '{}'".format(q))

                subSetEngine(cp,querySet,corona_df,

                             rel_threshold=threshold,

                             include_paragraphs = showPara,

                            focus_on_sentence= showSentence)

            if (user_response == "clear"):

                querySet=[]

            if (user_response == "exit"):

                flag=False

            if (user_response == "quit"):

                flag=False

            if (user_response == "list"):

                print("Context = '{}'".format(cp))

                print("Threshold = {}".format(threshold))

                print("ShowParagraphs = {}".format(showPara))

                print("ShowParagraphs = {}".format(showSentence))

                print("Queries:")

                for q in querySet:

                    print("    '{}'".format(q))

            if (user_response =="help"):

                print (" context:<regex-pattern>")

                print (" threshold:<relative-threshold>")

                print (" paragraphs:(on|off|true|false)")

                print (" sentence:(on|off|true|false)")

                print (" query:<query>")

                print (" clear")

                print (" run")

                print (" list")

                print (" 'exit' or 'quit'")

                print()

                
doIt =False



if (doIt):

    print("============================")

    for query in query_list:

        print('query: {}\n'.format(query))

        for a in ans_dict[query]:

          print('    ans: {}\n'.format(a))

        print('    keys: {}\n'.format(key_dict[query]))

        print('    sum: {}\n'.format(sum_dict[query]))

        print() 
!pip install vaderSentiment
doIt=False

if (doIt):

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    #get a sentiment Analyzer

    sentimentAnalyzer = SentimentIntensityAnalyzer()

    # provide answers as a dataframe

    ans_features = {"query": [], "answers": [], "keys": [],"summary": [],"sentiment": []}

    ans_df = pd.DataFrame.from_dict(ans_features)

    for query in query_list:

        vs = sentimentAnalyzer.polarity_scores(sum_dict[query])

        readQuery = query

        context=""

        if ('|' in query):

            queryp = query.split('|')

            context = queryp[0] #extra context keywords

            readQuery = queryp[1] # the NL query



        row = {"context":None, "query": None, "answers": None, "keys": None,"summary": None ,"sentiment":None}

        row["context"]=context

        row["query"]=readQuery

        row["answers"] = " , ".join(ans_dict[query])

        row["keys"]=key_dict[query]

        row["summary"]=sum_dict[query]

        row["sentiment"]= vs['compound']

        ans_df= ans_df.append(row, ignore_index=True)



    ans_df.head()

    ans_out = ans_df.to_csv('covid-19_answer_output.csv')
ans_features = {

                "context": [],

                "task":[],

                "query": [], 

                "answers": [], 

                "keys": [],

                "summary": []

                }

ans_df = pd.DataFrame.from_dict(ans_features)







query_list=[ 

            'SARS-CoV-2 covid coronavirus infected cases disease | What are common COVID-19 symptoms?'



            ,'disease pandemic increasing resistance | What presents a emerging threat to global health?'

            ,'SARS-CoV-2 covid coronavirus infected cases disease | When did COVID-19 appear?'

            ,'SARS-CoV-2 covid coronavirus infected cases disease | What animal did SARS-COV2 come from?'

            ,'What is the effective reproductive number of COVID-19?'

            ,'What is the basic reproductive number (r0) of COVID-19?'

            ,'What is the incubation period in days for COVID-19? '

            ,'What is the mean or average COVID-19 incubation period in days?'

            ,'What is the range of COVID-19 incubation periods in days?'

            ,'SARS-CoV-2 covid coronavirus drug therapy treatment study trials efficacy | What is an effective treatment for COVID-19?'

            ,'SARS-CoV-2 death fatality | What is the motality rate of COVID-19?'

            ,'SARS-CoV-2 death motality | What is the fatality rate of COVID-19?'

            ,'SARS-CoV-2 covid-19 coronavirus survive deactivate inactivation disinfection decontaminate nutralize surfaces sterilize| What is an effective disinfectant for coronavirus?'

            ,'SARS-CoV-2 covid-19 coronavirus drug therapy | What is the most effective anti-viral against coronavirus COVID-19?'

            ,'SARS-CoV-2 coronavirus lungs respritory COPD| Does pre-existing pulmonary disease increase risk of COVID-19?'

            ,'SARS-CoV-2 coronavirus  hypertension | How does high-blood pressure affect COVID-19?'

            ,'SARS-CoV-2 coronavirus | How does diabetes affect COVID-19?'

            ,'SARS-CoV-2 coronavirus factors | What increases the risk of COVID-19?'

            ,'SARS-CoV-2 coronavirus factors | What are common risk factors for COVID-19 patients?'

            ,'SARS-CoV-2 babies coronavirus maternal | Are neonates and pregnant women at greater risk of COVID-19?'



]

# first try a universal context

cp="(.*)"

threshold = 0.8

querySet=[]

showPara=False

showSentence=True

print("Running query set:")

print("  Context = '{}'".format(cp))

print("  Threshold = {}".format(threshold))

print("  ShowParagraphs = {}".format(showPara))

print("  Queries:")



current_task="risk_factors"

for q in query_list:

    print("      '{}'".format(q))



subSetEngine(cp,query_list,corona_df,

        rel_threshold=threshold,

        include_paragraphs = showPara,

        focus_on_sentence= showSentence)



#next try operation with a focused subset

current_task="death_rate"

contextPatterns=[

    " (death|died|fatal|mortal).*(rate)"

]

querySet=[

  'SARS-CoV-2 death fatality | What is the motality rate of COVID-19?'

 ,'SARS-CoV-2 death motality | What is the fatality rate of COVID-19?'

]

for cp in contextPatterns:

    subSetEngine(cp,querySet,corona_df,

        rel_threshold=threshold,

        include_paragraphs = showPara,

        focus_on_sentence=  showSentence)



#something more generic

current_task="risk_factors"

tasks = [

            'comorbidities'

            ,'risk factors'

            ,'lung cancer'

            ,'hypertension'

            ,'heart disease'

            ,'chronic bronchitis'

            ,'cerebral infarction'

            ,'diabetes'

            ,'copd'

            ,'chronic obstructive pulmonary disease'

            ,'cardiovascular diseases'

            ,'chronic kidney disease'

            ,'bacterial pneumonia'

            ,'blood type'

            ,'smoking'

            ,'pregnancy'

        ]

for t in tasks:

    query_list=[] 

    query = "How does {} affect COVID-19?".format(t)

    query_list.append(query)

    cp ="({})".format(t)

    subSetEngine(cp,query_list,corona_df,

        rel_threshold=threshold,

        include_paragraphs = showPara,

        focus_on_sentence= showSentence)

    

# how about the weather    

current_task="weather"

climate_synonyms = [

    'climate',

    'weather',

    'humidity',

    'sunlight',

    'air temperature',

    'meteorology', # picks up meteorology, meteorological, meteorologist

    'climatology', # as above

    'a dry environment',

    'a damp environment',

    'a moist environment',

    'a wet environment',

    'a hot environment',

    'a cold environment',

    'a cool environment'

]



for t in climate_synonyms:

    query_list=[] 

    query = "How does {} affect the transmission of COVID-19?".format(t)

    query_list.append(query)

    cp ="({})".format(t).replace(" ",".*")

    subSetEngine(cp,query_list,corona_df,

        rel_threshold=threshold,

        include_paragraphs = showPara,

        focus_on_sentence= showSentence)

    

#transmission

current_task="transmission"

hypothesis_list = ["aerisol","droplets","food","fecal matter",

                   "contact","water"]

for t in hypothesis_list:

    query_list=[] 

    query = "Is the virus transmitted by {} ?".format(t)

    query_list.append(query)

    cp ="({})".format(t).replace(" ",".*")

    subSetEngine(cp,query_list,corona_df,

        rel_threshold=threshold,

        include_paragraphs = showPara,

        focus_on_sentence= showSentence)

    

#final output

ans_df.head()

ans_out = ans_df.to_csv('covid-19_answer_output.csv')