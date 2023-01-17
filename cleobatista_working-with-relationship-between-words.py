## Defining the environment

#for windows and linux, change to your path
import sys
sys.path.append('C:\Python27\lib\site-packages')
sys.path.append('/home/cleo/anaconda2/lib/python2.7/site-packages')

## Packages to import and functions

#part of this functions was credited to https://www.kaggle.com/the1owl

import sframe
from nltk.stem.porter import *
import re
import Levenshtein as lv



stemmer = PorterStemmer()

strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing']

len_home=74067
len_test=166693


def str_stem(s): 
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        #s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        
        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"

def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))

def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

def map_similarity(arraya,arrayb,fn):
    from nltk.corpus import wordnet as wn
    arrayc=[]
    #if arrayb==None:##this is for attributes, because not all rows has attributes
     #   return None
    for a in arraya:
        for b in arrayb:
            if fn=='path':
                try:
                    arrayc.append(wn.synset(wn.synsets(a)[0].name()).path_similarity(wn.synset(wn.synsets(b)[0].name())))
                except:
                    arrayc.append(None)
            if fn=='lch':
                try:
                    arrayc.append(wn.synset(wn.synsets(a)[0].name()).lch_similarity(wn.synset(wn.synsets(b)[0].name())))
                except:
                    arrayc.append(None)
            if fn=='wup':
                try:
                    arrayc.append(wn.synset(wn.synsets(a)[0].name()).wup_similarity(wn.synset(wn.synsets(b)[0].name())))
                except:
                    arrayc.append(None)
    return arrayc

def sum_values(arrayx):
    return sum(filter(None,arrayx))

def max_values(arrayx):
    try:
        return max(filter(None,arrayx))
    except:
        return 0
    
def min_values(arrayx):
    try:
        return min(filter(None,arrayx))
    except:
        return 0

def count_values(arrayx):
    return len(filter(None,arrayx))


def distance_lv(arraya,arrayb):
    arrayc=[]
    for a in arraya:
        for b in arrayb:
            arrayc.append(lv.distance(a,b))
    return arrayc

def has_attribute(x):
    if x is None:
        return 0.0
    elif x is not None: return 1.0
    

## reading raw data

from sys import platform as _platform
if _platform=='linux2' or _platform=='linux':
    home=sframe.SFrame.read_csv('train.csv')
    descriptions=sframe.SFrame.read_csv('product_descriptions.csv')
    attributes=sframe.SFrame.read_csv('attributes.csv')
else:
    home=sframe.SFrame.read_csv('c:/users/csbatista/Downloads/train_home_depot.csv/train.csv')
    descriptions=sframe.SFrame.read_csv('c:/users/csbatista/Downloads/product_descriptions.csv')
    attributes=sframe.SFrame.read_csv('c:/users/csbatista/Downloads/attributes.csv/attributes.csv')

## refining some words in product title and search term

home['search_term']=home['search_term'].apply(lambda x:str_stem(x))

home['product_title']=home['product_title'].apply(lambda x:str_stem(x))

##  Defining intervals to split and load data when it's necessary

parts=5
len_data=len_home
part=len_data/parts
intervals=list()
for i in range(1,parts+1):
    intervals.append([(i-1)*part,i*part-1])
intervals[parts-1]=[(parts-1)*part,len_data]

## to home refined

### Product the home_join's files

#This part is to produce the top 15 tfidf terms for product title.

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english',encoding='utf-8',decode_error='ignore')

home_join=list()
for interval in intervals:#para cada intervalo
    home_part=home[interval[0]:interval[1]+1]#the '+1' is for the stop index
    tfidf_matrix =  tf.fit_transform(home_part['product_title'])
    feature_names = tf.get_feature_names() 
    dense = tfidf_matrix.todense()
    top_terms=list()
    for i in range(0,len(home_part['product_title'])):
        term = dense[i].tolist()[0]
        phrase_scores = [pair for pair in zip(range(0, len(term)), term) if pair[1] > 0]
        sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
        if len(sorted_phrase_scores )>15:
            phrase_range=15
        else: phrase_range = len(sorted_phrase_scores)
        list_phrase=list()
        #para cada conjunto de termos
        for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:phrase_range]:
            list_phrase.append(phrase)
        top_terms.append(list_phrase)
    home_part['top_terms_product_name']=top_terms
    home_part.save('home_joins/home_join'+str(interval[0])+'.csv', format='csv')
    #if len(home_join)==0:
     #   home_join=home_part[0:0]
    #home_join.append(home_part)

home=sframe.SFrame()
for i in intervals:
    partial_home=sframe.SFrame.read_csv('home_joins/home_join'+str(i[0])+'.csv',column_type_hints ={'id':int,'top_terms_product_name':list})
    home=home.append(partial_home)

#Home_refined file has all the similarities, also those sums, max and count, and levenshtein with sum and min. All values are between search term and product title.

#split search terms
home['search_term']=home['search_term'].apply(lambda x: x.split())

home['similarities_path']=home.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_name'],'path'))

home['similarities_lch']=home.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_name'],'lch'))

home['similarities_wup']=home.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_name'],'wup'))

home['sum_wup']=home['similarities_wup'].apply(lambda x:sum_values(x))

home['max_wup']=home['similarities_wup'].apply(lambda x:max_values(x))

home['count_wup']=home['similarities_wup'].apply(lambda x:count_values(x))

#home['mean_wup']=home['sum_wup']/home['count_wup']

home['sum_lch']=home['similarities_lch'].apply(lambda x:sum_values(x))

home['max_lch']=home['similarities_lch'].apply(lambda x:max_values(x))

home['count_lch']=home['similarities_lch'].apply(lambda x:count_values(x))

#home['mean_lch']=home['sum_lch']/home['count_lch']

home['sum_path']=home['similarities_path'].apply(lambda x:sum_values(x))

home['max_path']=home['similarities_path'].apply(lambda x:max_values(x))

home['count_path']=home['similarities_path'].apply(lambda x:count_values(x))

#home['mean_path']=home['sum_path']/home['count_path']

#to calculate levenshtein distance for each word in search term and top terms product name
home['levenshtein_dist_search_name']=home.apply(lambda x:distance_lv(x['search_term'],x['top_terms_product_name']))

home['min_lev_name']=home['levenshtein_dist_search_name'].apply(lambda x:min_values(x))

home['sum_lev_name']=home['levenshtein_dist_search_name'].apply(lambda x:sum_values(x))

from sframe import aggregate as agg
home=home.join(home.groupby(key_columns='product_uid',operations={'count':agg.COUNT()}),on='product_uid')

home['count']=(home['count']-min(home['count']))/(max(home['count']-min(home['count'])))

home['len_search_term']=home['search_term'].apply(lambda x:len(x))

home['len_product_title']=home['product_title'].apply(lambda x:len(x))

home=home.remove_columns(['top_terms_product_name','similarities_wup','similarities_path','similarities_lch','levenshtein_dist_search_name'])

home.save('home_joins/home_refined.csv',format='csv')

## to home refined 1

#This part is to produce the top 15 tfidf terms for descriptions.

part=len(descriptions)/20
intervals=list()
for i in range(1,21):
    intervals.append([(i-1)*part,i*part-1])
intervals[19]=[19*part,len(descriptions)]

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english',encoding='utf-8',decode_error='ignore')

desc_join=list()
for interval in intervals:#para cada intervalo
    desc_part=descriptions[interval[0]:interval[1]+1]#the '+1' is for the stop index
    tfidf_matrix =  tf.fit_transform(desc_part['product_description'])
    feature_names = tf.get_feature_names() 
    dense = tfidf_matrix.todense()
    top_terms=list()
    for i in range(0,len(desc_part['product_description'])):
        term = dense[i].tolist()[0]
        phrase_scores = [pair for pair in zip(range(0, len(term)), term) if pair[1] > 0]
        sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
        if len(sorted_phrase_scores )>15:
            phrase_range=15
        else: phrase_range = len(sorted_phrase_scores)
        list_phrase=list()
        #para cada conjunto de termos
        for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:phrase_range]:
            list_phrase.append(phrase)
        top_terms.append(list_phrase)
    desc_part['top_terms_product_description']=top_terms
    desc_part.save('home_joins/home_join_description'+str(interval[0])+'.csv', format='csv')
    #if len(home_join)==0:
     #   home_join=home_part[0:0]
    #home_join.append(home_part)

descriptions=sframe.SFrame.read_csv('home_joins/home_join_description0.csv',column_type_hints ={'product_uid':int,'top_terms_product_description':list})[['product_uid','top_terms_product_description']][0:0]
for i in intervals:
    partial=sframe.SFrame.read_csv('home_joins/home_join_description'+str(i[0])+'.csv',column_type_hints ={'product_uid':int,'top_terms_product_description':list})[['product_uid','top_terms_product_description']]
    descriptions=descriptions.append(partial)

descriptions.save('home_joins/descriptions_refined.csv',format='csv')

descriptions=sframe.SFrame.read_csv('home_joins/descriptions_refined.csv')

home=home.join(descriptions,on='product_uid')

home['similarities_path_description']=home.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_description'],'path'))

home['similarities_lch_description']=home.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_description'],'lch'))

home['similarities_wup_description']=home.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_description'],'wup'))

home['sum_wup_description']=home['similarities_wup_description'].apply(lambda x:sum_values(x))

home['max_wup_description']=home['similarities_wup_description'].apply(lambda x:max_values(x))

home['count_wup_description']=home['similarities_wup_description'].apply(lambda x:count_values(x))

#home['mean_wup_description']=home['sum_wup_description']/home['count_wup_description']

home['sum_lch_description']=home['similarities_lch_description'].apply(lambda x:sum_values(x))

home['max_lch_description']=home['similarities_lch_description'].apply(lambda x:max_values(x))

home['count_lch_description']=home['similarities_lch_description'].apply(lambda x:count_values(x))

#home['mean_lch_description']=home['sum_lch_description']/home['count_lch_description']

home['sum_path_description']=home['similarities_path_description'].apply(lambda x:sum_values(x))

home['max_path_description']=home['similarities_path_description'].apply(lambda x:max_values(x))

home['count_path_description']=home['similarities_path_description'].apply(lambda x:count_values(x))

#home['mean_path_description']=home['sum_path_description']/home['count_path_description']

home['levenshtein_dist_search_description']=home.apply(lambda x:distance_lv(x['search_term'],x['top_terms_product_description']))

home['min_lev_description']=home['levenshtein_dist_search_description'].apply(lambda x:min_values(x))

home['sum_lev_description']=home['levenshtein_dist_search_description'].apply(lambda x:sum_values(x))

home=home.remove_columns(['top_terms_product_description','similarities_path_description','similarities_lch_description','similarities_wup_description','levenshtein_dist_search_description'])

home.save('home_joins/home_refined1.csv',format='csv')

## to home refined 2

#This part is to check attributes. I had a problem manipulating 'na' values generated in products without attributes, so I decided to use only some informations about attributes

## to join attributes from the same product_uid
from sframe import aggregate as agg
attributes=attributes.groupby('product_uid',{'attribute':agg.CONCAT('name','value')})
attributes=attributes.sort('product_uid')

attributes['attribute']=attributes.apply(lambda x:str(x['attribute']))

attributes.save('home_joins/attributes_refined.csv',format='csv')

attributes=sframe.SFrame.read_csv('home_joins/attributes_refined.csv')

home=sframe.SFrame.read_csv('home_joins/home_refined1.csv')

home['has_attribute']=home['attribute'].apply(lambda x:has_attribute(x))

home=home.fillna('has_attribute',0)

home=home.join(attributes,on='product_uid',how='left')

home=home.remove_columns(['product_title','search_term','product_uid','attribute'])

home.save('home_joins/home_refined2.csv',format='csv')

home=sframe.SFrame.read_csv('home_joins/home_refined2.csv')

home.head()

## Split the data in training and testing sets

home_train,home_test=home.random_split(0.8,seed=42)

home_train.save('home_joins/home_train_refined.csv')
home_test.save('home_joins/home_test_refined.csv')

home_train=sframe.SFrame.read_csv('home_joins/home_train_refined.csv')

home_test=sframe.SFrame.read_csv('home_joins/home_test_refined.csv')

home_train=home_train.remove_columns(['id','product_uid','search_term'])

home_test=home_test.remove_columns(['id','product_uid','search_term'])

home_train=home_train.to_dataframe()

home_test=home_test.to_dataframe()

## creating a model

from sklearn import linear_model
clf = linear_model.BayesianRidge()
clf.fit(home_train.drop('relevance',1), home_train['relevance'])

pred=clf.predict(home_test.drop('relevance',1))

#to check rmse
sum(abs(pred-home_test['relevance']))/len(home_test)

## applying the model in the test set

test=sframe.SFrame.read_csv('test.csv')

test['search_term']=test['search_term'].apply(lambda x:str_stem(x))

test['product_title']=test['product_title'].apply(lambda x:str_stem(x))

parts=10
len_data=len_test
part=len_data/parts
intervals=list()
for i in range(1,parts+1):
    intervals.append([(i-1)*part,i*part-1])
intervals[parts-1]=[(parts-1)*part,len_data]

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english',encoding='utf-8',decode_error='ignore')

#test_join=list()
for interval in intervals:#para cada intervalo
    test_part=test[interval[0]:interval[1]+1]#the '+1' is for the stop index
    tfidf_matrix =  tf.fit_transform(test_part['product_title'])
    feature_names = tf.get_feature_names() 
    dense = tfidf_matrix.todense()
    top_terms=list()
    for i in range(0,len(test_part['product_title'])):
        term = dense[i].tolist()[0]
        phrase_scores = [pair for pair in zip(range(0, len(term)), term) if pair[1] > 0]
        sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
        if len(sorted_phrase_scores )>15:
            phrase_range=15
        else: phrase_range = len(sorted_phrase_scores)
        list_phrase=list()
        #para cada conjunto de termos
        for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:phrase_range]:
            list_phrase.append(phrase)
        top_terms.append(list_phrase)
    test_part['top_terms_product_name']=top_terms
    test_part.remove_column('product_title').save('home_joins/test_join'+str(interval[0])+'.csv', format='csv')
    #if len(home_join)==0:
     #   home_join=home_part[0:0]
    #home_join.append(home_part)

#complete_test=sframe.SFrame.read_csv('home_joins/test_join0.csv',column_type_hints ={'top_terms_product_name':list})[['product_uid','top_terms_product_name']][0:0]
test=sframe.SFrame()
for i in intervals:
    partial_test=sframe.SFrame.read_csv('home_joins/test_join'+str(i[0])+'.csv',column_type_hints ={'top_terms_product_name':list})
    test=test.append(partial_test)

test['search_term']=test['search_term'].apply(lambda x: x.split())

test.save('home_joins/test_refined.csv',format='csv')

test['similarities_path']=test.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_name'],'path'))

test['similarities_lch']=test.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_name'],'lch'))

test['similarities_wup']=test.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_name'],'wup'))

descriptions=sframe.SFrame.read_csv('home_joins/descriptions_refined.csv')

test=test.join(descriptions,on='product_uid')

test['similarities_path_description']=test.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_description'],'path'))

test['similarities_lch_description']=test.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_description'],'lch'))

test['similarities_wup_description']=test.apply(lambda x:map_similarity(x['search_term'],x['top_terms_product_description'],'wup'))

test['sum_wup']=test['similarities_wup'].apply(lambda x:sum_values(x))

test['max_wup']=test['similarities_wup'].apply(lambda x:max_values(x))

test['count_wup']=test['similarities_wup'].apply(lambda x:count_values(x))

#test['mean_wup']=test['sum_wup']/test['count_wup']

test['sum_lch']=test['similarities_lch'].apply(lambda x:sum_values(x))

test['max_lch']=test['similarities_lch'].apply(lambda x:max_values(x))

test['count_lch']=test['similarities_lch'].apply(lambda x:count_values(x))

#test['mean_lch']=test['sum_lch']/test['count_lch']

test['sum_path']=test['similarities_path'].apply(lambda x:sum_values(x))

test['max_path']=test['similarities_path'].apply(lambda x:max_values(x))

test['count_path']=test['similarities_path'].apply(lambda x:count_values(x))

#test['mean_path']=test['sum_path']/test['count_path']

test['sum_wup_description']=test['similarities_wup_description'].apply(lambda x:sum_values(x))

test['max_wup_description']=test['similarities_wup_description'].apply(lambda x:max_values(x))

test['count_wup_description']=test['similarities_wup_description'].apply(lambda x:count_values(x))

#test['mean_wup_description']=test['sum_wup_description']/test['count_wup_description']

test['sum_lch_description']=test['similarities_lch_description'].apply(lambda x:sum_values(x))

test['max_lch_description']=test['similarities_lch_description'].apply(lambda x:max_values(x))

test['count_lch_description']=test['similarities_lch_description'].apply(lambda x:count_values(x))

#test['mean_lch_description']=test['sum_lch_description']/test['count_lch_description']

test['sum_path_description']=test['similarities_path_description'].apply(lambda x:sum_values(x))

test['max_path_description']=test['similarities_path_description'].apply(lambda x:max_values(x))

test['count_path_description']=test['similarities_path_description'].apply(lambda x:count_values(x))

#test['mean_path_description']=test['sum_path_description']/test['count_path_description']

test['levenshtein_dist_search_description']=test.apply(lambda x:distance_lv(x['search_term'],x['top_terms_product_description']))

test['levenshtein_dist_search_name']=test.apply(lambda x:distance_lv(x['search_term'],x['top_terms_product_name']))

test['min_lev_description']=test['levenshtein_dist_search_description'].apply(lambda x:min_values(x))

test['min_lev_name']=test['levenshtein_dist_search_name'].apply(lambda x:min_values(x))

test['sum_lev_description']=test['levenshtein_dist_search_description'].apply(lambda x:sum_values(x))

test['sum_lev_name']=test['levenshtein_dist_search_name'].apply(lambda x:sum_values(x))

from sframe import aggregate as agg
test=test.join(test.groupby(key_columns='product_uid',operations={'count':agg.COUNT()}),on='product_uid')

test['count']=(test['count']-min(test['count']))/(max(test['count']-min(test['count'])))

test['len_search_term']=test['search_term'].apply(lambda x:len(x))

test['len_product_title']=test['product_title'].apply(lambda x:len(x))

test=test.remove_columns(['product_title','search_term','product_uid','top_terms_product_name','top_terms_product_description','similarities_wup','similarities_path','similarities_lch','similarities_path_description','similarities_lch_description','similarities_wup_description','levenshtein_dist_search_description','levenshtein_dist_search_name'])

test.save('home_joins/test_refined1.csv')#falta por se tem attributos ou nao

## Predicting in test set

test=sframe.SFrame.read_csv('home_joins/test_refined1.csv',column_type_hints={'levenshtein_dist_search_name':list})

attributes=sframe.SFrame.read_csv('home_joins/attributes_refined.csv')

test=test.join(attributes,on='product_uid',how='left')

test['has_attribute']=test['top_terms_product_attributes'].apply(lambda x:has_attribute(x))

test=test.fillna('has_attribute',0)

test=test.remove_columns(['product_title','search_term','product_uid','top_terms_product_name','top_terms_product_attributes','top_terms_product_description','similarities_wup','similarities_path','similarities_lch','similarities_path_description','similarities_lch_description','similarities_wup_description','levenshtein_dist_search_description','levenshtein_dist_search_name'])

test.save('home_joins/test_refined_final.csv')

test=sframe.SFrame.read_csv('home_joins/test_refined_final.csv')

test=test.to_dataframe()

pred=clf.predict(test[home_train.columns[1:27]])

result_final=sframe.SFrame([sframe.SArray(test['id']),sframe.SArray(pred)]).rename({'X1':'id','X2':'relevance'})

result_final['relevance']=result_final['relevance'].apply(lambda x:3 if x>3 else 1 if x<1 else x)

result_final.save('home_joins/result_final2.csv',format='csv')

result_final


