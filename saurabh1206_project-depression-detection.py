#Listing out all the dependencies
import os
import numpy as np
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm,trange
from bs4 import BeautifulSoup
import re,csv
from collections import defaultdict
import nltk
import math
from nltk.corpus import sentiwordnet as swn
import matplotlib.pyplot as plt
tok=WordPunctTokenizer()




def load_data():
    data=pd.read_csv('../input/nayadataset/nayadataset.txt',sep='\n',names=['tweet'])
    return data

def preprocess_data(data):
    ps=PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop=stopwords.words('english')
    mention_pat=re.compile(r'\B@[\w]+\b')
    url_pat=re.compile(r'https://[A-Za-z0-9./]+')
    hashtag_pat=re.compile(r'\B#\w*[a-zA-Z]+\w*')
    tqdm.pandas(desc='REMOVING STOPWORDS            ')
    data['tweet']=data['tweet'].progress_apply(lambda x:' '.join([word for word in tok.tokenize(x) if word not in (stop)]))
    tqdm.pandas(desc='TRANSLATING THE ABBREVIATIONS ')
    data['tweet']=data['tweet'].progress_apply(lambda x:translator(x))
    tqdm.pandas(desc='REMOVING PUNCTUATIONS         ')
    data['tweet']=data['tweet'].progress_apply(lambda x:' '.join([word for word in tok.tokenize(x) if word.isalpha()]))
    tqdm.pandas(desc='REMOVING @/mentions           ')
    data['tweet']=data['tweet'].progress_apply(lambda x:re.sub(mention_pat,'',x))
    tqdm.pandas(desc='REMOVING URLs                 ')
    data['tweet']=data['tweet'].progress_apply(lambda x:re.sub(url_pat,'',x))
    tqdm.pandas(desc='REMOVING HASHTAGS             ')
    data['tweet']=data['tweet'].progress_apply(lambda x:re.sub(hashtag_pat,'',x))
    tqdm.pandas(desc='APPLYING WORDNET LEMMATIZER     ')
    data['tweet']=data['tweet'].progress_apply(lambda x:' '.join([lemmatizer.lemmatize(word,simple_tag((nltk.pos_tag([word]))[0][1])) for word in tok.tokenize(x) if simple_tag((nltk.pos_tag([word]))[0][1])!='']))
    tqdm.pandas(desc='CAPITALIZING LETTERS          ')
    data['tweet']=data['tweet'].progress_apply(lambda x:' '.join([word.upper() for word in tok.tokenize(x)]))
    return data
#Removing the lines with a single word
def remove_single_word_lines(data):
    l=[]
    for i in range(0,data.shape[0]):
        if len(tok.tokenize(data.tweet[i]))<=3:
            l.append(i)
    data=data.drop(data.index[l]).reset_index()
    data.drop(['index'],axis=1,inplace=True)
    print('---------------------------------------------------------------------DATA PREPROCESSING COMPLETE-----------------------------------------------------------')
    return data

def translator(user_string):
    user_string=user_string.split(' ')
    j=0
    for _str in user_string:
        fileName='../input/dictionary1/slang.txt'
        with open(fileName,'r') as myCSVfile:
            dataFromFile=csv.reader(myCSVfile,delimiter=":")
            _str=re.sub('[^a-zA-Z0-9]+','',_str)
            for row in dataFromFile:
                if _str.upper()==row[0]:
                    user_string[j]=row[1]
            myCSVfile.close()
        j+=1
    return ' '.join(user_string)

def load_matrix(data,target_word):
    matrix=initialize_matrix()
    matrix=create_matrix(data,target_word,matrix)
    return matrix
	
def initialize_matrix():
    matrix=defaultdict(list)
    return matrix
	
def create_matrix(data,target_word,matrix):
    for i in trange(0,data.shape[0]):
        words_list=tok.tokenize(data.tweet[i])
        for j in words_list:
            try:
                si_tag=simple_tag((nltk.pos_tag([j]))[0][1])
                if si_tag!='':
                    word_tuple=(j,si_tag)
                    if word_tuple not in matrix:
                        matrix[word_tuple].append(0)
                        matrix[word_tuple].append(0)
                    if target_word in words_list:
  #                      print('Yes i afound it here')
                        matrix[word_tuple][0]+=1
                        matrix[word_tuple][1]+=len(words_list)-1
                    else:
                        matrix[word_tuple][1]+=len(words_list)-1
            except:
                continue
    return matrix
	
def simple_tag(str_tag):
    simpleDict={ 
        'NN' : 'n',
        'NNS' : 'n',
        'NNP' : 'n',
        'NNPS' : 'n',

        'JJ' : 's',
        'JJS' : 's',
        'JJR' : 's',
        
        'RB' : 'a',
        'RBR' : 'a',
        'RBS' : 'a',

        'VB' : 'v',
        'VBD': 'v',
        'VBG': 'v',
        'VBN': 'v',
        'VBP': 'v',
        'VBZ': 'v'
    }
    if str_tag in simpleDict:
        return simpleDict[str_tag]
    else:
        return ''

def normalized_radius_terms(matrix,target_word,data):
    senticircle=construct_senticircle(matrix,target_word,data)
    radius_list=[]
    for l in senticircle:
        radius_list.append(l[1][0])
    maxradius=max(radius_list)
    radius_list=[(radius_list[i]/maxradius) for i in range(0,len(radius_list))]
    for i in range(0,len(senticircle)):
        senticircle[i][1][0]=radius_list[i]
    return senticircle
	
def construct_senticircle(matrix,target_word,data):
    senticircle=[]
    c=0
    N=total_terms(data)
    length=len(matrix)
    for k,v in matrix.items():
      #  print('Percentage completed: '+str(c/length),end='\r')
        context_word_tuple=k
        radius=compute_TDOC(matrix,target_word,context_word_tuple,N)
        angle=getAngle(context_word_tuple)
        c+=1
        if radius!=0 and angle!=0:
            senticircle.append((context_word_tuple[0],[radius,angle]))
    return senticircle
	
def compute_TDOC(matrix,target_word,context_word_tuple,N):
    TDOC=0
    Nci=getNc(context_word_tuple,matrix)
    fcm=f(context_word_tuple,matrix)
    TDOC=(math.log(N/Nci))*fcm
    return TDOC
	
def getAngle(context_word_tuple):
    return math.pi*prior_sentiment_score(context_word_tuple)
	
def getNc(context_word_tuple,matrix):
    return matrix[context_word_tuple][1]
	
#Counting the total no of terms in the entire data of tweets
def total_terms(data):
    tok=WordPunctTokenizer()
    from tqdm import tqdm,trange
    count=0
    for i in trange(0,data.shape[0]):
        words=tok.tokenize(data.tweet[i])
        count+=len(words)-1
    return count
	
def f(context_word_tuple,matrix):
    return matrix[context_word_tuple][0]
	
def prior_sentiment_score(context_word_tuple):
    try:
        pos_tag=context_word_tuple[1]
        context_word=context_word_tuple[0]
        context_word=context_word.lower()
        context_word='{0}.{1}.01'.format(context_word,pos_tag)
        context_term=swn.senti_synset(context_word)
#        print('Here')
        return max_(context_term.pos_score(),context_term.neg_score(),context_term.obj_score())
#        if context_term.pos_score()>context_term.neg_score():
#            return context_term.pos_score()
#        elif context_term.neg_score()>context_term.pos_score():
#            return -1*context_term.neg_score()
#        else:
#            return context_term.pos_score()
#            '''
    except:
        return 0
    
def max_(a,b,c):
    print('dfghjk')
    if a>=b and a>=c:
        print('hello1')
 #       print('returning '+str(a)+' '+str(b)+' ' +str(c))
        return a
    elif b>=a and b>=c:
        print('hello2')
 #       print('ret '+str(a)+' ' +str(b)+' ' +str(c))
        return -b
    elif c>=a and c>=b:
        print('hello3')
 #       print('return '+str(a)+' '+str(b)+' ' +str(c))
        return 0
    
		
def convert_to_cartesian(senticircle):
    cartesian_senticircle=[]
    for i in range(0,len(senticircle)):
        x=senticircle[i][1][0]*math.cos(senticircle[i][1][1])
        y=senticircle[i][1][0]*math.sin(senticircle[i][1][1])
        cartesian_senticircle.append((senticircle[i][0],[x,y]))
    return cartesian_senticircle
	
def plot_points(cartesian_senticircle,target_word):
    x=[point[1][0] for point in cartesian_senticircle if point[0]!=target_word]
    y=[point[1][1] for point in cartesian_senticircle if point[0]!=target_word]
    labels=[point[0] for point in cartesian_senticircle if point[0]!=target_word]
    plt.title(target_word)
    plt.xlabel('SENTIMENT STRENGHT')
    plt.ylabel('SENTIMENT POLARITY')
    for i in range(len(x)):
        plt.annotate(labels[i], (x[i], y[i]), size=10)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size
    plt.scatter(x,y, s=10)
    plt.grid(True, which='both')
    plt.show()
    import csv
    path=str(target_word)+'.csv'
    myFile=open(path,'w')
    myFile.write("CONTEXT WORD         ,Sentiment Strength , Sentiment Polarity    \n\n")
    for i in range(len(x)):
        myFile.write(str(labels[i])+"  ,"+str(x[i])+"   ,   "+str(y[i])+" \n")
    
def listing_positive_negative_terms(cartesian_senticircle,target_word):
    pos_list_x=[point[1][0] for point in cartesian_senticircle if point[1][1]>=0]
    pos_list_y=[point[1][1] for point in cartesian_senticircle if point[1][1]>=0]
    pos_list_name=[point[0] for point in cartesian_senticircle if point[1][1]>=0]
    neg_list_x=[point[1][0] for point in cartesian_senticircle if point[1][1]<0]
    neg_list_y=[point[1][1] for point in cartesian_senticircle if point[1][1]<0]
    neg_list_name=[point[0] for point in cartesian_senticircle if point[1][1]<0]
    for i in range(len(pos_list_x)-1):
        for j in range(len(pos_list_x)-1-i):
            if(pos_list_x[j]>pos_list_x[j+1]):
                temp=pos_list_x[j]
                pos_list_x[j]=pos_list_x[j+1]
                pos_list_x[j+1]=temp
                temp=pos_list_name[j]
                pos_list_name[j]=pos_list_name[j+1]
                pos_list_name[j+1]=temp
                temp=pos_list_y[j]
                pos_list_y[j]=pos_list_y[j+1]
                pos_list_y[j+1]=temp            
    for i in range(len(neg_list_x)-1):
        for j in range(len(neg_list_x)-1-i):
            if(neg_list_x[j]>neg_list_x[j+1]):
                temp=neg_list_x[j]
                neg_list_x[j]=neg_list_x[j+1]
                neg_list_x[j+1]=temp
                temp=neg_list_name[j]
                neg_list_name[j]=neg_list_name[j+1]
                neg_list_name[j+1]=temp
                temp=neg_list_y[j]
                neg_list_y[j]=neg_list_y[j+1]
                neg_list_y[j+1]=temp
    import csv
    path1=str(target_word)+'_pos.csv'
    path2=str(target_word)+'_neg.csv'
    myFile1=open(path1,'w')
    myFile1.write("RANKING,CONTEXT WORD,Sentiment Polarity,Sentiment Strength\n\n")
    for i in range(len(pos_list_x)):
        myFile1.write(str(i+1)+","+str(pos_list_name[i])+" ,"+str(pos_list_y[i])+" ,"+str(pos_list_x[i])+"\n")
    myFile2=open(path2,'w')
    myFile2.write("RANKING,CONTEXT WORD,Sentiment Polarity,Sentiment Strength\n\n")
    for i in range(len(neg_list_x)):
        myFile2.write(str(i+1)+","+str(neg_list_name[i]+","+str(neg_list_y[i])+" ,"+str(neg_list_x[i])+"\n")) 
        
	
def create_lexicon_dictionaries():
    l=[]
    import csv
    pos_depression_dict={}
    with open('../input/dictionaries/pos_depression.csv','r') as f:
        reader=csv.reader(f,delimiter=',')
        c=0;
        for i in reader:
            if c>=2:
                pos_depression_dict[i[1].strip()]=(-(float(i[3]))+1)/2
            c+=1
    l.append(pos_depression_dict)
    neg_depression_dict={}
    with open('../input/dictionaries/neg_depression.csv','r') as f:
        reader=csv.reader(f,delimiter=',')
        c=0;
        for i in reader:
            if c>=2:
                neg_depression_dict[i[1].strip()]=(float(i[3])-1)/2
            c+=1
    l.append(neg_depression_dict)
    pos_anxiety_dict={}
    with open('../input/dictionaries/pos_anxiety.csv','r') as f:
        reader=csv.reader(f,delimiter=',')
        c=0;
        for i in reader:
            if c>=2:
                pos_anxiety_dict[i[1].strip()]=(-(float(i[3]))+1)/2
            c+=1
    l.append(pos_anxiety_dict)
    neg_anxiety_dict={}
    with open('../input/dictionaries/neg_anxiety.csv','r') as f:
        reader=csv.reader(f,delimiter=',')
        c=0;
        for i in reader:
            if c>=2:
                neg_anxiety_dict[i[1].strip()]=(float(i[3])-1)/2
            c+=1
    l.append(neg_anxiety_dict)
    pos_stress_dict={}
    with open('../input/dictionaries/pos_stress.csv','r') as f:
        reader=csv.reader(f,delimiter=',')
        c=0;
        for i in reader:
            if c>=2:
                pos_stress_dict[i[1].strip()]=(-(float(i[3]))+1)/2
            c+=1
    l.append(pos_stress_dict)
    neg_stress_dict={}
    with open('../input/dictionaries/neg_stress.csv','r') as f:
        reader=csv.reader(f,delimiter=',')
        c=0;
        for i in reader:
            if c>=2:
                neg_stress_dict[i[1].strip()]=(float(i[3])-1)/2
            c+=1
    l.append(neg_stress_dict)
    pos_fear_dict={}
    with open('../input/dictionaries/pos_fear.csv','r') as f:
        reader=csv.reader(f,delimiter=',')
        c=0;
        for i in reader:
            if c>=2:
                pos_fear_dict[i[1].strip()]=(-(float(i[3]))+1)/2
            c+=1
    l.append(pos_fear_dict)
    neg_fear_dict={}
    with open('../input/dictionaries/neg_fear.csv','r') as f:
        reader=csv.reader(f,delimiter=',')
        c=0;
        for i in reader:
            if c>=2:
                neg_fear_dict[i[1].strip()]=(float(i[3])-1)/2
            c+=1
    l.append(neg_fear_dict)
    return l

def scores_by_predefined(data,target_word,li):
    score_list=[]
    for i in li:
        score=0
#        print(data.tweet[i])
        for j in tok.tokenize(data.tweet[i]):
#            print('heer1')
            si_tag=simple_tag((nltk.pos_tag([j]))[0][1])
            if si_tag!='':
                word_tuple=(j,si_tag)
                score+=prior_sentiment_score(word_tuple)
        score_list.append(score)
    return score_list

def scores_by_own(data,target_word):
    l=[]
    li=[]
    l=create_lexicon_dictionaries()
    score_list=[]
    for i in range(0,data.shape[0]):
        score=0
        c=0
#        print(len(tok.tokenize(data.tweet[i])))
#        print('ghjk'+data.tweet[i]+'\n')
        for j in tok.tokenize(data.tweet[i]):
            ind_score_tuple=get_word_score(l,j)
            if ind_score_tuple[1]==True:
                c+=1
                score+=float(ind_score_tuple[0])
#        print(str(c/len(tok.tokenize(data.tweet[i]))))
        if(c/len(tok.tokenize(data.tweet[i])))>=0:
#            print('Appended')
            li.append(i)
            score_list.append(score)
    return (score_list,li)    

def get_word_score(l,j):
    ind_score_tuple=(False,False)
    for k in l:
        if j in k:
            ind_score_tuple=(k.get(j,None),True)
            break
    return ind_score_tuple

def comparison(data,target_word):
    import numpy as np
    new=()
    new=np.asarray(scores_by_own(data,target_word))
    predefined=np.asarray(scores_by_predefined(data,target_word,new[1]))
    tweets_own=[]
    tweets_own=(new[0]<0)
    tweets_predefined=[]
    tweets_predefined=(predefined<0)
    result=(tweets_own==tweets_predefined)
    output=sum(result)/len(result)
    print('Matching :'+str(output))
    print(np.sum(tweets_own))
    print(np.sum(tweets_predefined))
    print(len(tweets_own))
    

def all_operations(target_word):
    #Loading Data
    data=load_data()
#    print('---------------------------------------------------------------------------DATA LOADED--------------------------------------------------------------------')
    #Preprocessing Data
    data=preprocess_data(data)
    data=remove_single_word_lines(data)
    #Creating the co-occurence matrix
    matrix=load_matrix(data,target_word)
    print('---------------------------------------------------------------------CO-OCCURENCE MATRIX LOADED------------------------------------------------------------')
    #Construct the Senticircle
    senticircle=normalized_radius_terms(matrix,target_word,data)
    print('----------------------------------------------------------------------SENTICIRCLE CONSTRUCTED--------------------------------------------------------------')
    #Converting the Senticircle points from polar to cartesian
    cartesian_senticircle=convert_to_cartesian(senticircle)
    plot_points(cartesian_senticircle,target_word)
    listing_positive_negative_terms(cartesian_senticircle,target_word)
 #   comparison(data,target_word)
#Listing out all the dependencies
import os
import numpy as np
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm,trange
from bs4 import BeautifulSoup
import re,csv
from collections import defaultdict
import nltk
import math
from nltk.corpus import sentiwordnet as swn
import matplotlib.pyplot as plt
tok=WordPunctTokenizer()
def here():
    import csv
    path='pos_stress_new.csv'
    myFile99=open(path,'w')
    l=[]
    myFile99.write("CONTEXT WORD         ,Sentiment Strength , SentiWordNet Strength    \n\n")
    l=create_lexicon_dictionaries()
    for k in l:
        for j in k:
            try:
                lexword='{0}.n.01'.format(j)
                word=swn.senti_synset(lexword)
                print(word.pos_score())
                res=max_(word.pos_score(),word.neg_score(),word.obj_score())
                print(res)
                myFile99.write(str(j)+','+str(k.get(j))+','+str(res)+'\n')
            except:
                myFile99.write(str(j)+','+str(k.get(j))+',NOT FOUND'+'\n')

                
here()