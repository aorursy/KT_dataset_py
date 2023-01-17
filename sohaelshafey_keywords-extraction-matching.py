# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load spacy library

import spacy

from collections import Counter

import warnings

import warnings

warnings.filterwarnings("ignore")
nlp = spacy.load("en_core_web_lg")
#load ideal text

Golden_txt="Experts throughout both the developing and developed world have debated whether the advent of sophisticated modern technology such as mobile phones, laptops and iPad have helped to enhance and improve people’s social lives or whether the opposite has become the case. Personally, I strongly advocate the former view. This essay will discuss both sides using examples from the UK government and Oxford University to demonstrate points and prove arguments. On the one hand there is ample, powerful, almost daily evidence that such technology can be detrimental especially to the younger generation who are more easily affected by it’s addictive nature and which can result in people feeling more isolated from the society. The central reason behind this is twofold, firstly, the invention of online social media sites and apps, such as Twitter and Facebook have reduced crucial face-to-face interactions dramatically. Through use of these appealing and attractive mediums, people feel in touch and connected yet lack key social skills and the ability to communicate.  Secondly, dependence on such devices is built up frighteningly easily which may have a damaging effect on mental health and encourage a sedentary lifestyle. For example, recent scientific research by the UK government demonstrated that 90% of people in their 30s spend over 20 hours per week on Messenger and similar applications to chat with their friends instead of meeting up and spending quality time together or doing sport. As a result, it is conclusively clear that these technology advancements have decreased and diminished our real life interactions. On the other hand, although there are significant downsides to technological developments, its’ multifold advantages cannot be denied. This is largely because the popularity of technology such as cellphones allows people to connect freely and easily with no geographical barriers. People are able to share any type of news, information, photos and opinions with their loved ones whenever and wherever they want therefore keeping a feeling of proximity and closeness. For example, an extensive study by Oxford University illustrated that people who work, or study abroad and use applications like Facetime and WhatsApp to chat with their families, are less likely to experience loneliness and feel out of the loop than those  who do not. Consistent with this line of thinking is that businessmen are also undoubtedly able to benefit from these advances by holding virtual real -time meetings using Skype which may increase the chance of closing business deals without the need to fly. From the arguments and examples given I firmly believe that overall communication and mans’ sociability has been advanced enormously due to huge the huge technological progress of the past twenty years and despite some potentially serious health implications which governments should not fail to address, it is predicted that its popularity will continue to flourish in the future."

gld_doc=nlp(Golden_txt)
#load student text

std_txt="It is true that in recent years, technology influenced a lot in the way people interact with each other. Face to face meetings turned into a FaceTime and WhatsApp group video calls; however, it is hard to say whether the positive aspects outweigh the negative aspects or vice versa. On the positive side, new innovations in communication technology helped people to talk with their loved ones more easily than in the past. Nowadays, it has become cheaper and more hassle free to talk with anyone in the world from anywhere. This advancement helps people to maintain their relationships even when they are travelling or in the different countries. For example, in the current pandemic situation, many people are stuck in other countries due to lockdowns, but still they can easily interact with their family as well as they can maintain their business relationships by using virtual meeting applications. On the negative side, people have started preferring video calls over face to face meetings. They have started avoiding travel to meet their family members or business partners; however, physical meetings are way better than virtual meetings, and it always gives you much more realistic feeling and better interaction with the ones you are meeting compared to virtual meetings. In my opinion, new technological advancements removed all international and geographical barriers between people and enabled them to stay in constant touch; however, these innovations should be used only when they are necessary; whenever possible, people should go out and physically meet their loved ones; over use of technology might create many misunderstandings and it could also have negative impacts on relationships."

std_doc=nlp(std_txt)
#get number of each tag

Counter(([token.pos_ for token in std_doc]))
#get verb tenses count

tenses=dict([])

tenses['present']=len([nlp.vocab.morphology.tag_map[tkn.tag_].get("Tense_pres") for tkn in std_doc if tkn.pos_=="VERB" if nlp.vocab.morphology.tag_map[tkn.tag_].get("Tense_pres")])

tenses['past']=len([nlp.vocab.morphology.tag_map[tkn.tag_].get("Tense_past") for tkn in std_doc if tkn.pos_=="VERB" if nlp.vocab.morphology.tag_map[tkn.tag_].get("Tense_past")])

tenses['future']=len([tkn.text for tkn in std_doc if tkn.text.lower() in ("shall","will") and tkn.dep_.lower()=="aux"])

tenses
#Number of Active 

Act_tkn=[tok.text for tok in std_doc if (tok.dep_ == "nsubj")]

len(Act_tkn)
#Number of PAssive 

Act_tkn=[tok.text for tok in std_doc if (tok.dep_ == "nsubjpass")]

len(Act_tkn)
#get number of misspelled words

Counter(([token.is_oov for token in std_doc]))
#number of punctuations and words

Counter(([token.is_punct for token in std_doc]))
#number of stopwords

Counter(([token.is_stop for token in std_doc]))
#define a function to get the frequent words in a text which aren't stop words or punctuations

def get_frequent_words(sample_doc):

    words = [token.text.lower() for token in sample_doc if not token.is_stop and not token.is_punct]

    word_freq = Counter(words)

    return word_freq
#second extract the frequent words from the student text

std_wrd_freq = get_frequent_words(std_doc)

gld_wrd_freq = get_frequent_words(gld_doc)
#define the match function between the two lists

def matching_keywords(stdlst,gldlst):

        #matched list

        res=[]

        #unmatched list

        tmpres=[]

        for x in stdlst:

            if (x in gldlst):

                res.append(x)

            else:

                tmpres.append(x)

        return res,tmpres
#get matched and unmatched keywords lists

matched,unmatched=matching_keywords(std_wrd_freq,gld_wrd_freq)

#percentage of matched keywords from student common words to golden common words

print("matching percentage using common words:",len(matched)/len(gld_wrd_freq))

#ratio of unmatched keywords from student common words to all students common words

print("unmatching percentage using common words:",len(unmatched)/len(std_wrd_freq))
#get most similar tokens between unmatched list and the whole golden set

def most_similar(stdlst_unmatched,gld_lst):

    lst=[]

    for tkn1 in stdlst_unmatched:

        for tkn2 in gld_lst:

            lst.append([tkn1,tkn2,std_doc.vocab[tkn1].similarity(gld_doc.vocab[tkn2])])

                

    return pd.DataFrame(lst,columns=['tkn1','tkn2','similarity'])

df=most_similar(unmatched,gld_wrd_freq)
df[df['similarity']>0.7]
print("matched frequent keywords:", matched)
#second method extract unique words with only 1 occurrence

gld_uniq_wrds=[word for word in gld_wrd_freq if (gld_wrd_freq.get(word)==1)]

std_uniq_wrds=[word for word in std_wrd_freq if (std_wrd_freq.get(word)==1)]
#then apply match function

matched,unmatched=matching_keywords(std_uniq_wrds,gld_uniq_wrds)

#percentage of matched keywords from student unique words list and golden unique words list

print("matching percentage using unique words:",len(matched)/len(gld_uniq_wrds) )

#ratio of unmatched keywords from student unique words list

print("unmatching percentage using unique words:",len(unmatched)/len(std_uniq_wrds) )
print("matched unique keywords:", matched)
#get the most similar tokens between unmatched and golden list

df=most_similar(unmatched,gld_uniq_wrds)
df[df['similarity']>0.7]
#third method extract nouns chuncks and special tags like proper noun, noun, adjective

def extract_POS(sample_doc):

    res=[]

    for chk in sample_doc.noun_chunks:

        tmp=""

        for tkn in chk:

            if (tkn.pos_ in ['NOUN','PROPN','ADJ'] ):

                if (not(tkn.is_stop) and not(tkn.is_punct)):

                    tmp = tmp + tkn.text.lower() + " "

        if(tmp.strip()!=""):

            res.append(tmp.strip())

    return list(dict.fromkeys(res))
#extract noun chuncks from both golden and student text

gld_POS=extract_POS(gld_doc)

std_POS=extract_POS(std_doc)
#then apply match function

matched,unmatched=matching_keywords(std_POS,gld_POS)

#percentage of matched keywords

print("matching percentage using unique words:",len(matched)/len(gld_POS) )

#ratio of unmatchedd keywords

print("unmatching percentage using unique words:",len(unmatched)/len(std_POS))
print("matched Special POS keywords:", matched)
#get the most similar tokens between unmatched and golden list

df=most_similar(unmatched,gld_POS)
df[df['similarity']>0.7]
from nltk.stem import PorterStemmer

from itertools import groupby



def yule(sample_doc):

    # yule's I measure (the inverse of yule's K measure)

    # higher number is higher diversity - richer vocabulary

    d = {}

    ps = PorterStemmer()

    for w in sample_doc:

        if(not(w.is_punct)):

            try:

                w = ps.stem(w.text.lower())

            except:

                print(w)

            try:

                d[w] = d[w] + 1

            except KeyError:

                d[w] = 1

 

        M1 = float(len(d))

        M2 = float(sum([len(list(g))*(freq**2) for freq,g in groupby(sorted(d.values()))]))

 

    try:

        return float((M1*M1)/(M2-M1))

    except ZeroDivisionError:

        return float(0)

 

print("Yule's I measurement for vocabulary richness for golden answer:",str(round(yule(gld_doc))))

print("Yule's I measurement for vocabulary richness for student answer:",str(round(yule(std_doc))))