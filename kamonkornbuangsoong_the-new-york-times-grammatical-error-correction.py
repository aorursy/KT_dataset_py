!wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
!mkdir kenlm/build
%cd kenlm/build
!cmake ..
!make -j2
!conda install -y gdown
!gdown --id 1Rtaqxq_ahaocqva2ar7bIcyt39zmU8ZQ #gigaword 3/gigaword3_nyt_eng_2000.tar.gz
!unzip "/kaggle/working/kenlm/build/gigaword3_nyt_eng_2000 (extract.me).zip"
ls
!pip install https://github.com/kpu/kenlm/archive/master.zip

!bin/lmplz -o 6 --text nyt_eng_200012 --arpa sixgram.arpa
!bin/lmplz -o 6 --text nyt_eng_200001 --arpa sixgram.arpa
ls
import kenlm
model = kenlm.Model('sixgram.arpa')
import math
def print_score(model, s):
  tokens = s.split(' ')
  log_score = 0.0
  for i, (logprob, length, oov) in enumerate(model.full_scores(s)):
    #if i < len(tokens):
      #print(tokens[i], math.exp(logprob), oov)
    #else:
      #print('END', math.exp(logprob), oov)
  
    log_score += logprob
  return log_score
print_score(model, 'I look forward to meeting you')
print_score(model, 'I look forward to meet you')
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
nltk.download('tagsets')
from nltk.tokenize import word_tokenize
sentence = "My name is Jocelyn"
token = nltk.word_tokenize(sentence)
token

nltk.pos_tag(token)
#test some tagsets and create lists
article = ['the', 'an', 'a']
v_be = ['is', 'am', 'are', 'were', 'was', "isn't", "ain't", "weren't", "wasn't"]
do = ['do','does','did',"don't","doesn't","didn't"]

#noun --> noun.json ดูพหูพจน์
#prep.json
prep = [
    "aboard",
    "about",
    "above",
    "absent",
    "across",
    "after",
    "against",
    "along",
    "alongside",
    "amid",
    "amidst",
    "among",
    "amongst",
    "around",
    "as",
    "astride",
    "at",
    "atop",
    "before",
    "afore",
    "behind",
    "below",
    "beneath",
    "beside",
    "besides",
    "between",
    "beyond",
    "by",
    "circa",
    "despite",
    "down",
    "during",
    "except",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "less",
    "like",
    "minus",
    "near",
    "nearer",
    "nearest",
    "notwithstanding",
    "of",
    "off",
    "on",
    "onto",
    "opposite",
    "outside",
    "over",
    "past",
    "per",
    "save",
    "since",
    "through",
    "throughout",
    "to",
    "toward",
    "towards",
    "under",
    "underneath",
    "until",
    "up",
    "upon",
    "upside",
    "versus",
    "via",
    "with",
    "within",
    "without",
    "worth",
    "according to",
    "adjacent to",
    "ahead of",
    "apart from",
    "as of",
    "as per",
    "as regards",
    "aside from",
    "astern of",
    "back to",
    "because of",
    "close to",
    "due to",
    "except for",
    "far from",
    "inside of",
    "instead of",
    "left of",
    "near to",
    "next to",
    "opposite of",
    "opposite to",
    "out from",
    "out of",
    "outside of",
    "owing to",
    "prior to",
    "pursuant to",
    "rather than",
    "regardless of",
    "right of",
    "subsequent to",
    "such as",
    "thanks to",
    "up to",
    "as far as",
    "as opposed to",
    "as soon as",
    "as well as",
    "at the behest of",
    "by means of",
    "by virtue of",
    "for the sake of",
    "in accordance with",
    "in addition to",
    "in case of",
    "in front of",
    "in lieu of",
    "in place of",
    "in point of",
    "in spite of",
    "on account of",
    "on behalf of",
    "on top of",
    "with regard to",
    "with respect to",
    "with a view to"
]

# noun prep verb p'no
!git clone https://github.com/nozomiyamada/contest2
import json
with open('contest2/verbs-dictionaries.json') as f:
  verb = json.load(f)
#verb
f
with open('contest2/noun.json') as f:
  noun = json.load(f)
#noun
def article_candi(sentence):
  a_candi = []
  no_candi = []
  if sentence != None:
    tokens = sentence.split(' ')
    for i, (t) in enumerate(tokens):
      if tokens[i] in article:
        for a in article:
          a_candi.append(tokens[0:i]+[a]+tokens[i+1:])
  if a_candi == []:
    s = nltk.word_tokenize(sentence)
    no_candi.append(s)
    return no_candi
  else:
    return a_candi
a2 = article_candi('I love an kid') # in list
a2
a1= article_candi('I love this kid') #not in list
a1

#prototype for every pos finished!!!
def find_best_article(sentence):
  scd = {}
  all_cand = article_candi(sentence)
  all_cand
  s_cand = [' '.join(word) for word in all_cand]
  for i in s_cand:
    scores = print_score(model,i)
    scd[i] = scores
  max_key = max(scd, key=scd.get, default=None)
  return max_key
fa_1 = find_best_article('I love an dog')
fa_1
fa_2 = find_best_article('I lave a dog')
fa_2
#prototype of func for lst of lst <works!!!>
def noun_candi(sentence): #noun is a list of lists
  a_candi = []
  no_candi = []
  if sentence != None:
    tokens = sentence.split(' ')
    for i, (t) in enumerate(tokens):
      for forms in noun:
        if tokens[i] in forms:
          for word in forms:
            c = tokens[0:i]+[word]+tokens[i+1:] #ป้องกัน list ซ้ำ
            if c not in a_candi:
              a_candi.append(c)
  if a_candi == []:
    s = nltk.word_tokenize(sentence)
    no_candi.append(s)
    return no_candi
  else:
    return a_candi    
          
  return a_candi
n_cand = noun_candi('accountants is smart')
n_cand
n2 = noun_candi('Tu is dumb')
n2
nc2 = noun_candi('your academics performance is excellent')
nc2
#prototype for every pos finished!!!
def find_best_noun(sentence):
  scd = {}
  all_cand = noun_candi(sentence)
  all_cand
  s_cand = [' '.join(word) for word in all_cand]
  for i in s_cand:
    scores = print_score(model,i)
    scd[i] = scores
  max_key = max(scd, key=scd.get, default=None)
  return max_key
eg = 'your academics performance is excellent' #but accountants is smart won = incorrect
bfn = find_best_noun(eg)
bfn
def verb_candi(sentence): #noun is a list of lists
  a_candi = []
  no_candi = []
  if sentence != None:
    tokens = sentence.split(' ')
    for i, (t) in enumerate(tokens):
      for forms in verb:
        if tokens[i] in forms:
          for word in forms:
            c = tokens[0:i]+[word]+tokens[i+1:] #ป้องกัน list ซ้ำ
            if c not in a_candi:
              a_candi.append(c)
  if a_candi == []:
    s = nltk.word_tokenize(sentence)
    no_candi.append(s)
    return no_candi
  else:
    return a_candi   
          
  return a_candi
v1 = verb_candi('She swaggedd')
v1

def find_best_verb(sentence):
  scd = {}
  all_cand = verb_candi(sentence)
  all_cand
  s_cand = [' '.join(word) for word in all_cand]
  for i in s_cand:
    scores = print_score(model,i)
    scd[i] = scores
  max_key = max(scd, key=scd.get, default=None)
  return max_key
bfv = find_best_verb('I loves you')
bfv
be = 'He is the one'
vbb = find_best_verb(be)
vbb
def prep_candi(sentence):
  a_candi = []
  no_candi = []
  if sentence != None:
    tokens = sentence.split(' ')
    for i, (t) in enumerate(tokens):
      if tokens[i] in prep:
        for a in prep:
          a_candi.append(tokens[0:i]+[a]+tokens[i+1:])
  if a_candi == []:
    s = nltk.word_tokenize(sentence)
    no_candi.append(s)
    return no_candi
  else:
    return a_candi
      
  return a_candi
p1 = prep_candi('I fall into love')
p1
p2 = prep_candi('I fall im love')
p2
def find_best_prep(sentence):
  scd = {}
  all_cand = prep_candi(sentence)
  all_cand
  s_cand = [' '.join(word) for word in all_cand]
  for i in s_cand:
    scores = print_score(model,i)
    scd[i] = scores
  max_key = max(scd, key=scd.get, default=None)
  return max_key
s = 'We are afraid about climate change'
bfp = find_best_prep(s)
bfp
p = 'I look down to you'
pp = find_best_prep(p)
pp
def tok_pos(max_key):
  ans = nltk.word_tokenize(max_key)
  answer = nltk.pos_tag(ans)
  return answer
#max_key = 'This is the fish'
#answer = tok_pos(max_key)
#print(answer)
#>>>[('This', 'DT'), ('is', 'VBZ'), ('the', 'DT'), ('fish', 'NN')]
def find_best(sentence):
  s1 = find_best_article(sentence)
  s2 = find_best_noun(s1)
  s3 = find_best_verb(s2)
  s4 = find_best_prep(s3)
  return s4
sen = 'I loves to ate a apple'
fin = find_best(sen)
fin
two = 'I loves to eat a apple'
ans_2 = find_best(two)
ans_2
q = 'I eats a chicken'
a = find_best(q)
a
def break_into_sen(lst):
 sentence_s =  ' '.join(word for word in lst)
 return sentence_s

f = find_best('However , leading cancer specialists reviewed that animal test results do not necessarily apply to humans ( Lewan , 2007 ) .')
print(f)
g = 'she knew who I am'
h = find_best(g)
h
one = find_best('Humans have many basic needs and one of them is to have an environment that can sustain their lives .')
one
!gdown --id 1CHnRNybDYbq9xTZNCxf22PIQIyBe0Yfz
!rm -r contest2
!git clone https://github.com/nozomiyamada/contest2
with open('/kaggle/working/kenlm/build/contest2/dev_small.txt','r') as f: #encoding="utf-8", errors='ignore') as f: #remove ‘content/’ from path then use 
  lines = f.readlines()
  dev_lst =[]
  for line in lines:
    fi = find_best(line)
    dev_lst.append(fi)
ans_dev = '\n'
ans_dev = ans_dev.join(dev_lst)
print(ans_dev)
with open('/kaggle/working/kenlm/build/contest2/dev_small.txt') as f:
  dev = f.readlines()
  last = dev[-1]
  with open('/kaggle/working/ansdev_2.txt', 'w') as d:
    for line in dev:
      line = line.strip()
      ans = find_best(line)
      #last line
      if ans == last:
        d.write(ans)
      else:
        d.write(ans+'\n')

      
!wc '/kaggle/working/kenlm/build/contest2/dev_small.txt'
!wc '/kaggle/working/ansdev_2.txt'
!python /kaggle/working/kenlm/build/contest2/m2scorer/scripts/m2scorer.py /kaggle/working//ansdev_2.txt /kaggle/working/kenlm/build/contest2/dev_small_answer.txt