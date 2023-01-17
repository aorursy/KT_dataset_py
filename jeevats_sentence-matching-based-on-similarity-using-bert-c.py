!pip install -U sentence-transformers
!pip install openpyxl
!pip install googletrans
#df=pd.read_excel("../input/saved_google_t_mr.xlsx")

from openpyxl.workbook import Workbook
import pandas as pd
import os

import scipy
from sentence_transformers import SentenceTransformer

import re, math
from collections import Counter

df=pd.read_excel("../input/saved_google_t_mr.xlsx")
E=[]
H=[]
T=[]

print(df.head())
for i in range(len(df)):
  if df["English"][i]!="None" or df["Marati"][i]!="None":
    E.append(df["English"][i])
    H.append(df["Marati"][i])
    T.append(df["Translated"][i])
print(len(E),len(H),len(T),len(df))
df=pd.DataFrame()
df["English"]=E
df["Marati"]=H
df["Translated"]=T
len(df)
df.sample(30)

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

df1=pd.DataFrame()
e=[]
h=[]
t=[]
ind=[]
for i in range(len(df)):
    vector1 = text_to_vector(str(df['English'][i]))
    vector2 = text_to_vector(str(df['Translated'][i]))
    cosine = get_cosine(vector1, vector2)
    if cosine==0.65:
        ind.append(i)
        e.append(df['English'][i])
        h.append(df['Marati'][i])
        t.append(df['Translated'][i])
    #print('Cosine:', cosine)
df1["English"]=e
df1["Marati"]=h
df1["Translated"]=t


df1.to_excel("final_sent.xlsx")
df.drop(df.index[ind])
df.to_excel("remaining_sentence.xlsx")





df1
df.sample(30)
#
df=pd.read_excel("./remaining_sentence.xlsx")

df=pd.read_excel("./remaining_sentence.xlsx")
print(df.columns)



for i in range(len(df)):
  #df["English"][i]=re.sub("[\d]+","",str(df["English"][i]))
  df["English"][i]=str(df["English"][i])
  #df["Translated"][i]=re.sub("[\d]+","",str(df["Translated"][i]))
  df["Translated"][i]=str(df["Translated"][i])
  df["Marati"][i]=str(df["Marati"][i])
english_list=df["English"]
translate_list=df["Translated"]
count=0
eng=[]
hind=[]
trans=[]
eb=[]
hb=[]
tb=[]
def similarity_finding():
  embedder = SentenceTransformer('bert-base-nli-mean-tokens')
  global count
  global  eng
  global hind
  global trans
  # Corpus with example sentences
  corpus=df["English"]

  corpus_embeddings = embedder.encode(corpus)
  #print(1)
  for i in range(len(df["Translated"])):

    queries = [df["Translated"][i]]
    query_embeddings = embedder.encode(queries)
    #print(i)
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
        distances=list(distances)
        if min(distances)<0.20 and distances.index(min(distances))==i:
          count=count+1
          if df["English"][i]!="None" and df["Marati"][i]!="None" and len(re.findall("[A-Za-z]",str(df["English"][i])))>1 and len(re.findall("[\u0900-\u097F]",str(df["Marati"][i])))>1:
            eng.append(df["English"][i])
            hind.append(df["Marati"][i])
            trans.append(df["Translated"][i])
        else:
          eb.append(df["English"][i])
          hb.append(df["Marati"][i])
          tb.append(df["Translated"][i])
        print(min(distances))
        print(df["English"][i])
          #print(df["Translated"][i],"->",english_list[distances.index(min(distances))] )
similarity_finding()
print("Out of")
print(len(df))
print(len(df)-count)
print("Non Matching")
print("and")
print(count,"are Matching")

df1=pd.DataFrame()
df1["English"]=eng
df1["Marati"]=hind
df1["Translated"]=trans
df2=pd.read_excel("./final_sent.xlsx")
frames = [df2, df1]
result = pd.concat(frames)
result.to_excel("parallel_corpora.xlsx")


df45=pd.DataFrame()
df45["English"]=eb
df45["Marati"]=hb
df45["Translated"]=tb
df45.to_excel("remaining_sentence.xlsx")

from openpyxl.workbook import Workbook
import pandas as pd
import os

import scipy
from sentence_transformers import SentenceTransformer

import re, math
from collections import Counter

df=pd.read_excel("./remaining_sentence.xlsx")

df=df.rename(columns={"eng": "English", "hin": "Marati"})

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

df1=pd.DataFrame()
e=[]
h=[]
t=[]
ind=[]
for i in range(len(df)):
    vector1 = text_to_vector(str(df['English'][i]))
    sim=[]
    ind=[]
    j=i
    if i>100:
      j=i-100+2
    else:
      j=0
    while(j<i+100 and j>i-100):
      try:
        vector2 = text_to_vector(str(df['Translated'][j]))
        cosine = get_cosine(vector1, vector2)
        if cosine>0.90:
            ind.append(j)
            sim.append(cosine)
      except:
        print("hi")
      j=j+1
    print(i)
    if len(sim)!=0:
      print(df['English'][i])
      e.append(df['English'][i])
      h.append(df["Marati"][ind[sim.index(max(sim))]])
      t.append(df["Translated"][ind[sim.index(max(sim))]])
               
    
    #print('Cosine:', cosine)
df1["English"]=e
df1["Marati"]=h
df1["Translated"]=t


df1.to_excel("final_sent1_cosine.xlsx")





len(df1)


df1
df.drop(df.index[ind])
df.to_excel("remaining_sentence1.xlsx")
#df.sample(30)
import pandas as pd
df=pd.read_excel("./remaining_sentence1.xlsx")
print(df.columns)

for i in range(len(df)):
  #df["English"][i]=re.sub("[\d]+","",str(df["English"][i]))
  df["English"][i]=str(df["English"][i])
  #df["Translated"][i]=re.sub("[\d]+","",str(df["Translated"][i]))
  df["Translated"][i]=str(df["Translated"][i])
  df["Marati"][i]=str(df["Marati"][i])
english_list=df["English"]
translate_list=df["Translated"]
count=0
eng=[]
hind=[]
trans=[]


mise=[]
mish=[]
mist=[]
def similarity_finding():
  embedder = SentenceTransformer('bert-base-nli-mean-tokens')
  global count
  global  eng
  global hind
  global trans
  global df
  global mise
  global mish
  global mist
  # Corpus with example sentences
  corpus=df["English"]
  print(df.head())
  corpus_embeddings = embedder.encode(corpus)
  #print(1)
  for i in range(len(df["Translated"])):
    queries = [df["Translated"][i]]
    query_embeddings = embedder.encode(queries)
    #print(i)
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
        distances=list(distances)
        if min(distances)<0.1:
          count=count+1
          if df["English"][distances.index(min(distances))]!="None" and df["Marati"][i]!="None" and len(re.findall("[A-Za-z]",str(df["English"][distances.index(min(distances))])))>1 and len(re.findall("[\u0900-\u097F]",str(df["Marati"][i])))>1:
            eng.append(df["English"][distances.index(min(distances))])
            hind.append(df["Marati"][i])
            trans.append(df["Translated"][i])
        else:
          if df["English"][distances.index(min(distances))]!="None" and df["Marati"][i]!="None" and len(re.findall("[A-Za-z]",str(df["English"][distances.index(min(distances))])))>1 and len(re.findall("[\u0900-\u097F]",str(df["Marati"][i])))>1:
            mise.append(df["English"][distances.index(min(distances))])
            mish.append(df["Marati"][i])
            mist.append(df["Translated"][i])
            print(min(distances))
          #print(df["Translated"][i],"->",english_list[distances.index(min(distances))] )
similarity_finding()



print("Out of")
print(len(df))
print(len(df)-count)
print("Non Matching")
print("and")


df1=pd.DataFrame()
df1["English"]=eng
df1["Marati"]=hind
df1["Translated"]=trans
df2=pd.read_excel("./final_sent1_cosine.xlsx")
frames = [df2, df1]
result = pd.concat(frames)
result.to_excel("parallel_corpora_2.xlsx")
#result.sample(30)

df1=pd.DataFrame()
df1["English"]=mise
df1["Marati"]=mish
df1["Translated"]=mist
df1.to_excel("./remaining_sentences_2.xlsx")
#df1.sample(30)
#Completed
import pandas as pd
df=pd.read_excel("./remaining_sentences_2.xlsx")
print(df.columns)

for i in range(len(df)):
  #df["English"][i]=re.sub("[\d]+","",str(df["English"][i]))
  df["English"][i]=str(df["English"][i])
  #df["Translated"][i]=re.sub("[\d]+","",str(df["Translated"][i]))
  df["Translated"][i]=str(df["Translated"][i])
  df["Marati"][i]=str(df["Marati"][i])
english_list=df["English"]
translate_list=df["Translated"]
count=0
eng=[]
hind=[]
trans=[]
mise=[]
mish=[]
mist=[]
def similarity_finding():
  embedder = SentenceTransformer('bert-base-nli-mean-tokens')
  global count
  global  eng
  global hind
  global trans
  global df
  global mise
  global mish
  global mist
  # Corpus with example sentences
  corpus=df["English"]
  print(df.head())
  corpus_embeddings = embedder.encode(corpus)
  #print(1)
  for i in range(len(df["Translated"])):
    queries = [df["Translated"][i]]
    query_embeddings = embedder.encode(queries)
    #print(i)
    #print(query_embeddings)
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
        distances=list(distances)
        if min(distances)<0.30:
          count=count+1
          if df["English"][distances.index(min(distances))]!="None" and df["Marati"][i]!="None" and len(re.findall("[A-Za-z]",str(df["English"][distances.index(min(distances))])))>1 and len(re.findall("[\u0900-\u097F]",str(df["Marati"][i])))>1:
            eng.append(df["English"][distances.index(min(distances))])
            hind.append(df["Marati"][i])
            trans.append(df["Translated"][i])
        else:
          if df["English"][distances.index(min(distances))]!="None" and df["Marati"][i]!="None" and len(re.findall("[A-Za-z]",str(df["English"][distances.index(min(distances))])))>1 and len(re.findall("[\u0900-\u097F]",str(df["Marati"][i])))>1:
            mise.append(df["English"][distances.index(min(distances))])
            mish.append(df["Marati"][i])
            mist.append(df["Translated"][i])

          #print(df["Translated"][i],"->",english_list[distances.index(min(distances))] )
        print(min(distances))
    print(df["Translated"][i])
similarity_finding()
print("Out of")
print(len(df))
print(len(df)-count)
print("Non Matching")
print("and")
print(count,"are Matching")



df1=pd.DataFrame()
df1["English"]=eng
df1["Marati"]=hind
df1["Translated"]=trans

df1.to_excel("Doubtfull_sentences.xlsx")

len(df1)
print(len(df)-count)
df1.sample(30)
sent=0
English_all=[]
Marati_all=[]
trans_all=[]
cnt_90=0
cnt_80=0
English=[]
Marati=[]
trans=[]
df_section=pd.read_excel('./parallel_corpora.xlsx')
df_sentence=pd.read_excel('./parallel_corpora_2.xlsx')
df_doubt=pd.read_excel('./Doubtfull_sentences.xlsx')
English.extend(df_section['English'])
Marati.extend(df_section['Marati'])
trans.extend(df_section['Translated'])
English.extend(df_sentence['English'])
Marati.extend(df_sentence['Marati'])
trans.extend(df_sentence['Translated'])
sent=sent+len(trans)
print(len(trans))
Data_frame_Translation=pd.DataFrame()
Data_frame_Translation['English']=English
Data_frame_Translation['Marati']=Marati
Data_frame_Translation['Translated']=trans    
cnt_90=cnt_90+len(Data_frame_Translation)
Data_frame_Translation.to_excel('Matati_parallel_corpora_V1_90_percent.xlsx',index=False)
English.extend(df_doubt['English'])
Marati.extend(df_doubt['Marati'])
trans.extend(df_doubt['Translated'])
English_all.extend(English)
Marati_all.extend(Marati)
trans_all.extend(trans)
cnt_80=cnt_80+len(Data_frame_Translation)
Data_frame_Translation=pd.DataFrame()
Data_frame_Translation['English']=English
Data_frame_Translation['Marati']=Marati
Data_frame_Translation['Translated']=trans
Data_frame_Translation.to_excel('Matati_parallel_corpora_V1_80_percent.xlsx',index=False)
Data_frame_Translation=pd.DataFrame()
Data_frame_Translation['English']=English_all
Data_frame_Translation['Marati']=Marati_all
Data_frame_Translation['Translated']=trans_all
for column in Data_frame_Translation.columns:
            Data_frame_Translation = Data_frame_Translation[Data_frame_Translation[column]!='None']
Data_frame_Translation.to_excel('Marati_all.xlsx',index=False)
#************************************************* this is the final stage **************************************
sentence_matched=pd.read_excel('./Matati_parallel_corpora_V1_80_percent.xlsx')
len(sentence_matched)






"""for i in range(len(result)):
  result["English"][i]=re.sub("[\d]+","",str(result["English"][i]))
  result["Marati"][i]=re.sub("[\d]+","",str(result["Marati"][i]))
  result["Translated"][i]=re.sub("[\d]+","",result["Tranlsated"][i])
  result["English"][i]=re.sub('[#,:";\/<>()~`@$%&*-+=[]]','',str(result["English"][i]))
  result["Marati"][i]=re.sub('[#,:";\/<>()~`@$%&*-+=[]]','',str(result["Marati"][i]))
  result["Translated"][i]=re.sub('[#,:";\/<>()~`@$%&*-+=[]]','',result["Tranlsated"][i])"""






"""File1_Open=open("../input/5156_1_Untitled-2.txt",'r')
English_Txt_List=[]
for Line_in_File in File1_Open:
    English_Txt_List.append(Line_in_File)
#print(len(English_Txt_List))
#English_Book_Txt= ' '.join(map(str, English_Txt_List))"""

#print(English_Txt_List,Marati_Txt_List)

#"""en_val=' '.join(map(str, English_Txt_List))
#type(en_val)

#hn_val=' '.join(map(str, Marati_Txt_List))
#type(hn_val)

#import re
#en_val=re.sub("(TABLE \d.\d)","",en_val)

#en_val
"""
en_val=re.sub("(Figure \d.\d)","",en_val)
en_val=re.sub("(FIGURE \d.\d)","",en_val)
en_val=re.sub("(Table \d.\d)","",en_val)
en_val=re.sub("(TABLE \d.\d)","",en_val)
en_val=re.sub("Table \d.\d","",en_val)
en_val=re.sub("TABLE \d.\d","",en_val)
en_val=re.sub("(Figure \d.\d)","",en_val)
en_val=re.sub("(FIGURE \d.\d)","",en_val)
en_val=re.sub("(Fig. \d.\d)","",en_val)
en_val=re.sub("(FIG. \d.\d)","",en_val)
en_val=re.sub("(Fig. \d.\d)","",en_val)
en_val=re.sub("(FIG. \d.\d)","",en_val)"""
#en_val=re.sub("(\d.\d-\d.\d)","",en_val)
#en_val=re.sub("[a-zA-Z]+[\. ]+\d.\d","",en_val)
#en_val=re.sub("([a-zA-Z]+[\. ]+\d.\d)","",en_val)

#s=re.sub("\n","",en_val)
#type(s)

#el=[]

#tmp1=re.match('(.*?)\d\.\d',s)
#Value_English_Start=tmp1.group(0)
#s1=s[len(Value_English_Start)-3:]

#el.append(Value_English_Start)
"""
i=0
while(i<1):
	stvl=s1[0:s1.find(" ")]
	s1=s1[s1.find(" "):]
	try:
		val2=re.match('(.*?)\d.\d',s1).group(0)
	except:
		i=2
		break
	str_val=stvl+" "+val2
	print(str_val)
	el.append(str_val)
	s1=s1[len(re.match('(.*?)\d.\d',s1).group(0))-3:]"""

#el.append(s1)
#el
"""File2_Open=open("/content/5401_1.txt",encoding="utf8")
Marati_Txt_List=[]
for Line_in_File in File2_Open:
    Marati_Txt_List.append(Line_in_File)


hn_val=' '.join(map(str, Marati_Txt_List))
import re
hn_val=re.sub("(चित्र \d.\d)","",hn_val)
hn_val=re.sub("चित्र \d.\d","",hn_val)

hn_val=re.sub("(तालिका \d.\d)","",hn_val)
hn_val=re.sub("तालिका \d.\d","",hn_val)


hn_val=re.sub("[\u0900-\u097F]+[\. ]+\d.\d","",hn_val)
hn_val=re.sub("([\u0900-\u097F]+[\. ]+\d.\d)","",hn_val)

hn_val=re.sub("\d.\d मिलियन से लेकर \d.\d","",hn_val)


s=re.sub("\n","",hn_val)



hl=[]
tmp1=re.match('(.*?)\d\.\d',s)
Value_Marati_Start=tmp1.group(0)
s1=s[len(Value_Marati_Start)-3:]
hl.append(Value_Marati_Start)
i=0
while(i<1):
	stvl=s1[0:s1.find(" ")]
	s1=s1[s1.find(" "):]
	try:
		val2=re.match('(.*?)\d.\d',s1).group(0)
	except:
		i=2
		break
	str_val=stvl+" "+val2
	print(str_val)
	hl.append(str_val)
	s1=s1[len(re.match('(.*?)\d.\d',s1).group(0))-3:]
hl.append(s1)"""
#print(len(el),len(hl))
#el
#hl
"""chapter_no=str(input("Enter Chapter No"))

File1_Open=open("/content/5161_2_Chapter-2.txt",'r')
English_Txt_List=[]
for Line_in_File in File1_Open:
    English_Txt_List.append(Line_in_File)
    print(Line_in_File)"""
"""Final_list=[]
l2=[]
for i in range(0,len(English_Txt_List)):
  vl=re.search('^'+chapter_no+'\.\d',English_Txt_List[i])
  if vl:
    l2.append(vl)
    print(vl)
  else:
    print("Hi")
"""

#len(l2)
"""print("Enter The Subject Name \n 1. Biology \n2.")


chapter_no=str(input("Enter Chaptor No : "))
def data_English_Marati(English_f)
File2_Open=open("/content/5430_14_Chapter 14.txt",encoding="utf8")

English_Txt_List=[]
for Line_in_File in File2_Open:
    English_Txt_List.append(Line_in_File)
Final_list=[]
l2=[]
for i in range(0,len(English_Txt_List)):
  English_Txt_List[i]=re.sub("\\u0923\\u094d",".",English_Txt_List[i])
  if re.search('^'+chapter_no+'\.\d',English_Txt_List[i]):
    Final_list.append(l2)
    l2=[]
    l2.append(English_Txt_List[i])
    break
  else:
    l2.append(English_Txt_List[i])
for j in range(i+1,len(English_Txt_List)):
  English_Txt_List[j]=re.sub("\\u0923\\u094d",".",English_Txt_List[j])
  vl=re.search('^'+chapter_no+'\.\d',English_Txt_List[j])
  if vl:
    Final_list.append(l2)
    l2=[]
    l2.append(English_Txt_List[j])
  else:
    l2.append(English_Txt_List[j])
Final_list.append(l2)"""
#len(Final_list)

#Final_list
"""File1_Open=open("/content/5227_14_Chapter14.txt",'r')
English_Txt_List=[]
for Line_in_File in File1_Open:
    English_Txt_List.append(Line_in_File)
Final_list=[]
l2=[]
for i in range(0,len(English_Txt_List)):
  vl=re.search('^'+chapter_no+'\.\d',English_Txt_List[i])
  if vl:
    Final_list.append(l2)
    l2=[]
    l2.append(English_Txt_List[i])
    break
  else:
    l2.append(English_Txt_List[i])
for j in range(i+1,len(English_Txt_List)):
  vl=re.search('^'+chapter_no+'\.\d',English_Txt_List[j])
  if vl:
    Final_list.append(l2)
    l2=[]
    l2.append(English_Txt_List[j])
  else:
    l2.append(English_Txt_List[j])
Final_list.append(l2)"""
#len(Final_list)
#Final_list

#Science
"""chapter_no=str(input("Enter Chaptor No : "))

File2_Open=open("/content/5450_1_Chapter-1.txt",encoding="utf8")

English_Txt_List=[]
for Line_in_File in File2_Open:
    English_Txt_List.append(Line_in_File)
Final_list1=[]
l2=[]
for i in range(0,len(English_Txt_List)):
  English_Txt_List[i]=re.sub("\\u0923\\u094d",".",English_Txt_List[i])
  if re.search('^क्रियाकलाप '+chapter_no+'\.\d',English_Txt_List[i]) or re.search('^क्रियाकलाप \d',English_Txt_List[i]):
    Final_list1.append(l2)
    l2=[]
    l2.append(English_Txt_List[i])
    break
  else:
    l2.append(English_Txt_List[i])
for j in range(i+1,len(English_Txt_List)):
  English_Txt_List[j]=re.sub("\\u0923\\u094d",".",English_Txt_List[j])
  #re.search('^क्रियाकलाप '+chapter_no+'\.\d',English_Txt_List[j])
  if re.search('^क्रियाकलाप '+chapter_no+'\.\d',English_Txt_List[j]) or re.search('^क्रियाकलाप \d',English_Txt_List[j]):
    Final_list1.append(l2)
    l2=[]
    l2.append(English_Txt_List[j])
  else:
    l2.append(English_Txt_List[j])
Final_list1.append(l2)"""
#print(len(Final_list1),Final_list1)
#for i in Final_list1:
#  print(i)
"""File1_Open=open("/content/5236_1_Untitled-14.txt",'r')
English_Txt_List=[]
for Line_in_File in File1_Open:
    English_Txt_List.append(Line_in_File)
Final_list=[]
l2=[]
for i in range(0,len(English_Txt_List)):
  if re.search('^Activity '+chapter_no+'\.\d',English_Txt_List[i]) or re.search('^Activity \d',English_Txt_List[i]):
    Final_list.append(l2)
    l2=[]
    l2.append(English_Txt_List[i])
    break
  else:
    l2.append(English_Txt_List[i])
for j in range(i+1,len(English_Txt_List)):
  if re.search('^Activity '+chapter_no+'\.\d',English_Txt_List[j]) or re.search('^Activity \d',English_Txt_List[j]):
    Final_list.append(l2)
    l2=[]
    l2.append(English_Txt_List[j])
  else:
    l2.append(English_Txt_List[j])
Final_list.append(l2)"""
#len(Final_list)
#for i in Final_list:
#  print(i)
"""String_list=[]
for i in range(len(Final_list)):
  val=' '.join(map(str, Final_list[i]))
  String_list.append(val)"""
"""String_list1=[]
for i in range(len(Final_list1)):
  val=' '.join(map(str, Final_list1[i]))
  String_list1.append(val)"""
#print(len(String_list),len(String_list1))

#if len(String_list)>len(String_list1):
#  String_list
#String_list1
#len(String_list1)
"""l=[]
i=0
while(i< len(String_list1)-1):
  val=String_list1[i].find(" ", String_list1[i].find(" ") + 1)
  st=String_list1[i][0:val+1]
  val1=String_list1[i+1].find(" ", String_list1[i+1].find(" ") + 1)
  st1=String_list1[i+1][0:val1+1]
  #print(st.strip()==st1.strip())
  if st.strip()==st1.strip():
    val2=String_list1[i]+String_list1[i+1]
    l.append(val2)
    print(val2)
    i=i+1
  else:
    l.append(String_list1[i])
  i=i+1
if i==len(String_list1)-1:
  l.append(String_list1[i])
  
"""


#l
#len(l)
"""l=[]
for i in range(len(String_list1)-1):
  val=String_list1[i].find(" ")
  st=String_list1[i][0:val]
  val1=String_list1[i+1].find(" ")
  st1=String_list1[i+1][0:val1]
  if st==st1:
    val2=String_list1[i]+String_list1[i]
    l.append(val2)
    i=i+1
  else:
    l.append(String_list1[i])
if i==len(String_list1):
  l.append(String_list1[i])
  
"""


l

"""chapter_no=str(input("Enter Chapter No"))

File1_Open=open("/content/5161_1_Chapter-1.txt",'r')
English_Txt_List=[]
for Line_in_File in File1_Open:
    English_Txt_List.append(Line_in_File)
#print(len(English_Txt_List))
#English_Book_Txt= ' '.join(map(str, English_Txt_List))

print(English_Txt_List)

en_val=' '.join(map(str, English_Txt_List))

type(en_val)

import re
en_val=re.sub("(TABLE \d.\d)","",en_val)

en_val=re.sub("(\d.\d-\d.\d)","",en_val)

en_val=re.sub("[a-zA-Z]+[\. ]+\d.\d","",en_val)
en_val=re.sub("([a-zA-Z]+[\. ]+\d.\d)","",en_val)

s=re.sub("\n","",en_val)

type(s)

el=[]



tmp1=re.match('(.*?)'+chapter_no+'\.\d',s)
Value_English_Start=tmp1.group(0)
s1=s[len(Value_English_Start)-(len(chapter_no)+2):]

el.append(Value_English_Start)

i=0
while(i<1):
	stvl=s1[0:s1.find(" ")]
	s1=s1[s1.find(" "):]
	try:
		val2=re.match('(.*?)'+chapter_no+'\.\d',s1).group(0)
	except:
		i=2
		break
	str_val=stvl+" "+val2
	print(str_val)
	el.append(str_val)
	s1=s1[len(re.match('(.*?)'+chapter_no+'\.\d',s1).group(0))-(len(chapter_no)+2):]

el.append(s1)

el





File2_Open=open("/content/5432_1.txt",encoding="utf8")

Marati_Txt_List=[]
for Line_in_File in File2_Open:
    Marati_Txt_List.append(Line_in_File)


hn_val=' '.join(map(str, Marati_Txt_List))
import re

print(English_Txt_List)

en_val=' '.join(map(str, English_Txt_List))

type(en_val)

Marati_Txt_List=[]

hn_val=re.sub("[\u0900-\u097F]+[\. ]+\d.\d","",hn_val)
hn_val=re.sub("([\u0900-\u097F]+[\. ]+\d.\d)","",hn_val)

hn_val=re.sub("\d.\d मिलियन से लेकर \d.\d","",hn_val)


s=re.sub("\n","",hn_val)



hl=[]



tmp1=re.match('(.*?)'+chapter_no+'\.\d',s)
Value_Marati_Start=tmp1.group(0)
s1=s[len(Value_Marati_Start)-(len(chapter_no)+2):]

hl.append(Value_Marati_Start)

i=0
while(i<1):
	stvl=s1[0:s1.find(" ")]
	s1=s1[s1.find(" "):]
	try:
		val2=re.match('(.*?)'+chapter_no+'\.\d',s1).group(0)
	except:
		i=2
		break
	str_val=stvl+" "+val2
	print(str_val)
	hl.append(str_val)
	s1=s1[len(re.match('(.*?)'+chapter_no+'\.\d',s1).group(0))-(len(chapter_no)+2):]

hl.append(s1)


print(len(el),len(hl))"""
#for i in el:

#  print(i)
#print(len(el),len(hl))
#sorted(hl)
#sorted(el)
















#import re

#English_Book_txt=re.sub("\d\.\d","",en_val)

#English_Book_txt
#Marati_Book_Txt=re.sub("\d\.\d","",hn_val)

Marati_Book_Txt
#English_Book_txt=re.sub("\.\d","",English_Book_txt)
#Marati_Book_Txt=re.sub("\.\d","",Marati_Book_Txt)
#English_Book_txt=re.sub("\d\.",".",English_Book_txt)
#Marati_Book_Txt=re.sub("\d\.","।",Marati_Book_Txt)
#print(English_Book_txt)
#English_Book_txt=re.sub("\d\.",".",English_Book_txt)
#Marati_Book_Txt=re.sub("\d।","।",Marati_Book_Txt)


#print(English_Book_txt)
#print(Marati_Book_Txt)
#s = re.sub(r"[-()\"#/@;:<>{}`+=~|,]", "", English_Book_txt)
#s
#s1 = re.sub(r"[-()\"#/@;:<>{}`+=~|,]", "", Marati_Book_Txt)
#<div>
#s1=re.sub("\n","",s1)
#s1
#s=re.sub("\d","",s)

#s1=re.sub("\d","",s1)
#print(s)
#print(s1)
#s = re.sub(r"[!?]", ".", s)
#s1 = re.sub(r"[!?]", "।", s1)

len(list(s1.split("।")))
#s11=list(s.split("."))
#s21=list(s1.split("।"))
#s11.append("None")
#s11.append("None")
#s11.append("None")
#import pandas as pd
#df=pd.DataFrame()
#df["English"]=s11
#df["Marati"]=s21
#df.to_csv("Changed.csv")
#!pip install googletrans
"""from googletrans import Translator"""
"""translator = Translator()
translations = translator.translate(s11, dest='hi')
#translator.translate('hello.', dest='hi')"""
"""English_text=[]
Marati_text=[]"""
"""for translation in translations:
  English_text.append(translation.origin)
  Marati_text.append(translation.text)
  #print(translation.origin, ' -> ', translation.text)"""

#df1=pd.DataFrame()
#df1["English"]=English_text
#df1["Marati"]=Marati_text
#df1.to_csv("translated.csv")
#Marati_text


"""
English_Book_txt=re.sub("\.\d","",English_Book_txt)
Marati_Book_Txt=re.sub("\.\d","",Marati_Book_Txt)

English_Book_txt=re.sub("\d\.",".",English_Book_txt)
Marati_Book_Txt=re.sub("\d\.","।",Marati_Book_Txt)
print(English_Book_txt,Marati_Book_Txt)

English_Book_txt=English_Book_txt.split(".")
Marati_Book_Txt=Marati_Book_Txt.split("।")
print(English_Book_txt)
print(Marati_Book_Txt)

print(len(English_Book_txt),len(Marati_Book_Txt))



    
l1=list(English_Book_Txt.split('.'))
l2=list(Marati_Book_Txt.split('।'))
print(len(l1),len(l2))
print(l2)
print(l1)
l = []
for x in l1:
    if not x.isdigit():
      l.append(x)
print(len(l))
return [l1,l1]"""
    

#l1=read_file1("5156_1_Untitled-2.txt","5401_1.txt")
#print(l1)
"""File1_Open=open("/content/5215_1_Chapter 1.txt",'r')
English_Txt_List=[]
for Line_in_File in File1_Open:
    English_Txt_List.append(Line_in_File)
#print(len(English_Txt_List))
#English_Book_Txt= ' '.join(map(str, English_Txt_List))

print(English_Txt_List)

en_val=' '.join(map(str, English_Txt_List))
#en_val=str(English_Txt_List)
type(en_val)

chapter_No=str(input("Enern The Chapter No :"))

import re
en_val=re.sub("(TABLE \d.\d)","",en_val)

en_val=re.sub("(\d.\d-\d.\d)","",en_val)

en_val=re.sub("[a-zA-Z]+[\. ]+\d.\d","",en_val)
en_val=re.sub("([a-zA-Z]+[\. ]+\d.\d)","",en_val)

s=en_val

type(s)

print(s)

el=[]

Value_English_Start=re.findall('^(.*?)'+chapter_No+'\.\d',s)
s1=s[len(Value_English_Start):]

el.append(Value_English_Start)

i=0
while(i<1):
	stvl=s1[0:s1.find(" ")]
	s1=s1[s1.find(" "):]
	try:
		val2=re.findall('^(.*?)'+chapter_No+'\.\d',s1)
	except:
		i=2
		break
	str_val=stvl+" "+' '.join(map(str, val2))
	print(str_val)
	el.append(str_val)
  
	s1=s1[len(re.findall('^(.*?)'+chapter_No+'.\d',s1)):]

el.append(s1)

el

File2_Open=open("/content/5393_1.txt",encoding="utf8")
Marati_Txt_List=[]
for Line_in_File in File2_Open:
    Marati_Txt_List.append(Line_in_File)


hn_val=' '.join(map(str, Marati_Txt_List))
import re

hn_val=re.sub("[\u0900-\u097F]+[\. ]+\d.\d","",hn_val)
hn_val=re.sub("([\u0900-\u097F]+[\. ]+\d.\d)","",hn_val)

hn_val=re.sub("\d.\d मिलियन से लेकर \d.\d","",hn_val)


s=re.sub("\n","\n",hn_val)



hl=[]
tmp1=re.match('(.*?)\n\d\.\d',s)
Value_Marati_Start=tmp1.group(0)
s1=s[len(Value_Marati_Start)-3:]
hl.append(Value_Marati_Start)
i=0
while(i<1):
	stvl=s1[0:s1.find(" ")]
	s1=s1[s1.find(" "):]
	try:
		val2=re.match('(.*?)\d.\d',s1).group(0)
	except:
		i=2
		break
	str_val=stvl+" "+val2
	print(str_val)
	hl.append(str_val)
	s1=s1[len(re.match('(.*?)\d.\d',s1).group(0))-3:]
hl.append(s1)

print(len(el),len(hl))

el

hl"""