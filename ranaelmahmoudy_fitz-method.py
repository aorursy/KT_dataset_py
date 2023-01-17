!pip install pdfminer
!pip install --upgrade pymupdf
!pip install textdistance
import numpy as np 
import pandas as pd 
import sys
import re
#sys.path.append(os.path.abspath(os.path.dirname('/kaggle/input/curriculum-vitae-data-pdf/pdf')))
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, sent_tokenize

import os
import fitz
from operator import itemgetter

folderpath='/kaggle/input/curriculum-vitae-data-pdf/pdf'
test='/kaggle/input/curriculum-vitae-data-pdf/pdf/88.pdf'




def pdf2text(path):
    
    text=list()
    spans=list()
    doc = fitz.open(path)
    for page in doc:
        blocks = page.getText("dict")["blocks"]
       
        for b in blocks: 
            if b['type'] == 0: 
                for l in b["lines"]: 
                    for s in l["spans"]:
                        text.append(s['text'])
                        spans.append(s)
                       
    return spans,text


def get_pdf(folderpath):
    
    final=list()
    
    wholespans=list()
    wholetext=list()
    nFiles = len(os.listdir(folderpath))
 
    for f in os.listdir(folderpath):
        filepath = os.path.join(folderpath, f)
        
        try:
            
            spans,text=pdf2text(filepath)
            
            wholespans.append(spans)
            wholetext.append(text)
            final.append(get_headers(spans))
            
        except:
            continue
            
    print("No. of files = ", nFiles) 
    print("No. of parsed files = ", len(wholetext)) 
    return final ,wholetext ,wholespans

#needs str casting#
def get_email(final):
    email = None
    match = re.search(r'[\w\.-]+@[\w\.-]+',final)
    if match is not None:
        email = match.group(0)
    return email

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    return tokens
    


import re
import textdistance
import fitz
import xml.etree.cElementTree as ET


headings=['objective','aim','expertise','academia','profile','credentials','experience','qualification','knowledge','qualities','purview','assets','proficiency','personal','languages','projects','training','achievements','activities','hobbies','interest','declaration','industrial visit']

def clean_data(text):
    text=re.sub("[^a-zA-Z0-9 ]","",text)
    text=text.lower().lstrip().rstrip()
    return text

def onetime():
   
    s=set()
    f= open("/kaggle/input/allskills/allskills.txt","r")
    result=f.read().splitlines()
    for r in result:
        text=re.sub("[^a-zA-Z ]","",r)
        text=text.lower().lstrip().rstrip()
        s.add(text)
        
    f1= open("skills_ontology.txt","w+")
    for skill in s:
        f1.write(skill +'\n')
    return s

def get_headers(spans):
    headers=list()
    for s in spans:
        if "Bold" in s['font'] or "Italic" in s['font'] or "Black" in s['font'] :
            text=re.sub("[^a-zA-Z ]","",s['text'])
            text=text.lower().lstrip().rstrip()
            if len(text.split())<=3 and text != "" :
                add=text
                headers.append(add)
            
    return headers

def filter_headers(headers):
    skills=set()
    others=set()
  
    for h in headers:
        for i in h:
            if "skills" in i:
                skills.add(i)
                
            else:
                for x in headings:
                    if x in i:
                        others.add(i)
                        
    return skills,others                    

def tofile(skills,others):
    f= open("skills.txt","w+")
    f1= open("others.txt","w+")
    for skill in skills:
        f.write(skill +'\n')
    for other in others:
        f1.write(other +'\n')
        
def outoffile():
    f=open("skills.txt","r")
    result=f.read().splitlines()
    f1=open("others.txt","r")
    result1=f1.read().splitlines()
    
    return result,result1 

    
def real_headers(header,skills,others):
    final=set()
    real=list()
    real2=list()
    for h in header:
        weight=list()
        weight2=list()
        ##for skills ##
        for i in range (len(skills)):
            #weight.append((h,textdistance.jaro_winkler(h,skills[i])))
            weight.append((h,textdistance.ratcliff_obershelp(h,skills[i])))
            if i== len(skills)-1:
                weight.sort(key=lambda x: x[1],reverse = True)
                real.append(weight[0])
        ##for other headers ##        
        for j in range (len(others)):
            weight2.append((h,textdistance.ratcliff_obershelp(h,others[j])))
            if j== len(others)-1:
                weight2.sort(key=lambda y: y[1],reverse = True)
                real2.append(weight2[0]) 
        
    real.sort(key=lambda x: x[1],reverse = True)
    real2.sort(key=lambda y: y[1],reverse = True)  
    final.add(real[0][0])
    skill=real[0][0]
    for k in real2:
        if k[1]>0.5:
            final.add(k[0])
        
    return skill,final



def indices(doc,headers):
    indexes=list()
   
    for h in headers:
        for i in range(len(doc)):
            if h in doc[i]['text'].lower():
                if "Bold" in doc[i]['font'] or "Italic" in doc[i]['font'] or "Black" in doc[i]['font'] :
                    indexes.append((h,i))
                    break
                
    indexes.sort(key=lambda y: y[1])         
    return indexes


def extract_content(indexes,docx,skill):
    content=list()
    
    for i in range(len(indexes)):
        if i==len(indexes)-1:
            content.append((indexes[i][0],clean_data(str(docx[indexes[i][1]+1:]))))
            break
            
        content.append((indexes[i][0],clean_data(str(docx[indexes[i][1]+1:indexes[i+1][1]]))))
      
    return content

def run(result4,result5,doc,spans,testing):
    resultt=list()
    skills=list()
    others=list()
    for i in range(len(doc)):
       
       
        result6=set()
        index=list()
        
        skill=""
        
        try:
            
            #testing=get_headers(spans[i])
            skill,result6=real_headers(testing[i],result4,result5)
            skills.append(skill)
            #others.append(result6)
            index=indices(doc,result6)
            resultt=extract_content(indexes,docx,skill)

        except:
            continue
    
    return resultt

result,doc,spans=get_pdf(folderpath)
result2,result3=filter_headers(result) 
tofile(result2,result3)
result4,result5=outoffile()
#resultt=run(result4,result5,doc[0],spans,result)
skill,result6=real_headers(result[2],result4,result5)
index=indices(spans[2],result6)
resultt=extract_content(index,doc[2],skill)
print(resultt)
print(index)
#l=onetime()
#print(l)