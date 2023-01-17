!tar xzvf ../input/spf.tar.gz  -C /tmp
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import  datetime
date_depart=datetime.datetime.now()
duree_max=datetime.timedelta(hours=5,minutes=45)
date_limite= date_depart+duree_max
# Any results you write to the current directory are saved as output.
import xml.parsers.expat
import codecs
import unicodedata
import copy
import functools
import lzma

names=set()
courant=None
elements=[]
pile=[]
textes=""
keys=set()
# 3 handler functions
def start_element(name, attrs):
    global courant
    global keys
    global pile
    names.add(name)
    courant=attrs.copy()
    courant['el_type']=name
    courant['texte']=""
    elements.append(courant)
    pile.append(courant)
    keys.update(courant.keys())
def end_element(name):
    global pile
    global courant
    if len(pile)>0:
        pile.pop()
        if len(pile)>0:
            courant=pile[-1]
    
     
def char_data(data,filedesc):
    global textes
    global courant
    if courant is not None:
        courant['texte']+=data
    textes+=data
    print(data,file=filedesc)
    
def make_parser(filedesc):
    p = xml.parsers.expat.ParserCreate()

    p.StartElementHandler = start_element
    p.EndElementHandler = end_element
    p.CharacterDataHandler = functools.partial(char_data,filedesc=filedesc)
    return p

with lzma.open("texte.txt.xz","wt") as f_text:
    p=make_parser(f_text)
    pos=0
    print("debut")
    with open("/tmp/final_SPF_2.xml","rb") as f:
        while datetime.datetime.now()<date_limite:
            try:
                p.ParseFile(f)
            except xml.parsers.expat.ExpatError as e:
                posf=f.tell()
                #print(e.code,e,posf,p.ErrorByteIndex ,p.CurrentByteIndex)  
                if p.ErrorByteIndex>0:
                    f.seek(pos+p.ErrorByteIndex )
                    datal=b""
                    while True:
                        datal1=f.readline()
                        datal+=datal1
                        if b"<" in datal1:
                            break

                    data=datal.split(b"<",1)[0]
                    avance=len(data)
                    data=codecs.decode(data,encoding='utf8',errors='ignore')
                    data="".join(c for c in data if unicodedata.name('\x05',None) is not None)
                    textes+=data

                    if avance>0:
                        pos+=avance
                    else:
                        pos+=p.ErrorByteIndex 
                    f.seek(pos)
                else:
                    f.seek(pos)
                    datal=f.readline()
                    data=datal.split(b"<",1)[0]
                    if len(data)>0:
                        pos+=len(data)
                        textes+=codecs.decode(data,encoding='utf8',errors='ignore')
                        f.seek(pos)
                    elif datal[0]==b"<":
                        print (datal)
                        raise ValueError
                    else:
                        print (datal)
                        raise ValueError
                        pos=pos+1
                        f.seek(pos)







                p=p=make_parser(f_text)

            except ValueError:
                break
    articles=[e for e in elements if len(e['texte'])>0]

pd.DataFrame(elements).to_csv("elements.csv.xz",compression="xz",chunksize=4096)
!rm final_SPF_2.xml
import json
k=set((e['el_type'],)+tuple(e.keys()) for e in elements)
k=sorted(k)
el_s=[e for e in elements if e['el_type']=='s' ]
el_s1=pd.DataFrame(([tuple(e.values()) for e in elements if e['el_type']=='s' ]))
el_utt=[e for e in elements if e['el_type']=='utt' ]
el_utt1=pd.DataFrame(([tuple(e.values()) for e in elements if e['el_type']=='utt' ]))
el_s1.columns=pd.Index(k[1][1:])
el_utt1.columns=pd.Index(k[2][1:])
del el_s1["el_type"]
del el_utt1["el_type"]

el_utt1.to_csv("posts.csv")
el_s1.to_csv("threads.csv")
el_utt1

df=pd.DataFrame(elements)
df.to_csv("elements.csv.xz",compression="xz",chunksize=4096)