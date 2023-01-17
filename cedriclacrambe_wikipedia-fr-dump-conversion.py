# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import json,requests,glob
import gensim
import zipfile
import lzma,bz2
# Any results you write to the current directory are saved as output.


def fake_tk(t,*args):
  return t.split(" " )

dump_path=glob.glob("../input/**/frwiki*.xml.bz2",recursive=True)[0]
wiki = gensim.corpora.wikicorpus.WikiCorpus(dump_path,lemmatize=False ,dictionary ={} , lower=False ,
                  token_min_len=1,tokenizer_func =fake_tk )

n=1
with zipfile.ZipFile("frwiki.zip", mode="w",compression=zipfile.ZIP_DEFLATED) as zf:
    with bz2.open("frwiki.txt.bz2","wt") as textfile:
          for text in wiki.get_texts():
            text_s=' '.join(text)
            text=bytes(text_s, 'utf-8').decode('utf-8') + '\n'   
            zf.writestr(f"article{n}.txt",text)
            st=os.statvfs(".")
            textfile.write(text_s)
            if st.f_bavail*st.f_bsize<1e7:
                break



            if n%5000==0:
              print(n)
              print(text[:300])      
            elif n%1000==0:
              print(n)
            n+=1