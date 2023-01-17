keyword='养发|沐发|沐头|洗发|涂之立生|润发|立生|脱发|长发|发落'

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
print(os.listdir("../input/chinesemedicalbook/"))
filelist=sorted(glob.glob("../input/chinesemedicalbook/ChineseMedicalBook/*.txt"))
def agent(filelist):
    reader=pd.read_csv(filelist,
                       header=None,
                       sep='\r',
                       engine='python',
                       error_bad_lines=False
                      )
    reader.columns=['data']
    reader['book']=reader['data'][0]
    reader['item']=reader['data'].apply(lambda x: x if '<篇名>' in x else None  )
    reader['item']=reader['item'].fillna(method='ffill')
    reader['keyWords']=reader['data'].str.contains(keyword)
    temp=reader[reader['keyWords']==True]
    return temp[['data','item','book']]
tempall=pd.DataFrame(agent(filelist[0]))
for i in range(681):
    temp=agent(filelist[i])
    tempall=tempall.append(temp,ignore_index=True,)
tempall.to_csv('output.csv',index=None)
tempall