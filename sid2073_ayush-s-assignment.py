# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    x=pd.DataFrame()

    li=[]

    for filename in filenames:

        li.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
x=pd.DataFrame()

for i in li:

    y=pd.read_json(i,lines=True)

    x=pd.concat([x,y],axis=0,sort=False)

    
x.head()
x['msg']
sen='asjkdhajkhappyasdahHappysadSaDSADsAd'
def happy_or_sad(x):

    if 'happy' in x.lower():

        return 1

    elif 'sad' in x.lower():

        return 0

    else:

        return -1
x['label']=x['msg'].apply(happy_or_sad)
final=x[['msg','label']]

#final.to_csv('final.csv')
final['msg']=final['msg'].apply(lambda y:y.lower())

final['msg']=final['msg'].apply(lambda y:y.replace('sad',""))

final['msg']=final['msg'].apply(lambda y:y.replace('happy',""))
happy=final[final['label']==1]

sad=final[final['label']==0]

none=final[final['label']==-1]