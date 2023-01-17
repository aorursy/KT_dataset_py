

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df=pd.read_excel('../input/cian_sale_cost.xlsx')

df


def tel(x):

    x=''.join([i for i in str(x) if i.isdigit()])

    a=[]

    while len(x)>0:

        if x[0]=='7':

            x=x[1:]

        a+=['7'+x[:10]]

        x=x[10:]

    

   

    return a



df['tel_done']=df['tel'].map(lambda x: tel(x) )



       





df0=pd.DataFrame(df['tel_done'].tolist(), index=df.index)    

df0.columns=['tel_'+str(i) for i in df0.columns]



df=df0.join(df) 

df
