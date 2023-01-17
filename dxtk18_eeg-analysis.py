# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd 

import os

import random

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

filelist=[]

import os

for dirname, _, filenames in os.walk('/kaggle/input/Alcoholics/SMNI_CMI_TRAIN/Train/'):

    for filename in filenames:

        ((os.path.join(dirname, filename)))

        

        



scsv = pd.concat( [ pd.read_csv('/kaggle/input/Alcoholics/SMNI_CMI_TRAIN/Train/'+f) for f in filenames ] )





        

#         filelist=list.append(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
scsv.shape
s1a=scsv.loc[((scsv['subject identifier']=='a') & (scsv['matching condition']=='S1 obj'))]

s1c=scsv.loc[((scsv['subject identifier']=='c') & (scsv['matching condition']=='S1 obj'))]



s12ma=scsv.loc[((scsv['subject identifier']=='a') & (scsv['matching condition']=='S2 match'))]

s12mc=scsv.loc[((scsv['subject identifier']=='c') & (scsv['matching condition']=='S2 match'))]



s12nma=scsv.loc[((scsv['subject identifier']=='a') & (scsv['matching condition']=='S2 nomatch,'))]

s12nmc=scsv.loc[((scsv['subject identifier']=='c') & (scsv['matching condition']=='S2 nomatch,'))]





dfm1a = pd.DataFrame(columns = ['SN', 'SP', 'meanS']) 

dfd1a=s1a.groupby(['sample num', 'sensor position'])

for name,group in dfd1a:

    dfm1a=dfm1a.append({'SN' : str(name[0]), 'SP' : str(name[1]), 'meanS' : group['sensor value'].mean()},ignore_index = True) 





dfm1c = pd.DataFrame(columns = ['SN', 'SP', 'meanS']) 

dfd1c=s1c.groupby(['sample num', 'sensor position'])

for name,group in dfd1c:

    dfm1c=dfm1c.append({'SN' : str(name[0]), 'SP' : str(name[1]), 'meanS' : group['sensor value'].mean()},ignore_index = True) 







    

    

    

    

    

    

dfm2a = pd.DataFrame(columns = ['SN', 'SP', 'meanS']) 

dfd2a=s12ma.groupby(['sample num', 'sensor position'])

for name,group in dfd2a:

    dfm2a=dfm2a.append({'SN' : str(name[0]), 'SP' : str(name[1]), 'meanS' : group['sensor value'].mean()},ignore_index = True) 





dfm2c = pd.DataFrame(columns = ['SN', 'SP', 'meanS']) 

dfd2c=s12mc.groupby(['sample num', 'sensor position'])

for name,group in dfd2c:

    dfm2c=dfm2c.append({'SN' : str(name[0]), 'SP' : str(name[1]), 'meanS' : group['sensor value'].mean()},ignore_index = True) 



    

    

    

    

    

    

    

    

    

dfm3a = pd.DataFrame(columns = ['SN', 'SP', 'meanS']) 

dfd3a=s12nma.groupby(['sample num', 'sensor position'])

for name,group in dfd3a:

    dfm3a=dfm3a.append({'SN' : str(name[0]), 'SP' : str(name[1]), 'meanS' : group['sensor value'].mean()},ignore_index = True) 





dfm3c = pd.DataFrame(columns = ['SN', 'SP', 'meanS']) 

dfd3c=s12nmc.groupby(['sample num', 'sensor position'])

for name,group in dfd3c:

    dfm3c=dfm3c.append({'SN' : str(name[0]), 'SP' : str(name[1]), 'meanS' : group['sensor value'].mean()},ignore_index = True) 







x1=np.array(dfm1a['SP'].unique())

y1=np.array(dfm1a['SN'].unique())











a=0

b=63

ary2_1a=[]

ary2_1c=[]

ary2_2a=[]

ary2_3a=[]

ary2_2c=[]

ary2_3c=[]



for i in range(256): 

    ary1a=np.array(dfm1a['meanS'].loc[a:b])

    ary1c=np.array(dfm1c['meanS'].loc[a:b])

    ary2a=np.array(dfm2a['meanS'].loc[a:b]) 

    ary2c=np.array(dfm2c['meanS'].loc[a:b])

    ary3a=np.array(dfm3a['meanS'].loc[a:b]) 

    ary3c=np.array(dfm3c['meanS'].loc[a:b])    

    a=a+64

    b=b+64

    ary2_1a.append(ary1a)

    ary2_1c.append(ary1c)

    ary2_2a.append(ary2a)

    ary2_2c.append(ary2c)

    ary2_3a.append(ary3a)

    ary2_3c.append(ary3c)

    



    







print('Alcoholic group average : '+str(dfm1a['meanS'].mean()))

print('Control group average : '+str(dfm1c['meanS'].mean()))



fig = go.Figure(data=[go.Surface(z=ary2_1a, x=x1, y=y1)])

fig.update_layout(title='Group:a  S1', autosize=False,

                  width=800, height=800,

                  margin=dict(l=65, r=50, b=85, t=90))

fig.show()



fig2 = go.Figure(data=[go.Surface(z=ary2_1c, x=x1, y=y1)])

fig2.update_layout(title='Group:c  S1', autosize=False,

                  width=800, height=800,

                  margin=dict(l=65, r=50, b=85, t=90))

fig2.show()
print('Alcoholic group average : '+str(dfm2a['meanS'].mean()))

print('Control group average : '+str(dfm2c['meanS'].mean()))

fig3 = go.Figure(data=[go.Surface(z=ary2_2a, x=x1, y=y1)])

fig3.update_layout(title='Group:a  S2 MATCH', autosize=False,

                  width=800, height=800,

                  margin=dict(l=65, r=50, b=85, t=90))

fig3.show()







fig4 = go.Figure(data=[go.Surface(z=ary2_2c, x=x1, y=y1)])

fig4.update_layout(title='Group:c  S2 MATCH', autosize=False,

                  width=800, height=800,

                  margin=dict(l=65, r=50, b=85, t=90))

fig4.show()
print(dfm3a.head())
print('Alcoholic group average : '+str(dfm3a['meanS'].mean()))

print('Control group average : '+str(dfm3c['meanS'].mean()))





fig5 = go.Figure(data=[go.Surface(z=ary2_3a, x=x1, y=y1)])

fig5.update_layout(title='Group:a  S2 NO MATCH', autosize=False,

                  width=800, height=800,

                  margin=dict(l=65, r=50, b=85, t=90))

fig5.show()







fig6 = go.Figure(data=[go.Surface(z=ary2_3c, x=x1, y=y1)])

fig6.update_layout(title='Group:c  S2 NO MATCH', autosize=False,

                  width=800, height=800,

                  margin=dict(l=65, r=50, b=85, t=90))

fig6.show()