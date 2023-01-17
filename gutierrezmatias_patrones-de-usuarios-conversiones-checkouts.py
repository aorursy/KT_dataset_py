import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython.core.display import display, HTML
plt.style.use('default')
display(HTML("<style>.container { width:90% !important; }</style>"))
df = pd.read_csv('../input/events.csv',low_memory=False)
#convieto las fechas a formato fecha 
df['timestamp']= df['timestamp'].astype('datetime64')
dfa = df.loc[df['event']=='conversion'].groupby(['person','sku'])['event'].count()
dfa= dfa.reset_index()
dfb= df.loc[df['event']=='checkout'].groupby(['person','sku'])['event'].count()
dfb= dfb.reset_index()
result=pd.merge(dfa,dfb, on = ['person', 'sku'], how='inner')
result.rename(columns={"event_x": "conversion"}, inplace= True)
result.rename(columns={"event_y": "checkout"}, inplace= True)
result.sort_values(by= ['conversion','checkout'], ascending = False)
result.fillna(0)
result.sort_values('conversion', ascending = False).head(10)