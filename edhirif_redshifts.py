import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls
df=pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
df.head()
df['class'].value_counts()
df_ga = df[df['class']=='GALAXY']
df_st = df[df['class']=='STAR']
df_qo = df[df['class']=='QSO']
sns.boxplot( x=df["class"], y=df["redshift"] );
