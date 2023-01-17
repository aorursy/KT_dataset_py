import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
draft = pd.read_csv('../input/draft78.csv',encoding ='utf-8')
draft.head()
season = pd.read_csv('../input/season78.csv',encoding ='utf-8')
season.head()
#relationship between 'Pick' and 'Yrs'.
sns.jointplot(data=draft,x='Pick',y='Yrs',kind = 'hex',color ='r',size =8)
data = season.merge(draft,on = 'Player', how = 'inner')
data_grouped = data.groupby(['Player','Draft'])['WS'].mean()
data_grouped = data_grouped.reset_index()
f, ax= plt.subplots(figsize = (20, 10))
sns.boxplot(data=data_grouped,x='Draft',y='WS')