import numpy as np

import pandas as pd

pd.set_option('display.max_columns',0)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')

import gc
pth = '../input/datathon-belcorp-prueba/'

#prods = pd.read_csv(pth+'maestro_producto.csv',index_col=0)

cnslt = pd.read_csv(pth+'maestro_consultora.csv',index_col=0)

camps = pd.read_csv(pth+'campana_consultora.csv',index_col=0).sort_values(by=['IdConsultora','campana']).reset_index(drop=True)

hists = pd.read_csv(pth+'dtt_fvta_cl.csv').sort_values(by=['idconsultora','campana']).reset_index(drop=True)

submt = pd.read_csv(pth+'predict_submission.csv')
indx = pd.MultiIndex.from_tuples(list(zip(camps['campana'],camps['IdConsultora'])),names=['index','idconsultora'])

hist = pd.DataFrame(camps['Flagpasopedido'].values, index=indx).unstack().T.reset_index().iloc[:,1:]
X = hist[(hist.iloc[:,1:-1].isna().mean(axis=1)<1) & (~hist.iloc[:,-1].isna())].iloc[:,:-1]

y = hist[(hist.iloc[:,1:-1].isna().mean(axis=1)<1) & (~hist.iloc[:,-1].isna())].iloc[:,-1]
from sklearn.linear_model import LogisticRegression



logit = LogisticRegression()

logit.fit(X[X[[201807,201903,201904]].isna().sum(axis=1)<1].iloc[:,1:].fillna(0),y[X[[201807,201903,201904]].isna().sum(axis=1)<1])
logit.coef_[0]
def weighted_mean(x):

    weights = [0,-0.02859811,  0.05876883,  0.12493859,  0.1913275 ,  0.04148779,

        0.12260066,  0.12621394,  0.22652612,  0.15365354,  0.39404203,

        0.31334125,  0.04162932,  0.16122161,  0.31743629,  0.22749796,

        1.25904213,  2.26523606][-len(x):]

    return np.average(x,weights=weights)
submt = pd.merge(submt['idconsultora'],

         camps.rename(columns={'IdConsultora':'idconsultora'}).groupby('idconsultora').agg({'Flagpasopedido':[weighted_mean]}).reset_index(),

         on = 'idconsultora', how = 'left')

submt.columns = ['idconsultora','flagpasopedido']
submt['flagpasopedido'] = np.where(submt['idconsultora'].isin(cnslt[cnslt['campanaultimopedido']<201907]['IdConsultora'].values),0,submt['flagpasopedido'])

submt['flagpasopedido'] = np.where(submt['idconsultora'].isin(cnslt[cnslt['campanaultimopedido']==201907]['IdConsultora'].values),1,submt['flagpasopedido'])
submt.head()
submt['flagpasopedido'] = submt['flagpasopedido'].clip(0,1)
submt.to_csv('submission.csv',index=None, sep=',',encoding='utf-8')