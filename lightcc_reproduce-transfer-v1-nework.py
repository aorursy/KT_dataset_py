import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')



# subsample

df = pd.read_csv("../input/PS_20174392719_1491204439457_log.csv")#, nrows=int(1e6))

df = df.iloc[:, : 10] #删掉最后一列“isFlaggedFraud”

df = df.query('type=="TRANSFER"')#仅保留TRANSFER类型的记录

df = df[['step', 'amount', 'nameOrig', 'nameDest']]

df.head()
def  countAgents():

    agentsO = set(df['nameOrig'].tolist())

    agentsD = set(df['nameDest'].tolist())

    print('转账发起人数：', len(agentsO))

    print('转账接收人数：', len(agentsD))

    agents = agentsO | agentsD

    print('转账发起和接收的总人数：', len(agents))

    return agents



agents = countAgents()