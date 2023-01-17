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
df=df.iloc[:, : 10] #删掉最后一列“isFlaggedFraud”

df.head()
df.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])


df.groupby(['type', 'isFraud']).count()['step'] #5种交易类型的欺诈记录与正常记录数目统计
df_corr = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']]



data = [

    go.Heatmap(

        z=df_corr.corr().values,

        x=df_corr.columns.values,

        y=df_corr.columns.values,

        colorscale='Viridis',

        text = True ,

        opacity = 1.0

        

    )

]





layout = go.Layout(

    title='Pearson Correlation of all numeric features',

    #xaxis = dict(ticks='', nticks=36),

    #yaxis = dict(ticks='' ),

    #width = 900, height = 700,

    

)





fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='labelled-heatmap')
plt.figure(figsize=(12,8))

b1=sns.boxplot(x = 'isFraud', y = 'amount', data = df[df.amount < 1e5])

b1.set_xlabel("isFraud",fontsize=20) #字体调大便于论文展示

b1.set_ylabel("amount",fontsize=20)

b1.tick_params(labelsize=17)

sns.plt.show()
plt.figure(figsize=(12,8))

b1=sns.boxplot(hue = 'isFraud', x = 'type', y = 'amount', data = df[df.amount < 1e5])

b1.set_xlabel('type', fontsize=20) #字体调大便于论文展示

b1.set_ylabel('amount',fontsize=20)

b1.tick_params(labelsize=17)

sns.plt.show()
df = df.sample(636262) #数据量太大计算太慢

plt.figure(figsize=(12,8))

b1 = sns.pairplot(df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']], hue='isFraud')

b1.set_xlabel('type', fontsize=20) #字体调大便于论文展示

b1.set_ylabel('amount',fontsize=20)

b1.tick_params(labelsize=17)
11