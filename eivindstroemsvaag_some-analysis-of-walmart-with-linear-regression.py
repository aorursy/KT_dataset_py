#importing lib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
import plotly.graph_objs as go
init_notebook_mode()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
stocks = pd.read_csv('../input/fundamentals.csv')
stocks.head()
top_rev = stocks.groupby(by='Ticker Symbol').agg({'Total Revenue':sum})
g = top_rev['Total Revenue'].nlargest(5)
next = [Bar(
            y=g,
            x=g.keys(),
            marker = dict(
            color = 'lightsteelblue'
            ),
            name = "Contractor's amount earned per project"
    )]
layout1 = go.Layout(
    title="Top 10 Exporters",
    xaxis=dict(
        title='Company',
        titlefont=dict(
            family='Courier New, monospace',
            size=30,
            color='#7f7f7f'
               )
    ),
    yaxis=dict(
        title='Total Revenue',
        titlefont=dict(
            family='Courier New, monospace',
            size=22,
            color='#7f7f7f'
        )
    )
)
myFigure2 = go.Figure(data = next, layout = layout1)
iplot(myFigure2)
wmt = stocks[stocks['Ticker Symbol']=='WMT']
#sns.distplot(Aqua['Generation'],bins=28,kde=False,color='red')
wmt.head()
gir = ['Total Equity',
       'Total Revenue',
       'Accounts Payable',
       'Accounts Receivable',
      'Cost of Revenue',
      'Profit Margin',
      'Sale and Purchase of Stock',
      'Earnings Per Share']
tip = np.corrcoef(wmt[gir].values.T)

wmt.info()
sns.set(font_scale = 0.9)
map = sns.heatmap(tip, cbar = True,
                  cmap="YlGnBu",
                  annot = True, 
                  square= True,
                  fmt = '.1f',
                  annot_kws = {'size':11}, 
                 yticklabels = gir,
                 xticklabels = gir)
#Setting the arrays 

n = wmt['Total Revenue']
m = wmt[['Accounts Payable',
         'Cost of Revenue',
         'Sale and Purchase of Stock'
    
]]
# Splitting the into sets of training and test.
train,test,train_label,test_label=train_test_split(m,n,test_size=0.33,random_state=101)
# The model 
Linear = LinearRegression(fit_intercept=True)
mo = Linear.fit(train,train_label)
predi = mo.predict(test)
print(r2_score(test_label,predi))
print (Linear.intercept_)
# Finding the coefficient. (value of 1 unit increase) 
coef = pd.DataFrame(Linear.coef_,m.columns,columns=['Coefficient'])
coef

