# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly as py

import plotly.graph_objects as go

import ipywidgets as widgets

import plotly.express as px

import numpy as np

from datetime import timedelta

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
df = pd.read_excel('../input/poker7292020/PokerNow Data Clean 2020-07-29.xlsx', sheet_name='Clean Log')





df.drop(columns=['Unnamed: 0'], inplace=True)

df = df.set_index(('Timestamp'))

tbh = pd.pivot_table(df, index=["Player's Name"], values=['Hands'],aggfunc=[pd.Series.nunique], fill_value=0)

tbl = pd.pivot_table(df, index=["Player's Name"], values=['Winning Hand','Played?','Winning Amount'],aggfunc=['count',pd.Series.nunique,'sum','max','mean','last'], fill_value=0)

tbl =tbl[[  (      'count',        'Played?'),

            (      'count',   'Winning Hand'),

            (        'sum', 'Winning Amount'),

            (        'max', 'Winning Amount'),

            (       'mean', 'Winning Amount')]]





tbl.columns = ['No. Hands Played','Total Wins','Total Amount Won','Highest Amt Won','Avg Amt Won']

tbl['Total Amount Won'] = tbl['Total Amount Won'].map('${:,.2f}'.format)

tbl['Highest Amt Won'] = tbl['Highest Amt Won'].map('${:,.2f}'.format)

tbl['Avg Amt Won'] = tbl['Avg Amt Won'].map('${:,.2f}'.format)

tbl["Player's Winning %"] = tbl['Total Wins']/tbl['No. Hands Played']

tbl["Player's Winning %"] = tbl["Player's Winning %"].astype(float).map("{:.2%}".format)



tbl = tbl.reset_index()



tbh = tbh.reset_index()



tbh.columns = ["Player's Name", "Total Hands"]

tbl = pd.merge(tbl,tbh[["Player's Name","Total Hands"]],on="Player's Name")

tbl['Hands Played %'] = (tbl['No. Hands Played']/tbl['Total Hands']).astype(float).map("{:.2%}".format)

tbl = tbl.reindex(columns=["Player's Name",'No. Hands Played','Total Hands','Hands Played %','Total Wins',"Player's Winning %",'Total Amount Won','Highest Amt Won','Avg Amt Won'])

tbl
dfavgamt = df[(df["Action"].notnull()) & (df["Action"]!="collected")&(df["Action"]!="posts")&(df["Action"]!="folds")&(df["Action"]!="checks")]

dfavgamt = pd.pivot_table(dfavgamt, index=["Player's Name","Action"], values=['Amount'],aggfunc=['count',pd.Series.nunique,'max','mean'], fill_value=0)

dfavgamt=dfavgamt.reset_index()

dfavgamt.columns = ["Player's Name", "Action", "HandsC", "Action Count","Max Amount ","Avg Amt"]



dfavgamt['Avg Amt']=dfavgamt['Avg Amt'].map('{:,.2f}'.format)

dfavgamt.drop(columns=['HandsC'], inplace=True)

dfavgamt
tblpcount = pd.pivot_table(df, index=["Position","Player's Name"],values=['Played?','Hands'],aggfunc=['count',pd.Series.nunique] ,fill_value=0)

tblpcount.reset_index(inplace=True)

tblposition = pd.pivot_table(df, index=["Position","Player's Name"],values= ['Hands'],columns=['Action'],aggfunc=['count'] ,fill_value=0)

tblposition.reset_index(inplace=True)

tblpcount= tblpcount[[(     'Position',        ''),

            ("Player's Name",        ''),

            (        'count', 'Played?'),

            (      'nunique',   'Hands')]]





tblposition.columns = ['Position',"Player's Name", 'bets','calls','checks','wins','folds','posts','raises']

tblpcount.columns = ['Position',"Player's Name", "No. Hands Played","Total Hands"]



tblposition = pd.merge(tblposition, tblpcount[["Player's Name",'Position',"No. Hands Played",'Total Hands']], on=["Position","Player's Name"])

tblposition.drop(columns=['posts'],inplace=True)





tblposition['Total']= tblposition['checks'] +  tblposition['calls'] + tblposition['bets'] + tblposition['raises'] + tblposition['folds']

tblposition[['bets','calls','checks','folds','raises']] = tblposition[['bets','calls','checks','folds','raises']].div(tblposition['Total'].values,axis=0)

tblposition["Player's Winning %"] = (tblposition['wins']/tblposition['No. Hands Played']).astype(float).map("{:.2%}".format)

format_dict = {'bets': '{:.2%}','calls': '{:.2%}','checks': '{:.2%}','folds': '{:.2%}','raises': '{:.2%}'}

tblposition  = tblposition.reindex(columns=["Player's Name",'Position','checks','calls','bets',"raises",'folds','wins',"Player's Winning %",'No. Hands Played','Total Hands'])

tblposition.style.format(format_dict)
dfchipcount = pd.read_excel('../input/poker7292020/PokerNow Data Clean 2020-07-29.xlsx', sheet_name='Chip Count')

dfchipcount.drop(columns=['Unnamed: 0'], inplace=True)



fig =px.line(dfchipcount, x="Timestamp", y="Amount", color ="Player's Name", title ='''Chip Stack by Hand''', 

        labels={'Timestamp':'Time','Amount':'Chip Amount'}, hover_name='Hands', hover_data=['Amount'], width=1000, height=600)

fig.update_layout(

    title={

        'y':0.9,

        'x':0.5,

         'xanchor': 'center',

        })

fig.update_traces(mode='markers+lines')

fig.show()
dfWH = df[df['WH Rank'].notnull()]

dfWH = dfWH[["Player's Name", "Hands", "Amount","WH Rank","Winning Hand"]]

dfWH = dfWH.copy()

dfWH.columns = ["PName","Hands","Amount","WHRank","WHHand"]

fig = go.Figure(data=[go.Table(

    columnwidth = [400,800],

    header=dict(values=["<b>Player's Name</b>","<b>Hands</b>", "<b>Amount</b>","<b>WH Rank</b>","<b>Winning Hand</b>"],

                fill_color='forestgreen',

                align='center',

                font=dict(color='white', size=12)),

    cells=dict(values= [dfWH.PName, dfWH.Hands, dfWH.Amount, dfWH.WHRank, dfWH.WHHand],

               fill_color='lavender',

               align='center',

               font_size=14))

])



fig.show()
dfduration = dfchipcount.drop_duplicates(["Player's Name","Hands"],keep= 'last')

tbl2 = pd.pivot_table(dfduration, index=["Player's Name"],values=['Hands','Amount','Duration'],aggfunc=["last","count","sum"], fill_value=0)

tbl2 = tbl2.copy()

tbl2 = tbl2[[( 'last', 'Amount'), ('count',  'Hands'), (  'sum', 'Duration')]]

tbl2.columns = ['Ending Chips','Hands','Duration']

tbl2 = tbl2.reset_index(level="Player's Name")





buyinlog = pd.read_excel('../input/poker7292020/PokerNow Data Clean 2020-07-29.xlsx', sheet_name='Buy in Log')

tbl2 = pd.merge(tbl2, buyinlog[["Player's Name",'Buy in Chips']], on="Player's Name")

tbl2['BB per 100'] = (((tbl2['Ending Chips']-tbl2['Buy in Chips'])/2)/((tbl2['Hands'])/100))

tbl2['$ Per Hr'] = (((tbl2['Ending Chips']-tbl2['Buy in Chips'])/4)/((tbl2['Duration'])/60))

tbl2 = tbl2.reindex(columns=["Player's Name",'Hands','Buy in Chips','Ending Chips','Duration','BB per 100','$ Per Hr'])

tbl2.sort_values(by='BB per 100', ascending=False, inplace=True)

tbl2['Duration'] = tbl2['Duration'].map('{:,.2f}'.format)

tbl2['BB per 100'] = tbl2['BB per 100'].map('{:,.2f}'.format)

tbl2['$ Per Hr'] = tbl2['$ Per Hr'].map('{:,.2f}'.format)

tbl2
HvsH = pd.read_excel('../input/poker7292020/PokerNow Data Clean 2020-07-29.xlsx', sheet_name='HvsH', index_col=0)



HvsH
HvsHvar = pd.read_excel('../input/poker7292020/PokerNow Data Clean 2020-07-29.xlsx', sheet_name='HvsH Var', index_col = 0)



def color_n(number):

    color='red' if number < 0 else 'black'

    return f'color: {color}'

def df_style(val):

    return 'font-weight: bold'



HvsHvar.style.set_properties(**{'background-color':'lightcyan',

                               'border-color':'black',

                               'border-width':'1px',

                               'border-style':'solid',

                               'text-align': 'center'}).applymap(color_n).applymap(df_style,

                  subset=pd.IndexSlice[['Total']])

dfwhrank = df[df["WH Rank"].str.contains("collected")==False]

dfwhrank = dfwhrank.copy()







tbl4 = pd.pivot_table(dfwhrank, index=["Player's Name"],columns=['WH Rank'],values=["Log#"],aggfunc=["count"], fill_value=0)



tbl4.columns = tbl4.columns.droplevel(0)

tbl4.columns = tbl4.columns.droplevel(0)



tbl4 = tbl4.reindex(columns=["High Card","Pair",'Two Pair','Three of a Kind','Straight','Flush','Full House','Four of a Kind'])

tbl4 = tbl4.fillna(0)



tbl4.loc['Total',:]= tbl4.sum(axis=0)







tbl4.astype(int)

dfAction = df[(df["Action"].notnull()) & (df["Action"]!="collected")&(df["Action"]!="posts")]





tbl5 = pd.pivot_table(dfAction, index=["Player's Name",'Street'], columns=["Action"],values=['Log#',"3 bet"],aggfunc=["count"], fill_value=0)



tbl5 = tbl5[[('count', '3 bet', 'raises'),

            ('count',  'Log#',  'folds'),

            ('count',  'Log#',   'bets'),

            ('count',  'Log#',  'calls'),

            ('count',  'Log#', 'checks'),

            ('count',  'Log#', 'raises')]]

tbl5.reset_index(inplace=True)

tbl5.columns = ["Player's Name","Street","3 Bet","folds","bets","calls","checks","raises"]

dfact = tbl5[["Player's Name","Street","folds","bets","calls","checks","raises"]]

tbl5.sort_values(by='Street',ascending=True,inplace=True)

df3bet = df[df['3 bet']==True]

df3bet = pd.pivot_table(df3bet, index=["Player's Name",'Street'], values=["3 bet"],aggfunc=["count"], fill_value=0)

df3bet.reset_index(inplace=True)

df3bet.columns = ["Player's Name","Street","3 Bet"]

dfact = pd.merge(dfact, df3bet[["Player's Name","Street","3 Bet"]], on=["Player's Name","Street"], how='left')

dfact['3 Bet'] =dfact['3 Bet'].fillna(0).astype(int)

dfact.sort_values(by=['Street',"Player's Name"],inplace=True)



dfact= dfact.copy()

dfact['Total'] = dfact.iloc[:,2:7].sum(axis=1)



dfact[['folds','bets','calls','checks','raises']] = (dfact[['folds','bets','calls','checks','raises']].div(dfact['Total'].values,axis=0)*100).round(2)

dfact = dfact[["Player's Name",'Street','folds','bets','calls','checks','raises']]

dfactall = df[(df["Action"].notnull()) & (df["Action"]!="collected")&(df["Action"]!="posts")]





tbl6 = pd.pivot_table(dfactall, index=["Player's Name"], columns=["Action"],values=['Log#'],aggfunc=["count"], fill_value=0)



tbl6 = tbl6[[('count', 'Log#', 'folds'),

            ('count',  'Log#',   'bets'),

            ('count',  'Log#',  'calls'),

            ('count',  'Log#', 'checks'),

            ('count',  'Log#', 'raises')]]

tbl6.reset_index(inplace=True)

tbl6.columns = ["Player's Name","folds","bets","calls","checks","raises"]



dfacttotal = tbl6

dfacttotal['Total'] = dfacttotal.iloc[:,1:6].sum(axis=1)

dfacttotal[['folds','bets','calls','checks','raises']] = (dfacttotal[['folds','bets','calls','checks','raises']].div(dfacttotal['Total'].values,axis=0)*100).round(2)

dfacttotal = dfacttotal[["Player's Name",'folds','bets','calls','checks','raises']]



df1 = dfact.loc[(dfact["Street"] == "Preflop")]

df2 = dfact.loc[(dfact["Street"] == "flop")]

df3 = dfact.loc[(dfact["Street"] == "turn")]

df4 = dfact.loc[(dfact["Street"] == "river")]

dft = dfacttotal



fig = go.Figure()



trace1 = go.Bar(text=(dft["folds"].astype(str) + "%"),textposition='auto',name='Folds', x=dft["Player's Name"], y=dft["folds"])

trace2 = go.Bar(text=(dft["checks"].astype(str) + "%"),textposition='auto',name='Checks', x=dft["Player's Name"], y=dft["checks"])

trace3=go.Bar(text=(dft["bets"].astype(str) + "%"),textposition='auto',name='Bets', x=dft["Player's Name"], y=dft["bets"])

trace4=go.Bar(text=(dft["calls"].astype(str) + "%"),textposition='auto',name='Calls', x=dft["Player's Name"], y=dft["calls"])

trace5= go.Bar(text=(dft["raises"].astype(str) + "%"),textposition='auto',name='Raises', x=dft["Player's Name"], y=dft["raises"])



trace6 = go.Bar(text=(df1["folds"].astype(str) + "%"),textposition='auto',name='Folds', x=df1["Player's Name"], y=df1["folds"])

trace7 = go.Bar(text=(df1["checks"].astype(str) + "%"),textposition='auto',name='Checks', x=df1["Player's Name"], y=df1["checks"])

trace8=go.Bar(text=(df1["bets"].astype(str) + "%"),textposition='auto',name='Bets', x=df1["Player's Name"], y=df1["bets"])

trace9=go.Bar(text=(df1["calls"].astype(str) + "%"),textposition='auto',name='Calls', x=df1["Player's Name"], y=df1["calls"])

trace10= go.Bar(text=(df1["raises"].astype(str) + "%"),textposition='auto',name='Raises', x=df1["Player's Name"], y=df1["raises"])



trace11 = go.Bar(text=(df2["folds"].astype(str) + "%"),textposition='auto',name='Folds', x=df2["Player's Name"], y=df2["folds"])

trace12= go.Bar(text=(df2["checks"].astype(str) + "%"),textposition='auto',name='Checks', x=df2["Player's Name"], y=df2["checks"])

trace13=go.Bar(text=(df2["bets"].astype(str) + "%"),textposition='auto',name='Bets', x=df2["Player's Name"], y=df2["bets"])

trace14=go.Bar(text=(df2["calls"].astype(str) + "%"),textposition='auto',name='Calls', x=df2["Player's Name"], y=df2["calls"])

trace15= go.Bar(text=(df2["raises"].astype(str) + "%"),textposition='auto',name='Raises', x=df2["Player's Name"], y=df2["raises"])



trace16 = go.Bar(text=(df3["folds"].astype(str) + "%"),textposition='auto',name='Folds', x=df3["Player's Name"], y=df3["folds"])

trace17= go.Bar(text=(df3["checks"].astype(str) + "%"),textposition='auto',name='Checks', x=df3["Player's Name"], y=df3["checks"])

trace18=go.Bar(text=(df3["bets"].astype(str) + "%"),textposition='auto',name='Bets', x=df3["Player's Name"], y=df3["bets"])

trace19=go.Bar(text=(df3["calls"].astype(str) + "%"),textposition='auto',name='Calls', x=df3["Player's Name"], y=df3["calls"])

trace20= go.Bar(text=(df3["raises"].astype(str) + "%"),textposition='auto',name='Raises', x=df3["Player's Name"], y=df3["raises"])



trace21 = go.Bar(text=(df4["folds"].astype(str) + "%"),textposition='auto',name='Folds', x=df4["Player's Name"], y=df4["folds"])

trace22= go.Bar(text=(df4["checks"].astype(str) + "%"),textposition='auto',name='Checks', x=df4["Player's Name"], y=df4["checks"])

trace23=go.Bar(text=(df4["bets"].astype(str) + "%"),textposition='auto',name='Bets', x=df4["Player's Name"], y=df4["bets"])

trace24=go.Bar(text=(df4["calls"].astype(str) + "%"),textposition='auto',name='Calls', x=df4["Player's Name"], y=df4["calls"])

trace25= go.Bar(text=(df4["raises"].astype(str) + "%"),textposition='auto',name='Raises', x=df4["Player's Name"], y=df4["raises"])







list_updatemenus = [

    {'label': 'Total',

  'method': 'update',

  'args': [{'visible': [True, True, True, True, True,False, False, False, False, False,False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]}, {'title': 'Action Type Total'}]},

 {'label': 'Preflop',

  'method': 'update',

  'args': [{'visible': [False, False, False, False,False,True,True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False,False, False, False]}, {'title': 'Action Type by Preflop'}]},

 {'label': 'Flop',

  'method': 'update',

  'args': [{'visible': [False, False,False, False, False, False, False, False, False, False,True, True,True, True, True,False, False, False, False, False, False, False, False, False, False]}, {'title': 'Action Type by Flop'}]},

 {'label': 'Turn',

  'method': 'update',

  'args': [{'visible': [False, False, False, False, False, False, False,False, False, False, False,False, False, False, False,True,True, True, True, True, False, False, False, False, False]}, {'title': 'Action Type by Turn'}]},

 {'label': 'River',

  'method': 'update',

  'args': [{'visible': [False, False, False, False, False, False, False, False,False, False, False, False,False, False, False, False,False, False, False, False,True, True, True, True, True]}, {'title': 'Action Type by River'}]}]







data = [trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11,trace12,trace13,trace14,trace15,trace16,\

        trace17,trace18,trace19,trace20,trace21,trace22,trace23,trace24,trace25]



layout=go.Layout(title='Poker Action Type Breakdown',updatemenus=list([dict(buttons= list_updatemenus),]),\

                 width=1000,height=800,barmode='stack',xaxis=dict(title = "Player's Name"), yaxis=dict(title = "% of Player's Total Action"))







fig = go.Figure(data,layout)



fig.show()