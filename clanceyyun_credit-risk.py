import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
credit = pd.read_csv('../input/credit_data.csv',index_col = 0)
print(credit.info())
print('\n')
print(credit.head())
## looking unique values
print(credit.nunique())
import plotly.offline as py 
import plotly.graph_objs as go # it's like "plt" of matplot
import plotly.tools as tls # It's useful to we get some tools of plotly

py.init_notebook_mode(connected=True)  # make offline plotly version work

bar_good = go.Bar(
        x = credit[credit["Risk"] == 'good']["Risk"].value_counts().index.values,
        y = credit[credit["Risk"] == 'good']["Risk"].value_counts().values,
        name = 'Good Credit'
    )

bar_bad = go.Bar(
        x = credit[credit["Risk"] == 'bad']["Risk"].value_counts().index.values,
        y = credit[credit["Risk"] == 'bad']["Risk"].value_counts().values,
        name = 'Bad Credit'
    )
# x = df_credit[df_credit["Risk"]== 'bad']["Risk"].value_counts().index.values 
# print(x) ##= ['bad']
# y = df_credit[df_credit["Risk"]== 'bad']["Risk"].value_counts().values
# print(y) ##= [300]

# x1 = df_credit[df_credit["Risk"]== 'good']["Risk"].value_counts().index.values
# print(x1) ##=['good']
# y1 = df_credit[df_credit["Risk"]== 'good']["Risk"].value_counts().values
# print(y1) ##=[700]

bar = [bar_good,bar_bad]

layout = go.Layout(
        xaxis=dict( title = 'Risk'),
        yaxis=dict( title = 'Count'),
#         title = 'Risk variable'
)

fig = go.Figure(data = bar, layout=layout)
py.iplot(fig)
credit_age = credit['Age'].values.tolist()
# print(credit_age) # list
credit_good =credit[credit["Risk"] == 'good']['Age'].values.tolist()
#len(credit_good)
credit_bad =credit[credit["Risk"] == 'bad']['Age'].values.tolist()

H_good = go.Histogram(
        x= credit_good,
        histnorm= 'percent',
        name = 'Good Credit'
)

H_bad = go.Histogram(
        x= credit_bad,
        histnorm= 'percent',
        name = 'Bad Credit'
)

H_age = go.Histogram(
        x= credit_age,
        histnorm= 'percent',
        name = 'Age'
)

## 2*2 fig can find in example 3.
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Good','Bad', 'General Distribuition'))

## three fig need to show
fig.append_trace(H_good,1,1) ## x0 will display at the position row=1,col=1,
fig.append_trace(H_bad,1,2) ## x1 will display at the position row=1,col=2,
fig.append_trace(H_age,2,1) ## x2 will display at the position row=2,col=1,
    
fig['layout'].update(bargap=0.05) 
py.iplot(fig)
credit_good = credit[credit["Risk"] == 'good']
credit_bad = credit[credit["Risk"] == 'bad']

## creat the frame of fig
fig, ax = plt.subplots(nrows=2,figsize=(10,8))
plt.subplots_adjust(hspace = 0.5, top =0.8)

## draw the first fig
P1 = sns.distplot(credit_good["Age"],ax = ax[0],color = 'g')
P1 = sns.distplot(credit_bad["Age"],ax = ax[0],color = 'r')

## give the fig title and lable
P1.set_title('Age Distribution')
P1.set_xlabel('Age')
P1.set_ylabel('Frequency')

# draw the second fig
P2 = sns.countplot(x = "Age",data = credit, palette = "hls", 
                   ax = ax[1], hue = "Risk")

## give the fig title and lable
P2.set_title('Age Counting Distribution')
P2.set_xlabel('Age')
P2.set_ylabel('Counting')

plt.show() ## this row can hidden "Text(0, 0.5, 'Counting')"
credit_good = credit[credit["Risk"] == 'good']
credit_bad = credit[credit["Risk"] == 'bad']

Interv = (18, 25, 40, 60, 120)
stage = ['Young','Adult','Middle','Older']
credit['Age_stage'] = pd.cut(credit.Age, Interv, labels = stage)

b_good = go.Box(
        x = credit_good['Age_stage'],
        y = credit_good['Credit amount'],
        name = 'Good Credit'
) 

b_bad = go.Box(
        x = credit_bad['Age_stage'],
        y = credit_bad['Credit amount'],
        name = 'Bad Credit'
) 

data = [b_good,b_bad]
layout = go.Layout(
        xaxis=dict(title = 'Age Stage'),
        yaxis=dict(title = 'Credit Amount',zeroline = True),
        boxmode = 'group'
)

fig = go.Figure(data = data, layout =layout )

py.iplot(fig)
b_good = go.Bar(
        x = credit[credit["Risk"] == 'good']["Housing"].value_counts().index.values,
        y = credit[credit["Risk"] == 'good']["Housing"].value_counts().values,
        name = 'Good Credit'
)

b_bad = go.Bar(
        x = credit[credit["Risk"] == 'bad']["Housing"].value_counts().index.values,
        y = credit[credit["Risk"] == 'bad']["Housing"].value_counts().values,
        name = 'Bad Credit'
)

data  = [b_good,b_bad]

layout = go.Layout(
        title = 'Risk Housing Distribution'
)
## display the fig
fig =go.Figure(data = data, layout = layout)
py.iplot(fig)
b_good = go.Bar(
        x = credit[credit["Risk"] == 'good']["Housing"].value_counts().index.values,
        y = credit[credit["Risk"] == 'good']["Credit amount"].values,
        name = 'Good Credit'
)

b_bad = go.Bar(
        x = credit[credit["Risk"] == 'bad']["Housing"].value_counts().index.values,
        y = credit[credit["Risk"] == 'bad']["Credit amount"].values,
        name = 'Bad Credit'
)

data  = [b_good,b_bad]

layout = go.Layout(
        title = 'Credit amount Housing Distribution'
)
## display the fig
fig =go.Figure(data = data, layout = layout)
py.iplot(fig)
b_1 = go.Bar(
        x = credit[credit["Risk"] == 'good']["Sex"].value_counts().index.values,
        y = credit[credit["Risk"] == 'good']["Sex"].value_counts().values,
        name = 'Good Credit'
)

b_2 = go.Bar(
        x = credit[credit["Risk"] == 'bad']["Sex"].value_counts().index.values,
        y = credit[credit["Risk"] == 'bad']["Sex"].value_counts().values,
        name = 'Bad Credit'
)

b_3 = go.Box(
        x = credit[credit["Risk"] == 'good']["Sex"],
        y = credit[credit["Risk"] == 'good']["Credit amount"],
        name = 'Good Credit'
)

b_4 = go.Box(
        x = credit[credit["Risk"] == 'bad']["Sex"],
        y = credit[credit["Risk"] == 'bad']["Credit amount"],
        name = 'Bad Credit'
)


data  = [b_1,b_2,b_3,b_4]

fig = tls.make_subplots(rows=1, cols=2, 
                        subplot_titles=('Sex Count', 'Credit Amount by Sex'))

fig.append_trace(b_1,1,1)
fig.append_trace(b_2,1,1)
fig.append_trace(b_3,1,2)
fig.append_trace(b_4,1,2)

## display the fig
fig['layout'].update(height=400,width=800,title='Sex Distribution',
                     boxmode='group') 
py.iplot(fig)
b_good = go.Bar(
        x = credit[credit["Risk"] == 'good']["Job"].value_counts().index.values,
        y = credit[credit["Risk"] == 'good']["Job"].value_counts().values,
        name = 'Good Credit'
)

b_bad = go.Bar(
        x = credit[credit["Risk"] == 'bad']["Job"].value_counts().index.values,
        y = credit[credit["Risk"] == 'bad']["Job"].value_counts().values,
        name = 'Bad Credit'
)

data  = [b_good,b_bad]

layout = go.Layout(
        title = 'Job Distribution'
)
## display the fig
fig =go.Figure(data = data, layout = layout)
py.iplot(fig)
b_good = go.Bar(
        x = credit[credit["Risk"] == 'good']["Job"].value_counts().index.values,
        y = credit[credit["Risk"] == 'good']["Age"].values,
        name = 'Good Credit'
)

b_bad = go.Bar(
        x = credit[credit["Risk"] == 'bad']["Job"].value_counts().index.values,
        y = credit[credit["Risk"] == 'bad']["Age"].values,
        name = 'Bad Credit'
)

data  = [b_good,b_bad]

layout = go.Layout(
        title = 'Job Age '
)
## display the fig
fig =go.Figure(data = data, layout = layout)
py.iplot(fig)
B_good = go.Box(
    x=credit_good["Job"],
    y=credit_good["Credit amount"],
    name='Good credit'
)

B_bad = go.Box(
    x=credit_bad['Job'],
    y=credit_bad['Credit amount'],
    name='Bad credit'
)
    
data = [B_good, B_bad]

layout = go.Layout(
    yaxis=dict(
        title='Credit Amount'
    ),
    title = 'Credit Amount - Job',
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
import plotly.figure_factory as ff

d1 = np.log(credit_good['Credit amount'])
d2 = np.log(credit_bad['Credit amount'])

data = [d1,d2]

labels = ['Good Credit','Bad Credit']

fig = ff.create_distplot(data,labels,bin_size=0.2)

py.iplot(fig)
bar_good = go.Bar(
        x = credit_good["Saving accounts"].value_counts().index.values,
        y = credit_good["Saving accounts"].value_counts().values,
        name = 'Good Credit'
)

bar_bad = go.Bar(
        x = credit_bad["Saving accounts"].value_counts().index.values,
        y = credit_bad["Saving accounts"].value_counts().values,
        name = 'Bad Credit'
)

b_1 = go.Box(
    x=credit_good["Saving accounts"],
    y=credit_good["Credit amount"],
    name='Good credit'
)
b_2 = go.Box(
    x=credit_bad["Saving accounts"],
    y=credit_bad["Credit amount"],
    name='Bad credit'
)

s_1 = go.Box(
    x=credit_good["Saving accounts"],
    y=credit_good["Age"],
    name='Good credit'
)
s_2 = go.Box(
    x=credit_bad["Saving accounts"],
    y=credit_bad["Age"],
    name='Bad credit'
)

data = [bar_good, bar_bad, b_1, b_2, s_1, s_2]

fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Saving Accounts','Credit Amount by Savings', 
                                          'Saving accounts　by age'))

fig.append_trace(bar_good,1,1)
fig.append_trace(bar_bad,1,1)

fig.append_trace(b_2, 1, 2)
fig.append_trace(b_1, 1, 2)

fig.append_trace(s_1, 2, 1)
fig.append_trace(s_2, 2, 1)

fig['layout'].update(height=700, width=800, title='Saving Accounts Exploration', boxmode='group')

py.iplot(fig)
print(credit)
credit.columns
credit = pd.get_dummies(credit)
credit.columns
plt.figure(figsize=(25,25))
sns.heatmap(credit.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True,  linecolor='white', annot=True)
plt.show()