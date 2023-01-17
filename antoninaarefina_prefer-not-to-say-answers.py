# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

%matplotlib inline
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

inpdir = "../input"
import os
print(os.listdir(inpdir))

# Any results you write to the current directory are saved as output.
df = pd.read_csv(inpdir + '/multipleChoiceResponses.csv', low_memory=False)
df.drop(index=0, inplace=True)
df["Time from Start to Finish (seconds)"] = df["Time from Start to Finish (seconds)"].astype('int')

df.head()
pnts_qa = {
    'Q1': ('Prefer not to say', 'Gender'),
    'Q3': ('I do not wish to disclose my location', 'Location'),
    'Q4': ('I prefer not to answer', 'Education'),
    'Q9': ('I do not wish to disclose my approximate yearly compensation', 'Compensation'), 
}

def get_pnts_index(question):
    return df.loc[df[question] == pnts_qa[question][0]].index

print('Prefer-not-to-say Answers ')
print('Question\tCount\tProportion')
for question in pnts_qa:    
    ans, desc = pnts_qa[question]
    df1 = df.loc[get_pnts_index(question), question]
    print('{0}:{3}\t{1}\t{2}'.format(question, df1.count(), df1.count()/df[question].count(), desc))
prefer_not_to_say = pd.DataFrame(index=df.index)

prefer_not_to_say['Q1'] = (df['Q1'] == 'Prefer not to say').astype(int)
prefer_not_to_say['Q3'] = (df['Q3'] == 'I do not wish to disclose my location').astype(int)
prefer_not_to_say['Q4'] = (df['Q4'] == 'I prefer not to answer').astype(int)
#prefer_not_to_say['Q9'] = (df['Q9'] == 'I do not wish to disclose my approximate yearly compensation').astype(int)

A = prefer_not_to_say.values
pd.DataFrame(A.T.dot(A), index=['Q1', 'Q3', 'Q4'], columns=['Q1', 'Q3', 'Q4'])
prefer_not_to_say.sum(axis=1).value_counts().reset_index().rename(columns={
    'index': "Number of unanswered questions", 0:"Number of respondents"})
df['Q2'] = pd.Categorical(df['Q2'], categories=[
    '18-21', '22-24', '25-29', '30-34', '35-39', '40-44','45-49', '50-54', '55-59', 
    '60-69', '70-79', '80+'], ordered=True)
def plot_compare_plotly(idx1, idx2, col, idx1_label, idx2_label, title,
                        sort_by='index', figsize=(10, 5)):
    d1 = df.loc[idx1, col]
    d2 = df.loc[idx2, col]
    
    total_respondents = (d1.count(), d2.count())

    d1 = d1.value_counts()
    d2 = d2.value_counts()
    
    if sort_by=='index':
        d1 = d1.sort_index()
        d2 = d2.sort_index()
    else:
        d1 = d1.sort_values()
        d2 = d2.sort_values()        
        
    trace1 = go.Bar(
        y=d1.index,
        x=(d1.values / total_respondents[0]),
        text=d1.apply(lambda x: str(x) + ' respondents').values,
        name=idx1_label,
        orientation='h'
    )
    trace2 = go.Bar(
        y=d2.index,
        x=(d2.values / total_respondents[1]),
        text=d2.apply(lambda x: str(x) + ' respondents').values,
        name=idx2_label,
        orientation='h'
    )

    data = [trace1, trace2] if sort_by != 'second' else [trace2, trace1] 

    layout = go.Layout(
        barmode='group',
        title= title,
        xaxis= dict(title= 'Proportion',),
        yaxis=dict(tickmode='linear', showticklabels=True,),
        legend=dict(orientation="h", x=-0.1, y=-0.2),
        width=figsize[0] * 80, 
        height=figsize[1] * 80,
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig) 
    
def plot_compare_other_plotly(idx, col, idx_label, title, sort_by='index', figsize=(10, 5)):
    plot_compare_plotly(idx1=idx, idx2=df.drop(index=idx).index, col=col, 
                 idx1_label=idx_label, idx2_label='Other Kagglers', title=title,
                 sort_by=sort_by, figsize=figsize)
plot_compare_other_plotly(idx = get_pnts_index('Q1'), col = 'Q2', 
                   idx_label='Kagglers not willing to disclose their gender',
                          title='Age Distribution')
plot_compare_other_plotly(idx = get_pnts_index('Q3'), col = 'Q2', 
             idx_label='Kagglers not willing to disclose their location', title='Age Distribution')
plot_compare_other_plotly(idx = get_pnts_index('Q4'), col = 'Q2', 
             idx_label='Kagglers not willing to disclose their level of formal education', title='Age Distribution')
plot_compare_other_plotly(idx = (get_pnts_index('Q1') | get_pnts_index('Q3') | get_pnts_index('Q4')),
                   col = 'Q2', 
             idx_label='Kagglers not willing to disclose either their gender, or location, or educarton level', title='Age Distribution')
def plot_proportions(question, by_col, title, figsize=(10, 5)):
    df1 = pd.DataFrame(index=df.index)
    df1[by_col] = df[by_col]
    df1['NoQ'] = 0
    df1.loc[get_pnts_index(question), 'NoQ'] = 1

    gbc = df1.groupby(by=by_col).mean().sort_values(by='NoQ')

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y=gbc.index, width=gbc.values.reshape(-1), color='blue', alpha=0.4)

    ax.set_title(title)
    ax.set_xlabel('Proportion')
    ax.grid(axis='x')

plot_proportions('Q9', 'Q3', 
                'Proportion of respondents not willing to disclose their yearly compensation by Country', figsize=(15, 10))
df1 = pd.DataFrame(index=df.index)
df1['Q3'] = df['Q3']
df1['NoQ'] = 0
df1.loc[get_pnts_index('Q9'), 'NoQ'] = 1

gbc = df1.groupby(by='Q3').mean().sort_values(by='NoQ')

gbc.iloc[[0, -2, -1]]
plot_compare_other_plotly(idx=get_pnts_index('Q9'), col='Q3', 
                   idx_label='Kagglers not willing to disclose their yearly compensation',
                  title='Country', sort_by='second', figsize=(12, 10))
plot_compare_other_plotly(idx=get_pnts_index('Q9'), col='Q7', 
                   idx_label='Kagglers not willing to disclose their yearly compensation',
                  title='Industry', sort_by='second')
plot_compare_other_plotly(idx=get_pnts_index('Q9'), col='Q6', 
                   idx_label='Kagglers not willing to disclose their yearly compensation',
                  title='Role at Work', sort_by='second')
df1 = pd.DataFrame(index=df.index)
df1['Q3'] = df['Q3']
df1['NoQ9'] = 0
df1.loc[get_pnts_index('Q9'), 'NoQ9'] = 1
df1['IsStudent'] = (df['Q6'] == 'Student').astype(int)

gbc = df1.groupby(by='Q3').mean()
gbc['Q9_respondents'] = df[['Q3','Q9']].groupby(by='Q3').count()

print("Top-5 countries & territories by proportion of students")
print(gbc['IsStudent'].sort_values(ascending=False)[0:5].rename("Students' proportion"))
trace = go.Scatter(
    x=gbc['NoQ9'], 
    y=gbc['IsStudent'],
    mode='markers',
    marker=dict(size= np.log(gbc['Q9_respondents']) * 2),
    text=gbc.index.str.cat(gbc['Q9_respondents'].apply(lambda x: str(x) + ' Q9 respondents'), sep=': ') 
)    

layout = go.Layout(
    title='Proportion of respondents not willing to disclose their yearly compensation <br>vs Proportion of students <br>by Location',
    xaxis=dict(title='Proportion of respondents not willing to disclose their yearly compensation',),
    yaxis=dict(title='Proportion of students',),
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
stats.spearmanr(gbc['NoQ9'], gbc['IsStudent'])

