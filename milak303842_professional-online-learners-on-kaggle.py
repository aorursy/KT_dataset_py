# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
multi = pd.read_csv('../input/multipleChoiceResponses.csv')
multi.shape
multi_dict = multi.iloc[0,:].to_dict()
multi.drop(multi.index[0],inplace=True)
eda = multi.loc[(multi.Q36_Part_12.isna()==True)&(~multi.Q6.isin(['Student','Not employed']))&(multi.Q7!='I am a student'),:]
eda['Q25_part'] = eda.Q25.fillna('0').str.split('-').str[0]
eda = eda.loc[eda.Q25_part.isin(['2','3','4','5','10','20+ years']),:]
print('The resulting dataset has N of rows: '+str(eda.shape[0])+ ' and N of columns: '+str(eda.shape[1]))
#labels, that are better at the end of array
pop_lab = ['Other','None','I have never performed this task','I have never found any difficulty in this task','I have not used any cloud providers']

def other(old_list):
    new_list = sorted([l for l in old_list if ~np.isin(l,pop_lab)])
    if np.isin(old_list,'Other').any():
        new_list.insert(len(new_list),'Other')
    if np.isin(old_list,'None').any():
        new_list.insert(len(new_list),'None')
    if np.isin(old_list,'I have never performed this task').any():
        new_list.insert(len(new_list),'I have never performed this task')
    if np.isin(old_list,'I have never found any difficulty in this task').any():
        new_list.insert(len(new_list),'I have never found any difficulty in this task')
    if np.isin(old_list,'I have not used any cloud providers').any():
        new_list.insert(len(new_list),'I have not used any cloud providers')
    return new_list    

def multi_count(data, column):
    temp = data[column].apply(pd.value_counts)
    arr_flat = [e for sublist in temp.values for e in sublist]
    temp_new = pd.DataFrame({'names': temp.index,
                            'values' : [arr for arr in arr_flat if arr > 0 ]})
    temp_new.names = pd.Categorical(temp_new.names,
                                categories = other(temp.index),
                                ordered=False)
    temp_new = temp_new.sort_values('names')
    temp_new_fin = temp_new.loc[temp_new.names.isin(pop_lab)==False,:].sort_values('values',ascending=False)
    temp_new_fin = temp_new_fin.append(temp_new.loc[temp_new.names.isin(pop_lab)==True,:],ignore_index=True)
    return temp_new_fin

def get_title(variable_string):
    full_text = [ v for k,v in multi_dict.items() if variable_string in k].pop(0)
    question = full_text.split('?')[0]
    question = question.replace('(Select all that apply)','')
    question = question.replace('- Selected Choice','')
    question_full = ''.join([str(question),'?'])
    if len(question_full)>70:
        parag_place = math.floor(len(question_full.split(' '))/2)
        question_full = ' '.join([' '.join(question_full.split(' ')[:parag_place]),'<br>',' '.join(question_full.split(' ')[parag_place:])]) 
    else:
        question_full
    return question_full
#plotly library
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.offline.offline import _plot_html
import plotly.graph_objs as go
import cufflinks as cf
init_notebook_mode(connected=True)

colors = ['rgb(43,140,190)','rgb(166,189,219)','rgb(236,231,242)']
temp = pd.DataFrame(eda.Q1.value_counts())
data = [go.Pie(
    labels=temp.index,
    values=temp.iloc[:,0],
    marker =  dict(colors= colors),
    hoverinfo = 'label+value'
    )]
layout = go.Layout(
        title = "Gender")
    
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
s = pd.Categorical(eda.Q2,categories=['18-21','22-24','25-29','30-34','35-39','40-44','45-49',
                                          '50-54','55-59','60-69','70-79','80+'],ordered=False)
temp = pd.DataFrame(s.value_counts())
data = [go.Bar(
    x = temp.index,
    y= temp.iloc[:,0],
    hoverinfo = 'y'
)]
layout = go.Layout(
    title = get_title('Q2'),
    xaxis = dict(
        title = 'Years'
    ),
    yaxis = dict(
        title = 'Count'
    )
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
temp = pd.DataFrame(eda.Q3.value_counts())
data = [ dict(
    type = 'choropleth',
    locations = temp.index,
    locationmode = 'country names',
    z = temp.iloc[:,0],
    text = temp.index,
    colorscale = [[0,"rgb(4,90,141)"],[0.4,"rgb(43,140,190)"],\
                  [0.5,"rgb(116,169,207)"],[0.9,"rgb(189,201,225)"],[1,"rgb(241,238,246)"]],
    autocolorscale = False,
    reversescale = True,
    marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            #tickprefix = 'k',
            title = 'Number of participants'),
      ) ]

layout = dict(
    title = 'Countries',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        showcountries = True,
        countriescolor = "rgb(236,236,236)",
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.offline.iplot(fig, validate=False, filename='d3-world-map' )
temp = pd.DataFrame(eda.Q4.value_counts())
data = [go.Bar(
    x = ['Master’s<br> degree',
         'Bachelor’s<br> degree',
         'Doctoral<br> degree',
         'Some<br> college/university<br> study without earning<br> a bachelor’s degree',
         'Professional<br> degree',
         'I prefer<br> not to<br> answer',
         'No formal<br> education<br> past high school'],
    y= temp.iloc[:,0].values,
    hoverinfo = 'y'
)]
layout = go.Layout(
    title = get_title('Q4'),
    xaxis = dict(
        tickangle=0
    ),
    yaxis = dict(
        exponentformat = 'none'
    )
    
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
temp = pd.DataFrame(eda.Q5.value_counts()).sort_values('Q5')
temp_fin = temp.loc[temp.index.isin(['Other','I never declared a major']),:]
temp_fin = temp_fin.append(temp.loc[~temp.index.isin(['Other','I never declared a major']),:])

data = [go.Bar(
    y = ['I never declared a major', 'Other', 'Fine arts or performing arts',
       'Environmental science or geology',
       'Humanities',
       'Social sciences',
       'Medical or life sciences',
       'Information technology,<br> networking, or system administration',
       'Physics or astronomy',
       'A business discipline',
       'Mathematics or statistics', 'Engineering',
       'Computer science'],
    x= temp_fin.iloc[:,0].values,
    orientation = 'h',
    hoverinfo = 'x'
)]
layout = go.Layout(
    title = get_title('Q5'),
    margin = dict(l=300),
    xaxis = dict(
        exponentformat = 'B',
        #showticklabels = False,
    )
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
print('Crosstab of major and degree')
round(pd.crosstab(eda.Q4, eda.Q5, normalize = 'columns'),2)
temp = pd.DataFrame(eda.Q37.value_counts()).head(10)

data = [go.Bar(
    x = temp.index,
    y= temp.iloc[:,0],
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = get_title('Q37'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
#TOP10
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q38_Part_')])
plot_df_fin = plot_df.loc[~plot_df.names.isin(['Other','None/I do not know']),:].head(10)
plot_df_fin = plot_df_fin.append(plot_df.loc[plot_df.names.isin(['Other','None/I do not know']),:])

data = [go.Bar(
    x = plot_df_fin.names.tolist(),
    y= plot_df_fin['values'].tolist(),
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = str(get_title('Q38_Part_')+' - TOP 10'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
#online learning
s = pd.Categorical(eda.Q39_Part_1,categories = ['Much better',
                                        'Slightly better',
                                         'Neither better nor worse',
                                        'Slightly worse',
                                         'Much worse',
                                        'No opinion; I do not know'],ordered=False)
temp = pd.DataFrame(s.value_counts())
temp_fin = temp.loc[temp.index!='No opinion; I do not know',:]
temp_fin = temp_fin.append(temp.loc[temp.index=='No opinion; I do not know',:])

data = [go.Bar(
    x = ['Much better', 'Slightly better', 'Neither better<br> nor worse',
         'Slightly worse', 'Much worse', 'No opinion;<br> I do not know'],
    y= temp.iloc[:,0],
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = 'How do you perceive the quality of <b>online learning platforms</b><br>as compared to the quality of the education provided by<br> traditional brick and mortar institutions?')
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
#bootcaml
s = pd.Categorical(eda.Q39_Part_2,categories = ['Much better',
                                        'Slightly better',
                                         'Neither better nor worse',
                                        'Slightly worse',
                                         'Much worse',
                                        'No opinion; I do not know'],ordered=False)
temp = pd.DataFrame(s.value_counts())
temp_fin = temp.loc[temp.index!='No opinion; I do not know',:]
temp_fin = temp_fin.append(temp.loc[temp.index=='No opinion; I do not know',:])

data = [go.Bar(
    x = ['Much better', 'Slightly better', 'Neither better<br> nor worse',
         'Slightly worse', 'Much worse', 'No opinion;<br> I do not know'],
    y= temp_fin.iloc[:,0],
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = 'How do you perceive the quality of <b>in-person bootcamps</b><br> as compared to the quality of the education provided by<br> traditional brick and mortar institutions?')
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
temp = pd.DataFrame(eda.Q40.value_counts())
temp_fin = temp.loc[temp.index!='No opinion; I do not know',:]
temp_fin = temp_fin.append(temp.loc[temp.index=='No opinion; I do not know',:])

data = [go.Bar(
    x = ['Independent<br>projects are<br>equally<br>important',
       'Independent<br>projects are<br>much more<br> important',
       'Independent<br> projects are<br> slightly more<br> important',
       'Independent<br> projects are<br> slightly less<br> important',
       'Independent<br> projects are<br> much less<br> important<br>',
        'No opinion;<br> I do not know'],
    y= temp_fin.iloc[:,0],
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = get_title('Q40'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
temp = pd.DataFrame(eda.Q6.value_counts()).sort_values('Q6')
temp_fin = temp.loc[temp.index.isin(['Other','Not employed']),:]
temp_fin = temp_fin.append(temp.loc[~temp.index.isin(['Other','Not employed']),:])

data = [go.Bar(
    y = temp_fin.index,
    x= temp_fin.iloc[:,0].values,
    orientation = 'h',
    hoverinfo = 'x'
)]
layout = go.Layout(
    title = get_title('Q6'),
    margin = dict(l=200))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
s = pd.Categorical(eda.Q26,categories = ['Definitely not',
                                        'Probably not',
                                         'Maybe',
                                        'Probably yes',
                                         'Definitely yes'],ordered=False)
temp = pd.DataFrame(s.value_counts())

data = [go.Bar(
    x = temp.index,
    y= temp.iloc[:,0],
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = 'Do you consider yourself to be a data scientist?')
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
s = pd.Categorical(eda.Q10,categories = ['We have well established ML methods (i.e., models in production for more than 2 years)',
                                        'We recently started using ML methods (i.e., models in production for less than 2 years)',
                                        'We use ML methods for generating insights (but do not put working models into production)',
                                        'We are exploring ML methods (and may one day put a model into production)',
                                        'No (we do not use ML methods)','I do not know'],ordered=False)
temp = pd.DataFrame(s.value_counts())
temp.columns = ['size']
temp.sort_values('size', inplace=True)
temp_fin = temp.loc[temp.index.isin(['I do not know','No (we do not use ML methods)']),:]
temp_fin = temp_fin.append(temp.loc[~temp.index.isin(['I do not know','No (we do not use ML methods)']),:])

data = [go.Bar(
    y = ['I do not know', 'No',
         'We use ML methods<br> for generating insights',
         'We have well established ML methods',
         'We recently started using ML methods',
         'We are exploring ML methods'],
    x= temp_fin.iloc[:,0],
    orientation = 'h',
    hoverinfo = 'x'
)]
layout = go.Layout(
    title = get_title('Q10'),
    margin = dict(l=300))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
print('Crosstab of industry and use of ML methods')
round(pd.crosstab(eda.Q7, eda.Q10, normalize = 'index'),2)
plot_df = multi_count(eda, [col for col in eda if col.startswith('Q11_Part_')])
plot_df_fin = plot_df.loc[plot_df.names.isin(['Other','None of these activities are an important part of my role at work']),:]
plot_df_fin = plot_df_fin.append(plot_df.loc[~plot_df.names.isin(['Other','None of these activities are an important part of my role at work']),:])

data = [go.Bar(
    y = ['None of these activities<br> are an important part of my role at work',
         'Other','Analyze and understand data<br> to influence product or business decisions',
         'Build prototypes to explore<br> applying machine learning to new areas',
         'Build and/or run a machine learning service<br> that operationally improves<br> my product or workflows',
         'Build and/or run the data infrastructure<br> that my business uses for<br> storing, analyzing, and operationalizing data',
         'Do research<br> that advances the state of the art<br> of machine learning'],
    x= plot_df_fin['values'].tolist(),
    orientation = 'h',
    hoverinfo = 'x'
)]
layout = go.Layout(
    title = get_title('Q11_Part_'),
    margin = dict(l=300),
    yaxis = dict(
        showline = True
    )
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
s = pd.Categorical(eda.Q12_MULTIPLE_CHOICE,categories = ['Business intelligence software (Salesforce, Tableau, Spotfire, etc.)',
                                        'Basic statistical software (Microsoft Excel, Google Sheets, etc.)',
                                        'Advanced statistical software (SPSS, SAS, etc.)',
                                        'Local or hosted development environments (RStudio, JupyterLab, etc.)',
                                         'Cloud-based data software & APIs (AWS, GCP, Azure, etc.)',
                                        'Other'],ordered=False)
temp = pd.DataFrame(s.value_counts())
temp.columns = ['size']
temp.sort_values('size', inplace=True)
temp_fin = temp.loc[temp.index=='Other',:]
temp_fin = temp_fin.append(temp.loc[temp.index!='Other',:])


data = [go.Bar(
    y = ['Other',
         'Business intelligence software<br> (Salesforce, Tableau, Spotfire, etc.)',
         'Cloud-based data software & APIs<br> (AWS, GCP, Azure, etc.)',
         'Advanced statistical software<br> (SPSS, SAS, etc.)',
         'Basic statistical software<br> (Microsoft Excel, Google Sheets, etc.)',
         'Local or hosted development environments<br> (RStudio, JupyterLab, etc.)'],
    x= temp_fin.iloc[:,0],
    orientation = 'h',
    hoverinfo = 'x'
)]
layout = go.Layout(
    title = get_title('Q12_MULTIPLE_CHOICE'),
    margin = dict(l=300),
    xaxis = dict(type="category",
            categoryorder= 'array',
             categoryarray= ['Business intelligence software<br> (Salesforce, Tableau, Spotfire, etc.)',
         'Basic statistical software<br> (Microsoft Excel, Google Sheets, etc.)',
         'Advanced statistical software<br> (SPSS, SAS, etc.)',
         'Local or hosted development environments<br> (RStudio, JupyterLab, etc.)',
         'Cloud-based data software & APIs<br> (AWS, GCP, Azure, etc.)','Other'],
            tickangle=20))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q13_Part_')])

data = [go.Bar(
    x = plot_df.names.tolist(),
    y= plot_df['values'].tolist(),
    hoverinfo = 'y'
)]
layout = go.Layout(
    title = get_title('Q13_Part_'),
    yaxis = dict(
        exponentformat = 'none'
    )
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q14_Part_')])

data = [go.Bar(
    x = plot_df.names.tolist(),
    y= plot_df['values'].tolist(),
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = get_title('Q14_Part_'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q15_Part_')])

data = [go.Bar(
    x = ['AWS',
         'GCP',
         'Microsoft Azure',
         'IBM Cloud',
         'Alibaba Cloud',
         'Other','I have not used<br> any cloud providers',],
    y= plot_df['values'].tolist(),
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = get_title('Q15_Part_'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q16_Part_')])

data = [go.Bar(
    x = plot_df.names.tolist(),
    y= plot_df['values'].tolist(),
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = get_title('Q16_Part_'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
temp = pd.DataFrame(eda.Q17.value_counts())
temp.columns = ['counts']
temp_fin = temp.loc[temp.index!='Other',:]
temp_fin = temp_fin.append(temp.loc[temp.index=='Other',:])

data = [go.Bar(
    x = temp_fin.index,
    y= temp_fin.counts.values,
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = get_title('Q17'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
#TOP5
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q28_Part_')])
plot_df_fin = plot_df.loc[~plot_df.names.isin(['Other','None']),:].head(5)
plot_df_fin = plot_df_fin.append(plot_df.loc[plot_df.names.isin(['Other','None']),:])

data = [go.Bar(
    x = ['SAS',
 'Cloudera',
 'RapidMiner',
 'Azure ML Studio',
 'Google Cloud<br>ML Engine',
 'Other',
 'None'],
    y= plot_df_fin['values'].tolist(),
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = str(get_title('Q28_Part_')+' - TOP 5'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
#TOP5
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q29_Part_')])
plot_df_fin = plot_df.loc[~plot_df.names.isin(['Other','None']),:].head(5)
plot_df_fin = plot_df_fin.append(plot_df.loc[plot_df.names.isin(['Other','None']),:])

data = [go.Bar(
    x = plot_df_fin.names.tolist(),
    y= plot_df_fin['values'].tolist(),
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = str(get_title('Q29_Part_')+' - TOP 5'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
languages = ['Python','R','SQL','Java']
plot_df = pd.DataFrame(eda.loc[eda.Q17.isin(languages),['Q17','Q18']].groupby(['Q17','Q18']).size())
plot_df.columns = ['size']
g = plot_df['size'].groupby(level=0, group_keys=False)
plot_df_fin = g.apply(lambda x: x.sort_values(ascending=False).head(3))
fig = {
    'data': [
        {
            'labels': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='Python',:].index.get_level_values('Q18'),
            'values': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='Python',:].values,
            'type': 'pie',
            'name': 'Python',
            'domain': {'x': [0, .48],
                       'y': [.51, 0.92]},
            'hoverinfo':'percent',
            'textinfo':'label',
            'marker': {'colors': colors}
        },
        {
            'labels': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='R',:].index.get_level_values('Q18'),
            'values': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='R',:].values,
            'type': 'pie',
            'name': 'R',
            'domain': {'x': [.48, 1],
                       'y': [.51, 0.92]},
            'hoverinfo':'percent',
            'textinfo':'label',
            'marker': {'colors': colors}

        },
        {
            'labels': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='SQL',:].index.get_level_values('Q18'),
            'values': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='SQL',:].values,
            'type': 'pie',
            'name': 'SQL',
            'domain': {'x': [0, .47],
                       'y': [0, 0.42]},
            'hoverinfo':'percent',
            'textinfo':'label',
            'marker': {'colors': colors}
        },
        {
            'labels': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='Java',:].index.get_level_values('Q18'),
            'values': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='Java',:].values,
            'type': 'pie',
            'name':'Java',
            'domain': {'x': [.47, 1],
                       'y': [0, .42]},
            'hoverinfo':'percent',
            'textinfo':'label',
            'marker': {'colors': colors}
        }
    ],
    'layout': {'title': 'What programming language would you recommend an aspiring data scientist to learn first? <br> If you programme mainly...',
               'showlegend': False,
               "annotations": [
                   {
                       "font": {
                           "size": 16
                       },
                       'align':'center',
                       "showarrow": False,
                       "text": "in Python",
                       "x": 0.2,
                       "y": 1
                   },
                   {
                       "font": {
                           "size": 16
                       },
                       'align':'center',
                       "showarrow": False,
                       "text": "in R",
                       "x": 0.77,
                       "y": 1
                   },
                   {
                       "font": {
                           "size": 16
                       },
                       'align':'center',
                       "showarrow": False,
                       "text": "in SQL",
                       "x": 0.2,
                       "y": 0.46
                   },
                   {
                       "font": {
                           "size": 16
                       },
                       'align':'center',
                       "showarrow": False,
                       "text": "in Java",
                       "x": 0.77,
                       "y": 0.46
                   }
               ]
              }
}

py.offline.iplot(fig)
#TOP5
temp = pd.DataFrame(eda.Q20.value_counts())
temp.columns = ['counts']
temp_fin = temp.loc[temp.index!='Other',:].head(5)
temp_fin = temp_fin.append(temp.loc[temp.index=='Other',:])

data = [go.Bar(
    x = temp_fin.index,
    y= temp_fin.iloc[:,0],
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = str(get_title('Q20')+' - TOP 5'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
languages = ['Python','R','SQL','Java']
plot_df = pd.DataFrame(eda.loc[eda.Q17.isin(languages),['Q17','Q20']].groupby(['Q17','Q20']).size())
plot_df.columns = ['size']
g = plot_df['size'].groupby(level=0, group_keys=False)
plot_df_fin = g.apply(lambda x: x.sort_values(ascending=False).head(3))
colors = ['rgb(43,140,190)','rgb(166,189,219)','rgb(236,231,242)']

fig = {
    'data': [
        {
            'labels': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='Python',:].index.get_level_values('Q20'),
            'values': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='Python',:].values,
            'type': 'pie',
            'name': 'Python',
            'domain': {'x': [0, .48],
                       'y': [.51, 0.92]},
            'hoverinfo':'percent',
            'textinfo':'label',
            'marker': {'colors': colors}
        },
        {
            'labels': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='R',:].index.get_level_values('Q20'),
            'values': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='R',:].values,
            'type': 'pie',
            'name': 'R',
            'domain': {'x': [.48, 1],
                       'y': [.51, 0.92]},
            'hoverinfo':'percent',
            'textinfo':'label',
            'marker': {'colors': colors}

        },
        {
            'labels': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='SQL',:].index.get_level_values('Q20'),
            'values': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='SQL',:].values,
            'type': 'pie',
            'name': 'SQL',
            'domain': {'x': [0, .47],
                       'y': [0, 0.42]},
            'hoverinfo':'percent',
            'textinfo':'label',
            'marker': {'colors': colors}
        },
        {
            'labels': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='Java',:].index.get_level_values('Q20'),
            'values': plot_df_fin.loc[plot_df_fin.index.get_level_values('Q17')=='Java',:].values,
            'type': 'pie',
            'name':'Java',
            'domain': {'x': [.47, 1],
                       'y': [0, .42]},
            'hoverinfo':'percent',
            'textinfo':'label',
            'marker': {'colors': colors}
        }
    ],
    'layout': {'title': 'Which ML library have you used the most? <br> If you programme mainly...',
               'showlegend': False,
               "annotations": [
                   {
                       "font": {
                           "size": 16
                       },
                       'align':'center',
                       "showarrow": False,
                       "text": "in Python",
                       "x": 0.2,
                       "y": 1
                   },
                   {
                       "font": {
                           "size": 16
                       },
                       'align':'center',
                       "showarrow": False,
                       "text": "in R",
                       "x": 0.77,
                       "y": 1
                   },
                   {
                       "font": {
                           "size": 16
                       },
                       'align':'center',
                       "showarrow": False,
                       "text": "in SQL",
                       "x": 0.2,
                       "y": 0.46
                   },
                   {
                       "font": {
                           "size": 16
                       },
                       'align':'center',
                       "showarrow": False,
                       "text": "in Java",
                       "x": 0.77,
                       "y": 0.46
                   }
               ]
              }
}

py.offline.iplot(fig)
temp = pd.DataFrame(eda.Q22.value_counts())
temp.columns = ['counts']
temp_fin = temp.loc[temp.index!='Other',:].head(5)
temp_fin = temp_fin.append(temp.loc[temp.index=='Other',:])

data = [go.Bar(
    x = temp_fin.index,
    y= temp_fin.iloc[:,0],
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = str(get_title('Q22')+' - TOP 5'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
cross_temp  = pd.crosstab(eda.Q22, eda.Q32, normalize = 'columns')
print('Crosstab of type of data and visualization library the mostly used')
round(cross_temp.loc[cross_temp.index.isin(['Matplotlib','ggplot2','Seaborn','Plotly','Shiny']),:],2)
s = pd.Categorical(eda.Q23,categories = ['0% of my time',
                                        '1% to 25% of my time',
                                        '25% to 49% of my time',
                                        '50% to 74% of my time',
                                         '75% to 99% of my time',
                                        '100% of my time'],ordered=False)
temp = pd.DataFrame(s.value_counts())

data = [go.Bar(
    x = ['0%<br> of my time',
         '1% to 25%<br> of my time',
         '25% to 49%<br> of my time',
         '50% to 74%<br> of my time',
         '75% to 99%<br> of my time',
         '100%<br> of my time'],
    y= temp.iloc[:,0],
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = get_title('Q23'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
#TOP5
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q27_Part_')])
plot_df_fin = plot_df.loc[~plot_df.names.isin(['Other','None']),:].head(5)
plot_df_fin = plot_df_fin.append(plot_df.loc[plot_df.names.isin(['Other','None']),:])

data = [go.Bar(
    x = ['AWS<br> EC2',
 'Google<br> Compute Engine',
 'AWS Lambda',
 'Azure Virtual Machines',
 'Google App Engine',
 'Other',
 'None'],
    y= plot_df_fin['values'].tolist(),
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = str(get_title('Q27_Part_')+' - TOP 5'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
#TOP5
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q30_Part_')])
plot_df_fin = plot_df.loc[~plot_df.names.isin(['Other','None']),:].head(5)
plot_df_fin = plot_df_fin.append(plot_df.loc[plot_df.names.isin(['Other','None']),:])

data = [go.Bar(
    x = plot_df_fin.names.tolist(),
    y= plot_df_fin['values'].tolist(),
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = str(get_title('Q30_Part_')+' - TOP 5'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q31_Part_')])

data = [go.Bar(
    x = plot_df.names.tolist(),
    y= plot_df['values'].tolist()
)]
layout = go.Layout(
        title = get_title('Q31_Part_'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
temp = pd.DataFrame(eda.Q32.value_counts())

data = [go.Bar(
    x = temp.index,
    y= temp.iloc[:,0],
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = get_title('Q32'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
eda.loc[:,[col for col in eda if col.startswith('Q34_Part_')]] = eda.loc[:,[col for col in eda if col.startswith('Q34_Part_')]].astype(float)
temp = eda.loc[:,[col for col in eda if col.startswith('Q34_Part_')]].dropna(axis=0)

trace0 = go.Box(
    x = np.array(temp.Q34_Part_1),
    name='Gathering data',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(241,238,246)')
)
trace1 = go.Box(
    x = np.array(temp.Q34_Part_2),
    name = 'Cleaning data',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(208,209,230)')
)
trace2 = go.Box(
    x = np.array(temp.Q34_Part_3),
    name = 'Visualizing data',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(166,189,219)')
)
trace3 = go.Box(
    x = np.array(temp.Q34_Part_4),
    name = 'Model building/model selection',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(116,169,207)')
)
trace4 = go.Box(
    x = np.array(temp.Q34_Part_5),
    name = 'Putting the model into production',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(43,140,190)')
)
trace5 = go.Box(
    x = np.array(temp.Q34_Part_6),
    name = 'Finding insights in the data<br> and communicating with stakeholders',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(4,90,141)')
)
data = [trace0, trace1,trace2,trace3,trace4,trace5]

layout = go.Layout(
    title = get_title('Q34_Part_'),
    yaxis = dict(
        showticklabels = False
    ),
    legend = dict(traceorder = 'reversed')

)

fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
eda.loc[:,[col for col in eda if col.startswith('Q35_Part_')]] = eda.loc[:,[col for col in eda if col.startswith('Q35_Part_')]].astype(float)
temp = eda.loc[:,[col for col in eda if col.startswith('Q35_Part_')]].dropna(axis=0)

trace0 = go.Box(
    x = np.array(temp.Q35_Part_1),
    name='Gathering data',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(241,238,246)')
)
trace1 = go.Box(
    x = np.array(temp.Q35_Part_2),
    name = 'Cleaning data',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(208,209,230)')
)
trace2 = go.Box(
    x = np.array(temp.Q35_Part_3),
    name = 'Visualizing data',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(166,189,219)')
)
trace3 = go.Box(
    x = np.array(temp.Q35_Part_4),
    name = 'Model building/model selection',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(116,169,207)')
)
trace4 = go.Box(
    x = np.array(temp.Q35_Part_5),
    name = 'Putting the model into production',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(43,140,190)')
)
trace5 = go.Box(
    x = np.array(temp.Q35_Part_6),
    name = 'Finding insights in the data<br> and communicating with stakeholders',
    hoverinfo = 'x',
    boxmean = True,
    marker = dict(color = 'rgb(4,90,141)')
)
data = [trace0, trace1,trace2,trace3,trace4,trace5]

layout = go.Layout(
    title = get_title('Q35_Part_'),
    yaxis = dict(
        showticklabels = False
    ),
    legend = dict(traceorder = 'reversed')

)

fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
top_labels = ['Very important','Slightly important',
             'Not at all important','No opinion; I do not know']

eda['Q41_Part_1_new'] = pd.Categorical(eda.Q41_Part_1,categories = top_labels,ordered=False)
eda['Q41_Part_2_new'] = pd.Categorical(eda.Q41_Part_2,categories = top_labels,ordered=False)
eda['Q41_Part_3_new'] = pd.Categorical(eda.Q41_Part_3,categories = top_labels,ordered=False)

colors = ['rgb(5,112,176)','rgb(116,169,207)',
         'rgb(189,201,225)','rgb(241,238,246)']

x0 = eda.Q41_Part_1_new.value_counts().values/sum(eda.Q41_Part_1_new.value_counts().values)*100
x1 = eda.Q41_Part_2_new.value_counts().values/sum(eda.Q41_Part_2_new.value_counts().values)*100
x2 = eda.Q41_Part_3_new.value_counts().values/sum(eda.Q41_Part_3_new.value_counts().values)*100

x_data = [np.round(x0,0),
          np.round(x1,0),
          np.round(x2,0)]

y_data = ['Fairness and bias<br> in ML algorithms',
          'Being able to explain ML model<br> outputs and/or predictions',
          'Reproducibility<br> in data science']


traces = []

for i in range(0, len(x_data[0])):
    for xd, yd in zip(x_data, y_data):
        traces.append(go.Bar(
            x=[xd[i]],
            y=[yd],
            orientation='h',
            marker=dict(
                color=colors[i],
                line=dict(
                        color='rgb(248, 248, 249)',
                        width=1)
            ),
            name = top_labels[i],
        hoverinfo = 'name'
        ))

layout = go.Layout(
    title = get_title('Q41_Part_1'),
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        domain=[0.15, 1]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False
    ),
    barmode='stack',
    #paper_bgcolor='rgb(248, 248, 255)',
    #plot_bgcolor='rgb(248, 248, 255)',
    #margin=dict(
    #    l=120,r=0
    #),
    showlegend=False
)

annotations = []

for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref='paper', yref='y',
                            x=0.14, y=yd,
                            xanchor='right',
                            text=str(yd),
                            font=dict(family='Arial', size=12,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False, align='right'))
    # labeling the first percentage of each bar (x_axis)
    annotations.append(dict(xref='x', yref='y',
                            x=xd[0] / 2, y=yd,
                            text=str(xd[0]) + '%',
                            font=dict(family='Arial', size=14,
                                      color='rgb(248, 248, 255)'),
                            showarrow=False))
    # labeling the first Likert scale (on the top)
    #if yd == y_data[-1]:
        #annotations.append(dict(xref='x', yref='paper',
                                #x=xd[0] / 3, y=1.1,
                                #text=top_labels[0],
                                #font=dict(family='Arial', size=12,
                                #          color='rgb(67, 67, 67)'),
                                #showarrow=False))
    space = xd[0]
    for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=space + (xd[i]/2), y=yd, 
                                    text=str(xd[i]) + '%',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the Likert scale
            #if yd == y_data[-1]:
                #annotations.append(dict(xref='x', yref='paper',
                                        #x=space + (xd[i]/3), y=1.1,
                                        #text=top_labels[i],
                                        #font=dict(family='Arial', size=12,
                                        #          color='rgb(67, 67, 67)'),
                                        #showarrow=False))
            space += xd[i]

layout['annotations'] = annotations

fig = go.Figure(data=traces, layout=layout)
py.offline.iplot(fig)
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q42_Part_')])
plot_df_fin =  plot_df.loc[~plot_df.names.isin(['Other','Not applicable (I am not involved with an organization that builds ML models)']),:]
plot_df_fin =  plot_df_fin.append(plot_df.loc[plot_df.names.isin(['Other','Not applicable (I am not involved with an organization that builds ML models)']),:])

data = [go.Bar(
    x = ['Metrics<br> that consider accuracy',
         'Revenue and/or<br> business goals',
         'Metrics<br> that consider unfair bias',
         'Not applicable',
         'Other'],
    y= plot_df['values'].tolist(),
    hoverinfo = 'y'
)]
layout = go.Layout(
        title = get_title('Q42_Part_'))
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
s = pd.Categorical(eda.Q43,categories = ['0','0-10','10-20',
                                        '20-30','30-40','40-50',
                                         '50-60','60-70','70-80',
                                        '80-90','90-100'],ordered=False)
temp = pd.DataFrame(s.value_counts())

data = [go.Bar(
    x = temp.index,
    y= temp.iloc[:,0],
    hoverinfo = 'y'
)]
layout = go.Layout(
    title = get_title('Q43'),
    yaxis = dict(
        range = (0,2000)
    )
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q44_Part_')]).sort_values('values')
plot_df_fin = plot_df.loc[plot_df.names.isin(['I have never performed this task','I have never found any difficulty in this task']),:]
plot_df_fin = plot_df_fin.append(plot_df.loc[~plot_df.names.isin(['I have never performed this task','I have never found any difficulty in this task']),:])

data = [go.Bar(
    y = ['Difficulty in collecting<br> enough data about groups<br> that may be unfairly targeted',
         'Difficulty in<br> identifying and selecting<br> the appropriate evaluation metrics',
         'Difficulty in<br> identifying groups<br> that are unfairly targeted'
         'Lack of communication<br> between individuals who collect the data<br> and individuals who analyze the data'
         'I have never performed this task',
         'I have never found any difficulty in this task'
    ],
    x= plot_df_fin['values'].tolist(),
    orientation = 'h',
    hoverinfo = 'x'
)]
layout = go.Layout(
    title = get_title('Q44_Part_'),
    xaxis = dict(
        showticklabels = False
    ),
    margin = dict(l=300)
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q45_Part_')]).sort_values('values')
plot_df_fin = plot_df.loc[plot_df.names.isin(['I have never performed this task','I have never found any difficulty in this task']),:]
plot_df_fin = plot_df_fin.append(plot_df.loc[~plot_df.names.isin(['I have never performed this task','I have never found any difficulty in this task']),:])

data = [go.Bar(
    y = ['When building<br> a model that was specifically designed<br> to produce such insights',
         'When determining<br> whether it is worth<br> it to put the model into production',
         'When first exploring<br> a new ML model or dataset',
         'For all models<br> right before putting the model<br> in production',
         'Only for<br> very important models<br> that are already in production',
         'I do not<br> explore and interpret<br> model insights and predictions'],
    x= plot_df_fin['values'].tolist(),
    orientation = 'h',
    hoverinfo = 'x'
)]
layout = go.Layout(
    title = get_title('Q45_Part_'),
    xaxis = dict(
        showticklabels = False
    ),
    margin = dict(l=300)
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
s = pd.Categorical(eda.Q46,categories = ['0','0-10','10-20',
                                        '20-30','30-40','40-50',
                                         '50-60','60-70','70-80',
                                        '80-90','90-100'],ordered=False)
temp = pd.DataFrame(s.value_counts())

data = [go.Bar(
    x = temp.index,
    y= temp.iloc[:,0],
    hoverinfo = 'y'
)]

layout = go.Layout(
    title = get_title('Q46')
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
col_names = [col for col in eda if col.startswith('Q47_Part_')]
col_names = [col for col in col_names if col!='Q47_Part_16']
plot_df =  multi_count(eda, col_names).sort_values('values')
plot_df_fin = plot_df.loc[plot_df.names.isin(['Other','None/I do not use these model explanation techniques']),:]
plot_df_fin = plot_df_fin.append(plot_df.loc[~plot_df.names.isin(['Other','None/I do not use these model explanation techniques']),:])



data = [go.Bar(
    y = plot_df_fin.names.tolist(),
    x= plot_df_fin['values'].tolist(),
    orientation = 'h',
    hoverinfo = 'x'
)]
layout = go.Layout(
    title = get_title('Q47_Part_'),
    xaxis = dict(
        showticklabels = False
    ),
    margin = dict(l=300)
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
temp = pd.DataFrame(eda.Q48.value_counts())
temp.columns = ['count']
temp.sort_values('count', inplace=True)
temp_fin = temp.loc[temp.index=='I do not know; I have no opinion on the matter',:]
temp_fin = temp_fin.append(temp.loc[temp.index!='I do not know; I have no opinion on the matter',:])

data = [go.Bar(
    y = ['I do not know;<br> I have no opinion on the matter',
       'Yes,<br> most ML models are "black boxes"',
       'I am confident<br> that I can explain the outputs<br> of most if not all ML models',
       'I view ML models as "black boxes"<br> but I am confident that<br> experts are able to explain model outputs',
       'I am confident<br> that I can understand and explain<br> the outputs of many<br> but not all ML models'],
    x= temp_fin['count'].tolist(),
    hoverinfo = 'x',
    orientation = 'h'
)]

layout = go.Layout(
    title = get_title('Q48'),
    margin = dict(l=300)
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q49_Part_')]).sort_values('values')
plot_df_fin = plot_df.loc[plot_df.names.isin(['Other','None/I do not make my work easy for others to reproduce']),:]
plot_df_fin = plot_df_fin.append(plot_df.loc[~plot_df.names.isin(['Other','None/I do not make my work easy for others to reproduce']),:])


data = [go.Bar(
    y = ['Other',
         'None',
         'Share code, data, and environment<br> using virtual machines',
         'Share data, code, and environment<br> using a hosted service',
         'Share data, code, and environment<br> using containers',
         'Define relative rather<br> than absolute file paths',
         'Share both data and code on Github<br> or a similar code-sharing repository',
         'Include a text file describing all dependencies',
         'Define all random seeds',
         'Share code on Github<br> or a similar code-sharing repository',
         'Make sure the code is human-readable',
         'Make sure the code is well documented'],
    x= plot_df_fin['values'].tolist(),
    hoverinfo = 'x',
    orientation = 'h'
)]
layout = go.Layout(
    title = get_title('Q49_Part_'),
    margin = dict(l=300)
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)
plot_df =  multi_count(eda, [col for col in eda if col.startswith('Q50_Part_')]).sort_values('values')
plot_df_fin = plot_df.loc[plot_df.names=='Other',:]
plot_df_fin = plot_df_fin.append(plot_df.loc[plot_df.names!='Other',:])


data = [go.Bar(
    y = ['Other',
          'I had never considered<br> making my work easier for others to reproduce',
          'Too expensive',
          'Afraid that others will use my work<br> without giving proper credit',
          'Requires too much technical knowledge',
          'None of these reasons apply to me',
          'Not enough incentives to share my work',
          'Too time-consuming'],
    x= plot_df_fin['values'].tolist(),
    hoverinfo = 'x',
    orientation = 'h'
)]
layout = go.Layout(
    title = get_title('Q50_Part_'),
    margin = dict(l=300)
)
fig = go.Figure(data=data, layout = layout)

py.offline.iplot(fig)