import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from matplotlib import gridspec
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
from wordcloud import WordCloud, STOPWORDS

import warnings
warnings.filterwarnings('ignore')
#plt.style.use('fivethirtyeight') #fivethirtyeight
plt.rcParams.update({'font.size':12,
                    'xtick.labelsize':14,
                    'ytick.labelsize':14})
path = '../input/'
#path = 'dataset/'
df_2018 = pd.read_csv(path+ 'multipleChoiceResponses.csv',) #header=[0,1])
response = pd.read_csv(path + 'freeFormResponses.csv',) #header=[0,1])
schema = pd.read_csv(path+ 'SurveySchema.csv')
print('Number of rows and columns in multipleChoiceResponses 2018 dataset:', df_2018.shape)
print('Number of rows and columns in freeFormResponses  dataset:', response.shape)
print('Number of rows and columns in schema dataset:', schema.shape)
df = df_2018[1:]
df.head()
def Horizontal_bar_plot(df, column, name='',title='', limit=None , colorscale = 'Picnic',width = 900, height = 500):
    tmp = df[column].value_counts()[:limit]
    #tmp = tmp.sort_values()
    tmp_per = round(tmp * 100/ tmp.sum() , 2)
    tmp_per = [str(v)+' %' for v in tmp_per]
    # Plot
    trace1 = go.Bar(y = tmp.values, x = tmp.index, name=name,orientation='v',
        marker=dict(color=tmp.values, colorscale = colorscale, line=dict(color='rgb( 127, 140, 141)',width=2),),
        text = tmp_per, textposition='outside',
    )
    #Layout
    layout = dict(
        title=title,
        width = width,height = height,
        yaxis=dict(automargin=True,),
        paper_bgcolor='rgb(251, 252, 252)',
        plot_bgcolor='rgb(251, 252, 252)'
    )
    fig = {'data':[trace1], 'layout':layout}
    py.iplot(fig)
def Horitontal_Multi_Barplot(df, column, column_filter,title ='', height = 600, width = 850,
                             name = ['Student','Data Scientist','Data Analyst'],limit = None,):
    """ Bar plot"""
    colors = ['rgb (240,128,128)','rgb(0,255,255)','rgb(186,85,211)','rgb(210,105,30)','rgb(0,0,205)',
              'rgb(124,252,0)','rgb(255,99,71)',]
    # Layout
    fig = tools.make_subplots(rows= len(name), cols=1, #subplot_titles= tuple(name),
                              vertical_spacing = 0.05, horizontal_spacing = 0.05,
                              print_grid= False,shared_xaxes = True)
    
    fig['layout'].update(dict( 
        showlegend =False,
        height = height,
        width = width,
        title = title,
        paper_bgcolor='rgb(251, 252, 252)',
        plot_bgcolor='rgb(250, 250, 255)'))

    # Multi Plot
    for i, c in enumerate(name):
        #tmp = df[column].value_counts()[:limit]
        tmp = df[df[column_filter] == c][column].value_counts()[:limit]
        tmp_per = round(tmp * 100/ tmp.sum() , 2)
        tmp_per = [str(v)+' %' for v in tmp_per]
        
        # Plot
        trace1 = go.Bar(y = tmp.values, x = tmp.index, name= c,orientation='v',
            marker=dict(color = colors[i],line=dict(color='rgb( 127, 140, 141)',width=2),),
            text = tmp_per, textposition='auto', textfont = dict(size =13,family = 'Droid')
        )
        
        fig.append_trace(trace1, i+1,1)
        fig.layout[f'yaxis{i+1}'].update(title = c)

    # Final plot
    py.iplot(fig)
# 4 Pie plot
def Pie_plot_agg(filter_column = '',column = '',name = [], title = '', width = 1000, height= 600):
    
    """Draw four pie plot of
    filter_column is to filter out the perticual category mentioned in name variable
    column: This target column upon which pie plot is drwan
    name: Four different category of filter_column
    """
    # trace1
    tmp = (df[df[filter_column] == name[0]][column])
    tmp = tmp.value_counts(ascending = True)
    trace1 = go.Pie(labels= tmp.index, values= tmp.values, hoverinfo='label+percent+name', 
                    name = name[0],hole= .5, domain= dict(x = [0, 0.46], y = [0.54, 1]))
    
    # trace2
    tmp = (df[df[filter_column] == name[1]][column])
    tmp = tmp.value_counts(ascending = True)
    trace2= go.Pie(labels= tmp.index, values= tmp.values, hoverinfo='label+percent+name', 
                    name = name[1],hole= .5, domain= dict(x = [0.54,1],y = [0.54, 1]))
    #trace3
    tmp = (df[df[filter_column] == name[2]][column])
    tmp = tmp.value_counts(ascending = True)
    trace3 = go.Pie(labels= tmp.index, values= tmp.values, hoverinfo='label+percent+name', 
                    name = name[2],hole= .5, domain= dict(x = [0, 0.46],y = [0, 0.46]))
    #trace4
    tmp = (df[df[filter_column] == name[3]][column])
    tmp = tmp.value_counts(ascending = True)
    trace4 = go.Pie(labels= tmp.index, values= tmp.values, hoverinfo='label+percent+name', 
                    name = name[3],hole= .5, domain= dict(x = [0.54, 1],y = [0, 0.46]))

    # Layout
    layout = go.Layout(title = title, width = width, height = height,
                       annotations = [dict(font = dict(size=20)),
                                      dict(showarrow =False, text=name[0],x = 0.18, y=0.78),
                                      dict(font = dict(size=20)),
                                      dict(showarrow= False, text=name[1],x = 0.82, y=0.78),
                                      dict(font = dict(size=20)),
                                      dict(showarrow= False, text=name[2],x = 0.17, y=0.2),
                                      dict(font = dict(size=20)),
                                      dict(showarrow= False, text=name[3],x = 0.82, y=0.2),
                                ])
    fig = go.Figure(data = [trace1, trace2, trace3, trace4], layout= layout)
    py.iplot(fig)
# Venn diagram
def Venn2_diagram(df,columns):
    """ Venn diagram of 2 sets"""
    # Subset count
    label = df[columns].mode().values[0]
    subsets = (
        len(df[(df[columns[0]] == label[0]) & (df[columns[1]] != label[1])]), #A
        len(df[(df[columns[0]] != label[0]) & (df[columns[1]] == label[1])]), #B
        len(df[(df[columns[0]] == label[0]) & (df[columns[1]] == label[1])]), #A.B
             )
    return venn2(subsets = subsets, set_labels= label)

def Venn3_diagram(df,columns):
    """ Venn diagram of 3 sets"""
    # Subset count
    label = df[columns].mode().values[0]
    subsets = (
        len(df[(df[columns[0]] == label[0]) & (df[columns[1]] != label[1]) & (df[columns[2]] != label[2])]), #A
        len(df[(df[columns[0]] != label[0]) & (df[columns[1]] == label[1]) & (df[columns[2]] != label[2])]), #B
        len(df[(df[columns[0]] == label[0]) & (df[columns[1]] == label[1]) & (df[columns[2]] != label[2])]), #A.B
        len(df[(df[columns[0]] != label[0]) & (df[columns[1]] != label[1]) & (df[columns[2]] == label[2])]), #C
        len(df[(df[columns[0]] == label[0]) & (df[columns[1]] != label[1]) & (df[columns[2]] == label[2])]), #A.C
        len(df[(df[columns[0]] != label[0]) & (df[columns[1]] == label[1]) & (df[columns[2]] == label[2])]), #B.C
        len(df[(df[columns[0]] == label[0]) & (df[columns[1]] == label[1]) & (df[columns[2]] == label[2])]), #A.B.C
             )
    return venn3(subsets = subsets, set_labels= label)
#
tmp = df['Time from Start to Finish (seconds)'].astype('int')/60
tmp = tmp[tmp<100]
print('Mean time to anwser the quetions is:',round(np.mean(tmp),2), 'minutes')

# Plot
trace1 = go.Histogram(x = tmp, #nbinsx= 30, 
                      marker= dict(color='rgb(255, 65, 54)', line=dict(color='rgb( 127, 140, 141)',width=0.5)))
layout = dict(
        title='Duration in minute',
        width = 800,
        height = 400,
        xaxis = dict(autorange=True),
        yaxis=dict(automargin=True),
        paper_bgcolor='rgb(251, 252, 252)',
        plot_bgcolor='rgb(251, 252, 252)'
        )
fig = {'data':[trace1], 'layout':layout}
py.iplot(fig)
def Map(tmp, title = '', colorscale = 'Viridis',):
    """ Geo map:"""
    data =  dict( type = 'choropleth',
                locations = tmp.index,
                z = tmp.values,
                text = tmp.index,
                locationmode = 'country names',
                colorscale = colorscale,
                autocolorscale = False,
                reversescale = True,
                marker = dict( line = dict (
                        color = 'rgb(180,180,180)',width = 0.3
                    ) ),
                colorbar = dict( autotick = False,
                    title = 'Response'),
          ) 

    layout = dict(
        title = title,
        geo = dict(showland = True,
                   landcolor = "rgb(250, 250, 250)",
            showframe = False,
            showcoastlines = True,
            projection = dict( type = 'Mercator')
        ))

    fig = dict( data=[data], layout=layout )

    py.iplot( fig, validate=False, filename='world-map' )
print(df_2018['Q3'][0])
tmp = df['Q3'].value_counts()
title = '2018 Kaggle Survey - Response'
Map(tmp, title = title, colorscale= 'Viridis')
print(df_2018['Q3'][0])
Horizontal_bar_plot(df, column= 'Q3',
                   title = 'Top 20 Countries Response', limit = 20)
df['Q3'].nunique()
print(df_2018['Q6'][0])
title = 'Current Role'
Horizontal_bar_plot(df, column = 'Q6', title = title, colorscale= 'Rainbow')
print(df_2018['Q3'][0],'\n',df_2018['Q6'][0],)
title = 'Country Vs Current Role'
Pie_plot_agg(filter_column= 'Q3',
            column = 'Q6',
            title = title,
            name = ['United States of America','India', 'China', 'Russia'],
            width = 1000, height =800)
print(df_2018['Q7'][0])
column = 'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'
title = 'Current Employer'
Horizontal_bar_plot(df, column= 'Q7', title = title)
title = 'Current Industry'
Horitontal_Multi_Barplot(df, column= 'Q7', column_filter= 'Q6',title = title,limit=None, width= 1000)
title = 'Work Experience'
Horizontal_bar_plot(df, column= 'Q8', name='Year',title= title, limit=None, )
title = 'Work Experience of Data Professionals'
Horitontal_Multi_Barplot(df, column= 'Q8', column_filter= 'Q6',title = title,limit=None, width= 1000)
# Role, Industry, Country
#print(df_2018['Q3'])
#df.groupby().agg({column[2]:'count'}).rename(columns={column[2]:'count'}).reset_index()
print(df_2018['Q1'][0],'\n',df_2018['Q2'][0],)
tmp = df['Q1'].value_counts(sort =True)
tmp_per = round(tmp/df.shape[0] * 100, 2)
tmp_per = [str(v)+' %' for v in tmp_per]
trace1 = go.Bar(x = tmp.index, y = tmp.values, name='People',
               marker = dict(color = 'rgb(93, 164, 214)',line = dict(color = 'rgb( 127, 140, 141)', width =2)),
                text= tmp_per, textposition='auto')

tmp = df['Q2'].value_counts(sort =False)
tmp = tmp.sort_index()
tmp_per = round(tmp/df.shape[0] * 100,2)
tmp_per = [str(v)+' %' for v in tmp_per]
trace2 = go.Bar(x = tmp.index, y = tmp.values, name = 'People',
                marker = dict(color = 'rgb(255, 65, 54)',line = dict(color = 'rgb( 127, 140, 141)', width =2)),
               text = tmp_per, textposition='auto')

fig  = tools.make_subplots(rows= 1, cols=2, subplot_titles = ('Gender', 'Age'))
fig.append_trace(trace1, 1,1)
fig.append_trace(trace2, 1,2)
fig.layout.showlegend =False
fig.layout.height = 500
py.iplot(fig)
male = (df
       .loc[(df['Q1'] == 'Male') ]
      .groupby(['Q2'])
       .agg({'Q2':'count'})
      .rename(columns={'Q2':'count'})
      .reset_index()
      ) 

female = (df
       .loc[(df['Q1'] == 'Female') ]
      .groupby(['Q2'])
       .agg({'Q2':'count'})
      .rename(columns={'Q2':'count'})
      .reset_index()
      ) 

# Plot
values = male['count'].values /(male['count'].values + female['count'].values)
trace1 = go.Bar(x = male['Q2'], y = values,name = 'Male')

values = female['count'].values /(male['count'].values + female['count'].values)
trace2 = go.Bar(x = female['Q2'], y = values, name = 'Female')

layout = go.Layout(barmode= 'stack', height =300 ,title = 'Percentage male and female by age')
fig = go.Figure(data = [trace1, trace2], layout= layout)
py.iplot(fig)
title = 'Gender'
Horitontal_Multi_Barplot(df, column= 'Q1', column_filter= 'Q6',title = title,height=500)
title = 'Age'
Horitontal_Multi_Barplot(df, column= 'Q2', column_filter= 'Q6',title = title)
print(df_2018['Q5'][0])
title = 'Graduate Major'
Horizontal_bar_plot(df, column= 'Q5', name='Year',title= title, height= 500)
title = 'Graduate Major'
Horitontal_Multi_Barplot(df, column= 'Q5', column_filter= 'Q6',title = title,)
Pie_plot_agg(filter_column= 'Q3',
            column = 'Q5',
             title = 'Graduate major from top 4 countries',
            name = ['United States of America','India', 'China', 'Russia'],
            width = 1100, height =800)
print(df_2018['Q4'][0])
title = 'Highest Level Of Education'
Horizontal_bar_plot(df, column= 'Q4', name='Year',title= title)
title = 'Highest level of education'
Horitontal_Multi_Barplot(df, column= 'Q4', column_filter= 'Q6',title = title,)
print(df_2018['Q9'][0])
tmp = df['Q9'].value_counts()
#tmp = tmp.sort_index()
index = ['I do not wish to disclose my approximate yearly compensation', '0-10,000', '10-20,000', 
         '20-30,000', '30-40,000', '40-50,000', '50-60,000',  '60-70,000', '70-80,000', 
         '80-90,000', '90-100,000', '100-125,000', '125-150,000',  '150-200,000', '200-250,000', 
         '250-300,000', '300-400,000', '400-500,000', '500,000+']
tmp = tmp.reindex(index[::-1])
tmp_per = round(tmp * 100/ tmp.sum() , 2)
tmp_per = [str(v)+' %' for v in tmp_per]
tmp = tmp.rename(index={'I do not wish to disclose my approximate yearly compensation': "Don't Disclose"})
# Plot
trace1 = go.Bar(
    x = tmp.values, y = tmp.index,
    marker = dict(color = 'rgb(255, 65, 54)',line = dict(color = 'rgb( 127, 140, 141)', width =2)),
    name='$',orientation='h',
    text = tmp_per, textposition='inside'
)
#Layout
layout = dict(
    title='Current yearly compensation in $',
    width = 900,height = 700,
    yaxis=dict(automargin=True),
    paper_bgcolor='rgba(245,245,245,0.4)',
    plot_bgcolor='rgba(255,250,250,0.5)'
    )
fig = {'data':[trace1], 'layout':layout}
py.iplot(fig)
title = 'Current Yearly Compensation in USD $'
Horitontal_Multi_Barplot(df, column= 'Q9', column_filter= 'Q6',title = title)
print(df_2018['Q10'][0])
tmp = df['Q10'].value_counts(ascending = True)
trace1 = go.Pie(labels= tmp.index, values= tmp.values, hoverinfo='label+percent+name', 
                    marker = dict(colors =['magma']), name = 'Employer',
                    hole= .5, domain= dict(x = [0, 0.38], ))
# Layout
layout = go.Layout(title = 'Incorporate machine learning methods into business', width = 900, height = 500, 
                   annotations = [dict(font = dict(size=20)),
                                  dict(showarrow =False, text= 'Deploy ML',x = 0.15, y=0.5),
                            ])
fig = go.Figure(data = [trace1], layout= layout)
py.iplot(fig)
#column_filter = 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice'
#column = 'Does your current employer incorporate machine learning methods into their business?'
#title = '' 
#Horitontal_Stacked_barplot(df, column= column, column_filter= column_filter,title = title)
def Selection_choice_bar_plot(column_start = '', title = '', color = 'rgb(255, 65, 54)',width= 1000, height=400):
    """Selection_choice_bar_plot: For Multplie choice questions"""
    columns = df.columns[df.columns.str.startswith(column_start)]

    # Count options
    option_count = pd.DataFrame()
    for c in columns[:-1]:
        value = df[c].value_counts()
        option_count = pd.concat([option_count,value])

    option_count = option_count.rename(columns={0:'Count'})
    option_count = option_count.sort_values(by='Count')
    tmp_per = round(option_count * 100/ option_count.sum(), 2)
    tmp_per = [str(v)+' %' for v in  tmp_per['Count']]
    
    # Bar Plot
    trace1 = go.Bar(x = option_count['Count'].values, y = option_count.index, orientation='h',
                    text = tmp_per, textposition='inside',
                    marker=dict(color= option_count['Count'].values, colorscale = 'Rainbow', 
                               line=dict(color='rgb( 127, 140, 141)',width=2),)
                   )
    
    # Layout
    layout = dict(
            title=title,
            width = width,
            height = height,
            yaxis = dict(
                automargin=True,
                showticklabels=True,
            ),
            paper_bgcolor='rgb(251, 252, 252)',
            plot_bgcolor='rgb(251, 252, 252)'
            )
    fig = {'data':[trace1], 'layout':layout}
    py.iplot(fig)
def Selection_choice_Pie_plot(column_start, title = '', text = '', width= 900, height= 500):
    
    columns = df.columns[df.columns.str.startswith(column_start)]
    # Count options
    option_count = pd.DataFrame()
    for c in columns[:-1]:
        value = df[c].value_counts()
        option_count = pd.concat([option_count,value])

    option_count = option_count.rename(columns={0:'Count'})
    option_count = option_count.sort_values(by='Count')
    
    # Plot
    trace1 = go.Pie(labels= option_count.index, values= option_count['Count'].values, 
                    hoverinfo='label+percent+name', marker = dict(colors =['magma']), name = '',
                        hole= .5, domain= dict(x = [0, 0.5], ))
    
    # Layout
    layout = go.Layout(title = title, width = width, height = height, 
                       annotations = [dict(font = dict(size=20)),
                                      dict(showarrow =False, text=text,x = 0.2, y=0.5),
                                ])
    fig = go.Figure(data = [trace1], layout= layout)
    py.iplot(fig)
def Barplot_of_Q_Regular_MostOften(df, column1 ='', column2 = '', title = '', height = 500, width=900):
    """ Barplot of Most often and Regualar """
    columns = df.columns[df.columns.str.startswith(column1)]

    fig  = tools.make_subplots(rows= 1, cols=2, subplot_titles = ('Regular', 'Most Often'),
                               print_grid= False)

    # Count options
    option_count = pd.DataFrame()
    for c in columns[:-1]:
        value = df[c].value_counts()
        option_count = pd.concat([option_count,value])

    option_count = option_count.rename(columns={0:'Count'})
    option_count = option_count.sort_values(by='Count')
    tmp_per = round(option_count * 100/ option_count.sum(), 2)
    tmp_per = [str(v)+' %' for v in  tmp_per['Count']]

    # Bar Plot 1
    trace1 = go.Bar(x = option_count['Count'].values, y = option_count.index, orientation='h', name = '',
                    text = tmp_per, textposition='inside',
                   marker = dict(color = 'rgb(65,105,225)', line = dict(color = 'rgb( 127, 140, 141)', width =2)))
    # Bar Plot 2
    #column= 'Q17'
    tmp = df[column2].value_counts(ascending = True)
    tmp_per = round(tmp * 100/ tmp.sum(), 2)
    tmp_per = [str(v)+' %' for v in tmp_per]
    trace2 = go.Bar(x = tmp.values, y = tmp.index, orientation='h', name = '',
                    text = tmp_per, textposition='inside',
                   marker = dict(color = 'rgb(255, 65, 54)', line = dict(color = 'rgb( 127, 140, 141)', width =2)))

    fig.append_trace(trace1, 1,1)
    fig.append_trace(trace2, 1,2)
    fig['layout'].update(dict( 
            showlegend =False,
            height = height,
            width = width,
            title = title,
            paper_bgcolor='rgb(251, 252, 252)',
            plot_bgcolor='rgb(250, 250, 255)'))

    py.iplot(fig)
title = "Integrated Development Environments (IDE's)"
column_start = 'Q13_Part'
Selection_choice_bar_plot(column_start= column_start, title=title, width=1000, height= 600, 
                          color= 'rgb(65,105,225)')
columns = ['Q13_Part_1','Q13_Part_2','Q13_Part_9']
filter_label = ['Student', 'Data Scientist', 'Data Analyst']

plt.figure(figsize = (14,10))
gs = gridspec.GridSpec(3,3)

# Make Venn diagram
ax = plt.subplot(gs[0:2,:])
Venn3_diagram(df, columns)
plt.title("Integrated Development Environments (IDE's): Total")

# bottom 3 plot
for i, c in enumerate(filter_label):
    tmp = df[df['Q6'] == filter_label[i]]
    ax = plt.subplot(gs[2,i])
    Venn3_diagram(tmp, columns)
    plt.title(filter_label[i])
print(df_2018['Q16_Part_1'][0][:100])
title = 'Programming Laguage preference'
Barplot_of_Q_Regular_MostOften(df, column1='Q16_Part', column2= 'Q17', title=title,height=700)
columns = ['Q16_Part_1','Q16_Part_2','Q16_Part_3']
filter_label = ['Student', 'Data Scientist', 'Data Analyst']

plt.figure(figsize = (14,10))
gs = gridspec.GridSpec(3,3)

# Make Venn diagram
ax = plt.subplot(gs[0:2,:])
Venn3_diagram(df, columns)
plt.title("Programming Laguage preference: Total")

# bottom 3 plot
for i, c in enumerate(filter_label):
    tmp = df[df['Q6'] == filter_label[i]]
    ax = plt.subplot(gs[2,i])
    Venn3_diagram(tmp, columns)
    plt.title(filter_label[i])
plt.savefig('language.png')
tmp = df['Q18'].value_counts(ascending = True)

# Plot
trace1 = go.Pie(labels= tmp.index, values= tmp.values, hoverinfo='label+percent+name', 
                    marker = dict(colors =['magma']), name = 'Language',
                    hole= .5, domain= dict(x = [0, 1]))

# Layout
layout = go.Layout(title = 'Programming Language', width = 900, height = 500, 
                   annotations = [dict(font = dict(size=20)),
                                  dict(showarrow =False, text= 'Language',x = 0.5, y=0.5),
                            ])
fig = go.Figure(data = [trace1], layout= layout)
py.iplot(fig)
print(df_2018['Q19_Part_1'][0][:100])
title = 'Machine Learnig Libraries'
Barplot_of_Q_Regular_MostOften(df, column1='Q19_Part', column2= 'Q20', title=title,height=700)
columns = ['Q19_Part_2','Q19_Part_3','Q19_Part_4']
filter_label = ['Student', 'Data Scientist', 'Data Analyst']

plt.figure(figsize = (14,10))
gs = gridspec.GridSpec(3,3)

# Make Venn diagram
ax = plt.subplot(gs[0:2,:])
Venn3_diagram(df, columns)
plt.title("Deep Learnig Library: Total")

# bottom 3 plot
for i, c in enumerate(filter_label):
    tmp = df[df['Q6'] == filter_label[i]]
    ax = plt.subplot(gs[2,i])
    Venn3_diagram(tmp, columns)
    plt.title(filter_label[i])
print(df_2018['Q21_Part_1'][0][:100])
title = 'Machine Learnig Libraries'
Barplot_of_Q_Regular_MostOften(df, column1='Q21_Part', column2= 'Q22', title=title,height=600)
columns = ['Q21_Part_8','Q21_Part_2','Q21_Part_1']
filter_label = ['Student', 'Data Scientist', 'Data Analyst']

plt.figure(figsize = (14,10))
gs = gridspec.GridSpec(3,3)

# Make Venn diagram
ax = plt.subplot(gs[0:2,:])
Venn3_diagram(df, columns)
plt.title("Data visualization: Total")

# bottom 3 plot
for i, c in enumerate(filter_label):
    tmp = df[df['Q6'] == filter_label[i]]
    ax = plt.subplot(gs[2,i])
    Venn3_diagram(tmp, columns)
    plt.title(filter_label[i])
columns = ['Q23','Q24','Q25','Q26']
print(df_2018[columns].iloc[0].values)

name = ['Coding','Analyze Data','ML in School', 'Data Scientist?']
title = ''

# Plot
trace = []
for c in columns:
    tmp = df[c].value_counts(sort = True)
    tmp_per = round(tmp *100 / tmp.sum(),2)
    trace1 = go.Bar(x = tmp.index, y = tmp.values, text = tmp_per.values, textposition = 'outside', name = '',
                   marker=dict(color= tmp.values, colorscale = 'Rainbow', 
                               line=dict(color='rgb( 127, 140, 141)',width=2))
                   )
    trace.append(trace1)

# Layout
fig  = tools.make_subplots(rows= 2, cols=2, subplot_titles = ('Coding in school','Analyze Data',
                                                              'ML in School', 'Data Scientist?'))
fig.append_trace(trace[0], 1,1)
fig.append_trace(trace[1], 1,2)
fig.append_trace(trace[2], 2,1)
fig.append_trace(trace[3], 2,2)
fig.layout.showlegend =False
fig.layout.height = 1000
fig.layout.xaxis.automargin = True
py.iplot(fig)

print(df_2018['Q33_Part_1'][0])
title = 'Public data resource'
text = 'Data Repository'
Selection_choice_Pie_plot(column_start= 'Q33_Part', title = title, text=text,height = 500, width= 900)
columns = df.columns[df.columns.str.startswith('Q34_Part')]
#print(df_2018[columns].iloc[0].values)

# Plot
trace = []
name = pd.Series(df_2018[columns].iloc[0].values).str.split('-').apply(lambda x: x[1])
for c in columns:
    tmp = df[c]
    trace1 = go.Histogram(x = tmp, nbinsx= 20, name = '',
                      marker = dict(color='rgb(255, 65, 54)',
                                    line = dict(color='rgb( 127, 140, 141)',width=1)))
    trace.append(trace1)

# Layout
fig  = tools.make_subplots(rows= 2, cols=3, subplot_titles = (name.values[:-1]))
fig.append_trace(trace[0], 1,1)
fig.append_trace(trace[1], 1,2)
fig.append_trace(trace[2], 1,3)
fig.append_trace(trace[3], 2,1)
fig.append_trace(trace[4], 2,2)
fig.append_trace(trace[5], 2,3)
fig.layout.showlegend =False
fig.layout.height = 600
fig.layout.width = 1200
fig.layout.xaxis.automargin = False
py.iplot(fig)
print(df_2018['Q36_Part_1'][0][:100])
title = 'Machine Learnig Libraries'
Barplot_of_Q_Regular_MostOften(df, column1='Q36_Part', column2= 'Q37', title=title,height=600)
columns = ['Q36_Part_2','Q36_Part_1','Q36_Part_6']
filter_label = ['Student', 'Data Scientist', 'Data Analyst']

plt.figure(figsize = (14,10))
gs = gridspec.GridSpec(3,3)

# Make Venn diagram
ax = plt.subplot(gs[0:2,:])
Venn3_diagram(df, columns)
plt.title("MOOC: Total")

# bottom 3 plot
for i, c in enumerate(filter_label):
    tmp = df[df['Q6'] == filter_label[i]]
    ax = plt.subplot(gs[2,i])
    Venn3_diagram(tmp, columns)
    plt.title(filter_label[i])
# What percentage of your current machine learning/data science training falls under each category?
columns = df.columns[df.columns.str.startswith('Q35_Part')]

name = pd.Series(df_2018[columns].iloc[0].values).str.split(')').apply(lambda x: x[1])

# Plot
trace = []
for i,c in enumerate(columns):
    tmp = df[c]
    #tmp = tmp[tmp>0]
    trace1 = go.Histogram(x = tmp, nbinsx= 20, name = name[i],
                     marker = dict(color='rgb(65,105,225)',line = dict(color='rgb( 127, 140, 141)',width=1)))
    trace.append(trace1)

# Layout
fig  = tools.make_subplots(rows= 2, cols=3, subplot_titles = tuple(name.values))
fig.append_trace(trace[0], 1,1)
fig.append_trace(trace[1], 1,2)
fig.append_trace(trace[2], 1,3)
fig.append_trace(trace[3], 2,1)
fig.append_trace(trace[4], 2,2)
fig.append_trace(trace[5], 2,3)

fig.layout.showlegend =False
fig.layout.height = 600
fig.layout.xaxis.automargin = False
py.iplot(fig)
#column_start= 'How do you perceive the quality of online learning plcolumns = df.columns[df.columns.str.startswith(column_start)'
#title = 'Quality of MOOC vs Traditional Institution'
#Selection_choice_Pie_plot(column_start= column_start, title= title,width=900, height=500)
columns = df.columns[df.columns.str.startswith('Q39_Part')]
title = 'Quality of MOOC vs Traditional Institution'

# Count options
option_count = pd.DataFrame()
for c in columns[:-1]:
    value = df[c].value_counts()
    option_count = pd.concat([option_count,value])

option_count = option_count.rename(columns={0:'Count'})
option_count = option_count.sort_values(by='Count')

# Plot
trace1 = go.Pie(labels= option_count.index, values= option_count['Count'].values, 
                hoverinfo='label+percent+name', marker = dict(colors =['magma']), name = 'Language',
                hole= .5, domain= dict(x = [0, 1]))

layout = go.Layout(title = title, width = 900, height = 500, 
                   annotations = [dict(font = dict(size=20)),
                                  dict(showarrow =False, text='',x = 0.2, y=0.5),
                            ])
fig = go.Figure(data = [trace1], layout= layout)
py.iplot(fig)
title = 'Top Data Science Blog'
Selection_choice_bar_plot(column_start='Q38_Part',title = title, height = 600)
columns = df.columns[df.columns.str.startswith('Q38')]
df[columns].mode()
columns = ['Q38_Part_4','Q38_Part_18','Q38_Part_11']
filter_label = ['Student', 'Data Scientist', 'Data Analyst']

plt.figure(figsize = (14,10))
gs = gridspec.GridSpec(3,3)

# Make Venn diagram
ax = plt.subplot(gs[0:2,:])
Venn3_diagram(df, columns)
plt.title("Machine Learnig Blog: Total")

# bottom 3 plot
for i, c in enumerate(filter_label):
    tmp = df[df['Q6'] == filter_label[i]]
    ax = plt.subplot(gs[2,i])
    Venn3_diagram(tmp, columns)
    plt.title(filter_label[i])
print(df_2018['Q15_Part_1'][0])
title = 'Cloud Computing for Machine Learning Work'
text = 'Cloud'
Selection_choice_Pie_plot(column_start='Q15_Part',title= title, text = text, width=800, height=400)
# ML frame work
print(df_2018['Q27_Part_1'][0])
column_start= 'Q27'
title = 'Cloud'
columns = df.columns[df.columns.str.startswith(column_start)]

# Count options
option_count = pd.DataFrame()
for c in columns[:-1]:
    value = df[c].value_counts()
    option_count = pd.concat([option_count,value])

option_count = option_count.rename(columns={0:'Count'})
option_count = option_count.sort_values(by='Count')
tmp_per = round(option_count * 100/ option_count.sum(), 2)

# Bar Plot 1
trace1 = go.Bar(x = option_count['Count'].values, y = option_count.index, orientation='h', name = '',
                text = tmp_per['Count'].values, textposition='inside',
               marker = dict(color = 'rgb(65,105,225)', line = dict(color = 'rgb( 127, 140, 141)', width =2)))

# Ploting library
print(df_2018['Q28_Part_1'][0])
column_start= 'Q28_Part'
#title = 'Programming Laguage preference'
columns = df.columns[df.columns.str.startswith(column_start)]

# Count options
option_count = pd.DataFrame()
for c in columns[:-1]:
    value = df[c].value_counts()
    option_count = pd.concat([option_count,value])

option_count = option_count.rename(columns={0:'Count'})
option_count = option_count.sort_values(by='Count')
tmp_per = round(option_count * 100/ option_count.sum(), 2)

# Bar Plot 1
trace2 = go.Bar(x = option_count['Count'].values, y = option_count.index, orientation='h', name = '',
                text = tmp_per['Count'].values, textposition='inside',
               marker = dict(color = 'rgb(255, 65, 54)', line = dict(color = 'rgb( 127, 140, 141)', width =2)))

fig  = tools.make_subplots(rows= 1, cols=2, subplot_titles = ('Cloud Resource', 'Machine Learning Product'))
fig.append_trace(trace1, 1,1)
fig.append_trace(trace2, 1,2)
fig.layout.showlegend =False
fig.layout.yaxis.automargin = True
fig.layout.height = 800
fig.layout.width = 1100
py.iplot(fig)
# ML frame work
column_start= 'Which of the following relational database products have you used at work or school in the last 5 years?'
title = 'Cloud'
columns = df.columns[df.columns.str.startswith('Q29_Part')]

# Count options
option_count = pd.DataFrame()
for c in columns[:-1]:
    value = df[c].value_counts()
    option_count = pd.concat([option_count,value])

option_count = option_count.rename(columns={0:'Count'})
option_count = option_count.sort_values(by='Count')
tmp_per = round(option_count * 100/ option_count.sum(), 2)

# Bar Plot 1
trace1 = go.Bar(x = option_count['Count'].values, y = option_count.index, orientation='h', name = '',
                text = tmp_per['Count'].values, textposition='inside',
               marker = dict(color = 'rgb(65,105,225)', line = dict(color = 'rgb( 127, 140, 141)', width =2)))

# Ploting library
column_start= 'Which of the following big data and analytics products have you used at work or school in the last 5 years?'
#title = 'Programming Laguage preference'
columns = df.columns[df.columns.str.startswith('Q30_Part')]

# Count options
option_count = pd.DataFrame()
for c in columns[:-1]:
    value = df[c].value_counts()
    option_count = pd.concat([option_count,value])

option_count = option_count.rename(columns={0:'Count'})
option_count = option_count.sort_values(by='Count')
tmp_per = round(option_count * 100/ option_count.sum(), 2)

# Bar Plot 1
trace2 = go.Bar(x = option_count['Count'].values, y = option_count.index, orientation='h', name = '',
                text = tmp_per['Count'].values, textposition='inside',
               marker = dict(color = 'rgb(255, 65, 54)', line = dict(color = 'rgb( 127, 140, 141)', width =2)))

fig  = tools.make_subplots(rows= 1, cols=2, subplot_titles = ('Relational database management system', 
                                                              'Big Data'))
fig.append_trace(trace1, 1,1)
fig.append_trace(trace2, 1,2)
fig.layout.showlegend =False
fig.layout.yaxis.automargin = True
fig.layout.height = 700
fig.layout.width = 1000
py.iplot(fig)
title = 'Hosted Notebooks'
text = 'Site'
print(df_2018['Q14_Part_1'][0])
Selection_choice_Pie_plot(column_start= 'Q14_Part', title = title, text=text,height = 500, width= 900)
print(df_2018['Q42_Part_1'].iloc[0])

title = 'The metric to determine Model Performance'
text = 'Model Performance Metric'
Selection_choice_Pie_plot(column_start= 'Q42_Part', title= title,text =text)
print(df_2018['Q11_Part_1'].iloc[0][:100])
title = 'Select any activities that make up an important part of your role at work'
text = 'Activity'
Selection_choice_Pie_plot(column_start= 'Q11_Part', title= title, text= text, width = 1100)
tmp = (df['Q43'].value_counts())

# Plot
trace1 = go.Bar(x = tmp.index, y = tmp.values, 
                marker= dict(color='rgb(220,20,60)', line=dict(color='rgb( 0, 0, 20)',width=2)))
layout = dict(
        title='Exploring unfair bias',
        width = 600,
        height = 400,
        xaxis = dict(autorange=True),
        yaxis=dict(automargin=True),
         paper_bgcolor='rgba(245,245,245,0.4)',
        plot_bgcolor='rgba(255,250,250,0.5)'
        )
fig = {'data':[trace1], 'layout':layout}
py.iplot(fig)
print(df_2018['Q44_Part_1'].iloc[0][:90])
title = 'Difficulty in analyzing algorithm is fair and unbiased' 
text = ''
Selection_choice_Pie_plot(column_start= 'Q44_Part', title= title, width= 1000,)
print(df_2018['Q45_Part_1'][0][:95])
title = 'Model interpretion' 
text = 'Model interpretion'
Selection_choice_Pie_plot(column_start=  'Q45_Part', title= title, width= 900,)
tmp = (df['Q46'].value_counts(sort = True))

# Plot
trace1 = go.Bar(x = tmp.index, y = tmp.values, 
                marker= dict(color='rgb(220,20,60)', line=dict(color='rgb( 0, 0, 20)',width=2)))
layout = dict(
        title='Exporing model insights',
        width = 600,
        height = 400,
        xaxis = dict(autorange=True),
        yaxis=dict(automargin=True),
         paper_bgcolor='rgba(245,245,245,0.4)',
        plot_bgcolor='rgba(255,250,250,0.5)'
        )
fig = {'data':[trace1], 'layout':layout}
py.iplot(fig)
print(df_2018['Q47_Part_1'][0])
title = 'Evaluation Metirc'
Selection_choice_bar_plot(column_start= 'Q47_Part',title = title, width = 900, height = 600)
print(df_2018['Q49_Part_1'][0])
Selection_choice_Pie_plot(column_start='Q49_Part',width= 1200)
column_start = 'What barriers prevent you from making your work even easier to reuse and reproduce?'
title = 'Barriers in reuse and reproduce the previous work'
text = ''
Selection_choice_Pie_plot(column_start= 'Q50_Part', title= title,text =text, width= 800, height= 400)
#column_start = 'Do you consider ML models to be "black boxes" with outputs that are difficult or impossible to explain?'
#Selection_choice_Pie_plot(column_start= column_start,)
tmp = (df['Q48'].value_counts())
trace1 = go.Bar(x = tmp.values, y = tmp.index, orientation= 'h',
               marker = dict(color = 'rgb(10,30,231)'),
               text = tmp.values, textposition='outside')
layout = dict(
        title=' Is ML model Black boxes or White boxes',
        width = 1300,
        height = 400,
        yaxis=dict(automargin=True),
         paper_bgcolor='rgba(245,245,245,0.4)',
        plot_bgcolor='rgba(255,250,250,0.5)'
        )
fig = {'data':[trace1], 'layout':layout}
py.iplot(fig)
columns = df.columns[df.columns.str.startswith('Q12')]
print(df_2018['Q12_MULTIPLE_CHOICE'][0])
# Count options
option_count = pd.DataFrame()
for c in columns:
    value = df[c].value_counts()
    option_count = pd.concat([option_count,value])

option_count = option_count.rename(columns={0:'Count'})
option_count = option_count.sort_values(by='Count')
option_count = option_count.loc[['Business intelligence software (Salesforce, Tableau, Spotfire, etc.)',
                'Cloud-based data software & APIs (AWS, GCP, Azure, etc.)',
                 'Advanced statistical software (SPSS, SAS, etc.)',
                 'Other',
                 'Basic statistical software (Microsoft Excel, Google Sheets, etc.)',
                 'Local or hosted development environments (RStudio, JupyterLab, etc.)']]
# Plot
trace1 = go.Pie(labels= option_count.index, values= option_count['Count'].values, 
                hoverinfo='label+percent+name', marker = dict(colors =['magma']), name = '',
                    hole= .5, domain= dict(x = [0, 0.5], ))

# Layout
layout = go.Layout(title = 'Primary tool to analyze dataset', width = 800, height = 500, 
                   annotations = [dict(font = dict(size=20)),
                                  dict(showarrow =False, text='Tools',x = 0.2, y=0.5),
                            ])
fig = go.Figure(data = [trace1], layout= layout)
py.iplot(fig)
# other than above type mentioned
column_start = 'Q12_OTHER_TEXT'
wc = (WordCloud(height=400,width=1400, max_words=1000, stopwords=STOPWORDS,
                colormap='rainbow',background_color='White'
              ).generate(' '.join(response[column_start].dropna().astype(str))))

plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.savefig('wc.png')
plt.axis('off')
plt.title('Activity');
#pd.crosstab(df['Q6'],df['Q5']).style.background_gradient(cmap='cool')