import pandas as pd
import numpy as np

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools

init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

#print(__version__) 
import os
#print(os.listdir('kaggle_survey/'))

mcq = pd.read_csv('../input/multipleChoiceResponses.csv')

col_replace = {'Time from Start to Finish (seconds)' : 'Time required'}
col_tuple = [(14, 21), (22, 28), (29, 44), (45, 56), (57, 64), (65, 83), (88, 107), (110, 123), (130, 150), (151, 194), (195, 223), (224, 249), (250, 262), (265, 276), (277, 283), (284,290), (291, 304), (307, 329), (336, 341), (343, 349), (349, 355), (356, 371), (373, 385), (386, 394)]

for i in col_tuple:
    for j in range(i[0], i[1]):
        col_replace[mcq.columns[j]] = mcq.columns[j][:3] + '_' + mcq[mcq.columns[j]].iloc[0][mcq[mcq.columns[j]].iloc[0].rindex('- ')+2:]

mcq.drop(index = 0, inplace = True)
mcq.rename(columns = col_replace, inplace = True)

mcq['Q3'] = mcq['Q3'].replace('Iran, Islamic Republic of...', 'Iran') 


code_dict = {'Argentina': 'ARG',
 'Australia': 'AUS',
 'Austria': 'AUT',
 'Bangladesh': 'BGD',
 'Belarus': 'BLR',
 'Belgium': 'BEL',
 'Brazil': 'BRA',
 'Canada': 'CAN',
 'Chile': 'CHL',
 'China': 'CHN',
 'Colombia': 'COL',
 'Czech Republic': 'CZE',
 'Denmark': 'DNK',
 'Egypt': 'EGY',
 'Finland': 'FIN',
 'France': 'FRA',
 'Germany': 'DEU',
 'Greece': 'GRC',
 'Hungary': 'HUN',
 'India': 'IND',
 'Indonesia': 'IDN',
 'Iran': 'IRN',
 'Ireland': 'IRL',
 'Israel': 'ISR',
 'Italy': 'ITA',
 'Japan': 'JPN',
 'Kenya': 'KEN',
 'Malaysia': 'MYS',
 'Mexico': 'MEX',
 'Morocco': 'MAR',
 'Netherlands': 'NLD',
 'New Zealand': 'NZL',
 'Nigeria': 'NGA',
 'Norway': 'NOR',
 'Pakistan': 'PAK',
 'Peru': 'PER',
 'Philippines': 'PHL',
 'Poland': 'POL',
 'Portugal': 'PRT',
 'Romania': 'ROU',
 'Russia': 'RUS',
 'Singapore': 'SGP',
 'South Africa': 'ZAF',
 'Spain': 'ESP',
 'Sweden': 'SWE',
 'Switzerland': 'CHE',
 'Thailand': 'THA',
 'Tunisia': 'TUN',
 'Turkey': 'TUR',
 'Ukraine': 'UKR',
 'Hong Kong (S.A.R.)': 'HKG',
 'Republic of Korea': 'PRK',
 'South Korea': 'KOR',
 'United Kingdom of Great Britain and Northern Ireland': 'GBR',
 'United States of America': 'USA',
 'Viet Nam': 'VNM',
 'I do not wish to disclose my location': 'Do not wish to disclose',
 'Other': 'OTHER'}

mcq['Q3_CODE'] = mcq['Q3'].apply(lambda l : code_dict[l])

#mcq.head()
default_codes = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#7cfc00', '#ffa500', '#ff1493', '#adff2f', '#0000cd']

def pie_with_bar(x, y, labels, values, title, xtitle, ytitle, dx = [0.20, 1], dy = [0.20, 1], showlegend = True, legend_pos = 'v', rotation = 0):
    
    
    if legend_pos == 'v':
        legend = dict(orientation = 'v')
    else:
        legend = dict(orientation = 'h', x = 0, y = 0)

    trace1 = go.Bar(x = x, 
                    y = y, 
                    text = y,
                    hoverinfo = 'text',
                    marker = dict(color = default_codes),
                    textposition = 'auto',
                    showlegend = False)


    trace2 = go.Pie(labels = labels, 
                    values = values, 
                    domain = dict(x = dx, 
                                  y = dy),
                    hoverinfo = 'label+percent',
                    marker = dict(colors = default_codes),
                    hole = 0.40,
                    sort = False,
                    showlegend = showlegend,
                    rotation = rotation)

    layout = go.Layout(dict(title = title,
                           xaxis = dict(title = xtitle),
                           yaxis = dict(title = ytitle),
                           legend = legend))
    fig = dict(data = [trace1, trace2], layout = layout)
    iplot(fig)
    
def stacked_bar(index, column, title, legend_pos = 'v', extra_suffix = 'Object', showlegend = True):
    c_mat = count_percent_mat(index, column, 12, extra_suffix)
    p_mat = c_mat[1]
    c_mat = c_mat[0]
    data = []
    
    if legend_pos == 'v':
        legend = dict(orientation = 'v')
    else:
        legend = dict(orientation = 'h')
    for i in c_mat.columns:
        data.append(go.Bar(x = p_mat.index,
                           y = p_mat[i],
                           name = i,
                           text = c_mat[i].apply(str) + '<br>' + p_mat[i].apply(lambda l : format(l, '.2f')) + '%',
                           hoverinfo = 'text+name',
                           showlegend = showlegend)
                   )
    layout = go.Layout(dict(barmode = 'stack',
                           title = title,
                           yaxis = dict(title = 'Percentage'),
                           legend = legend))
    fig = go.Figure(data, layout)
    iplot(fig)
    
def multi_stacked_bar(index, column, title, legend_pos = 'v', extra_suffix = 'Objects'):
    c_mat = multi_count_percent_mat(index, column, 12, extra_suffix)
    p_mat = c_mat[1]
    c_mat = c_mat[0]
    data = []
    if legend_pos == 'v':
        legend = dict(orientation = 'v')
    else:
        legend = dict(orientation = 'h')
    for i in c_mat.columns:
        data.append(go.Bar(x = p_mat.index,
                           y = p_mat[i],
                           name = i,
                           text =  c_mat[i].apply(str) + '<br>' + p_mat[i].apply(lambda l : format(l, '.2f')) + '%',
                           hoverinfo = 'text+name',
                           showlegend = True)
                   )
    layout = go.Layout(dict(barmode = 'stack', 
                           title = title,
                           yaxis = dict(title = 'Percentage'),
                           legend = legend))
    fig = go.Figure(data, layout)
    iplot(fig)


from sklearn.preprocessing import LabelEncoder

def draw_map(index, title):
    if type(index) == str:
        c_mat = count_percent_mat('Q3', index)
    else:
        c_mat = multi_count_percent_mat('Q3', index)
    p_mat = c_mat[1].transpose()
    c_mat = c_mat[0].transpose()

    del c_mat['Other']
    del c_mat['I do not wish to disclose my location']
    del p_mat['Other']
    del p_mat['I do not wish to disclose my location']
    
    c_mat.sort_index(inplace = True)
    p_mat.sort_index(inplace = True)

    l = LabelEncoder()
    l.fit(c_mat.index)


    c_list = []
    l_list = []
    z_list = []
    t_list = []

    for i in c_mat.columns:
        c_list.append(i)
        z_list.append(l.transform([c_mat[i].idxmax()])[0])
        t = i+'<br>Max count, '+c_mat[i].idxmax()+' : '+str(max(c_mat[i]))+', '+format(max(p_mat[i]), '.2f')+' %'
        for (x, y, z) in zip(c_mat.index, c_mat[i], p_mat[i]):
            t += '<br>'+x+' : '+str(y)+', '+format(z, '.2f')+' %'
        t_list.append(t)

    l_list = list(map(lambda l : code_dict[l], c_list))

    data = dict(type='choropleth',
                locations = l_list,
                z = z_list,
                text = t_list,
                hoverinfo = 'text',
                autocolorscale = False,
                colorscale = 'Jet',
                showscale = False
                ) 
    
    title = '<b>' + title + '</b><br>Hover over for more details'
    layout = dict(title = title,
                  geo = dict(
                showframe = False,
                showcoastlines = False,
                showocean = True,
                oceancolor = '#3f3f4f',
                projection = dict(
                type = 'robinson')))
    choromap = go.Figure(data = [data],layout = layout)
    iplot(choromap)

def box_dist(index, columns, title):
    columns = list(columns)
    columns.append(index)
    d = mcq[columns].sort_values(index)
    columns.remove(index)

    traces = []

    for i in range(len(columns)):
        traces.append(go.Box(
                             x = d[index],
                             y = d[columns[i]],
                             fillcolor = default_codes[i],
                             showlegend = False))

    columns = list(map(lambda l : l[4:], columns))
    for i in range(len(columns)):
        if len(columns[i]) > 30:
            t = list(map(lambda l : l + ' ', columns[i].split()))
            j = 0
            k = 0
            columns[i] = ''
            while j <= 30:
                j += len(t[k])
                j += 1
                k += 1
            columns[i] = ''.join(t[:k - 1]) + '<br>' + ''.join(t[k - 1:])
            columns[i].rstrip()
    
    fig = tools.make_subplots(rows = len(columns) // 2,
                              cols = 2,  
                              shared_xaxes = True,
                              subplot_titles = columns,
                              vertical_spacing = 0.05)
                  
    for i in range(len(traces)):
           fig.append_trace(traces[i], (i // 2) + 1, (i % 2) + 1)
        
    fig['layout'].update(height = 1200, width = 900, title = title)        
    iplot(fig)            


# for single column

def count_percent_mat(index, column, limit = 0, suffix = 'Objects'):

    group = mcq.groupby([column, index]).count()['Time required']

    indexu = mcq[index].unique()
    indexu = indexu[~pd.isnull(indexu)]
    indexu.sort()
    colu = mcq[column].unique()
    colu = colu[~pd.isnull(colu)].tolist()
    
    if limit == 0 or limit >= len(colu):
        limit = len(colu)

    col_list = mcq.groupby(column).count().sort_values('Time required', ascending = False).index.tolist()
    if 'Other' in col_list:
        col_list.remove('Other')
        col_list.append('Other')
    col_list = col_list[:limit]

    col_len = len(col_list)

    if limit > 0 and limit < len(colu):
        others = 'Other ' + suffix
        col_list += [others]
        
    count_mat = pd.DataFrame(np.zeros((len(indexu), len(col_list))), index = indexu, columns = col_list)
    per_mat = pd.DataFrame(np.zeros((len(indexu), len(col_list))), index = indexu, columns = col_list)

    for i in range(limit):
        for j in group.loc[col_list[i]].index:
            count_mat.loc[j][col_list[i]] = group.loc[col_list[i]][j]
        colu.remove(col_list[i])

    # for 'other<suffix>' columns, if limit 0, nothing left in colu
    for i in colu:
        for j in group.loc[i].index:
            count_mat.loc[j][others] += group.loc[i][j]
    
    for i in count_mat.index:
        total = sum(count_mat.loc[i])
        for j in count_mat.columns:
            per_mat.loc[i][j] = (count_mat.loc[i][j] / total) * 100
    
    return (count_mat, per_mat)


# for (select all that apply) questions
# below method calculates percentage w.r.t. total non-nan values.
# For example, if a group has members a, b, c, d and there an attribute T with multiple selectable values x, y and z.
# a selects x and y, b selects y, z, c selects all x, y, z.
# percentage users of x is 50.0%, y is 75.0%, z is 50.0%.
# But with multiple options selected, it becomes more as a set problem and a bit complicated to interpret with large number of values of T.
# Thus, I have simply considered non - nan values for each values of x, y, z, and calculate percentage w.r.t. sum of all those non - nan values.
# This gives more direct comparison between x, y, z where I calculate which value (x, y, z) has majority.
# It also dissloves an uncertainty of 'd' not selecting any value, which is possible if question never appeared to 'd' as d could have selected negative answer to some 
# previous question OR could have selected values like 'None' or 'Other' which I have not considered for some questions. 
# Also makes easy for comparison between different groups like (a, b, c, d). 

def multi_count_percent_mat(index, columns, limit = 0, suffix = 'Objects'):

    count_mat = mcq.groupby(index).count()[columns]
    count_mat.columns = list(map(lambda l : l[4:], count_mat.columns))

    if limit >= len(count_mat.columns):
        limit = 0

    if limit > 0 and limit < len(count_mat.columns):
        l = []
        for i in count_mat.columns:
            l.append(sum(count_mat[i]))
        
        count_mat.loc['Total'] = l
        count_mat.sort_values('Total', axis = 1, ascending = False, inplace = True)
        if 'Other' in count_mat.columns:
            t = count_mat['Other']
            del count_mat['Other']
            count_mat['Other'] = t
        
        others = 'Other '+suffix
        count_mat[others] = np.zeros(len(count_mat))
    
        for i in count_mat.columns[limit:-1]:
            count_mat[others] += count_mat[i]
            del count_mat[i]
        count_mat.drop(index = 'Total', inplace = True)  

    per_mat = pd.DataFrame(np.zeros((len(count_mat.index), len(count_mat.columns))), index = count_mat.index, columns = count_mat.columns)
    for i in count_mat.index:
        total = sum(count_mat.loc[i])
        for j in count_mat.columns:
            per_mat.loc[i][j] = (count_mat.loc[i][j] / total) * 100
    return (count_mat, per_mat)

data = mcq.groupby('Q3').count()[['Time required']]
data.sort_values('Time required', ascending = False, inplace = True)

t = data.loc['Other']
data.drop(index = 'Other', inplace = True)
data = data.append(t)

data.drop(index = 'I do not wish to disclose my location', inplace = True)

c = 0
for i in data.index[13:]:
    c += data.loc[i]['Time required']

data = data[:13]
t.name = 'Rest of World'
t['Time required'] = c
data = data.append(t)

pie_with_bar(data.index, data['Time required'], data.index, data['Time required'], 'Countries', '', 'Count', [0.35, 1], [0.25, 0.9], showlegend=True, legend_pos = 'h', rotation = 180)
draw_map('Q1', 'Gender')
d = count_percent_mat('Q3', 'Q1')[1]
d.sort_values('Female', ascending = False)
# to hide O/P
draw_map('Q4', 'Highest Level of Education')
draw_map('Q5', 'Undergraduate Major')
draw_map('Q6', 'JobTitles')
draw_map('Q8', 'Years of Experience in current role')
draw_map(mcq.columns[29:44], 'IDEs (by use)')
draw_map(mcq.columns[45:54], 'Hosted Notebooks (by use)')
draw_map(mcq.columns[57:62], 'Cloud Services (by use)')
draw_map('Q17', 'Programming Languages')
draw_map('Q18', 'Recommended Language to young aspirants')
draw_map('Q20', 'ML Framework')
draw_map('Q22', 'Data Visualization Libraries')
draw_map('Q23', 'Time spent on coding')
draw_map('Q24', 'Years writing code to analyse data')
draw_map('Q25', 'Years using Machine Learning Algorithm')
draw_map(mcq.columns[130:148], 'Cloud Computing Framework (by use)')
draw_map(mcq.columns[195:221], 'Relational Database (by use)')
draw_map(mcq.columns[224:247], 'BigData and Analytics Products (by use)')
draw_map('Q32', 'Type of Data')
draw_map(mcq.columns[265:274], 'Find Public Datasets (by use)')
draw_map('Q37', 'Online Learning Platform')
draw_map('Q40', 'Independent Projects v/s Academic Achievements')
draw_map(mcq.columns[336:341], 'Metrics that determine ML model\'s sucess (by use)')
d = mcq['Q2'].unique()
d.sort()
for i in d[:-2]:
    print(i, end = ', ')
print(d[-2], 'and', d[-1],'\b.')
data = mcq.groupby('Q2').count()
pie_with_bar(data.index, data['Time required'], data.index, data['Time required'], 'Age-wise count', 'Age Group', 'Count', [0.55, 1], showlegend=True)

draw_map('Q2', 'What country has what majority of age group on Kaggle?')
stacked_bar('Q2', 'Q4', 'Highest Level of Education', legend_pos = 'h')
# is not displayed well when notebook is published
stacked_bar('Q2', 'Q5', 'Undergraduate major<br>Hover over bars for legends', legend_pos = 'h', extra_suffix = 'Undergraduate majors', showlegend = False)
stacked_bar('Q2', 'Q6', 'Job Titles', extra_suffix = 'JobTitles')
stacked_bar('Q2', 'Q8', 'Years of Experience in Current Role')
multi_stacked_bar('Q2', mcq.columns[45:54], 'Hosted Notebooks')
stacked_bar('Q2', 'Q17', 'Programming Languages', extra_suffix = 'Languages')
stacked_bar('Q2', 'Q18', 'Recommended Language for Young Aspirants', legend_pos = 'h', extra_suffix = 'Languages')
stacked_bar('Q2', 'Q20', 'Machine Language Framework', extra_suffix = 'ML Frmaeworks')
stacked_bar('Q2', 'Q22', 'Data Visualization Library')
stacked_bar('Q2', 'Q32', 'Types of Data', legend_pos = 'h')
box_dist('Q2', mcq.columns[277:283], 'Proportion of time devoted to various Data Science Tasks')
box_dist('Q2', mcq.columns[284:290], 'Proportion of training')
stacked_bar('Q2', 'Q48', 'Do you consider ML models as black box?', legend_pos = 'h')
multi_stacked_bar('Q2', mcq.columns[343:349], 'Difficulty in identifying if ML model is fair/unbiased', legend_pos = 'h')
multi_stacked_bar('Q2', mcq.columns[356:371], 'Methods for explaining ML models output', extra_suffix = 'methods')
multi_stacked_bar('Q2', mcq.columns[386:394], 'Barriers Preventing from Share Coding', legend_pos = 'h')
from wordcloud import WordCloud

import matplotlib.pyplot as plt 
%matplotlib inline

import string
from nltk.corpus import stopwords

def normalize_text(text):
    
    # lowercase it
    text = text.lower()
    # remove punctuation
    text = ''.join([t if t not in string.punctuation else ' ' for t in text])
    # remove stopwords
    text = [t for t in text.split() if t not in stopwords.words('english')]
    # return text
    return ' '.join(text)

def make_text(col):
    text = ''
    for i in col:
        if pd.notnull(i):
            text += ' ' + normalize_text(i)
    return text

def generate_wordcloud(col):
    wordcloud = WordCloud(background_color = 'black', height = 1500, width = 2350, random_state = 21)
    wordcloud.generate(make_text(col))
    plt.figure(figsize=(15, 7))
    plt.axis('off')
    plt.imshow(wordcloud)

freeres = pd.read_csv('../input/freeFormResponses.csv')
freeres.drop(index = 0, inplace = True)
generate_wordcloud(freeres['Q1_OTHER_TEXT'])
generate_wordcloud(freeres['Q6_OTHER_TEXT'])
generate_wordcloud(freeres['Q7_OTHER_TEXT'])
generate_wordcloud(freeres['Q11_OTHER_TEXT'])
generate_wordcloud(freeres['Q12_OTHER_TEXT'])
generate_wordcloud(freeres['Q13_OTHER_TEXT'])
generate_wordcloud(freeres['Q14_OTHER_TEXT'])
generate_wordcloud(freeres['Q15_OTHER_TEXT'])
generate_wordcloud(freeres['Q16_OTHER_TEXT'])
generate_wordcloud(freeres['Q17_OTHER_TEXT'])
generate_wordcloud(freeres['Q18_OTHER_TEXT'])
generate_wordcloud(freeres['Q19_OTHER_TEXT'])
generate_wordcloud(freeres['Q20_OTHER_TEXT'])
generate_wordcloud(freeres['Q21_OTHER_TEXT'])
generate_wordcloud(freeres['Q22_OTHER_TEXT'])
generate_wordcloud(freeres['Q27_OTHER_TEXT'])
generate_wordcloud(freeres['Q28_OTHER_TEXT'])
generate_wordcloud(freeres['Q29_OTHER_TEXT'])
generate_wordcloud(freeres['Q30_OTHER_TEXT'])
generate_wordcloud(freeres['Q31_OTHER_TEXT'])
generate_wordcloud(freeres['Q32_OTHER'])
generate_wordcloud(freeres['Q33_OTHER_TEXT'])
generate_wordcloud(freeres['Q35_OTHER_TEXT'])
generate_wordcloud(freeres['Q36_OTHER_TEXT'])
generate_wordcloud(freeres['Q37_OTHER_TEXT'])
generate_wordcloud(freeres['Q38_OTHER_TEXT'])
generate_wordcloud(freeres['Q42_OTHER_TEXT'])
generate_wordcloud(freeres['Q49_OTHER_TEXT'])
generate_wordcloud(freeres['Q50_OTHER_TEXT'])
