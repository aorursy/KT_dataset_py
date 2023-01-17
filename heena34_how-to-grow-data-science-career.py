import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
import itertools
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from numpy import array
from matplotlib import cm
import missingno as msno
import cufflinks as cf
from wordcloud import WordCloud, STOPWORDS

from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 5000)
pd.set_option('display.max_rows', None)
cf.go_offline()
freeFormResponse = pd.read_csv('../input/freeFormResponses.csv')
mcqResponse = pd.read_csv('../input/multipleChoiceResponses.csv')
schema = pd.read_csv('../input/SurveySchema.csv')
mcqResponse = mcqResponse[1:].reset_index(drop=True)
def filter_data(df):
    return df[~df['Q3'].isin(['Other', 'I do not wish to disclose my location'])].reset_index(drop=True)
world_map = filter_data(mcqResponse)
world_map_count = world_map['Q3'].value_counts()
data = [ dict(
        type = 'choropleth',
        locations = world_map_count.index,
        locationmode = 'country names',
        z = world_map_count.values,
        text = world_map_count.values,
        colorscale = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(190,190,190)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Survey Participation'),
      ) ]

layout = dict(
    title = 'Participation in survey from different countries',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False)


print('Other: ', mcqResponse[mcqResponse['Q3'] == 'Other']['Q3'].count(), ', Dont want to disclose: ',
      mcqResponse[mcqResponse['Q3'] == 'I do not wish to disclose my location']['Q3'].count())
mcqResponse['Q1'].value_counts().plot.bar(title='Gender Count')
role = mcqResponse[mcqResponse['Q6'] != 'Other']['Q6'].value_counts()
role.iplot(kind='barh', title='Developer Type', margin=go.Margin(l=160))

print('Other: ', mcqResponse[mcqResponse['Q6'] == 'Other']['Q6'].count())
client_specific_indistry_DS = mcqResponse[((mcqResponse['Q6'] == 'Data Scientist')&(mcqResponse['Q7'] != 'Other'))]['Q7']
client_specific_indistry = mcqResponse[mcqResponse['Q7'] != 'Other']['Q7'].value_counts().to_frame().reset_index()
role = client_specific_indistry_DS.value_counts().to_frame().reset_index()


trace0 = go.Bar(
    x = list(client_specific_indistry['Q7']),
    y = list(client_specific_indistry['index']),
    orientation = 'h'
)

trace1 = go.Bar(
    x = list(role['Q7']),
    y = list(role['index']),
    orientation = 'h'
)


fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Client Specific Industry', 'Client Specific Industry for Data Science'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=1000,margin=go.Margin(l=250))
py.iplot(fig)


print('Other: ', mcqResponse[mcqResponse['Q7'] == 'Other']['Q7'].count())
formal_edu_DS = mcqResponse[mcqResponse['Q6'] == 'Data Scientist']['Q4'].value_counts().to_frame().reset_index()
formal_edu = mcqResponse['Q4'].value_counts().to_frame().reset_index()


trace0 = go.Bar(
    x = list(formal_edu['Q4']),
    y = list(formal_edu['index']),
    orientation = 'h'
)

trace1 = go.Bar(
    x = list(formal_edu_DS['Q4']),
    y = list(formal_edu_DS['index']),
    orientation = 'h'
)


fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Formal Education', 'Formal Education of Data Scientist'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=500,margin=go.Margin(l=370))
py.iplot(fig)


experience_DS = mcqResponse[mcqResponse['Q6'] == 'Data Scientist']['Q8'].value_counts().to_frame().reset_index()
experience = mcqResponse['Q8'].value_counts().to_frame().reset_index()


trace0 = go.Bar(
    x = list(experience['Q8']),
    y = list(experience['index']),
    orientation = 'h'
)

trace1 = go.Bar(
    x = list(experience_DS['Q8']),
    y = list(experience_DS['index']),
    orientation = 'h'
)


fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Experience (in yrs)', 'Experience of Data Scientists (in yrs)'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=500,margin=go.Margin(l=50))
py.iplot(fig)


series = mcqResponse['Q10'].value_counts()
series.iplot(kind='barh', title='Does Current Employer Incorporate Machine Learning?', margin=go.Margin(l=560))
cols = ['Q11_Part_1', 'Q11_Part_2', 'Q11_Part_3', 'Q11_Part_4', 'Q11_Part_5', 'Q11_Part_6', 'Q11_Part_7']
skills_df = {}
skills_df['Skills'] = []
skills_df['Count'] = []
for col in cols:
    name = str(mcqResponse[col].dropna().unique()).strip("[]''")
    count = mcqResponse[mcqResponse[col] == name][col].count()
    skills_df['Skills'].append(name)
    skills_df['Count'].append(count)
    
skills_df = pd.DataFrame(skills_df)
skills_df.sort_values(by=['Count'], inplace=True)

trace0 = go.Bar(
    x = skills_df['Count'],
    y = skills_df['Skills'],
    orientation = 'h'
)

layout = go.Layout(
    margin = dict(l=700),
    title = 'Respondents important role at work'
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)

print('Select all that apply')
tools_used = mcqResponse[mcqResponse['Q12_MULTIPLE_CHOICE'] != 'Other']['Q12_MULTIPLE_CHOICE'].value_counts()
tools_used.iplot(kind='barh', title='Mostly used Tool', margin=go.Margin(l=450))

print('Other: ', mcqResponse[mcqResponse['Q12_MULTIPLE_CHOICE'] == 'Other']['Q12_MULTIPLE_CHOICE'].count())
cols = ['Q13_Part_1', 'Q13_Part_2', 'Q13_Part_3', 'Q13_Part_4', 'Q13_Part_5', 'Q13_Part_6', 'Q13_Part_7',
        'Q13_Part_8', 'Q13_Part_9', 'Q13_Part_10', 'Q13_Part_11', 'Q13_Part_12', 'Q13_Part_13', 'Q13_Part_14',
        'Q13_Part_15']

ide_df = {}
ide_df['IDE'] = []
ide_df['Count'] = []
for col in cols:
    name = str(mcqResponse[col].dropna().unique()).strip("[]''")
    count = mcqResponse[mcqResponse[col] == name][col].count()
    ide_df['IDE'].append(name)
    ide_df['Count'].append(count)
    
ide_df = pd.DataFrame(ide_df)
ide_df.sort_values(by=['Count'], inplace=True)
trace0 = go.Bar(
    x = ide_df['Count'],
    y = ide_df['IDE'],
    orientation = 'h'
)

layout = go.Layout(
    margin = dict(l=120),
    title = 'Most Preferred IDE in last 5 years'
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)

print('Select all that apply')
def generate_choices_df(cols, to_skip_cols):
    df = {}
    df['Choice'] = []
    df['Count'] = []
    for col in cols:
        name = str(mcqResponse[~mcqResponse[col].isin(to_skip_cols)][col].dropna().unique()).strip("[]''")
        count = mcqResponse[mcqResponse[col] == name][col].count()
        df['Choice'].append(name)
        df['Count'].append(count)

    df = pd.DataFrame(df)
    df.sort_values(by=['Count'], inplace=True)
    
    return df

def plot_choices(df, title, left_margin=50):
    trace0 = go.Bar(
        x = df['Count'],
        y = df['Choice'],
        orientation = 'h'
    )

    layout = go.Layout(
        margin = dict(l=left_margin),
        title = title
    )
    trace = [trace0]
    fig = go.Figure(data=trace, layout=layout)
    py.iplot(fig)

    print('Select all that apply')
    return None
columns =  ['Q14_Part_1', 'Q14_Part_2', 'Q14_Part_3', 'Q14_Part_4', 'Q14_Part_5', 'Q14_Part_6',
            'Q14_Part_7', 'Q14_Part_8', 'Q14_Part_9', 'Q14_Part_10', 'Q14_Part_11']
notebook_choices_df = generate_choices_df(columns, ['Other', 'None'])
plot_choices(notebook_choices_df, 'Most Preferred Notebook in last 5 years', left_margin=150)
columns =  ['Q15_Part_1', 'Q15_Part_2', 'Q15_Part_3', 'Q15_Part_4', 'Q15_Part_5', 'Q15_Part_6', 'Q15_Part_7']
cloud_choices_df = generate_choices_df(columns, ['I have not used any cloud providers'])
plot_choices(cloud_choices_df, 'Most Preferred Cloud Computing Service in last 5 years', left_margin=250)

print(mcqResponse[mcqResponse['Q15_Part_6'] == 'I have not used any cloud providers']['Q15_Part_6'].count(), ''' people have not used any Cloud Provider''')
lang_used_columns =  ['Q16_Part_1', 'Q16_Part_2', 'Q16_Part_3', 'Q16_Part_4', 'Q16_Part_5', 'Q16_Part_6', 'Q16_Part_7',
                       'Q16_Part_8', 'Q16_Part_9', 'Q16_Part_10', 'Q16_Part_11', 'Q16_Part_12', 'Q16_Part_13', 'Q16_Part_14',
                       'Q16_Part_15', 'Q16_Part_16', 'Q16_Part_17', 'Q16_Part_18']
lang_choices_df = generate_choices_df(lang_used_columns, ['None'])
lang_recommended = mcqResponse['Q18'].value_counts().to_frame().reset_index()

f, ax = plt.subplots(2, 2, figsize=(50,40))

sns.barplot(list(lang_choices_df['Count']), list(lang_choices_df['Choice']), ax=ax[0,0])

for index, value in enumerate(lang_choices_df['Count']):
    ax[0,0].text(0.8, index, value, color='k', fontsize=25)
    
ax[0,0].set_title('Programming Languages USed', fontsize=35)
ax[0,0].set_yticklabels(lang_choices_df['Choice'], fontsize=25)


sns.barplot(list(lang_recommended['Q18']), list(lang_recommended['index']), ax=ax[1,0])

for index, value in enumerate(lang_recommended['Q18']):
    ax[1,0].text(0.8, index, value, color='k', fontsize=25)
    
ax[1,0].set_title('Recommended Programming Languages', fontsize=35)
ax[1,0].set_yticklabels(lang_recommended['index'], fontsize=25)


word_cloud = WordCloud(height=460, width=300).generate("".join(mcqResponse['Q17'].dropna()))
ax3 = plt.subplot2grid((2,2), (0,1), rowspan=2)
ax3.imshow(word_cloud)
ax3.axis('off')
ax3.set_title('Specific Programming Language Used', fontsize=45)
plt.show()


columns_ml =  ['Q19_Part_1', 'Q19_Part_2', 'Q19_Part_3', 'Q19_Part_4', 'Q19_Part_5', 'Q19_Part_6', 'Q19_Part_7',
           'Q19_Part_8', 'Q19_Part_9', 'Q19_Part_10', 'Q19_Part_11', 'Q19_Part_12', 'Q19_Part_13', 'Q19_Part_14',
           'Q19_Part_15', 'Q19_Part_16', 'Q19_Part_17', 'Q19_Part_18', 'Q19_Part_19']

columns_vis =  ['Q21_Part_1', 'Q21_Part_2', 'Q21_Part_3', 'Q21_Part_4', 'Q21_Part_5', 'Q21_Part_6', 'Q21_Part_7',
           'Q21_Part_8', 'Q21_Part_9', 'Q21_Part_10', 'Q21_Part_11', 'Q21_Part_12', 'Q21_Part_13']

framework_choices_df = generate_choices_df(columns_ml, ['None'])
vis_lib_choices_df = generate_choices_df(columns_vis, ['None'])

trace1 = go.Bar(
    x=framework_choices_df['Count'],
    y=framework_choices_df['Choice'],
    orientation = 'h'
)
trace2 = go.Bar(
    x=vis_lib_choices_df['Count'],
    y=vis_lib_choices_df['Choice'],
    orientation = 'h'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Most Preferred Machine Learning Framework',
                                                          'Mostly Used Visualization Library'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig['layout'].update(height=500, width=900, title='Frameworks preferred by Developers in last 5 years', 
                     margin=go.Margin(l=100, r=0))

py.iplot(fig)


print(mcqResponse[mcqResponse['Q19_Part_18'] == 'None']['Q19_Part_18'].count(), ''' people have not used any ML Framework specified in the list''')
print(mcqResponse[mcqResponse['Q21_Part_12'] == 'None']['Q21_Part_12'].count(), ''' people have not used any Visualization Library specified in the list''')
columns =  ['Q27_Part_1', 'Q27_Part_2', 'Q27_Part_3', 'Q27_Part_4', 'Q27_Part_5', 'Q27_Part_6', 'Q27_Part_7',
           'Q27_Part_8', 'Q27_Part_9', 'Q27_Part_10', 'Q27_Part_11', 'Q27_Part_12', 'Q27_Part_13', 'Q27_Part_14',
           'Q27_Part_15', 'Q27_Part_16', 'Q27_Part_17', 'Q27_Part_18', 'Q27_Part_19', 'Q27_Part_20']
cloud_prod_choices_df = generate_choices_df(columns, ['None'])
plot_choices(cloud_prod_choices_df, 'Mostly Used Cloud Computing Products in last 5 years', left_margin=220)

print(mcqResponse[mcqResponse['Q27_Part_19'] == 'None']['Q27_Part_19'].count(), ''' people have not used any Cloud Computing Product specified in the list''')
columns_ml_product =  list(mcqResponse.columns[mcqResponse.columns.str.startswith('Q28')])
columns_ml_product.remove('Q28_OTHER_TEXT')
ml_prod_choices_df = generate_choices_df(columns_ml_product, ['None'])

columns_db_product =  list(mcqResponse.columns[mcqResponse.columns.str.startswith('Q29')])
columns_db_product.remove('Q29_OTHER_TEXT')
db_choices_df = generate_choices_df(columns_db_product, ['None'])

trace1 = go.Bar(
    x=ml_prod_choices_df['Count'],
    y=ml_prod_choices_df['Choice'],
    orientation = 'h'
)
trace2 = go.Bar(
    x=db_choices_df['Count'],
    y=db_choices_df['Choice'],
    orientation = 'h'
)

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Most Used ML Products in last 5 years',
                                                          'Most Used Database Products in last 5 years'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig['layout'].update(height=950, width=800, 
                     margin=go.Margin(l=240))

py.iplot(fig)


print(mcqResponse[mcqResponse['Q28_Part_42'] == 'None']['Q28_Part_42'].count(), ''' people have not used any ML Product specified in the list''')
print(mcqResponse[mcqResponse['Q29_Part_27'] == 'None']['Q29_Part_27'].count(), ''' people have not used any Database Product specified in the list''')
def plot_choices_graph(col, to_skip_cols, title, print_title=None, col_num=None, left_margin=50):
    columns =  list(mcqResponse.columns[mcqResponse.columns.str.startswith(col)])
    columns.remove(col+'_OTHER_TEXT')
    db_choices_df = generate_choices_df(columns, to_skip_cols)
    plot_choices(db_choices_df, title, left_margin=left_margin)
    
    if col_num:
        print(mcqResponse[mcqResponse[col+'_Part_'+col_num] == 'None'][col+'_Part_'+col_num].count(), print_title)
    
    return None
plot_choices_graph('Q30', ['None'], 'Most Used Big Data and Analytics Products in last 5 years',
                  'people have not used any Big Data and Anaytics Product specified in the list', '24', 200)
plot_choices_graph('Q31', ['None'], 'Most interacted Data', left_margin=110)
plot_choices_graph('Q33', ['None'], 'Where People find Public Data', left_margin=500)
def find_avg_proportion(col, names):
    columns = list(mcqResponse.columns[mcqResponse.columns.str.startswith(col)])
    columns.remove(col+'_OTHER_TEXT')
    proportions = []
    for c in columns:
        proportions.append(mcqResponse[c].astype(float).mean())
    
    df = pd.DataFrame({'Name': names, 'Proportion': proportions})
    df.sort_values(by=['Proportion'], inplace=True)
    return df
ml_project_distribution = find_avg_proportion('Q34', ['Data Gathering', 'Data Cleaning', 'Visualizing Data',
                                                     'Model Building/Model Selection', 'Putting model in production',
                                                     'Finding insights and communicating with stakeholders'])

def plot_proportions(df, title, left_margin=50):
    trace0 = go.Bar(
        x = df['Proportion'],
        y = df['Name'],
        orientation = 'h'
    )

    layout = go.Layout(
        margin = dict(l=left_margin),
        title = title
    )
    trace = [trace0]
    fig = go.Figure(data=trace, layout=layout)
    py.iplot(fig)

    return None

plot_proportions(ml_project_distribution, 'Distribution of Time in ML Project in %', left_margin=350)
ml_learning_distribution = find_avg_proportion('Q35', ['Self-taught', 'Online Courses', 'Work',
                                                     'University', 'Kaggle Competitions',
                                                     'Other'])

plot_proportions(ml_learning_distribution, 'Distribution of Learning in ML Project', left_margin=150)
plot_choices_graph('Q36', ['None'], 'Distribution of Learning from Online Platforms', left_margin=165)
most_used_online_platforms = mcqResponse['Q37'].value_counts()
most_used_online_platforms.iplot(title='Most Used Online platform for learning AI', margin=go.Margin(b=150))
plot_choices_graph('Q38', ['None'], title = 'Favourite Social Media platform that report on Data Science', left_margin=200)
online_learning = mcqResponse['Q39_Part_1'].value_counts().to_frame().reset_index()
institutional_learning = mcqResponse['Q39_Part_2'].value_counts().to_frame().reset_index()

trace0 = go.Bar(
    x = online_learning['Q39_Part_1'],
    y = online_learning['index'],
    orientation = 'h'
)
trace1 = go.Bar(
    x = institutional_learning['Q39_Part_2'],
    y = institutional_learning['index'],
    orientation = 'h'
)

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Online Learning and MOOCs',
                                                          'In-person Bootcamp'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=450, width=1000, 
                     margin=go.Margin(l=200))

py.iplot(fig)
series = mcqResponse['Q40'].value_counts()
series.iplot(kind='barh', title='Academic Achievement V/S Independent Projects', margin=go.Margin(l=500))
columns = mcqResponse.columns[mcqResponse.columns.str.startswith('Q41')]

ml_algo = mcqResponse[columns[0]].value_counts().to_frame().reset_index()
explaining_ml = mcqResponse[columns[1]].value_counts().to_frame().reset_index()
reproducibility_DS = mcqResponse[columns[2]].value_counts().to_frame().reset_index()

trace0 = go.Bar(
    x = ml_algo['index'],
    y = ml_algo['Q41_Part_1'],
    name = 'Fairness in ML algorithm'
)
trace1 = go.Bar(
    x = explaining_ml['index'],
    y = explaining_ml['Q41_Part_2'],
    name = 'Explaining ML model'
)
trace2 = go.Bar(
    x = reproducibility_DS['index'],
    y = reproducibility_DS['Q41_Part_3'],
    name = 'Reproducibility in Data Science'
)
trace = [trace0, trace1, trace2]

layout = go.Layout(
    barmode = 'group',
    title='Importance of Different ML Topics'
)

fig = go.Figure(data=trace, layout = layout)
py.iplot(fig)
plot_choices_graph('Q42', ['None'], 'Metrics preferred which determine the success of model', left_margin=500)
exploring_data = mcqResponse['Q43'].value_counts().to_frame().reset_index()
trace0 = go.Bar(
    x = exploring_data['Q43'],
    y = exploring_data['index'],
    orientation = 'h'
)

layout = go.Layout(
    title = 'Proportion of data project involved exploring unfair bias in the dataset'
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)
columns =  list(mcqResponse.columns[mcqResponse.columns.str.startswith('Q44')])
ensure_fairness_df = generate_choices_df(columns, ['None'])
plot_choices(ensure_fairness_df, title='Difficulty in ensuring the algorithm is fair and unbiased', left_margin=650)
columns =  list(mcqResponse.columns[mcqResponse.columns.str.startswith('Q45')])
model_df = generate_choices_df(columns, ['None'])
plot_choices(model_df, title='When to explore model insights about model prediction?', left_margin=500)
exploring_data = mcqResponse['Q46'].value_counts().to_frame().reset_index()
trace0 = go.Bar(
    x = exploring_data['Q46'],
    y = exploring_data['index'],
    orientation = 'h'
)

layout = go.Layout(
    title = 'How much data projects involve exploring model insights?'
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)
columns =  list(mcqResponse.columns[mcqResponse.columns.str.startswith('Q47')])
interpret_model_df = generate_choices_df(columns, ['None'])
plot_choices(interpret_model_df, title='Methods preferred to interpret decisions of ML model', left_margin=500)
black_box_or_not = mcqResponse['Q48'].value_counts().to_frame().reset_index()
trace0 = go.Bar(
    x = black_box_or_not['Q48'],
    y = black_box_or_not['index'],
    orientation = 'h'
)

layout = go.Layout(
    title = 'If ML box is black box or not?',
    margin = go.Margin(
        l = 650
    ),
    height = 400
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)
plot_choices_graph('Q49', ['None'], 'Tools that can make work easy and reproducible', left_margin=484)
plot_choices_graph('Q50', ['None'], 'Barriers that prevent from making work easy and reproducible', left_margin=450)
salary = mcqResponse[mcqResponse['Q9'] != 'I do not wish to disclose my approximate yearly compensation']['Q9'].value_counts().to_frame().reset_index()
salary_DS = mcqResponse[((mcqResponse['Q9'] != 'I do not wish to disclose my approximate yearly compensation')&(mcqResponse['Q6'] == 'Data Scientist'))]['Q9'].value_counts().to_frame().reset_index()

trace0 = go.Bar(
    x = list(salary['Q9']),
    y = list(salary['index']),
    orientation = 'h'
)

trace1 = go.Bar(
    x = list(salary_DS['Q9']),
    y = list(salary_DS['index']),
    orientation = 'h'
)


fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Salary (in $USD)', 'Salary of Data Scientists (in $USD)'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=700,margin=go.Margin(l=100))
py.iplot(fig)

