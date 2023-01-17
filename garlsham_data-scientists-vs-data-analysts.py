import pandas as pd
import numpy as np
import seaborn as sns
import squarify
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
import plotly.figure_factory as ff
#Always run this the command before at the start of notebook
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
import warnings
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
def question_info(index):
    survey_headers = pd.read_csv("../input/multipleChoiceResponses.csv", nrows=1)
    survey_headers.head()
    print(survey_headers.columns[index], ' : ',survey_headers[survey_headers.columns[index]][0])
    
def question_info2(index):
    survey_headers = pd.read_csv("../input/multipleChoiceResponses.csv", nrows=1)
    survey_headers.head()
    return survey_headers[survey_headers.columns[index]][0]
    
def get_data():
    survey = pd.read_csv("../input/multipleChoiceResponses.csv", skiprows=2,header=None)
    data = survey.loc[(survey[7] == 'Data Scientist') | (survey[7] == 'Data Analyst')]
    return data

def get_ds_data():
    survey = pd.read_csv("../input/multipleChoiceResponses.csv", skiprows=2,header=None)
    ds_data = survey[survey[7] == 'Data Scientist']
    return ds_data

def get_da_data():
    survey = pd.read_csv("../input/multipleChoiceResponses.csv", skiprows=2,header=None)
    da_data = survey[survey[7] == 'Data Analyst']
    return da_data

ds_data = get_ds_data()
da_data = get_da_data()
question_info(5)
ds_education = ds_data[5].value_counts()
ds_education = pd.DataFrame(ds_education).reset_index().rename(columns={'index':'Education_lvl', 5:"Count"})
ds_education = ds_education.replace('Some college/university study without earning a bachelor’s degree', 'Partial college/university study')
ds_education['Proportion'] = ds_education['Count'] / ds_education['Count'].sum()
ds_education['Job'] = 'Data Scientist'

da_education = da_data[5].value_counts()
da_education = pd.DataFrame(da_education).reset_index().rename(columns={'index':'Education_lvl', 5:"Count"})
da_education = da_education.replace('Some college/university study without earning a bachelor’s degree', 'Partial college/university study')
da_education['Proportion'] = da_education['Count'] / da_education['Count'].sum()
da_education['Job'] = 'Data Analyst'

education = ds_education.append(da_education)

fig,axes = plt.subplots(1,1,figsize=(10, 10))
pal = sns.light_palette((315, 49, 49),n_colors=2, input='husl', reverse=True)
sns.set_palette(pal)
ax = sns.barplot(x='Proportion', y='Education_lvl', data=education, hue='Job')
ax.set_ylabel('Education Level',fontsize=15)
ax.set_xlabel('Percentage of Respondentss',fontsize=15)
ax.set_title('Education Levels' ,fontsize=15)
ax.tick_params(labelsize=12.5)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title

plt.show()
question_info(6)
ds_education = ds_data[6].value_counts()
ds_education = pd.DataFrame(ds_education).reset_index().rename(columns={'index':'degree', 6:"Count"})
ds_education['Percentage'] = ds_education['Count'] / ds_education['Count'].sum()
ds_education['Job'] = 'Data Scientist'

da_education = da_data[6].value_counts()
da_education = pd.DataFrame(da_education).reset_index().rename(columns={'index':'degree', 6:"Count"})
da_education['Percentage'] = da_education['Count'] / da_education['Count'].sum()
da_education['Job'] = 'Data Analyst'

undergrad = ds_education.append(da_education)
fig,axes = plt.subplots(1,1,figsize=(10, 10))
pal = sns.light_palette((115, 49, 49),n_colors=2, input='husl', reverse=True)
sns.set_palette(pal)
ax = sns.barplot(x='Percentage', y='degree', data=undergrad, hue='Job')
ax.set_ylabel('Undergraduate Degree',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title('Percentage of Undergraduate Degrees per Job' ,fontsize=15)
ax.tick_params(labelsize=12.5)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title

plt.show()
ds_location = ds_data[4].value_counts()
ds_location = pd.DataFrame(ds_location).reset_index().rename(columns={'index':'Loc', 4:"Count"})
ds_location = ds_location.sort_values(['Count'], ascending=False)
ds_location['Proportion'] = ds_location['Count'] / ds_location['Count'].sum()
ds_location = ds_location[:20]

my_colours = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]]
data = [ dict(
        type = 'choropleth',
        colorscale = my_colours,
        locations=ds_location['Loc'],
        locationmode = 'country names', #matches locations to country names
        z = ds_location['Proportion'],
        text = ds_location['Count'], #hover info
        hoverinfo = 'text',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar=dict(
            tickprefix='',
            title='Respondent Proportion')
) ]

layout = dict(
    title = 'Top 20 Locations for Data Scientists',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'equirectangular'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot(fig)
da_location = da_data[4].value_counts()
da_location = pd.DataFrame(da_location).reset_index().rename(columns={'index':'Loc', 4:"Count"})
da_location = da_location.sort_values(['Count'], ascending=False)
da_location['Proportion'] = da_location['Count'] / da_location['Count'].sum()
da_location = da_location[:20]

data = [ dict(
        type = 'choropleth',
        colorscale = my_colours,
        locations=da_location['Loc'],
        locationmode = 'country names', #matches locations to country names
        z = da_location['Proportion'],
        text = da_location['Count'], #hover info
        hoverinfo = 'text',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar=dict(
            tickprefix='',
            title='Respondent Proportion')
) ]

layout = dict(
    title = 'Top 20 Locations for Data Analysts',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'equirectangular'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot(fig)
question_info(9)
ds_emp = ds_data[9].value_counts()
ds_emp = pd.DataFrame(ds_emp).reset_index().rename(columns={'index':'industry', 9:"Count"})
ds_emp['Proportion'] = ds_emp['Count'] / ds_emp['Count'].sum()
ds_emp['Job'] = 'Data Scientist'

da_emp = da_data[9].value_counts()
da_emp = pd.DataFrame(da_emp).reset_index().rename(columns={'index':'industry', 9:"Count"})
da_emp['Proportion'] = da_emp['Count'] / da_emp['Count'].sum()
da_emp['Job'] = 'Data Analyst'

employment = ds_emp.append(da_emp)

fig,axes = plt.subplots(1,1,figsize=(10, 10))
pal = sns.light_palette((325, 49, 49),n_colors=2, input='husl', reverse=True)
sns.set_palette(pal)
ax = sns.barplot(x='Proportion', y='industry', data=employment, hue='Job')
ax.set_ylabel('Top Employment Industries',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title('Most Popular Industries per Job Title' ,fontsize=15)
ax.tick_params(labelsize=12.5)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title

plt.show()
question_info(11)
ds_exp = ds_data[11].value_counts()
ds_exp = pd.DataFrame(ds_exp).reset_index().rename(columns={'index':'exp', 11:"Count"})
ds_exp['Proportion'] = ds_exp['Count'] / ds_exp['Count'].sum()
ds_exp['Job'] = 'Data Scientist'

da_exp = da_data[11].value_counts()
da_exp = pd.DataFrame(da_exp).reset_index().rename(columns={'index':'exp', 11:"Count"})
da_exp['Proportion'] = da_exp['Count'] / da_exp['Count'].sum()
da_exp['Job'] = 'Data Analyst'

exp = ds_exp.append(da_exp)

fig,axes = plt.subplots(1,1,figsize=(10, 10))
pal = sns.light_palette((5, 76, 49),n_colors=1, input='husl', reverse=True)
sns.set_palette(pal)
ax = sns.barplot(x='Proportion', y='exp', data=exp, hue='Job')
ax.set_ylabel('Years Experience',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title('Level of Work Experience' ,fontsize=15)
ax.tick_params(labelsize=12.5)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title

plt.show()
question_info(12)
ds_pay = ds_data[12].value_counts()
ds_pay = pd.DataFrame(ds_pay).reset_index().rename(columns={'index':'Wages', 12:"Count"})
ds_pay = ds_pay.replace('I do not wish to disclose my approximate yearly compensation', 'Not Disclosing')
ds_pay['Proportion'] = ds_pay['Count'] / ds_pay['Count'].sum()
ds_pay['Job'] = 'Data Scientist'

da_pay = da_data[12].value_counts()
da_pay = pd.DataFrame(da_pay).reset_index().rename(columns={'index':'Wages', 12:"Count"})
da_pay = da_pay.replace('I do not wish to disclose my approximate yearly compensation', 'Not Disclosing')
da_pay['Proportion'] = da_pay['Count'] / da_pay['Count'].sum()
da_pay['Job'] = 'Data Analyst'

pay = ds_pay.append(da_pay)

fig,axes = plt.subplots(1,1,figsize=(15, 15))
pal = sns.diverging_palette(250, 151, s=94, l=85,n=2)
sns.set_palette(pal)
ax = sns.barplot(x="Proportion", y="Wages", hue="Job", data=pay)
ax.set_ylabel('Wage Range',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Comparison of Compensation" ,fontsize=15)
ax.tick_params(labelsize=10)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title

plt.show()
pay_v_edu = ds_data.iloc[:,[11,12]].rename(columns={11:'Exp', 12:'Pay'})
temp = pay_v_edu.groupby(by=['Exp','Pay'], as_index=False).size().reset_index()
pay_v_edu = pd.DataFrame(temp).rename(columns={'Exp':'Exp', 'Pay':'Pay', 0:'Count'})
pay_v_edu['Proportion'] = pay_v_edu['Count'] / pay_v_edu['Count'].sum()
pay_v_edu = pay_v_edu.replace('I do not wish to disclose my approximate yearly compensation', 'Not Disclosing')

ax = sns.FacetGrid(pay_v_edu, col='Exp', height=6, aspect=.75,col_wrap=4, hue='Exp', palette="viridis")
                                                                
ax.map(sns.barplot, 'Proportion', 'Pay')
pay_v_edu = da_data.iloc[:,[11,12]].rename(columns={11:'Exp', 12:'Pay'})

temp = pay_v_edu.groupby(by=['Exp','Pay'], as_index=False).size().reset_index()
pay_v_edu = pd.DataFrame(temp).rename(columns={'Exp':'Exp', 'Pay':'Pay', 0:'Count'})
pay_v_edu['Proportion'] = pay_v_edu['Count'] / pay_v_edu['Count'].sum() * 100
pay_v_edu = pay_v_edu.replace('I do not wish to disclose my approximate yearly compensation', 'Not Disclosing')

ax1 = sns.FacetGrid(pay_v_edu, col='Exp', height=6, aspect=.75,col_wrap=4, hue='Exp', palette = sns.diverging_palette(4, 239, s=74, l=50,
                                                                n=len(pd.unique(da_exp['exp'])), center='dark'))
ax1.map(sns.barplot, 'Proportion', 'Pay')
del(ax1)
question_info(13)
ml_data = get_data()
ml_data = ml_data.replace('We are exploring ML methods (and may one day put a model into production)', 'Exploring ML methods')
ml_data = ml_data.replace('We recently started using ML methods (i.e., models in production for less than 2 years)', 'Recently started using ML')
ml_data = ml_data.replace('We have well established ML methods (i.e., models in production for more than 2 years)', 'Established ML Methods')
ml_data = ml_data.replace('We use ML methods for generating insights (but do not put working models into production)', 'Used for Insights')
ml_data = ml_data.replace('No (we do not use ML methods)', 'No')

#ml_data = ml_data.iloc[:,[7,13]].rename(columns={7:'Job', 13:'Ml'})
#ml_data = ml_data.groupby(by='Ml',as_index=False)['Job'].count()

ml_data = ml_data.iloc[:,[7,13]].rename(columns={7:'Job',13:'Ml'})
ml_DsData = ml_data[ml_data['Job']=='Data Scientist']
ml_DaData = ml_data[ml_data['Job']=='Data Analyst']

ml_DsData = ml_DsData['Ml'].value_counts()
ml_DsData = pd.DataFrame(ml_DsData).reset_index().rename(columns={'index':'Ml', 'Ml':"Count"})
ml_DsData.iplot(kind='pie',labels='Ml',values='Count',pull=.2,hole=.2,
          colorscale='greens',textposition='outside',textinfo='value+percent', title='Data Scientist ML Usage')

ml_DaData = ml_DaData['Ml'].value_counts()
ml_DaData = pd.DataFrame(ml_DaData).reset_index().rename(columns={'index':'Ml', 'Ml':"Count"})
ml_DaData.iplot(kind='pie',labels='Ml',values='Count',pull=.2,hole=.2,
          colorscale='blues',textposition='outside',textinfo='value+percent', title='Data Analyst ML Usage')
question_info(14)
data = get_ds_data()
job_activity = data[14].value_counts()
job_activity = pd.DataFrame(job_activity).reset_index().rename(columns={'index':'Activity', 14:"Count"})
for item in range(15,21):
    temp = data[item].value_counts()
    temp = pd.DataFrame(temp).reset_index().rename(columns={'index':'Activity', item:"Count"})
    job_activity = job_activity.append(temp)

job_activity['Job'] = 'Data Scientist'
job_activity['Percentage'] = job_activity['Count'] / job_activity['Count'].sum() * 100
job_activity['Category'] = ['Category 0','Category 1','Category 2','Category 3','Category 4','Category 5','Category 6']

trace = go.Table(
    header=dict(values=['Category', 'Activity'],
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[job_activity.Category, job_activity.Activity],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))

data = [trace] 
iplot(data, filename = 'pandas_table')

job_activity.iplot(kind='bubble', categories='Category', x='Percentage', y='Category', size='Percentage', text='Activity', colorscale='YlGn', 
            xTitle='Percentage of Respondents', title='ML Usage for Data Scientists')

data = get_da_data()
temp_activity = data[14].value_counts()
temp_activity = pd.DataFrame(temp_activity).reset_index().rename(columns={'index':'Activity', 14:"Count"})
for item in range(15,21):
    temp = data[item].value_counts()
    temp = pd.DataFrame(temp).reset_index().rename(columns={'index':'Activity', item:"Count"})
    temp_activity = temp_activity.append(temp)

temp_activity['Job'] = 'Data Analyst'
temp_activity['Percentage'] = temp_activity['Count'] / temp_activity['Count'].sum() * 100
temp_activity['Category'] = ['Category 0','Category 1','Category 2','Category 3','Category 4','Category 5','Category 6']


job_activity = job_activity.append(temp_activity)
temp_activity.iplot(kind='bubble',categories='Category', x='Percentage', y='Category', size='Percentage', text='Activity', colorscale='YlOrRd' ,
            xTitle='Percentage of Data Analyst Respondents', title='ML Usage for Data Analysts' )


question_info(22)
def change_string(item):
    if len(item.split('(')) > 1:
        item  = item.split('(')[1].strip(')')
        return item
    else:
        item = item
        return item
data = get_ds_data()
ds_tools = data[22].value_counts()
ds_tools = pd.DataFrame(ds_tools).reset_index().rename(columns={'index':'Tool', 22:"Count"})    
ds_tools['Job'] = 'Data Scientist'
ds_tools['Percentage'] = ds_tools['Count'] / ds_tools['Count'].sum() * 100

data = get_da_data()
da_tools = data[22].value_counts()
da_tools = pd.DataFrame(da_tools).reset_index().rename(columns={'index':'Tool', 22:"Count"})
da_tools['Job'] = 'Data Analyst'
da_tools['Percentage'] = da_tools['Count'] / da_tools['Count'].sum() * 100

tools = ds_tools.append(da_tools)

tools['Tool'] = tools['Tool'].apply(change_string)

ig,axes = plt.subplots(1,1,figsize=(10, 10))
pal = sns.light_palette((95, 96, 49),n_colors=2, input='husl', reverse=True)
sns.set_palette(pal)
ax = sns.barplot(x='Percentage', y='Tool', data=tools, hue='Job')
ax.set_ylabel('Primary Analysis Tool',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title('Most Popular Data Analysis Tools Used in Last 5 Years' ,fontsize=15)
ax.tick_params(labelsize=12.5)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title

plt.show()


question_info(29)
def agg_data(data,index, end, column,job):
    ds_data = data[index].value_counts().reset_index().rename(columns={'index':column,index:'Count'})
    
    for item in range(index +1, end + 1):
        ds_data = ds_data.append(data[item].value_counts().reset_index().rename(columns={'index':column,item:'Count'}))
        
    ds_data['Job'] = job
    ds_data['Percentage'] = ds_data['Count'] / ds_data['Count'].sum() * 100
    return ds_data
ds_ide = agg_data(ds_data,29, 43, 'IDE','Data Scientist')
da_ide = agg_data(da_data,29, 43, 'IDE','Data Analyst')

ide = ds_ide.append(da_ide)

fig,axes = plt.subplots(1,1,figsize=(15, 15))
pal = sns.diverging_palette(118, 309, s=74, l=50,n=2)
sns.set_palette(pal)
ax = sns.barplot(x="Percentage", y="IDE", hue="Job", data=ide)
ax.set_ylabel('IDE',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Most Popular IDE'S" ,fontsize=15)
ax.tick_params(labelsize=10)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
question_info(55)
ds_note = agg_data(ds_data,45, 55, 'Notebook','Data Scientist')
da_note = agg_data(da_data,45, 55, 'Notebook','Data Analyst')

notebooks = ds_note.append(da_note)
fig,axes = plt.subplots(1,1,figsize=(15, 15))
sns.set_palette("viridis")
ax = sns.barplot(x="Percentage", y="Notebook", hue="Job", data=notebooks)
ax.set_ylabel('Hosted Notebook',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Most Popular Hosted Notebook" ,fontsize=15)
ax.tick_params(labelsize=10)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
question_info(57)
ds_ccs = agg_data(ds_data,57, 63, 'Cloud','Data Scientist')
da_ccs = agg_data(da_data,57, 63, 'Cloud','Data Analyst')

ccs = ds_ccs.append(da_ccs)

fig,axes = plt.subplots(1,1,figsize=(10, 10))
pal = sns.light_palette((164, 99, 51),n_colors=len(pd.unique(ccs['Job'])), input='husl', reverse=True)
sns.set_palette(pal)
ax = sns.barplot(x="Percentage", y="Cloud", hue="Job", data=ccs)
ax.set_ylabel('Cloud Computing Provider',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Most Popular Cloud Computing Provider" ,fontsize=15)
ax.tick_params(labelsize=12.5)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
question_info(65)
ds_prog = agg_data(ds_data,65, 82, 'Prog','Data Scientist')
da_prog = agg_data(da_data,65, 82, 'Prog','Data Analyst')

prog = ds_prog.append(da_prog)

fig,axes = plt.subplots(1,1,figsize=(15, 15))
pal = sns.diverging_palette(220, 10, s=74, l=50,n=2)
sns.set_palette(pal)
ax = sns.barplot(x="Percentage", y="Prog", hue="Job", data=prog)
ax.set_ylabel('Programming Language',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Most Popular Programming Languages" ,fontsize=15)
ax.tick_params(labelsize=10)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
question_info(84)
ds_prog = agg_data(ds_data,84, 84, 'Prog','Data Scientist')
da_prog = agg_data(da_data,84, 84, 'Prog','Data Analyst')

prog = ds_prog.append(da_prog)

fig,axes = plt.subplots(1,1,figsize=(20, 10))
pal = sns.diverging_palette(150, 10, s=74, l=50, n=6)
sns.set_palette(pal)
ax = squarify.plot(sizes=ds_prog['Percentage'], label=ds_prog['Prog'], alpha=.4,color=pal )
ax.set_title('Most Popular Programming Language for Data Scientists' ,fontsize=20)
plt.axis('off')
plt.show()

fig,axes = plt.subplots(1,1,figsize=(20, 10))
pal = sns.diverging_palette(294, 244, s=74, l=50, n=6)
sns.set_palette(pal)
ax = squarify.plot(sizes=da_prog['Percentage'], label=da_prog['Prog'], alpha=.4, color=pal )
ax.set_title('Most Popular Programming Language for Data Analysts' ,fontsize=20)
plt.axis('off')
plt.show()

question_info(88)
ds_ml = agg_data(ds_data,88, 106, 'Ml_framework','Data Scientist')
da_ml = agg_data(da_data,88, 106, 'Ml_framework','Data Analyst')

frameworks = ds_ml.append(da_ml)
ds_ml.sort_values('Percentage', inplace=True)
da_ml.sort_values('Percentage', inplace=True)
fig,ax = plt.subplots(1,1,figsize=(10, 10))

plt.scatter(ds_ml['Percentage'], ds_ml['Ml_framework'], color='blue', alpha=1, label='Data Scientist')
plt.scatter(da_ml['Percentage'], da_ml['Ml_framework'], color='green', alpha=1, label='Data Analyst' )
ax.set_ylabel('ML Framework',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Most Popular Machine Learning Frameworks (Last 5 Years)" ,fontsize=15)
ax.tick_params(labelsize=10)
plt.legend()
plt.show()

question_info(108)
ds_ml = agg_data(ds_data,108, 108, 'Ml_framework','Data Scientist')
da_ml = agg_data(da_data,108, 108, 'Ml_framework','Data Analyst')

frameworks = ds_ml.append(da_ml)

ds_ml.iplot(kind='bubble', colorscale='PRGn', categories='Ml_framework', x='Percentage', y='Ml_framework', size='Percentage',
            xTitle='Percentage of Data Scientist Respondents', title='Most Popular ML Framework for Data Scientists')

da_ml.iplot(kind='bubble', categories='Ml_framework', x='Percentage', y='Ml_framework', size='Percentage', colorscale='RdYlBu',
            xTitle='Percentage of Data Analyst Respondents', title='Most Popular ML Framework for Data Analysts')

question_info(110)
ds_viz = agg_data(ds_data,110, 122, 'Viz','Data Scientist')
da_viz = agg_data(da_data,110, 122, 'Viz','Data Analyst')

viz = ds_viz.append(da_viz)
viz.sort_values('Percentage', inplace=True)

fig,axes = plt.subplots(1,1,figsize=(15, 15))
pal = sns.diverging_palette(120, 10, s=74, l=50,n=2)
sns.set_palette(pal)
ax = sns.barplot(x="Percentage", y="Viz", hue="Job", data=viz)
ax.set_ylabel('Visualization Libraries',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Most Popular Visualization Libraries" ,fontsize=15)
ax.tick_params(labelsize=15)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
question_info(124)
ds_viz = agg_data(ds_data,124, 124, 'Viz','Data Scientist')
da_viz = agg_data(da_data,124, 124, 'Viz','Data Analyst')

ds_viz.sort_values('Percentage', ascending=False, inplace=True)
da_viz.sort_values('Percentage', ascending=False, inplace=True)

ds_viz[:8].iplot(kind='pie',labels='Viz',values='Count',pull=.1,hole=.1, 
          colorscale='Greens',textposition='outside',textinfo='value+percent', title='Most Popular Visualization Library used by Data Scientists (Top 8)')


da_viz[:8].iplot(kind='pie',labels='Viz',values='Count',pull=.1,hole=.1, 
          colorscale='Reds',textposition='outside',textinfo='value+percent', title='Most Popular Visualization Library used by Data Analysts (Top 8)')


question_info(126)
ds_code = agg_data(ds_data,126, 126, 'Coding','Data Scientist')
da_code = agg_data(da_data,126, 126, 'Coding','Data Analyst')

fig,axes = plt.subplots(1,1,figsize=(20, 10))
pal = sns.diverging_palette(350, 10, s=74, l=50, n=6)
sns.set_palette(pal)
ax = squarify.plot(sizes=ds_code['Percentage'], label=ds_code['Coding'], alpha=.4,color=pal )
ax.set_title('Time Spent Coding for Data Scientists' ,fontsize=20)
plt.axis('off')
plt.show()

fig,axes = plt.subplots(1,1,figsize=(20, 10))
pal = sns.diverging_palette(394, 114, s=74, l=50, n=6)
sns.set_palette(pal)
ax = squarify.plot(sizes=da_code['Percentage'], label=da_code['Coding'], alpha=.4, color=pal )
ax.set_title('Time Spent Coding for Data Analysts' ,fontsize=20)
plt.axis('off')
plt.show()
question_info(127)
question_info(128)
ds_analyse = agg_data(ds_data,127, 127, 'Analyse','Data Scientist')
da_analyse = agg_data(da_data,127, 127, 'Analyse','Data Scientist')

ds_ml = agg_data(ds_data,128, 128, 'Ml','Data Scientist')
da_ml = agg_data(da_data,128, 128, 'Ml','Data Scientist')

ds_analyse = ds_analyse.replace('I have never written code but I want to learn', 'Never wrote code. Want to Learn')
ds_analyse = ds_analyse.replace('I have never written code and I do not want to learn', 'Never wrote code. Dont want to Learn')

agg = ds_ml.append(da_ml)

values = da_analyse['Percentage'] * -1
values = values.append(ds_analyse['Percentage'])
values = values.sort_values(ascending=True)

min_v = int(values.min())
max_v = int(values.max())
values = [int(i) for i in values]

labels = da_analyse['Percentage'] * -1
labels = labels.append(ds_analyse['Percentage'])
labels = labels.sort_values(ascending=True)
labels = [int(i) for i in labels]

new_labels =[]
for item in labels:
    if item < 0:
        item = item * -1
        new_labels.append(item)
    else:
        new_labels.append(item)

da_analyse['Percentage'] = da_analyse['Percentage'] * -1


layout = go.Layout(title='Experience in Analysing Data',
                   yaxis=go.layout.YAxis(tickangle=-15),
                   xaxis=go.layout.XAxis(
                       tickangle=-55,
                       range=[min_v, max_v],
                       tickvals= [int(i) for i in values],
                       ticktext= new_labels,
                       title='Percentage of Respondents'),
                   barmode='overlay',
                   bargap=0.5,
                   height=500,
                  width=900, 
                  margin=go.layout.Margin(l=225, r=0))

data = [go.Bar(y=ds_analyse['Analyse'],
               x=ds_analyse['Percentage'],
               orientation='h',
               name='Data Scientists',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=ds_analyse['Analyse'],
               x=da_analyse['Percentage'],
               orientation='h',
               name='Data Analysts',
               marker=dict(color='seagreen')
               )]

iplot(dict(data=data, layout=layout), filename='EXAMPLES/bar_pyramid') 

ds_ml = ds_ml.replace('I have never studied machine learning and I do not plan to', 'Never used ML')
ds_ml = ds_ml.replace('I have never studied machine learning but plan to learn in the future', 'Never used ML but want to')

values = da_ml['Percentage'] * -1
values = values.append(ds_ml['Percentage'])
values = values.sort_values(ascending=True)

min_v = int(values.min())
max_v = int(values.max())
values = [int(i) for i in values]

labels = da_ml['Percentage'] * -1
labels = labels.append(ds_ml['Percentage'])
labels = labels.sort_values(ascending=True)
labels = [int(i) for i in labels]

new_labels =[]
for item in labels:
    if item < 0:
        item = item * -1
        new_labels.append(item)
    else:
        new_labels.append(item)

da_ml['Percentage'] = da_ml['Percentage'] * -1


layout = go.Layout(title='Experience in ML Modelling',
            yaxis=go.layout.YAxis(tickangle=-15),
                   xaxis=go.layout.XAxis(
                       tickangle=-55,
                       range=[min_v, max_v],
                       tickvals= [int(i) for i in values],
                       ticktext= new_labels,
                       title='Percentage of Respondents'),
                   barmode='overlay',
                   bargap=0.5,
                   height=500,
                  width=900, 
                  margin=go.layout.Margin(l=160, r=0))

data = [go.Bar(y=ds_ml['Ml'],
               x=ds_ml['Percentage'],
               orientation='h',
               name='Data Scientists',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=ds_ml['Ml'],
               x=da_ml['Percentage'],
               orientation='h',
               name='Data Analysts',
               marker=dict(color='seagreen')
               )]

iplot(dict(data=data, layout=layout), filename='EXAMPLES/bar_pyramid') 
question_info(130)
ds_ccs = agg_data(ds_data,130, 149, 'CCS','Data Scientist')
da_ccs = agg_data(da_data,130, 149, 'CCS','Data Analyst')

s1 = ds_ccs['Percentage'] * 40
s2 = da_ccs['Percentage'] * 40

fig,ax = plt.subplots(1,1,figsize=(10, 10))

plt.scatter(ds_ccs['Percentage'], ds_ccs['CCS'], color='red', alpha=1, label='Data Scientist', s=s1)
plt.scatter(da_ccs['Percentage'], da_ccs['CCS'], color='green', alpha=1, label='Data Analyst', s=s2 )
ax.set_ylabel('Cloud Computing Service',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Most Popular Cloud Computing Services in the last 5 years" ,fontsize=15)
ax.tick_params(labelsize=15)
plt.legend(loc=0, markerscale=.5)
plt.show()

question_info(151)
ds_ccs = agg_data(ds_data,151, 193, 'ML','Data Scientist')
da_ccs = agg_data(da_data,151, 193, 'ML','Data Analyst')

ds_ccs = ds_ccs[:20]
da_ccs = da_ccs[:20]

fig,axes = plt.subplots(1,1,figsize=(25, 20))
pal = sns.diverging_palette(350, 10, s=74, l=50, n=20)
sns.set_palette(pal)
ax = squarify.plot(sizes=ds_ccs['Percentage'], label=ds_ccs['ML'], alpha=.4,color=pal )
ax.set_title('Most Popular Machine Learning Products (Last 5 Years) for Data Scientists' ,fontsize=20)
plt.axis('off')
plt.show()

fig,axes = plt.subplots(1,1,figsize=(25, 20))
pal = sns.diverging_palette(250, 44, s=74, l=50, n=20)
sns.set_palette(pal)
ax = squarify.plot(sizes=da_ccs['Percentage'], label=da_ccs['ML'], alpha=.4,color=pal )
ax.set_title('Most Popular Machine Learning Products (Last 5 Years) for Data Analysts' ,fontsize=20)
plt.axis('off')
plt.show()
question_info(195)
ds_db = agg_data(ds_data,195, 222, 'DB','Data Scientist')
da_db = agg_data(da_data,195, 222, 'DB','Data Analyst')

db = ds_db.append(da_db)
fig,axes = plt.subplots(1,1,figsize=(15, 15))
pal = sns.diverging_palette(280, 10, s=44, l=50,n=2)
sns.set_palette(pal)
ax = sns.barplot(x="Percentage", y="DB", hue="Job", data=db)
ax.set_ylabel('Relational Database Products',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Most Popular Relational Database Products (Last 5 Years)" ,fontsize=15)
ax.tick_params(labelsize=15)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title

question_info(224)
ds_bigData = agg_data(ds_data,224, 248, 'bigData','Data Scientist')
da_bigData = agg_data(da_data,224, 248, 'bigData','Data Analyst')

ds_bigData.sort_values('Percentage', ascending=False, inplace=True)
da_bigData.sort_values('Percentage', ascending=False, inplace=True)


ds_bigData[:15].iplot(kind='pie',labels='bigData',values='Count',pull=.1,hole=.1, 
          colorscale='Paired',textposition='outside',textinfo='value+percent', title='Most Popular Big Data Products for Data Scientists (Top 15)')

da_bigData[:15].iplot(kind='pie',labels='bigData',values='Count',pull=.1,hole=.1, 
          colorscale='Set3',textposition='outside',textinfo='value+percent', title='Most Popular Big Data Products for Data Analysts (Top 15)')

question_info(250)
ds_dtype = agg_data(ds_data,250, 261, 'Data','Data Scientist')
da_dtype = agg_data(da_data,250, 261, 'Data','Data Analyst')

agg = ds_dtype.append(da_dtype)

values = da_dtype['Percentage'] * -1
values = values.append(ds_dtype['Percentage'])
values = values.sort_values(ascending=True)

min_v = int(values.min())
max_v = int(values.max())
values = [int(i) for i in values]

labels = da_dtype['Percentage'] * -1
labels = labels.append(ds_dtype['Percentage'])
labels = labels.sort_values(ascending=True)
labels = [int(i) for i in labels]

new_labels =[]
for item in labels:
    if item < 0:
        item = item * -1
        new_labels.append(item)
    else:
        new_labels.append(item)

da_dtype['Percentage'] = da_dtype['Percentage'] * -1


layout = go.Layout(title='Most Popular types of data worked on in the PAST',
                   yaxis=go.layout.YAxis(tickangle=-15),
                   xaxis=go.layout.XAxis(
                       tickangle=-55,
                       range=[min_v, max_v],
                       tickvals= [int(i) for i in values],
                       ticktext= new_labels,
                       title='Percentage of Respondents'),
                   barmode='overlay',
                   bargap=0.5,
                   height=500,
                  width=900, 
                  margin=go.layout.Margin(l=225, r=0))

data = [go.Bar(y=ds_dtype['Data'],
               x=ds_dtype['Percentage'],
               orientation='h',
               name='Data Scientists',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=ds_dtype['Data'],
               x=da_dtype['Percentage'],
               orientation='h',
               name='Data Analysts',
               marker=dict(color='seagreen')
               )]

iplot(dict(data=data, layout=layout), filename='EXAMPLES/bar_pyramid') 
ds_dtype = agg_data(ds_data,263, 263, 'Data','Data Scientist')
da_dtype = agg_data(da_data,263, 263, 'Data','Data Analyst')

agg = ds_dtype.append(da_dtype)

values = da_dtype['Percentage'] * -1
values = values.append(ds_dtype['Percentage'])
values = values.sort_values(ascending=True)

min_v = int(values.min())
max_v = int(values.max())
values = [int(i) for i in values]

labels = da_dtype['Percentage'] * -1
labels = labels.append(ds_dtype['Percentage'])
labels = labels.sort_values(ascending=True)
labels = [int(i) for i in labels]

new_labels =[]
for item in labels:
    if item < 0:
        item = item * -1
        new_labels.append(item)
    else:
        new_labels.append(item)

da_dtype['Percentage'] = da_dtype['Percentage'] * -1


layout = go.Layout(title='Most Popular types of data CURRENTLY worked on',
                   yaxis=go.layout.YAxis(tickangle=-15),
                   xaxis=go.layout.XAxis(
                       tickangle=-55,
                       range=[min_v, max_v],
                       tickvals= [int(i) for i in values],
                       ticktext= new_labels,
                       title='Percentage of Respondents'),
                   barmode='overlay',
                   bargap=0.5,
                   height=500,
                  width=900, 
                  margin=go.layout.Margin(l=225, r=0))

data = [go.Bar(y=ds_dtype['Data'],
               x=ds_dtype['Percentage'],
               orientation='h',
               name='Data Scientists',
               marker=dict(color='red')
               ),
        go.Bar(y=ds_dtype['Data'],
               x=da_dtype['Percentage'],
               orientation='h',
               name='Data Analysts',
               marker=dict(color='purple')
               )]

iplot(dict(data=data, layout=layout), filename='EXAMPLES/bar_pyramid') 
question_info(265)
ds_dtype = agg_data(ds_data,265, 275, 'Data','Data Scientist')
da_dtype = agg_data(da_data,265, 275, 'Data','Data Analyst')

ds_dtype = ds_dtype.replace('None/I do not work with public data', 'None')
ds_dtype = ds_dtype.replace('Dataset aggregator/platform (Socrata, Kaggle Public Datasets Platform, etc.)', 'Kaggle, Socrata, etc.')
ds_dtype = ds_dtype.replace('I collect my own data (web-scraping, etc.)', 'I collect my own data')

da_dtype = ds_dtype.replace('None/I do not work with public data', 'None')
da_dtype = ds_dtype.replace('Dataset aggregator/platform (Socrata, Kaggle Public Datasets Platform, etc.)', 'Kaggle, Socrata, etc.')
da_dtype = ds_dtype.replace('I collect my own data (web-scraping, etc.)', 'I collect my own data')


agg = ds_dtype.append(da_dtype)

ds_dtype.iplot(kind='pie',labels='Data',values='Count',pull=.1,hole=.1, 
          colorscale='RdYlGn',textposition='outside',textinfo='value+percent', 
        title='Most Popular Public Dataset Source for Data Scientists')

ds_dtype.iplot(kind='pie',labels='Data',values='Count',pull=.1,hole=.1, 
          colorscale='RdYlBu',textposition='outside',textinfo='value+percent', 
         title='Most Popular Public Dataset Source for Data Analysts')

question_info(278)
def agg_data_work(data,index, end, column,job, label_5):
    
    labels=[]
    for i in range(index,end+1):
        temp_str = str(question_info2(i))
        s = temp_str.split("-")[1]
        labels.append(s)
    
    if len(label_5) > 1:
        labels[5] = 'Finding and Communicating Insights'
    
    ds_data = data[index].value_counts().reset_index().rename(columns={'index':column,index:'Count'})
    ds_data['Activity'] = labels[0]

    for ind,item in enumerate(range(index +1, end + 1)):
        temp = data[item].value_counts().reset_index().rename(columns={'index':column,item:'Count'})
        temp['Activity'] = labels[ind+1]
        ds_data = ds_data.append(temp)
        #ds_data = ds_data.append(data[item].value_counts().reset_index().rename(columns={'index':column,item:'Count'}))
    
        
    ds_data['Job'] = job
    ds_data['Percentage_of_Respondents'] = ds_data['Count'] / ds_data['Count'].sum() * 100
    return ds_data
def widths(data, w):
    widths = []
    for item in range(0, len(data)):
        widths.append(w)
    
    return widths
def colors(data, c):
    colors = []
    for item in range(0, len(data)):
        colors.append(c)
    
    return colors
def gen_plot(ds_work, Job):
    gathering_data = ds_work[ds_work['Activity'] == ' Gathering data']
    trace_gather = go.Scatter(x=list(gathering_data['Percentage_of_Respondents']),
                            y=list(gathering_data['Percentage_of_Time']),
                            name='Gathering data',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(gathering_data, '#33CFA5') ))
    
    cleaning_data = ds_work[ds_work['Activity'] == ' Cleaning data']
    trace_clean = go.Scatter(x=list(cleaning_data['Percentage_of_Respondents']),
                            y=list(cleaning_data['Percentage_of_Time']),
                            name='Cleaning data',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(cleaning_data, '#4286f4') ))

    viz_data = ds_work[ds_work['Activity'] == ' Visualizing data']
    trace_viz = go.Scatter(x=list(viz_data['Percentage_of_Respondents']),
                            y=list(viz_data['Percentage_of_Time']),
                            name='Visualizing data',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(viz_data, '#f2435d') ))

    model_data = ds_work[ds_work['Activity'] == ' Model building/model selection']
    trace_model = go.Scatter(x=list(model_data['Percentage_of_Respondents']),
                            y=list(model_data['Percentage_of_Time']),
                            name='Model building',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(model_data, '#de75ff') ))

    prod_data = ds_work[ds_work['Activity'] == ' Putting the model into production']
    trace_prod = go.Scatter(x=list(prod_data['Percentage_of_Respondents']),
                            y=list(prod_data['Percentage_of_Time']),
                            name='Putting the model into production',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(prod_data, '#8dfcc4') ))

    insight_data = ds_work[ds_work['Activity'] == 'Finding and Communicating Insights']
    trace_insight = go.Scatter(x=list(insight_data['Percentage_of_Respondents']),
                            y=list(insight_data['Percentage_of_Time']),
                            name='Finding and Communicating Insights',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(insight_data, '#f7cb79') ))   

    other_data = ds_work[ds_work['Activity'] == ' Other']
    trace_other = go.Scatter(x=list(other_data['Percentage_of_Respondents']),
                            y=list(other_data['Percentage_of_Time']),
                            name='Other',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(other_data, '#8dfcf5') ))

    data = [trace_gather, trace_clean, trace_viz, trace_model, trace_prod, trace_insight, trace_other]
    
    gather=[dict(x=gathering_data['Percentage_of_Respondents'].mean(),
                       y=gathering_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. Time at Activity: ' + str(round(gathering_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]
    
    clean=[dict(x=cleaning_data['Percentage_of_Respondents'].mean(),
                       y=cleaning_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. Time at Activity: ' + str(round(cleaning_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]
    
    viz=[dict(x=viz_data['Percentage_of_Respondents'].mean(),
                       y=viz_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. Time at Activity: ' + str(round(viz_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]
    
    model=[dict(x=model_data['Percentage_of_Respondents'].mean(),
                       y=model_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. Time at Activity: ' + str(round(model_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]
    
    prod=[dict(x=prod_data['Percentage_of_Respondents'].mean(),
                       y=prod_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. Time at Activity: ' + str(round(prod_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]
    
    insights=[dict(x=insight_data['Percentage_of_Respondents'].mean(),
                       y=insight_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. Time at Activity: ' + str(round(insight_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]
    
    other=[dict(x=other_data['Percentage_of_Respondents'].mean(),
                       y=other_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. Time at Activity: ' + str(round(other_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]


    updatemenus = list([
        dict(active=0,
             buttons=list([  
                dict(label = 'All Activities',
                     method = 'update',
                     args = [{'visible': [True, True, True, True,True, True, True]},
                             {'title': Job + 'Work Activity: All Activities'}]),
                dict(label = 'Gathering data',
                     method = 'update',
                     args = [{'visible': [True, False, False, False,False, False, False]},
                             {'title': Job + 'Work Activity: Gathering data',
                             'annotations': gather}]),
                dict(label = 'Cleaning data',
                     method = 'update',
                     args = [{'visible': [False, True, False, False,False, False, False]},
                             {'title': Job + 'Work Activity: Cleaning data',
                             'annotations': clean}]),
                dict(label = 'Visualizing data',
                     method = 'update',
                     args = [{'visible': [False, False, True, False,False, False, False]},
                             {'title': Job + 'Work Activity: Visualizing data',
                             'annotations': viz}]),

                dict(label = 'Model building',
                     method = 'update',
                     args = [{'visible': [False, False, False, True,False, False, False]},
                             {'title': Job + 'Work Activity: Model building',
                             'annotations': model}]),
                dict(label = 'Putting the model into production',
                     method = 'update',
                     args = [{'visible': [False, False, False, False,True, False, False]},
                             {'title': Job + 'Work Activity: Putting the model into production',
                             'annotations': prod}]),
                dict(label = 'Finding and Communicating Insights',
                     method = 'update',
                     args = [{'visible': [False, False, False, False,False, True, False]},
                             {'title': Job + 'Work Activity: Finding and Communicating Insights',
                             'annotations': insights}]), 
                 dict(label = 'Other',
                     method = 'update',
                     args = [{'visible': [False, False, False, False,False, False, True]},
                             {'title': Job + 'Work Activity: Other',
                             'annotations': other}]), 

                dict(label = 'Reset',
                     method = 'update',
                     args = [{'visible': [True, True, True, True,True, True, True]},
                             {'title':  Job + 'Work Activity: All Activities'}])
            ]),
        )
    ])

    Job2 = '% of ' + Job.strip() + 's '
    layout = dict(title=Job + 'Work Activities', showlegend=False,
                  updatemenus=updatemenus, 
                  xaxis=dict(
                    title=Job2 + 'That Perform Selected Activity',
                  ),
                  yaxis=dict(
                    title='% of Time Spent at Selected Activity',
                  ))

    fig = dict(data=data, layout=layout)
    filename='update_dropdown'
    return [fig, filename]   
labels_5 = 'Finding and Communicating Insights'

ds_work = agg_data_work(ds_data,277, 283, 'Percentage_of_Time','Data Scientist', labels_5)
da_work = agg_data_work(da_data,277, 283, 'Percentage_of_Time','Data Analyst', labels_5)

Job = 'Data Scientist '
plot = gen_plot(ds_work, Job)
iplot(plot[0], plot[1])

Job = 'Data Analyst '
plot = gen_plot(da_work, Job)
iplot(plot[0], plot[1])
question_info(284)
def gen_plot_ml(ds_work, Job):
    self_data = ds_work[ds_work['Activity'] == ' Self']
    trace_self = go.Scatter(x=list(self_data['Percentage_of_Respondents']),
                            y=list(self_data['Percentage_of_Time']),
                            name='Self Taught',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(self_data, '#33CFA5') ))
    
    online_data = ds_work[ds_work['Activity'] == ' Online courses (Coursera, Udemy, edX, etc.)']
    trace_online = go.Scatter(x=list(online_data['Percentage_of_Respondents']),
                            y=list(online_data['Percentage_of_Time']),
                            name='Online courses',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(online_data, '#4286f4') ))

    work_data = ds_work[ds_work['Activity'] == ' Work']
    trace_work = go.Scatter(x=list(work_data['Percentage_of_Respondents']),
                            y=list(work_data['Percentage_of_Time']),
                            name='Learned Through Work',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(work_data, '#f2435d') ))

    uni_data = ds_work[ds_work['Activity'] == ' University']
    trace_uni = go.Scatter(x=list(uni_data['Percentage_of_Respondents']),
                            y=list(uni_data['Percentage_of_Time']),
                            name='Learned at University',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(uni_data, '#de75ff') ))

    comp_data = ds_work[ds_work['Activity'] == ' Kaggle competitions']
    trace_kaggle = go.Scatter(x=list(comp_data['Percentage_of_Respondents']),
                            y=list(comp_data['Percentage_of_Time']),
                            name='Learned by doing Kaggle competitions',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(comp_data, '#8dfcc4') ))  

    other_data = ds_work[ds_work['Activity'] == ' Other']
    trace_other = go.Scatter(x=list(other_data['Percentage_of_Respondents']),
                            y=list(other_data['Percentage_of_Time']),
                            name='Other',
                            mode = 'markers',
                            marker=dict(size = 10,color= colors(other_data, '#8dfcf5') ))

    data = [trace_self, trace_online, trace_work, trace_uni, trace_kaggle, trace_other]
    
    self=[dict(x=self_data['Percentage_of_Respondents'].mean(),
                       y=self_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. amount of ML knowledge gained at selected activity: ' + str(round(self_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]
    
    online=[dict(x=online_data['Percentage_of_Respondents'].mean(),
                       y=online_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. amount of ML knowledge gained at selected activity: ' + str(round(online_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]
    
    work=[dict(x=work_data['Percentage_of_Respondents'].mean(),
                       y=work_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. amount of ML knowledge gained at selected activity: ' + str(round(work_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]
    
    uni=[dict(x=uni_data['Percentage_of_Respondents'].mean(),
                       y=uni_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. amount of ML knowledge gained at selected activity: ' + str(round(uni_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]
    
    kaggle=[dict(x=comp_data['Percentage_of_Respondents'].mean(),
                       y=comp_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. amount of ML knowledge gained at selected activity: ' + str(round(comp_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]

    
    other=[dict(x=other_data['Percentage_of_Respondents'].mean(),
                       y=other_data['Percentage_of_Time'].mean(),
                       xref='x', yref='y',
                       text='Avg. amount of ML knowledge gained at selected activity: ' + str(round(other_data['Percentage_of_Time'].mean(),2)),
                           ax=200, ay=-50)]


    updatemenus = list([
        dict(active=0,
             buttons=list([  
                dict(label = 'All ML Learning Sources',
                     method = 'update',
                     args = [{'visible': [True, True, True, True,True, True]},
                             {'title': Job + 'ML knowledge: All Sources'}]),
                dict(label = 'Self Taught',
                     method = 'update',
                     args = [{'visible': [True, False, False, False,False, False]},
                             {'title': Job + 'ML knowledge got by: Self Teaching',
                             'annotations': self}]),
                dict(label = 'Online Courses',
                     method = 'update',
                     args = [{'visible': [False, True, False, False,False, False]},
                             {'title': Job + 'ML knowledge got by: Online Courses',
                             'annotations': online}]),
                dict(label = 'Learning Through Work',
                     method = 'update',
                     args = [{'visible': [False, False, True, False,False, False]},
                             {'title': Job + 'ML knowledge got by: Learning Through Work',
                             'annotations': work}]),

                dict(label = 'University Course',
                     method = 'update',
                     args = [{'visible': [False, False, False, True,False, False]},
                             {'title': Job + 'ML knowledge got by: University Course',
                             'annotations': uni}]),
                dict(label = 'Partaking in Kaggle Competitions',
                     method = 'update',
                     args = [{'visible': [False, False, False, False,True, False]},
                             {'title': Job + 'ML knowledge got by: Partaking in Kaggle Competitions',
                             'annotations': kaggle}]),
                 dict(label = 'Other',
                     method = 'update',
                     args = [{'visible': [False, False, False, False,False, True]},
                             {'title': Job + 'ML knowledge got by: Other Sources',
                             'annotations': other}]), 

                dict(label = 'Reset',
                     method = 'update',
                     args = [{'visible': [True, True, True, True,True, True]},
                             {'title':  Job + 'ML knowledge: All Sources'}])
            ]),
        )
    ])

    Job2 = '% of ' + Job.strip() + 's '
    layout = dict(title=Job + 'ML Learning Sources', showlegend=False,
                  updatemenus=updatemenus, 
                  xaxis=dict(
                    title=Job2 + 'Responses',
                  ),
                  yaxis=dict(
                    title='% of ML knowledge gained',
                  ))

    fig = dict(data=data, layout=layout)
    filename='update_dropdown'
    return [fig, filename] 
label_5 = ''
ds_ml = agg_data_work(ds_data,284, 289, 'Percentage_of_Time','Data Scientist', label_5)
da_ml = agg_data_work(da_data,284, 289, 'Percentage_of_Time','Data Analyst', label_5)

Job = 'Data Scientist '
plot = gen_plot_ml(ds_ml, Job)
iplot(plot[0], plot[1])

Job = 'Data Analyst '
plot = gen_plot_ml(da_ml, Job)
iplot(plot[0], plot[1])

question_info(291)
ds_dtype = agg_data(ds_data,291, 303, 'Course','Data Scientist')
da_dtype = agg_data(da_data,291, 303, 'Course','Data Analyst')

agg = ds_dtype.append(da_dtype)

values = da_dtype['Percentage'] * -1
values = values.append(ds_dtype['Percentage'])
values = values.sort_values(ascending=True)

min_v = int(values.min())
max_v = int(values.max())
values = [int(i) for i in values]

labels = da_dtype['Percentage'] * -1
labels = labels.append(ds_dtype['Percentage'])
labels = labels.sort_values(ascending=True)
labels = [int(i) for i in labels]

new_labels =[]
for item in labels:
    if item < 0:
        item = item * -1
        new_labels.append(item)
    else:
        new_labels.append(item)

da_dtype['Percentage'] = da_dtype['Percentage'] * -1


layout = go.Layout(title='Most Popular online courses for data science',
                   yaxis=go.layout.YAxis(tickangle=-15),
                   xaxis=go.layout.XAxis(
                       tickangle=-55,
                       range=[min_v, max_v],
                       tickvals= [int(i) for i in values],
                       ticktext= new_labels,
                       title='Percentage of Respondents'),
                   barmode='overlay',
                   bargap=0.5,
                   height=500,
                  width=900, 
                  margin=go.layout.Margin(l=225, r=0))

data = [go.Bar(y=ds_dtype['Course'],
               x=ds_dtype['Percentage'],
               orientation='h',
               name='Data Scientists',
               marker=dict(color='green')
               ),
        go.Bar(y=ds_dtype['Course'],
               x=da_dtype['Percentage'],
               orientation='h',
               name='Data Analysts',
               marker=dict(color='orange')
               )]

iplot(dict(data=data, layout=layout), filename='EXAMPLES/bar_pyramid')
question_info(330)
ds_qual= agg_data(ds_data,330, 330, 'Quality','Data Scientist')
ds_qual2= agg_data(ds_data,331, 331, 'Quality','Data Scientist')

ds_qual = ds_qual.append(ds_qual)

da_qual = agg_data(da_data,330, 330, 'Quality','Data Analyst')
da_qual2 = agg_data(da_data,331, 331, 'Quality','Data Analyst')

agg = ds_qual.append(da_qual)
agg1 = ds_qual2.append(da_qual2)

fig,axes = plt.subplots(1,1,figsize=(10, 10))
pal = sns.light_palette((105, 70, 84),n_colors=2, input='husl', reverse=True)
sns.set_palette(pal)
ax = sns.barplot(x="Percentage", y="Quality", hue="Job", data=agg)
ax.set_ylabel('Media Sources',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Opinion: Online Courses VS. Traditional Learning Methods" ,fontsize=15)
ax.tick_params(labelsize=15)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title


fig,axes = plt.subplots(1,1,figsize=(10, 10))
pal = sns.light_palette((145, 50, 84),n_colors=2, input='husl', reverse=True)
sns.set_palette(pal)
ax = sns.barplot(x="Percentage", y="Quality", hue="Job", data=agg1)
ax.set_ylabel('Media Sources',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Opinion: Bootcamps VS. Traditional Learning Methods" ,fontsize=15)
ax.tick_params(labelsize=15)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
question_info(307)
ds_media= agg_data(ds_data,307, 328, 'Media','Data Scientist')
da_media = agg_data(da_data,307, 328, 'Media','Data Analyst')

agg = ds_media.append(da_media)

fig,axes = plt.subplots(1,1,figsize=(15, 15))
pal = sns.diverging_palette(120, 10, s=74, l=50,n=2)
sns.set_palette(pal)
ax = sns.barplot(x="Percentage", y="Media", hue="Job", data=agg)
ax.set_ylabel('Media Sources',fontsize=15)
ax.set_xlabel('Percentage of Respondents',fontsize=15)
ax.set_title("Most Popular Media Sources for Data Science" ,fontsize=15)
ax.tick_params(labelsize=15)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
question_info(332)
ds_expertise = agg_data(ds_data,332, 332, 'Exp','Data Scientist')
da_expertise = agg_data(da_data,332, 332, 'Exp','Data Analyst')

ds_expertise = ds_expertise.replace('Independent projects are slightly more important than academic achievements', 'Independent projects are slightly more important')
ds_expertise = ds_expertise.replace('Independent projects are much more important than academic achievements', 'Independent projects are much more important')
ds_expertise = ds_expertise.replace('Independent projects are equally important as academic achievements', 'Independent projects are equally important')
ds_expertise = ds_expertise.replace('Independent projects are slightly less important than academic achievements', 'Independent projects are slightly less important')
ds_expertise = ds_expertise.replace('Independent projects are much less important than academic achievements', 'Independent projects are much less important')

da_expertise = ds_expertise.replace('Independent projects are slightly more important than academic achievements', 'Independent projects are slightly more important')
da_expertise = ds_expertise.replace('Independent projects are much more important than academic achievements', 'Independent projects are much more important')
da_expertise = ds_expertise.replace('Independent projects are equally important as academic achievements', 'Independent projects are equally important')
da_expertise = ds_expertise.replace('Independent projects are slightly less important than academic achievements', 'Independent projects are slightly less important')
da_expertise = ds_expertise.replace('Independent projects are much less important than academic achievements', 'Independent projects are much less important')

ds_expertise.iplot(kind='pie',labels='Exp',values='Count',pull=.1,hole=.1, 
          colorscale='Greens',textposition='outside',textinfo='value+percent', 
        title='Academic achievement or independent projects?. Data Scientist Responses')

da_expertise.iplot(kind='pie',labels='Exp',values='Count',pull=.1,hole=.1, 
          colorscale='Reds',textposition='outside',textinfo='value+percent', 
         title='Academic achievement or independent projects?. Data Analyst Responses')

