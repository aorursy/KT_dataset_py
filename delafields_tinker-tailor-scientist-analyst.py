# import packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as mtick

import seaborn as sns

from itertools import chain

from fuzzywuzzy import fuzz

import math



%matplotlib inline

sns.set()



# setting some global styles for matplotlib & pandas

plt.rcParams['font.family'] = 'monospace'

sns.set_style({"axes.facecolor": "1.0", 'grid.linestyle': '--', 'grid.color': '.8'})

colors = ["#fcd74e", "#0b84a5"]

pd.set_option("display.max_columns", 300)

pd.set_option('display.max_colwidth', -1)



# dev vs prod settings

dev = False



if dev == True:

    import os   

    # Print input files

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))

else:            

    import warnings 

    warnings.filterwarnings('ignore')

            

# import data

mult_choice_responses = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

other_responses       = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')
###############################################################################################

# replace mult_choice OTHER columns with their actual text responses (from other_responses)

OTHERS_to_replace = ['Q9_OTHER_TEXT', 'Q14_Part_1_TEXT', 'Q14_Part_2_TEXT', 'Q14_Part_3_TEXT',

                     'Q14_Part_4_TEXT', 'Q14_Part_5_TEXT', 'Q18_OTHER_TEXT', 'Q19_OTHER_TEXT',

                     'Q20_OTHER_TEXT', 'Q24_OTHER_TEXT', 'Q27_OTHER_TEXT', 'Q28_OTHER_TEXT', 

                     'Q34_OTHER_TEXT']



for col in OTHERS_to_replace:

    mult_choice_responses[col] = other_responses[col]

###############################################################################################



###############################################################################################

# rename columns for easier access

name_mapping = {

    # demographics

    'Q1': 'Age',

    'Q2': 'Gender',

    'Q3': 'Country',

    # professional

    'Q4': 'Education',

    'Q5': 'Title',

    'Q10': 'Annual Compensation',

    'Q15': 'Years_Exp_Data',

    'Q23': 'Years_Exp_ML',

    # day-to-day work roles

    'Q9_Part_1': 'Analyze data to influence biz decisions',

    'Q9_Part_2': 'Build/run data infrastructure',

    'Q9_Part_3': 'Prototype ML applications',

    'Q9_Part_4': 'Build/run an internal ML service',

    'Q9_Part_5': 'Improve existing ML models',

    'Q9_Part_6': 'Research state-of-the-art ML',

    'Q9_Part_7': 'None of these',

    'Q9_Part_8': 'Other',

    # tools used

    'Q14_Part_1_TEXT': 'Basic_stats_software',

    'Q14_Part_2_TEXT': 'Advanced_stats_software',

    'Q14_Part_3_TEXT': 'BI_software',

    'Q14_Part_4_TEXT': 'Local_envs',

    'Q14_Part_5_TEXT': 'Cloud_software'}



mult_choice_responses = mult_choice_responses.rename(columns = name_mapping)

###############################################################################################



###############################################################################################

# aggregate multi-column questions for plotting ease

text_question_cols = {

    'work_roles':    ['Analyze data to influence biz decisions', 'Build/run data infrastructure', 

                      'Prototype ML applications', 'Build/run an internal ML service',

                      'Improve existing ML models', 'Research state-of-the-art ML',

                      'None of these', 'Other'],

    'Programming Language':     list(mult_choice_responses.filter(like='Q18').columns), 

    'beginner_lang': list(mult_choice_responses.filter(like='Q19').columns),

    'Visualization Tools':      list(mult_choice_responses.filter(like='Q20').columns),

    'Algorithms':    list(mult_choice_responses.filter(like='Q24').columns),

    'NLP Tools':     list(mult_choice_responses.filter(like='Q27').columns),

    'ML Frameworks': list(mult_choice_responses.filter(like='Q28').columns),

    'Relational DB Tools':  list(mult_choice_responses.filter(like='Q34').columns)}

###############################################################################################



###############################################################################################

# renaming tool

def renamer(col, old_name, new_name):

    mult_choice_responses[col] = mult_choice_responses[col].replace(regex=old_name, value=new_name)



renamer(col='Country', 

        new_name='United Kingdom', 

        old_name='United Kingdom of Great Britain and Northern Ireland')    

renamer(col='Education', 

        new_name='Some college', 

        old_name='Some college/university study without earning a bachelor’s degree')

renamer(col='Education', 

        new_name='High school', 

        old_name='No formal education past high school')

renamer(col='Education', 

        new_name='Never', 

        old_name='I have never written code')

renamer(col='Analyze data to influence biz decisions', 

        new_name='Analyze data to influence biz decisions', 

        old_name='Analyze and understand data to influence product or business decisions')

renamer(col='Build/run data infrastructure', 

        new_name='Build/run data infrastructure', 

        old_name='Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data')

renamer(col='Prototype ML applications', 

        new_name='Prototype ML applications', 

        old_name='Build prototypes to explore applying machine learning to new areas')

renamer(col='Build/run an internal ML service', 

        new_name='Build/run an internal ML service', 

        old_name='Build and/or run a machine learning service that operationally improves my product or workflows')

renamer(col='Improve existing ML models', 

        new_name='Improve existing ML models', 

        old_name='Experimentation and iteration to improve existing ML models')

renamer(col='Research state-of-the-art ML', 

        new_name='Research state-of-the-art ML', 

        old_name='Do research that advances the state of the art of machine learning')

renamer(col='None of these', 

        new_name='None of these', 

        old_name='None of these activities are an important part of my role at work')

###############################################################################################



###############################################################################################

# bucket compensation col

compensation_replace_dict = {

    '$0-999': '< 10,000','1,000-1,999': '< 10,000','2,000-2,999': '< 10,000','3,000-3,999': '< 10,000',

    '4,000-4,999': '< 10,000','5,000-7,499': '< 10,000','7,500-9,999': '< 10,000','10,000-14,999': '10,000 - 50,000',

    '15,000-19,999': '10,000 - 50,000','20,000-24,999': '10,000 - 50,000','25,000-29,999': '10,000 - 50,000',

    '30,000-39,999': '10,000 - 50,000','40,000-49,999': '10,000 - 50,000','50,000-59,999': '50,000 - 99,000',

    '60,000-69,999': '50,000 - 99,000','70,000-79,999': '50,000 - 99,000','80,000-89,999': '50,000 - 99,000',

    '90,000-99,999': '50,000 - 99,000','100,000-124,999': '> 100,000','125,000-149,999': '> 100,000',

    '150,000-199,999': '> 100,000','200,000-249,999': '> 100,000','250,000-299,999': '> 100,000',

    '300,000-500,000': '> 100,000','> $500,000': '> 100,000'}



mult_choice_responses['Annual Compensation'] = mult_choice_responses['Annual Compensation'].replace(compensation_replace_dict)

###############################################################################################



###############################################################################################

# preprocess the responses to "tools used"

for col in ['Basic_stats_software', 'Advanced_stats_software', 'BI_software', 'Local_envs', 'Cloud_software']:

    mult_choice_responses[col] = mult_choice_responses[col].str.strip().str.lower().fillna('')

###############################################################################################



###############################################################################################

# only keep necessary columns for profiling

non_text_cols = list(name_mapping.values())

text_cols     = list(chain(*text_question_cols.values()))

# list + set to remove duplicate work_roles

keep          = list(set(non_text_cols + text_cols))



mult_choice_responses = mult_choice_responses[keep]

###############################################################################################
# Looking @ just Data Scientists, Data Analysts

data_jobs = mult_choice_responses[mult_choice_responses['Title'].isin(['Data Scientist', 'Data Analyst'])]



num_analysts = len(data_jobs[data_jobs['Title'] == 'Data Analyst'])

num_scientists = len(data_jobs[data_jobs['Title'] == 'Data Scientist'])
tools_lookup_dict = {

    'Basic_stats_software':    ['excel', 'python', 'sheets', 'r', 'power bi', 'sql', 'libra', 'tableau', 'weka'],

    'Advanced_stats_software': ['sas', 'spss', 'python', 'r', 'matlab', 'sap'],

    'BI_software':             ['tableau', 'power bi', 'qlik'], 

    'Local_envs':              ['jupyter', 'rstudio', 'pycharm', 'spyder', 'visual studio', 'vscode', 'anaconda'],

    'Cloud_software':          ['aws', 'amazon', 'azure', 'gcp', 'bigquery', 'colab', 'watson', 'ibm', 

                                'databricks', 'paperspace', 'sagemaker']}



tools_rename_dict = {

    'Basic_stats_software':    {'excel': 'Excel', 'python': 'Python', 'sheets': 'Sheets', 'r': 'R', 'power bi': 'Power BI', 

                                'sql': 'SQL', 'libra': 'Libra', 'tableau': 'Tableau', 'weka': 'Weka'},

    

    'Advanced_stats_software': {'sas': 'SAS', 'spss': 'SPSS', 'python': 'Python', 'r': 'R', 'matlab': 'Matlab', 'sap': 'SAP'},

    

    'BI_software':             {'tableau': 'Tableau', 'power bi': 'Power BI', 'qlik': 'Qlik'}, 

    

    'Local_envs':              {'jupyter': 'Jupyter', 'rstudio': 'RStudio', 'pycharm': 'PyCharm', 'spyder': 'Spyder', 

                                'visual studio': 'Visual Studio', 'vscode': 'Visual Studio', 'anaconda': 'Anaconda'},

    

    'Cloud_software':          {'aws': 'AWS', 'amazon': 'AWS', 'azure': 'Azure', 'gcp': 'GCP', 'bigquery': 'GCP', 

                                'colab': 'Colab', 'watson': 'IBM Watson', 'ibm': 'IBM Watson', 'databricks': 'Databricks',

                                'paperspace': 'Paperspace', 'sagemaker': 'Sagemaker'}}



def fuzzy_match(row, tool_category, match):

    '''function to fuzzy match values via the dict above'''

    name = row[tool_category]

    return fuzz.partial_ratio(name, match)



def create_tool_df(tool_category):

    '''creates a DataFrame of a certain tool category'''

    

    dataframe = pd.DataFrame()

    

    for tool in tools_lookup_dict[tool_category]:

        if tool != 'r':

            # fuzzy match cell values (threshold = > 70)

            temp = data_jobs[data_jobs.apply(fuzzy_match, tool_category=tool_category, match=tool, axis=1) > 70]

        else:

            # no good way to fuzzy match r

            temp = data_jobs[data_jobs[tool_category] == 'r']

            

        # narrow down columns

        temp = temp[['Title', tool_category]]

        # rename columns to their appropriate tool

        temp[tool_category] = tool

        # append to dataframe

        dataframe = dataframe.append(temp)

        

    # rename    

    dataframe[tool_category] = dataframe[tool_category].replace(tools_rename_dict[tool_category])

        

    return dataframe



Basic_stats_software = create_tool_df('Basic_stats_software')

Advanced_stats_software = create_tool_df('Advanced_stats_software')

BI_software = create_tool_df('BI_software')

Local_envs = create_tool_df('Local_envs')

Cloud_software = create_tool_df('Cloud_software')
def get_single_col_freq_percents(dataframe, col):

    '''Get the % of responses per value (single column of df)'''

    # get counts by job title

    count_per_col = dataframe.groupby(['Title', col]).size()

    

    # get frequency percentage via the counts

    col_freq_percents = pd.DataFrame(count_per_col.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())))

    

    # reset and rename index

    col_freq_percents.reset_index(inplace=True)

    col_freq_percents = col_freq_percents.rename(columns={0: '% of Responses'})

    

    # sort DataFrame

    order = {

        'Education': ['High school', 'Some college', 'Professional degree', 

                      'Bachelor’s degree', 'Master’s degree',  'Doctoral degree'],

        'Years_Exp_Data': ['< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years'],

        'Years_Exp_ML': ['< 1 years', '1-2 years', '2-3 years', '3-4 years', 

                         '4-5 years', '5-10 years', '10-15 years', '20+ years'],

        'Gender': ['Female', 'Male'],

        'Annual Compensation': ['< 10,000', '10,000 - 50,000', '50,000 - 99,000', '> 100,000']}

    

    if col in list(order.keys()):

        col_freq_percents[col] = pd.Categorical(col_freq_percents[col], order[col])

        col_freq_percents.sort_values(col)

    

    return col_freq_percents





def get_multicol_freq_percents(question_list):

    '''Get frequency % of responses (over multiple columns)'''

    

    # get counts for Data Scientists and Data Analysts

    DA = data_jobs[data_jobs.Title == 'Data Analyst'][question_list].fillna('').stack().value_counts()

    DS = data_jobs[data_jobs.Title == 'Data Scientist'][question_list].fillna('').stack().value_counts()

    

    # calculate frequency percents

    DA = DA.apply(lambda x: (x / num_analysts) * 100)

    DS = DS.apply(lambda x: (x / num_scientists) * 100)

    

    # combine titles

    df = pd.DataFrame({'Data Analyst': DA, 'Data Scientist': DS}).drop(labels='')

    

    # select top 10 responses

    top10_idx = df.sum(axis=1).sort_values(ascending=False).head(10).index

    df = df.loc[top10_idx]

    

    return df



def single_col_bar_plotter(category, plot_size=(6,5), axis_limit=5.0, legend_loc='lower right', axes=None):

    '''plot a single column response'''

    

    #plt.figure(figsize=plot_size)

    if category == 'Country':

        df = get_single_col_freq_percents(data_jobs, 'Country')

        temp = df[df['Country'] != 'Other'] # Remove 'Others'

        temp = temp.groupby(['Country']).sum() # sum by % of respondants

        temp = temp.reset_index().sort_values('% of Responses', ascending=False).head(10) # top 10

        top_10_countries = list(temp['Country'])



        df = df[df['Country'].isin(top_10_countries)]

        ax = sns.barplot(x='% of Responses', y=category, data=df, 

                         hue='Title', palette=sns.color_palette(colors), edgecolor='.2', ax=axes)

        

    elif category in ['Basic_stats_software','Advanced_stats_software','BI_software','Local_envs','Cloud_software']:

        ax = sns.barplot(x='% of Responses', y=category, data=get_single_col_freq_percents(globals()[category], category),

                         hue='Title', palette=sns.color_palette(colors), edgecolor='.2', ax=axes)

        

    else:

        ax = sns.barplot(x='% of Responses', y=category, data=get_single_col_freq_percents(data_jobs, category),

                         hue='Title', palette=sns.color_palette(colors), edgecolor='.2', ax=axes)

  

    # title formatting

    if category == 'Country':

        ax.set_title('Top 10 Responding Countries\n', fontsize=14)

    elif category == 'Years_Exp_Data':

        ax.set_title('Years Using Code to Analyze Data\n', fontsize=14)

    elif category == 'Years_Exp_ML':

        ax.set_title('Years Using Machine Learning Methods\n', fontsize=14)

    elif category == 'Annual Compensation':

        ax.set_title(category + ' (US$) of Respondents \n', fontsize=14)

    elif category == 'Basic_stats_software':

        ax.set_title('Basic Statistical Software Used by Respondents\n', fontsize=14)

    elif category == 'Advanced_stats_software':

        ax.set_title('Advanced Statistical Software Used by Respondents\n', fontsize=14)        

    elif category == 'BI_software':

        ax.set_title('Business Intelligence Software Used by Respondents\n', fontsize=14)

    elif category == 'Local_envs':

        ax.set_title('Local Environments Used by Respondents\n', fontsize=14)    

    elif category == 'Cloud_software':

        ax.set_title('Cloud Software Used by Respondents\n', fontsize=14)          

    else:

        ax.set_title(category + ' of Respondents \n', fontsize=14)



    # legend formatting

    legend = plt.legend(frameon=True)

    legend_frame = legend.get_frame()

    legend_frame.set_facecolor('white')

    legend_frame.set_edgecolor('black')

    plt.legend(loc=legend_loc)

    

    # axis formatting

    ax.yaxis.label.set_visible(False)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    x_ax_lim = int(math.ceil(ax.get_xlim()[1] / axis_limit) * axis_limit) # end at a 5.0%

    ax.set(xlim=(0.0, x_ax_lim))

    

    return ax



def multi_col_bar_plotter(category, plot_size=(6,5), axis_limit=5.0, axes=None):

    '''plots multi column responses'''

    

    #plt.figure(figsize=plot_size)

    

    # get frequency percents

    df = get_multicol_freq_percents(text_question_cols[category])

    

    # melt multiple columns

    df = df.reset_index()

    df = pd.melt(df,

                 id_vars=['index'], var_name='Title',

                 value_vars=['Data Analyst', 'Data Scientist'], value_name='% of Responses')

    

    ax = sns.barplot(x='% of Responses', y='index', data=df, hue='Title', 

                 palette=sns.color_palette(colors), edgecolor=".2", ax=axes)



    # title formatting

    if category == 'beginner_lang':

        ax.set_title('Respondents\' suggested programming language for beginners\n', fontsize=14)

    elif category == 'work_roles':

        ax.set_title('Respondents\' roles at work \n', fontsize=14)

    else:

        ax.set_title(category + ' used by Respondents \n', fontsize=14)

    

    # legend formatting

    ax.legend(loc='lower right')

    

    # axis formatting

    ax.yaxis.label.set_visible(False)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    x_ax_lim = int(math.ceil(ax.get_xlim()[1] / axis_limit) * axis_limit) # end at a 5.0%

    ax.set(xlim=(0.0, x_ax_lim))

    

    return ax
ax = sns.countplot(x='Title', data=data_jobs, order=['Data Analyst', 'Data Scientist'],

              palette=sns.color_palette(colors), edgecolor='.2')



ax.set_title('Number of Respondents by Title\n')

ax.set_ylabel('')

ax.set_xlabel('')



for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2.,p.get_height()), 

                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
fig, axarr = plt.subplots(1, 2, figsize=(20, 6))

plt.subplots_adjust(wspace=0.4)

single_col_bar_plotter('Age', legend_loc='upper right', axes=axarr[0])

single_col_bar_plotter('Gender', legend_loc='upper right', axes=axarr[1]);
fig, axarr = plt.subplots(1, figsize=(10, 4))

single_col_bar_plotter('Country', legend_loc='upper right');
fig, axarr = plt.subplots(1, figsize=(10, 4))

single_col_bar_plotter('Education', legend_loc='upper right');
fig, axarr = plt.subplots(1, figsize=(10, 4))

single_col_bar_plotter('Annual Compensation', legend_loc='lower right');
fig, axarr = plt.subplots(1, 2, figsize=(20, 6))

plt.subplots_adjust(wspace=0.4)

single_col_bar_plotter('Years_Exp_Data', axes=axarr[0])

single_col_bar_plotter('Years_Exp_ML', legend_loc='lower right', axes=axarr[1]);
fig, axarr = plt.subplots(1, figsize=(12, 6))

multi_col_bar_plotter('work_roles');
fig, axarr = plt.subplots(1, 2, figsize=(20, 6))

plt.subplots_adjust(wspace=0.4)

multi_col_bar_plotter('Programming Language', axes=axarr[0])

multi_col_bar_plotter('beginner_lang', axes=axarr[1]);
fig, axarr = plt.subplots(2, 2, figsize=(20, 10))

plt.subplots_adjust(wspace=0.4, hspace=0.6)

single_col_bar_plotter('BI_software', axes=axarr[0][0])

single_col_bar_plotter('Cloud_software', axes=axarr[0][1], legend_loc='lower right')

single_col_bar_plotter('Basic_stats_software', axes=axarr[1][0])

single_col_bar_plotter('Advanced_stats_software', axes=axarr[1][1], legend_loc='upper right');
fig, axarr = plt.subplots(1, figsize=(10, 4))

single_col_bar_plotter('Local_envs', legend_loc='lower right');
fig, axarr = plt.subplots(1, figsize=(10, 8))

multi_col_bar_plotter('Algorithms');
fig, axarr = plt.subplots(1, figsize=(10, 4))

multi_col_bar_plotter('ML Frameworks');
fig, axarr = plt.subplots(1, figsize=(10, 4))

multi_col_bar_plotter('Visualization Tools');
fig, axarr = plt.subplots(1, figsize=(10, 4))

multi_col_bar_plotter('Relational DB Tools');
fig, axarr = plt.subplots(1, figsize=(10, 8))

multi_col_bar_plotter('NLP Tools');