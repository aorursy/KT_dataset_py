import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import umap as umap



# hide setting with copy warning to keep things clean

import warnings

warnings.filterwarnings('ignore')



pd.set_option("max_columns", 102)





sns.set_style('whitegrid')
UK_COLOUR = '#C8102E'

EU_COLOUR = '#1E448A'

color_dict = {'EU': EU_COLOUR, 'UK':UK_COLOUR}

eu_countries = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark','Estonia','Finland','France',

               'Germany','Greece','Hungary','Ireland','Italy','Latvia','Lithuania','Luxembourg','Malta','Netherlands',

               'Poland','Portugal','Romania','Slovakia','Slovenia','Spain','Sweden']



survey_years = np.r_[2019, 2018, 2017]
# some helper functions used throughout the notebook

def map_ages_to_intervals(x):

    """

    Bins a continuous range of ages into discrete intervals.



    Parameters:

    -----------

    x : pd.Series,

        Series of float values of ages

        

    Returns:

    --------

    x_binned, pd.Series,

        The binned ages.

    """



    age_bins = pd.IntervalIndex.from_tuples([(18, 21), (22, 24), (25, 29),

                                             (30, 34), (35, 39), (40, 44),

                                             (45, 49), (50, 54), (55, 59),

                                             (60,64), (65,69), (70, 100)])

    

    x = pd.cut(x, age_bins).astype(str)

    x = (x.str.replace(', ', '-')

          .str.strip('(]')

          .str.replace('.0-','-')

          .apply(lambda x: x[:-2]))

    return x



def turn_off_lines(ax):

    """

    Turns off lines on a matplotlib axes object.

    """

    ax.spines['right'].set_visible(False)

    ax.spines['top'].set_visible(False)

    ax.yaxis.set_ticks_position('left')

    ax.xaxis.set_ticks_position('bottom')

    ax.grid(False)



def load_and_process(eu_countries, drop_non_eu=True):

    """

    Loads the kaggle ML & DS survey results from 2017, 2018 and 2019.

    It casts to the results to a DataFrame with a structure consistent 

    across the years. It only extracts a subset of the total questions

    and only takes countries in the EU (including the UK).

    

    Currently, in the interest of time, it is not robust to the addition 

    of future years (i.e., 2020 onwards).

    

    Parameters:

    -----------

    eu_countries: iterable, 

        List-like object containing the names of each country (excluding the UK) in the EU

    

    Yields:

    --------

    df: 

        Returns a list of the survey results where each entry s a pandas DataFrame with the

        results from a given year.

    """

    

    # load the data

    prefix = '/kaggle/input/kaggle-survey'

    df_2019 = pd.read_csv(prefix + '-2019/multiple_choice_responses.csv')

    df_2018 = pd.read_csv(prefix + '-2018/multipleChoiceResponses.csv')

    df_2017 = pd.read_csv(prefix + '-2017/multipleChoiceResponses.csv', encoding="ISO-8859-1")

    

    # align the columns we want to create in out dataframes

    cols_to_generate = ['country','gender','age','education',

                        'employment_status','salary']

    col_names_2019 = ['Q3','Q2','Q1','Q4','Q5','Q10']

    col_names_2018 = ['Q3','Q1','Q2','Q4','Q6','Q9']

    col_names_2017 = ['Country','GenderSelect','Age','FormalEducation',

                      'EmploymentStatus','CompensationAmount']



    for df, cols in zip([df_2019, df_2018, df_2017], 

                        [col_names_2019, col_names_2018, col_names_2017]):

        # instantiate our new df

        processed_df = pd.DataFrame({'body': df[cols[0]].values})

        

        # classify countries as EU member state or UK

        EU_mask = processed_df['body'].isin(eu_countries)

        UK_mask = processed_df['body'].str.contains('United Kingdom', case=False)==True

        processed_df.loc[UK_mask,'body'] = 'UK'

        processed_df.loc[EU_mask,'body'] = 'EU'



        # create the other columns

        for new_col, old_col in zip(cols_to_generate, cols):

            processed_df[new_col] = df[old_col].values

            

        # perform some manual corrections

        

        # map the age column for 2017 to intervals

        if 'Age' in cols:

            processed_df.loc[processed_df['age']<17, 'age'] = np.nan

            processed_df['age'] = map_ages_to_intervals(processed_df['age'])

        

        # group 'out' of bound ages and fill NaNs

        high_ages = ['80+','70-100','70-79']

        processed_df.loc[processed_df['age'].isin(high_ages), 'age'] = '70+'

        sixties = ['60-64','65-69']

        processed_df.loc[processed_df['age'].isin(sixties), 'age'] = '60-69'

        processed_df.loc[processed_df['age']=='n','age'] = np.nan

        

        # process the salary column for the 2019 survey

        if cols[5]=='Q10':



            for i, col in enumerate(['salary_lower', 'salary_higher']):

                # remove the row featuring the questions

                processed_df = processed_df[~(processed_df.salary.str.contains('What is your current')==True)]

                # format the salary columns (remove $,commas and > signs)

                processed_df[col] = (processed_df['salary']

                                    .str.split('-', expand=True)[i]

                                    .str.replace('$','')

                                    .str.replace(',','')

                                    .str.replace('>','')

                                    .astype(float)

                                    )

            

            # add media question responses

            media_questions = df.columns[df.columns.str.contains('Q12')]

            processed_df[media_questions] = df[media_questions]

            # add IDE questions

            IDE_questions = df.columns[df.columns.str.contains('Q16')]

            processed_df[IDE_questions] = df[IDE_questions]

            # add programming questions

            prog_questions = df.columns[df.columns.str.contains('Q18')]

            processed_df[prog_questions] = df[prog_questions]

            processed_df['job'] = df['Q5']

        

        # corrections for 2018 and 2019 survies

        if 'Q1' in cols:

             # add the survery response time

            processed_df['completion_time'] = df['Time from Start to Finish (seconds)']

        

            # Add coding experience

            experience_mapping = {'0-1':'0-1','< 1':'0-1','1-2':'1-2','2-3':'3-5',

                                  '3-4':'3-5','4-5':'3-5','3-5':'3-5','5-10':'5-10',

                                  '10-15':'10-20','10-20':'10-20','15-20':'10-20',

                                  '20+':'20-+','20-25':'20-+','25-30':'20-+','30 +':'20-+',

                                  'I have never written code': '0-1'}

            try:

                # 2019 survey, map it to single categories

                processed_df['coding_experience'] = df['Q15']

                processed_df['coding_experience'] = (processed_df['coding_experience']

                                                     .str.replace('years','')

                                                     .str.strip()

                                                     .map(experience_mapping))

            except:

                # 2018 survey, map it to single categories

                processed_df['coding_experience'] = df['Q8']

                processed_df['coding_experience'] = (processed_df['coding_experience']

                                                     .str.strip()

                                                     .map(experience_mapping))

        

        if drop_non_eu:

            # drop countries which are not EU or UK

            processed_df = processed_df[processed_df['body'].isin(['UK','EU'])]  

            

        yield processed_df

        

def autolabel(rects):

    """

    Attach a text label above a bar in a bar chart displaying its height.

    

    This is taken from the matplotlib example gallery:

    https://matplotlib.org/examples/api/barchart_demo.html

    """

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,

                '%d' % int(height),

                ha='center', va='bottom')

     
# a list of dfs for each survey [2019, 2018, 2017]

yearly_dfs = list(load_and_process(eu_countries))
# prepare data for this analysis

eu_respondents = []

uk_respondents = []

for df in yearly_dfs:

    counts = df.groupby('body')['country'].count()

    eu_respondents.append(counts['EU'])

    uk_respondents.append(counts['UK'])

    

eu_respondents = np.array(eu_respondents)

uk_respondents = np.array(uk_respondents)
fig, (ax, ax2) = plt.subplots(1, 2)

# plot bar chart and offset them along x axis to make them side by side

eu_rects = ax.bar(survey_years-0.2, eu_respondents, 0.4, color=EU_COLOUR, label='EU', alpha=0.8)

uk_rects = ax.bar(survey_years+0.2, uk_respondents, 0.4, color=UK_COLOUR, label='UK', alpha=0.8)





# add labels and sort out tick labels

ax.legend()

ax.set_ylabel('Number of responses')

ax.set_xlabel('Year')

ax.set_xticklabels(['','','2017','','2018','','2019'])

ax.set_ylim(0,4500)

ax.set_title('Net Participation')



# add counts to the top of the bars

autolabel(eu_rects)

autolabel(uk_rects)





# plot the normalised data on the second axis



# population figures from wikipedia

eu_pop = np.r_[513481691,(513481691+511643456)/2,511643456]

uk_pop = np.r_[66647112,(66647112+65808573)/2,65808573]

eu_pop = eu_pop - uk_pop



eu_rects = ax2.bar(survey_years-0.2, eu_respondents/eu_pop * 100, 0.4, color=EU_COLOUR, label='EU', alpha=0.8)

uk_rects = ax2.bar(survey_years+0.2, uk_respondents/uk_pop * 100, 0.4, color=UK_COLOUR, label='UK', alpha=0.8)





# add labels and sort out tick labels

ax2.legend()

ax2.set_ylabel('Fraction of population responding (%)')

ax2.set_xlabel('Year')

_ = ax2.set_xticklabels(['','','2017','','2018','','2019'])

ax2.set_title('Participation normalised by population')



for ax in [ax,ax2]:

    turn_off_lines(ax)



fig.set_size_inches(13,5)
fig, axes = plt.subplots(ncols=2, sharey=True)

max_comp_time = 7500 # cut-off for max response time to plot

bins = np.linspace(0,max_comp_time,50)

for i, ax in enumerate(axes):

    for body in ['EU','UK']:

        body_mask = yearly_dfs[i]['body']==body

        time_2019 = yearly_dfs[i].loc[body_mask,'completion_time'].astype(float, errors='ignore')

        sns.kdeplot(time_2019[time_2019<max_comp_time], color=color_dict[body],alpha=0.3, label=body, ax=ax, shade=True)

        #ax.hist(time_2019[time_2019<max_comp_time], bins=bins, color=color_dict[body],

                #density=True, alpha=0.4, label=body)

        ax.set_title(survey_years[i])

        ax.set_xlabel('Survey completion time (seconds)')

        turn_off_lines(ax)

# finishing touches

axes[0].set_ylabel('Probabilty density')

axes[1].legend()

fig.set_size_inches(10,5)
fig, axes = plt.subplots(3,3, gridspec_kw= {'wspace':0.5, 'hspace':0,'height_ratios': [4, 4,2]})



residuals = []

for axs, body in zip(axes, ('EU','UK')):

    for i, ax in enumerate(axs):

        # make the plot

        if body=='UK':

            ax.invert_yaxis()

            #ax.axhline(0.12,color='grey', alpha=0.5)

        grouped_by_age = yearly_dfs[i][yearly_dfs[i]['body']==body].groupby('age')['body'].count() 

        grouped_by_age = grouped_by_age / grouped_by_age.sum() # normalise

        ax.bar(grouped_by_age.index, grouped_by_age.values, color=color_dict[body], label=body, alpha=0.5,lw=2)

        turn_off_lines(ax)

        ax.grid(axis='y')

        if i==2:

            ax.legend()

        # store values for residual plot 

        residuals.append(grouped_by_age.values)

        if body=='EU':

            ax.set_title(survey_years[i])



# plot the difference graph

residuals = np.array(residuals).reshape(33,2)      

differences = (residuals[:,0]-residuals[:,1])#/(residuals[:,0]+residuals[:,1])

differences = differences.reshape(3,11)



for ax, resid in zip(axes[2], differences):

    # make the colours for the differance graph

    colors = pd.Series(resid.copy()) # be flamboyent with types

    mask = colors>0

    colors[mask] = color_dict['EU']

    colors[~mask] = color_dict['UK']

    ax.bar(grouped_by_age.index, resid, color=colors, alpha=0.3)

    turn_off_lines(ax)

    plt.setp(ax.get_xticklabels(), rotation=90)

    ax.set_ylim(-0.2,0.2)

    ax.set_xlabel('Age group (years)')

    ax.spines['left'].set_visible(False)

    ax.yaxis.set_visible(False)

    ax.axhline(0, color='grey', alpha=0.3)

axes[0][0].set_ylabel('Fraction of respondents')

axes[0][0].yaxis.set_label_coords(-0.175, 0)



fig.set_size_inches(15,4)
UK_MULTI_COLOURS = ['#810a1e','#ef3654','#b00e29','#e01233']

EU_MULTI_COLOURS = ['#10254b','#2c63c9','#193a75','#234e9f']

pi_labels = ['UK: Female','UK: Male','UK: Other','EU: Female','EU: Male','EU: Other']

size = 0.3 # used to determine size of the doughnuts

# kwargs shared across the charts

shared_pi_formats = {'wedgeprops':dict(width=size, edgecolor='w'),

                     'textprops':dict(color='white', fontsize=8),

                     'autopct':'%.1f %%'}



fig, axes = plt.subplots(ncols=3)

for i, ax in enumerate(axes):

    # group all the other genders to 'Other' for simplicity see disclaimer above

    yearly_dfs[i].loc[~yearly_dfs[i]['gender'].isin(['Male','Female']), 'gender'] = 'Other'

    UK_mask = yearly_dfs[i]['body']=='UK'

    EU_mask = yearly_dfs[i]['body']=='EU'

    ax.pie(yearly_dfs[i][UK_mask].groupby('gender')['body'].count(), radius=1,

           colors=UK_MULTI_COLOURS, pctdistance=0.85, **shared_pi_formats)

    ax.pie(yearly_dfs[i][EU_mask].groupby('gender')['body'].count(), radius=1-size,

           colors=EU_MULTI_COLOURS, pctdistance=0.78, **shared_pi_formats)

    ax.set(aspect="equal", title=survey_years[i])



    

axes[2].legend(loc=(0.8,0.8), labels=pi_labels)

fig.set_size_inches(16,6)
# create a dictionary with eductation data in

EU_dict = {'Bachelor':[],'Master':[],'Doctoral':[]}

UK_dict = {'Bachelor':[],'Master':[],'Doctoral':[]}

for i in range(0,3):

    uk_mask = yearly_dfs[i]['body']=='UK'

    eu_mask = yearly_dfs[i]['body']=='EU'

    eu_degree_fractions = (yearly_dfs[i][eu_mask].groupby('education')['body'].count() / 

                           yearly_dfs[i][eu_mask].groupby('education')['body'].count().sum())

    uk_degree_fractions = (yearly_dfs[i][uk_mask].groupby('education')['body'].count() / 

                           yearly_dfs[i][uk_mask].groupby('education')['body'].count().sum())

    # extract values for each degree and append to global dicts

    for degree in ['Bachelor','Master','Doctoral']:

        eu_degree_mask = eu_degree_fractions.index.str.contains(degree)

        uk_degree_mask = uk_degree_fractions.index.str.contains(degree)

        EU_dict[degree].append(eu_degree_fractions[eu_degree_mask].values)

        UK_dict[degree].append(uk_degree_fractions[uk_degree_mask].values)

        

# plot the data we created



fig, ax = plt.subplots(1, 1)

ax.bar(survey_years-0.2,  np.ravel(EU_dict['Doctoral']), 0.4, color=EU_MULTI_COLOURS[0],

       alpha=0.8)

ax.bar(survey_years+0.2, np.ravel(UK_dict['Doctoral']), 0.4, color=UK_MULTI_COLOURS[0],

       alpha=0.8)

ax.bar(survey_years-0.2, np.ravel(EU_dict['Master']), 0.4, color=EU_MULTI_COLOURS[1],

       label='EU', alpha=0.8, bottom = np.ravel(EU_dict['Doctoral']))

ax.bar(survey_years+0.2, np.ravel(UK_dict['Master']), 0.4, color=UK_MULTI_COLOURS[1],

       label='UK', alpha=0.8, bottom = np.ravel(UK_dict['Doctoral']))

ax.bar(survey_years-0.2, np.ravel(EU_dict['Bachelor']), 0.4, color=EU_MULTI_COLOURS[3],

       alpha=0.5,  bottom = np.ravel(EU_dict['Doctoral']) +  np.ravel(EU_dict['Master']))

ax.bar(survey_years+0.2, np.ravel(UK_dict['Bachelor']), 0.4, color=UK_MULTI_COLOURS[3], 

       alpha=0.5,  bottom = np.ravel(UK_dict['Doctoral']) + np.ravel(UK_dict['Master']))





# add labels and sort out tick labels

ax.legend(loc=(1.01,0.878))

ax.set_ylabel('Fraction of responses')

ax.set_xlabel('Year')

ax.text(0.09,0.1,'PhD', transform=ax.transAxes, color='white', rotation=0, fontsize=10)

ax.text(0.07,0.47,'Master', transform=ax.transAxes, color='white', rotation=0, fontsize=10)

ax.text(0.057,0.82,'Bachelor', transform=ax.transAxes, color='white', rotation=0, fontsize=10)

_ = ax.set_xticklabels(['','','2017','','2018','','2019'])

ax.set_ylim(0,1)

# turn off all the lines to remove clutter

turn_off_lines(ax)

fig.set_size_inches(7,5)
# only looking at students for now.

jobs_to_use = ['Student','Other']



# map all the other categories to 'other'

yearly_dfs[0].loc[~yearly_dfs[0]['employment_status'].isin(jobs_to_use), 'employment_status'] = 'Other'

yearly_dfs[1].loc[~yearly_dfs[1]['employment_status'].isin(jobs_to_use), 'employment_status'] = 'Other'



EU_students = []

UK_students = []

students = {'UK':UK_students, 'EU':EU_students}

for body in ['UK','EU']:

    student_2019  = yearly_dfs[0][yearly_dfs[0]['body']==body].groupby('employment_status')['body'].count()

    student_2019 = student_2019 / student_2019.sum()

    student_2018 = yearly_dfs[1][yearly_dfs[1]['body']==body].groupby('employment_status')['body'].count()

    student_2018 = student_2018 / student_2018.sum()

    students[body].append([student_2019.values[1], student_2018.values[1]])

    

fig, ax = plt.subplots(1, 1)

ax.bar(survey_years[:-1]-0.2,  np.ravel(students['EU'])*100, 0.4, color=EU_COLOUR, alpha=0.8, label='EU')

ax.bar(survey_years[:-1]+0.2, np.ravel(students['UK'])*100, 0.4, color=UK_COLOUR, alpha=0.8, label='UK')



ax.legend()

ax.set_ylabel('Fraction of responses from students (%)')

ax.set_xlabel('Year')

_ = ax.set_xticklabels(['','','2018','','','','2019'])

ax.set_ylim(0,25)

turn_off_lines(ax)

fig.set_size_inches(7,5)
fig, axes = plt.subplots(1,2, sharey=True)

offset = {'UK':0.2, 'EU':-0.2}

for i, ax in enumerate(axes):

    yearly_dfs[i]['coding_lower'] =  yearly_dfs[i]['coding_experience'].str.split('-', expand=True)[0].astype(float)

    yearly_dfs[i]['coding_higher'] =  yearly_dfs[i]['coding_experience'].str.split('-', expand=True)[1]

    for body in ['EU','UK']:

        coding_grouping = yearly_dfs[i][yearly_dfs[i]['body']==body].groupby(['coding_lower','coding_higher'])['body'].count()

        coding_grouping = coding_grouping / coding_grouping.sum()

        coding_grouping = coding_grouping.reset_index()

        coding_grouping['bins'] = coding_grouping['coding_lower'].astype(int).astype(str) + '-' + coding_grouping['coding_higher'].astype(str)

        labels = coding_grouping['bins'].values

        ax.bar(np.arange(0, len(labels))-offset[body], coding_grouping['body'], alpha=0.6, color=color_dict[body], label=body, width=0.4)

        ax.set_title(survey_years[i])

        ax.set_xlabel('Coding experience (years)')

        # turn off all the lines to remove clutter

        ax.set_xticklabels(['']+list(labels[:-1])+['20+'])

        turn_off_lines(ax)

        ax.grid(axis='y', alpha=0.4)

    

# tidy up the axes

axes[0].set_ylabel('Fraction of respondents')

axes[1].legend()



fig.set_size_inches(14,5.5)
fig, axes = plt.subplots(1,3, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0,

                                                         'width_ratios': [3, 3,1]})

groupings = []

for ax, body in zip(axes,['UK','EU']):

    salary_grouping = yearly_dfs[0][yearly_dfs[0]['body']==body].groupby(['salary_lower','salary_higher'])['body'].count()

    salary_grouping = salary_grouping / salary_grouping.sum()

    salary_grouping = salary_grouping.reset_index()

    salary_grouping['salary_lower'] = salary_grouping['salary_lower'] 

    salary_grouping['bins'] = salary_grouping['salary_lower'].astype(str) + '-' + salary_grouping['salary_higher'].astype(str)

    groupings.append((salary_grouping['bins'], salary_grouping['body']))

    ax.barh(salary_grouping['bins'], salary_grouping['body'], alpha=0.5, color=color_dict[body], label=body)

    ax.set_title(body, fontsize=14)

    plt.setp(ax.spines.values(), visible=False)

    ax.patch.set_visible(False)

    ax.grid(False)

    ax.set_xlabel('Fraction of respondents')



axes[0].invert_xaxis()

axes[0].set_ylabel('Compensation amount ($)')



#ax.set_title('Compensation amount in 2019')



# calulate and plot the residuals on the third axis

# calculate normalised difference

asymmetry = (groupings[1][1][:-1]-groupings[0][1])/(groupings[1][1][:-1]+groupings[0][1])

# make the colours of the bars

colors = asymmetry.copy()

mask = asymmetry>0

colors[mask] = color_dict['EU']

colors[~mask] = color_dict['UK']

# plot

axes[2].barh(groupings[0][0], asymmetry, alpha=0.3,color=colors)

axes[2].axvline(0, alpha=0.2, color='gray')

axes[2].grid(False)

plt.setp(axes[2].spines.values(), visible=False)

plt.setp(axes[2].get_xticklabels(), visible=False)

axes[2].patch.set_visible(False)

axes[2].set_title('Normalized differences', fontsize=14)

plt.suptitle('Reported income by body', fontsize=16)

fig.set_size_inches(15,12)


# group the compensation level in low, medium or high income.

salary_mapping = {'$0-999':'low', '1,000-1,999':'low', 

                  '10,000-14,999':'low', '100,000-124,999':'high',

                  '125,000-149,999':'high', '15,000-19,999':'low', 

                  '150,000-199,999':'high', '2,000-2,999':'low',

                  '20,000-24,999':'low', '200,000-249,999':'high', 

                  '25,000-29,999':'low', '250,000-299,999':'high',

                  '3,000-3,999':'low','30,000-39,999':'medium',

                  '300,000-500,000':'high', '4,000-4,999':'low',

                  '40,000-49,999':'medium', '5,000-7,499':'low', 

                  '50,000-59,999':'medium', '60,000-69,999':'medium',

                  '7,500-9,999':'low', '70,000-79,999':'medium', 

                  '80,000-89,999':'medium', '90,000-99,999':'medium',

                  '> $500,000':'high'}

# make the salary grouping

yearly_dfs[0]['income_group'] = yearly_dfs[0]['salary'].map(salary_mapping)

    

fig, axes = plt.subplots(ncols=3)

for i, (ax, income) in enumerate(zip(axes, ['low', 'medium','high'])):

    # group all the other genders to 'Other' for simplicity see disclaimer above

    yearly_dfs[0].loc[~yearly_dfs[0]['gender'].isin(['Male','Female']), 'gender'] = 'Other'

    

    # respondent masks

    UK_mask = yearly_dfs[0]['body']=='UK'

    EU_mask = yearly_dfs[0]['body']=='EU'

    income_mask = yearly_dfs[0]['income_group']==income

    

    # Count respondents of each gender in subgroup

    EU_gender_count = yearly_dfs[0][EU_mask & income_mask].groupby('gender')['body'].count()

    UK_gender_count = yearly_dfs[0][UK_mask & income_mask].groupby('gender')['body'].count()



    # plot

    ax.pie(UK_gender_count, radius=1, colors=UK_MULTI_COLOURS, pctdistance=0.85, **shared_pi_formats)

    ax.pie(EU_gender_count, radius=1-size, colors=EU_MULTI_COLOURS, pctdistance=0.78, **shared_pi_formats)

    ax.set(aspect="equal", title=f'{income.capitalize()} income')





axes[1].legend(loc=(2.1,0.8), labels=pi_labels)

fig.set_size_inches(16,6)
# extract the question on engagement with media sources

df_media = yearly_dfs[0].loc[:,(yearly_dfs[0].columns.str.contains('Q12')==True) | yearly_dfs[0].columns.str.contains('body')]



# extract name of websites for each response

col_names = df_media.mode().values[0]

engagement_types = [str(name).split('(')[0].strip() for name in col_names]



# make the column names the name of the media

df_media.columns = ['body'] + engagement_types[1:]



# drop the free field text column

df_media.drop('-1', axis=1, inplace=True)



# count the number of each media type

media_grouped = df_media.groupby('body').count()-1 # -1 for the question row



# make the plot

fig, ax = plt.subplots()

for body in ['EU','UK']:

    data = media_grouped.loc[body,:] / media_grouped.loc[body,:].sum()

    data.sort_values(inplace=True)

    ax.bar(data.index, data.values, alpha=0.35, color=color_dict[body], label=body)

    turn_off_lines(ax)

ax.legend()

plt.setp(ax.get_xticklabels(), rotation=90)    

ax.set_ylabel('Fraction of total engagements')

ax.set_title('2019 Media engagements')

fig.set_size_inches(8,5)
body_series = df_media['body'].copy()

df_media[~df_media.isna()] = 1

df_media[df_media.isna()] = 0



fig, axes = plt.subplots(1,2)



for ax, body, cmap in zip(axes, ['EU','UK'], ['Blues','Reds']):

    mask = body_series==body

    # create the cross occurances df

    cross_occ_df = pd.DataFrame(index=df_media.columns[1:], columns=df_media.columns[1:])

    cols = df_media.columns[1:]

    for col_1 in cols:

        for col_2 in cols:

            # calculate the co-occurances (i.e, when respondent said they used them both)

            cross_occ = pd.crosstab(df_media.loc[mask,col_1], df_media.loc[mask,col_2])

            try:

                cross_occ_df.loc[col_1,col_2] = cross_occ.loc[1,1]

            except:

                cross_occ_df.loc[col_1, col_2] = 0

    # plot and format the graph

    ax.imshow(cross_occ_df, cmap=cmap)

    ax.set_title(body)

    ax.set_xticks(np.arange(0,12))

    ax.set_yticks(np.arange(0,12))

    ax.set_xticklabels(cols, rotation=90)

    ax.set_yticklabels(cols)

    fig.set_size_inches(10,10)

    turn_off_lines(ax)

    if body=='UK':

        ax.set_yticklabels([])

        ax.spines['left'].set_visible(False)

        ax.set_yticks([])

        

plt.suptitle('Co-occurances of media use', fontsize=15)

fig.set_size_inches(10,5)
cols_for_analysis = ['job'] + list(yearly_dfs[0].columns[yearly_dfs[0].columns.str.contains('Q18')])

job_titles = ['Data Scientist', 'Software Engineer', 'Student', 'Data Analyst', 'Research Scientist']



fig, axes = plt.subplots(1,2, facecolor=(0.99, 0.99, 0.99), sharey=True)



for ax, body, cmap in zip(axes,['EU','UK'], ['Blues','Reds']):

    # extract the data

    df_IDE = yearly_dfs[0][cols_for_analysis]

    # make the column names those of the tick box

    ide_names = yearly_dfs[0][cols_for_analysis].mode().values[0]

    df_IDE.columns = ['job'] + list(ide_names[1:])

    # make a mask for the body

    body_mask = yearly_dfs[0].body==body

    

    # count number of ide used by each job type

    grouped_df = df_IDE.loc[body_mask,:].groupby('job').count().loc[job_titles, :'None']

    values = grouped_df.values / grouped_df.values.sum(axis=1)[:,np.newaxis]

    # make a grid to plot on

    X, Y = np.meshgrid(np.arange(1,12), np.arange(1,len(job_titles)+1))

    

    # get and clean up the tick labels

    x_tick_labels = [x.split('(')[0].split('/')[0].strip() for x in grouped_df.columns]

    y_tick_labels = list(grouped_df.index)



    ax.scatter(x=np.ravel(X), y=np.ravel(Y), s=np.ravel(values)*2500, c=np.ravel(values), cmap=cmap)



    # tidy up axis, make ticks and lines invsible.

    plt.setp(ax.spines.values(), visible=False)

    ax.patch.set_visible(False)

    ax.grid(False)

    ax.set_yticks(np.arange(1,len(job_titles)+1))

    ax.set_yticklabels(y_tick_labels)

    ax.set_xticks(np.arange(1,12))

    ax.set_xticklabels(x_tick_labels, rotation=90)

    ax.set_title(f'{body} respondents')

plt.suptitle('Programming by Profession', fontsize=14)

    

fig.set_size_inches(15,5)
cols_for_analysis = ['job'] + list(yearly_dfs[0].columns[yearly_dfs[0].columns.str.contains('Q16')])

job_titles = ['Data Scientist', 'Software Engineer', 'Student', 'Data Analyst', 'Research Scientist']



fig, axes = plt.subplots(1,2, facecolor=(0.99, 0.99, 0.99), sharey=True)



for ax, body, cmap in zip(axes,['EU','UK'], ['Blues','Reds']):

    # extract the data

    df_IDE = yearly_dfs[0][cols_for_analysis]

    # make the column names those of the tick box

    ide_names = yearly_dfs[0][cols_for_analysis].mode().values[0]

    df_IDE.columns = ['job'] + list(ide_names[1:])

    # make a mask for the body

    body_mask = yearly_dfs[0].body==body

    

    # count number of ide used by each job type

    grouped_df = df_IDE.loc[body_mask,:].groupby('job').count().loc[job_titles, :'None']

    values = grouped_df.values / grouped_df.values.sum(axis=1)[:,np.newaxis]

    # make a grid to plot on

    X, Y = np.meshgrid(np.arange(1,12), np.arange(1,len(job_titles)+1))

    

    # get and clean up the tick labels

    x_tick_labels = [x.split('(')[0].split('/')[0].strip() for x in grouped_df.columns]

    y_tick_labels = list(grouped_df.index)



    ax.scatter(x=np.ravel(X), y=np.ravel(Y), s=np.ravel(values)*2500, c=np.ravel(values), cmap=cmap)



    # tidy up axis, make ticks and lines invsible.

    plt.setp(ax.spines.values(), visible=False)

    ax.patch.set_visible(False)

    ax.grid(False)

    ax.set_yticks(np.arange(1,len(job_titles)+1))

    ax.set_yticklabels(y_tick_labels)

    ax.set_xticks(np.arange(1,12))

    ax.set_xticklabels(x_tick_labels, rotation=90)

    ax.set_title(f'{body} respondents')

    

plt.suptitle('IDE use by Profession', fontsize=14)    

fig.set_size_inches(15,5)