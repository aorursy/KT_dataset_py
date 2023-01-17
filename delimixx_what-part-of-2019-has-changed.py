from IPython.display import Image
Image("../input/kaggle-survey-image/Kaggle.png", width ='1000')
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import squarify

plt.style.use('fivethirtyeight')

# You Can Change 

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import base64

import io

from scipy.misc import imread

import codecs

from IPython.display import HTML

from matplotlib_venn import venn3

import re
survey_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

question_2019 = pd.read_csv('../input/kaggle-survey-2019/questions_only.csv')

columns_multiple_2019 = [col for col in list(survey_2019.columns) if re.search('Part_\d{1,2}$', col)]

multiple_columns_list_2019 = [ [col]+col.split('_') for col in columns_multiple_2019 ]

qa_multiple_2019 = pd.DataFrame(multiple_columns_list_2019).groupby([1])[0].apply(list)

question_numbers_list_2019 = sorted([int(i.split('Q')[1]) for i in list(qa_multiple_2019.index)])

question_list_2019 = [ 'Q{}'.format(i) for i in question_numbers_list_2019]

#questions_2019 = ''.join([f'<li>{i}</li>' for i in question_list_2019])

survey_2019['year'] = '2019'



survey_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')

question_2018 = pd.read_csv('../input/kaggle-survey-2018/SurveySchema.csv').iloc[0:1]

del question_2018['2018 Kaggle Machine Learning and Data Science Survey']

columns_multiple_2018 = [col for col in list(survey_2018.columns) if re.search('Part_\d{1,2}$', col)]

multiple_columns_list_2018 = [ [col]+col.split('_') for col in columns_multiple_2018 ]

qa_multiple_2018 = pd.DataFrame(multiple_columns_list_2018).groupby([1])[0].apply(list)

question_numbers_list_2018 = sorted([int(i.split('Q')[1]) for i in list(qa_multiple_2018.index)])

question_list_2018 = [ 'Q{}'.format(i) for i in question_numbers_list_2018]

#questions_2018 = ''.join([f'<li>{i}</li>' for i in question_list_2018])

survey_2018['year'] = '2018'



survey_2017 = pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')

survey_2017['year'] = '2017'
plt.subplots(figsize = (20, 20))

gender_2019 = survey_2019[['year','Q2']].rename(columns = {'Q2' : 'GenderSelect'}).iloc[1:,]

gender_2018 = survey_2018[['year','Q1']].rename(columns = {'Q1' : 'GenderSelect'}).iloc[1:,]

gender_2017 = survey_2017[['year','GenderSelect']]

gender_data = pd.concat([gender_2019,gender_2018,gender_2017])

gender_data_prop = gender_data['GenderSelect'].groupby(gender_data['year']).value_counts(normalize = True).rename ('Prop').reset_index()



sns.barplot(gender_data_prop['Prop'], gender_data_prop['GenderSelect'],palette='inferno_r', hue =gender_data_prop['year'])

plt.legend()

plt.show()
# tranform percentage

country_2019 = survey_2019[['year','Q3']].rename(columns = {'Q3' : 'Country'}).iloc[1:,]

country_2018 = survey_2018[['year','Q3']].rename(columns = {'Q3' : 'Country'}).iloc[1:,]

country_2017 = survey_2017[['year','Country']]

country_2019_prop = country_2019['Country'].groupby(country_2019['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending=False)[:10].reset_index()

country_2018_prop = country_2018['Country'].groupby(country_2018['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending=False)[:10].reset_index()

country_2017_prop = country_2017['Country'].groupby(country_2017['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending=False)[:10].reset_index()



# plot

f,ax=plt.subplots(1,3,figsize=(25,15))

sns.barplot('Prop','Country', data=country_2019_prop , palette='inferno',ax=ax[0])

ax[0].set_xlabel('')

ax[0].set_title('Top 10 Countries by number of Response 2019')

sns.barplot('Prop','Country', data=country_2018_prop , palette='inferno',ax=ax[1])

ax[1].set_xlabel('')

ax[1].set_ylabel('')

ax[1].set_title('Top 10 Countries by number of Response 2018')

sns.barplot('Prop','Country', data=country_2017_prop , palette='inferno',ax=ax[2])

ax[2].set_xlabel('')

ax[2].set_ylabel('')

ax[2].set_title('Top 10 Countries by number of Response 2017')

plt.subplots_adjust(wspace=1.0)

plt.show()
age = {}

for i in survey_2019['Q1'].iloc[1:,].unique()[:-1]:

    min = int(i.split('-')[0])

    max = int(i.split('-')[1])

    age.update({i : list(range(min,max+1))})

    

def chage_categori_age(x):

    for i in age.items():

        if x in i[1]:

            return i[0]

        

survey_2018['Q2'] = survey_2018['Q2'].iloc[1:,].apply(lambda x: '70+' if (x == '80+') | (x == '70-79') else x)

survey_2017['Age'] = survey_2017['Age'].apply(chage_categori_age)
# tranform percentage

age_2019_prop = survey_2019['Q1'].groupby(country_2019['year']).value_counts(normalize = True).rename ('Prop').reset_index().rename(columns = {'Q1' : 'Age'})

age_2018_prop = survey_2018['Q2'].groupby(country_2018['year']).value_counts(normalize = True).rename ('Prop').reset_index().rename(columns = {'Q2' : 'Age'})

age_2017_prop = survey_2017['Age'].groupby(country_2017['year']).value_counts(normalize = True).rename ('Prop').reset_index()



# plot

f,ax=plt.subplots(1,3,figsize=(25,15))

sns.barplot('Prop','Age', data=age_2019_prop , palette='summer',ax=ax[0])

ax[0].set_xlabel('')

ax[0].set_title('Age 2019')

ax[0].axvline(0.25, linestyle='dashed')

ax[0].axvline(0.10, linestyle='dashed', color = 'r')

ax[0].axhspan(2.5,3.5 ,facecolor='Blue', alpha=0.2) # hilight space



sns.barplot('Prop','Age', data=age_2018_prop , palette='summer',ax=ax[1])

ax[1].set_xlabel('')

ax[1].set_ylabel('')

ax[1].set_title('Age 2018')

ax[1].axvline(0.25, linestyle='dashed')

ax[1].axvline(0.10, linestyle='dashed', color = 'r')

ax[1].axhspan(2.5,3.5 ,facecolor='Blue', alpha=0.2) # hilight space



sns.barplot('Prop','Age', data=age_2017_prop , palette='summer',ax=ax[2])

ax[2].set_xlabel('')

ax[2].set_ylabel('')

ax[2].set_title('Age 2017')

ax[2].axvline(0.25, linestyle='dashed')

ax[2].axvline(0.10, linestyle='dashed', color = 'r')

ax[2].axhspan(3.5,4.5 ,facecolor='Blue', alpha=0.2) # hilight space



plt.subplots_adjust(wspace=0.6)

plt.show()
currentjob_2019_prop = survey_2019['Q5'].groupby(survey_2019['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending = False).reset_index().rename(columns = {'Q5' : 'CurrentJobTitleSelect'})[:10]

currentjob_2018_prop = survey_2018['Q6'].groupby(survey_2018['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending = False).reset_index().rename(columns = {'Q6' : 'CurrentJobTitleSelect'})[:10]

currentjob_2017_prop = survey_2017['CurrentJobTitleSelect'].groupby(survey_2017['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending = False).reset_index()[:10]



# plot

f,ax=plt.subplots(1,3,figsize=(25,15))



sns.barplot('Prop','CurrentJobTitleSelect', data=currentjob_2019_prop , palette='BrBG',ax=ax[0])

ax[0].set_xlabel('')

ax[0].set_title('Top10 CurrentJobTitle 2019')

ax[0].axvline(0.20, linestyle='dashed')

ax[0].axvline(0.15, linestyle='dashed', color = 'r')

ax[0].axhspan(0.5,1.5 ,facecolor='Red', alpha=0.5) # hilight space



sns.barplot('Prop','CurrentJobTitleSelect', data=currentjob_2018_prop , palette='BrBG',ax=ax[1])

ax[1].set_xlabel('')

ax[1].set_ylabel('')

ax[1].set_title('Top10 CurrentJobTitle 2018')

ax[1].axvline(0.20, linestyle='dashed')

ax[1].axvline(0.15, linestyle='dashed', color = 'r')

ax[1].axhspan(-0.5,0.5 ,facecolor='Red', alpha=0.5) # hilight space



sns.barplot('Prop','CurrentJobTitleSelect', data=currentjob_2017_prop , palette='BrBG',ax=ax[2])

ax[2].set_xlabel('')

ax[2].set_ylabel('')

ax[2].set_title('Top10 CurrentJobTitle 2017')

ax[2].axvline(0.20, linestyle='dashed')

ax[2].axvline(0.15, linestyle='dashed', color = 'r')



plt.subplots_adjust(wspace=0.6)

plt.show()
survey_2019_e = survey_2019[survey_2019['Q5'] == 'Student'].iloc[1:,]

survey_2018_e = survey_2018[survey_2018['Q6'] == 'Student'].iloc[1:,]

education_2019_prop = survey_2019_e['Q4'].groupby(survey_2019_e['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending = False).reset_index().rename(columns = {'Q4' : 'Formal Education'})[:10]

education_2018_prop = survey_2018_e['Q4'].groupby(survey_2018_e['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending = False).reset_index().rename(columns = {'Q4' : 'Formal Education'})[:10]



# plot

f,ax=plt.subplots(1,2,figsize=(25,15))



sns.barplot('Prop','Formal Education', data=education_2019_prop , palette='RdYlGn',ax=ax[0])

ax[0].set_xlabel(' ')

ax[0].set_ylabel(' ')

ax[0].set_title('Formal Education 2019')

ax[0].axvline(0.40, linestyle='dashed')

ax[0].axhspan(-0.5,0.5 ,facecolor='Gray', alpha=0.5) # hilight space



sns.barplot('Prop','Formal Education', data=education_2018_prop , palette='RdYlGn',ax=ax[1])

ax[1].set_xlabel(' ')

ax[1].set_ylabel(' ')

ax[1].set_title('Formal Education 2018')

ax[1].axvline(0.40, linestyle='dashed')

ax[1].axhspan(0.5,1.5 ,facecolor='Gray', alpha=0.5) # hilight space



plt.subplots_adjust(wspace=1.0)

plt.show()
question = 'Q13' # On which platforms have you begun or completed data science courses?

columns_list_2019 = qa_multiple_2019[question]

survey_2019 ['LearningPlatformSelect'] = survey_2019[columns_list_2019].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)



question = 'Q36' # On which platforms have you begun or completed data science courses?

columns_list_2018 = qa_multiple_2018[question]

survey_2018['LearningPlatformSelect'] = survey_2018[columns_list_2018].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)
learn_2019 = survey_2019['LearningPlatformSelect'].iloc[1:,].str.split(',')

learn_2018 = survey_2018['LearningPlatformSelect'].iloc[1:,].str.split(',')

learn_2017 = survey_2017['LearningPlatformSelect'].iloc[1:,].str.split(',')



platform_2019 = []

platform_2018 = []

platform_2017 = []



for i in learn_2019.dropna():

    platform_2019.extend(i)

    

for i in learn_2018.dropna():

    platform_2018.extend(i)

    

for i in learn_2017.dropna():

    platform_2017.extend(i)



    

f, ax = plt.subplots(1,3, figsize = (18,8))



pd.Series(platform_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('winter',15),ax = ax[0])

ax[0].set_title('Top 10 Platforms to Learn 2019', size = 15)

pd.Series(platform_2018).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('winter',15),ax = ax[1])

ax[1].set_title('Top 10 Platforms to Learn 2018', size = 15)

pd.Series(platform_2017).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('winter',15),ax = ax[2])

ax[2].set_title('Top 10 Platforms to Learn 2017', size = 15)

plt.show()
question = 'Q24' # Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice

columns_list_2019 = qa_multiple_2019[question]

survey_2019 ['MLTechniquesSelect'] = survey_2019[columns_list_2019].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)
mlTech_2019 = survey_2019['MLTechniquesSelect'].iloc[1:,].str.split('?')

mlTech_2017 = survey_2017['MLTechniquesSelect'].iloc[1:,].str.split(',')

mlTech_ds_2019 = survey_2019[survey_2019['Q5'] =='Data Scientist']['MLTechniquesSelect'].iloc[1:,].str.split('?')

mlTech_ds_2017 = survey_2017[survey_2017['CurrentJobTitleSelect'] =='Data Scientist']['MLTechniquesSelect'].iloc[1:,].str.split(',')





ml_2019 = []

ml_2017 = []

ml_ds_2019 = []

ml_ds_2017 = []



for i in mlTech_2019.dropna():

    ml_2019.extend(i)

    

for i in mlTech_2017.dropna():

    ml_2017.extend(i)

    

for i in mlTech_ds_2019.dropna():

    ml_ds_2019.extend(i)

    

for i in mlTech_ds_2017.dropna():

    ml_ds_2017.extend(i)

    

    

f, ax = plt.subplots(2,2, figsize = (25,15))

pd.Series(ml_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('spring',15),ax = ax[0][0])

ax[0][0].set_title('Top 10 MLTech 2019', size = 15)

ax[0][0].axhspan(6.5,7.5 ,facecolor='Gray', alpha=0.5) # hilight space

ax[0][0].axhspan(5.5,6.5 ,facecolor='Gray', alpha=0.5) # hilight space

ax[0][0].axvline(0.10, linestyle='dashed', color= 'r')

pd.Series(ml_2017).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('spring',15),ax = ax[0][1])

ax[0][1].set_title('Top 10 MLTech 2017', size = 15)

ax[0][1].axhspan(1.5,2.5 ,facecolor='Gray', alpha=0.5) # hilight space

ax[0][1].axhspan(3.5,4.5 ,facecolor='Gray', alpha=0.5) # hilight space

ax[0][1].axvline(0.10, linestyle='dashed', color= 'r')

pd.Series(ml_ds_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('winter',15),ax = ax[1][0])

ax[1][0].set_title('Top 10 Data Scientist MLTech 2019', size = 15)

ax[1][0].axhspan(6.5,7.5 ,facecolor='Gray', alpha=0.5) # hilight space

ax[1][0].axhspan(5.5,6.5 ,facecolor='Gray', alpha=0.5) # hilight space

ax[1][0].axvline(0.10, linestyle='dashed', color= 'r')

pd.Series(ml_ds_2017).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('winter',15),ax = ax[1][1])

ax[1][1].set_title('Top 10 Data Scientist MLTech 2017', size = 15)

ax[1][1].axhspan(1.5,2.5 ,facecolor='Gray', alpha=0.5) # hilight space

ax[1][1].axhspan(3.5,4.5 ,facecolor='Gray', alpha=0.5) # hilight space

ax[1][1].axvline(0.10, linestyle='dashed', color= 'r')

plt.subplots_adjust(wspace=.6)

plt.show()
question = 'Q28' # Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice

columns_list_2019 = qa_multiple_2019[question]

survey_2019 ['MLFramework'] = survey_2019[columns_list_2019].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)



question = 'Q19' # What machine learning frameworks have you used in the past 5 years?

columns_list_2018 = qa_multiple_2018[question]

survey_2018['MLFramework'] = survey_2018[columns_list_2018].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)
mlFrame_2019 = survey_2019['MLFramework'].iloc[1:,].str.split(',')

mlFrame_2018 = survey_2018['MLFramework'].iloc[1:,].str.split(',')



mlfr_2019 = []

mlfr_2018 = []



for i in mlFrame_2019.dropna():

    mlfr_2019.extend(i)

    

for i in mlFrame_2018.dropna():

    mlfr_2018.extend(i)

    

f, ax = plt.subplots(1,2, figsize = (18,8))



pd.Series(mlfr_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('inferno_r',15),ax = ax[0])

ax[0].set_title('Top 10 ML Framework 2019', size = 15)

pd.Series(mlfr_2018).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('inferno_r',15),ax = ax[1])

ax[1].set_title('Top 10 ML Framework 2018', size = 15)



plt.show()
question = 'Q29' # Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice

columns_list_2019 = qa_multiple_2019[question]

survey_2019 ['Cloud'] = survey_2019[columns_list_2019].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)



question = 'Q15' # What machine learning frameworks have you used in the past 5 years?

columns_list_2018 = qa_multiple_2018[question]

survey_2018['Cloud'] = survey_2018[columns_list_2018].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)
cloud_2019 = survey_2019['Cloud'].iloc[1:,].str.split('?')

cloud_2018 = survey_2018['Cloud'].iloc[1:,].str.split('?')



cl_2019 = []

cl_2018 = []



for i in cloud_2019.dropna():

    cl_2019.extend(i)

    

for i in cloud_2018.dropna():

    cl_2018.extend(i)

    

f, ax = plt.subplots(1,2, figsize = (18,15))



pd.Series(cl_2019).value_counts(normalize=True)[:5].sort_values(ascending=True).plot.barh(width = 0.9,ax = ax[0])

ax[0].set_title('Top 5 Cloud 2019', size = 15)

ax[0].axvline(0.2, linestyle='dashed', color= 'r')

pd.Series(cl_2018).value_counts(normalize=True)[:5].sort_values(ascending=True).plot.barh(width = 0.9, ax = ax[1])

ax[1].set_title('Top 5 Cloud  2018', size = 15)

ax[1].axvline(0.2, linestyle='dashed', color= 'r')

plt.subplots_adjust(wspace=.6)

plt.show()
question = 'Q33'# Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?

columns_list_2019 = qa_multiple_2019[question]

survey_2019 ['AutoML'] = survey_2019[columns_list_2019].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)
auto_2019 = survey_2019['AutoML'].iloc[1:,].str.split('?')



aml_2019 = []



for i in auto_2019.dropna():

    aml_2019.extend(i)

    

pd.Series(aml_2019).value_counts(normalize=True)[:5].sort_values(ascending=True).plot.barh(width = 0.9)

plt.title('Top 5 Cloud 2019', size = 15)

plt.show()
question = 'Q18' # What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice 

columns_list_2019 = qa_multiple_2019[question]

survey_2019 ['WorkToolsSelect'] = survey_2019[columns_list_2019].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)



question = 'Q16' # What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice 

columns_list_2018 = qa_multiple_2018[question]

survey_2018['WorkToolsSelect'] = survey_2018[columns_list_2018].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)
programm_2019 = survey_2019.dropna(subset = ['WorkToolsSelect']).iloc[1:,]

programm_2018 = survey_2018.dropna(subset = ['WorkToolsSelect']).iloc[1:,]

programm_2017 = survey_2017.dropna(subset = ['WorkToolsSelect'])
python_2019 = programm_2019[(programm_2019['WorkToolsSelect'].str.contains('Python')) & (~programm_2019['WorkToolsSelect'].str.contains('R')) & (~programm_2019['WorkToolsSelect'].str.contains('SQL'))]

R_2019 = programm_2019[(~programm_2019['WorkToolsSelect'].str.contains('Python')) & (programm_2019['WorkToolsSelect'].str.contains('R')) & (~programm_2019['WorkToolsSelect'].str.contains('SQL'))]

SQL_2019 = programm_2019[(~programm_2019['WorkToolsSelect'].str.contains('Python')) & (~programm_2019['WorkToolsSelect'].str.contains('R'))& (programm_2019['WorkToolsSelect'].str.contains('SQL'))]

python_R_2019 = programm_2019[(programm_2019['WorkToolsSelect'].str.contains('Python')) & (programm_2019['WorkToolsSelect'].str.contains('R'))& (~programm_2019['WorkToolsSelect'].str.contains('SQL'))]

python_SQL_2019 = programm_2019[(programm_2019['WorkToolsSelect'].str.contains('Python')) & (~programm_2019['WorkToolsSelect'].str.contains('R'))& (programm_2019['WorkToolsSelect'].str.contains('SQL'))]

R_SQL_2019 = programm_2019[(~programm_2019['WorkToolsSelect'].str.contains('Python')) & (programm_2019['WorkToolsSelect'].str.contains('R'))& (programm_2019['WorkToolsSelect'].str.contains('SQL'))]

ALL_2019 = programm_2019[(programm_2019['WorkToolsSelect'].str.contains('Python')) & (programm_2019['WorkToolsSelect'].str.contains('R'))& (programm_2019['WorkToolsSelect'].str.contains('SQL'))]

OTHER_2019 = programm_2019[(~programm_2019['WorkToolsSelect'].str.contains('Python')) & (~programm_2019['WorkToolsSelect'].str.contains('R'))& (~programm_2019['WorkToolsSelect'].str.contains('SQL'))]



python_2018 = programm_2018[(programm_2018['WorkToolsSelect'].str.contains('Python')) & (~programm_2018['WorkToolsSelect'].str.contains('R')) & (~programm_2018['WorkToolsSelect'].str.contains('SQL'))]

R_2018 = programm_2018[(~programm_2018['WorkToolsSelect'].str.contains('Python')) & (programm_2018['WorkToolsSelect'].str.contains('R')) & (~programm_2018['WorkToolsSelect'].str.contains('SQL'))]

SQL_2018 = programm_2018[(~programm_2018['WorkToolsSelect'].str.contains('Python')) & (~programm_2018['WorkToolsSelect'].str.contains('R'))& (programm_2018['WorkToolsSelect'].str.contains('SQL'))]

python_R_2018 = programm_2018[(programm_2018['WorkToolsSelect'].str.contains('Python')) & (programm_2018['WorkToolsSelect'].str.contains('R'))& (~programm_2018['WorkToolsSelect'].str.contains('SQL'))]

python_SQL_2018 = programm_2018[(programm_2018['WorkToolsSelect'].str.contains('Python')) & (~programm_2018['WorkToolsSelect'].str.contains('R'))& (programm_2018['WorkToolsSelect'].str.contains('SQL'))]

R_SQL_2018 = programm_2018[(~programm_2018['WorkToolsSelect'].str.contains('Python')) & (programm_2018['WorkToolsSelect'].str.contains('R'))& (programm_2018['WorkToolsSelect'].str.contains('SQL'))]

ALL_2018 = programm_2018[(programm_2018['WorkToolsSelect'].str.contains('Python')) & (programm_2018['WorkToolsSelect'].str.contains('R'))& (programm_2018['WorkToolsSelect'].str.contains('SQL'))]

OTHER_2018 = programm_2018[(~programm_2018['WorkToolsSelect'].str.contains('Python')) & (~programm_2018['WorkToolsSelect'].str.contains('R'))& (~programm_2018['WorkToolsSelect'].str.contains('SQL'))]



python_2017 = programm_2017[(programm_2017['WorkToolsSelect'].str.contains('Python')) & (~programm_2017['WorkToolsSelect'].str.contains('R')) & (~programm_2017['WorkToolsSelect'].str.contains('SQL'))]

R_2017 = programm_2017[(~programm_2017['WorkToolsSelect'].str.contains('Python')) & (programm_2017['WorkToolsSelect'].str.contains('R')) & (~programm_2017['WorkToolsSelect'].str.contains('SQL'))]

SQL_2017 = programm_2017[(~programm_2017['WorkToolsSelect'].str.contains('Python')) & (~programm_2017['WorkToolsSelect'].str.contains('R'))& (programm_2017['WorkToolsSelect'].str.contains('SQL'))]

python_R_2017 = programm_2017[(programm_2017['WorkToolsSelect'].str.contains('Python')) & (programm_2017['WorkToolsSelect'].str.contains('R'))& (~programm_2017['WorkToolsSelect'].str.contains('SQL'))]

python_SQL_2017 = programm_2017[(programm_2017['WorkToolsSelect'].str.contains('Python')) & (~programm_2017['WorkToolsSelect'].str.contains('R'))& (programm_2017['WorkToolsSelect'].str.contains('SQL'))]

R_SQL_2017 = programm_2017[(~programm_2017['WorkToolsSelect'].str.contains('Python')) & (programm_2017['WorkToolsSelect'].str.contains('R'))& (programm_2017['WorkToolsSelect'].str.contains('SQL'))]

ALL_2017 = programm_2017[(programm_2017['WorkToolsSelect'].str.contains('Python')) & (programm_2017['WorkToolsSelect'].str.contains('R'))& (programm_2017['WorkToolsSelect'].str.contains('SQL'))]

OTHER_2017 = programm_2017[(~programm_2017['WorkToolsSelect'].str.contains('Python')) & (~programm_2017['WorkToolsSelect'].str.contains('R'))& (~programm_2017['WorkToolsSelect'].str.contains('SQL'))]
f, ax = plt.subplots(1,3, figsize = (18,8))



venn3(subsets = (round(python_2019.shape[0]/len(programm_2019),2) , 

                 round(R_2019.shape[0]/len(programm_2019),2), 

                 round(SQL_2019.shape[0]/len(programm_2019),2) ,

                 round(python_R_2019.shape[0]/len(programm_2019),2) ,

                 round(python_SQL_2019.shape[0]/len(programm_2019),2) ,

                 round(R_SQL_2019.shape[0]/len(programm_2019) ,2) ,

                 round(ALL_2019.shape[0]/len(programm_2019) ,2)),

       set_labels = ('Python','R','SQL','Python + R','Python + SQL' , 'R + SQL', 'ALL' ) , ax = ax[0] )

ax[0].set_title('Percent of Users 2019')



venn3(subsets = (round(python_2018.shape[0]/len(programm_2018),2) , 

                 round(R_2018.shape[0]/len(programm_2018),2), 

                 round(SQL_2018.shape[0]/len(programm_2018),2) ,

                 round(python_R_2018.shape[0]/len(programm_2018),2) ,

                 round(python_SQL_2018.shape[0]/len(programm_2018),2) ,

                 round(R_SQL_2018.shape[0]/len(programm_2018) ,2) ,

                 round(ALL_2018.shape[0]/len(programm_2018) ,2)),

       set_labels = ('Python','R','SQL','Python + R','Python + SQL' , 'R + SQL', 'ALL') , ax = ax[1])

ax[1].set_title('Percent of Users 2018')



venn3(subsets = (round(python_2017.shape[0]/len(programm_2017),2) , 

                 round(R_2017.shape[0]/len(programm_2017),2), 

                 round(SQL_2017.shape[0]/len(programm_2017),2) ,

                 round(python_R_2017.shape[0]/len(programm_2017),2) ,

                 round(python_SQL_2017.shape[0]/len(programm_2017),2) ,

                 round(R_SQL_2017.shape[0]/len(programm_2017) ,2) ,

                 round(ALL_2017.shape[0]/len(programm_2017) ,2)),

       set_labels = ('Python','R','SQL','Python + R','Python + SQL' , 'R + SQL', 'ALL') , ax = ax[2])

ax[2].set_title('Percent of Users 2017')

plt.show()



print('2019 OTHER Percentage : ' + str(round(OTHER_2019.shape[0]/len(programm_2019) ,2)))

print('2018 OTHER Percentage : ' + str(round(OTHER_2018.shape[0]/len(programm_2018) ,2)))

print('2017 OTHER Percentage : ' + str(round(OTHER_2017.shape[0]/len(programm_2017) ,2)))
f, ax = plt.subplots(1,2, figsize = (18,8))



# 2019-Q19 or 2018-Q18 : What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice 

survey_2019.iloc[1:,].dropna(subset =['Q19'])['Q19'].value_counts(normalize = True, ascending = True).plot.barh(width = 0.9, color =sns.color_palette('inferno_r',15) ,ax = ax[0])

ax[0].set_title('Recommend Programming Tool 2019', size = 15)

survey_2018.iloc[1:,].dropna(subset =['Q18'])['Q18'].value_counts(normalize = True, ascending = True).plot.barh(width = 0.9, color =sns.color_palette('inferno_r',15) ,ax = ax[1])

ax[1].set_title('Recommend Programming Tool 2018', size = 15)

plt.show()
#2019

python_2019_1 = python_2019.copy()

r_2019_1 = R_2019.copy()

sql_2019_1 = SQL_2019.copy()



python_2019_1['WorkToolsSelect_1'] = 'Python'

r_2019_1['WorkToolsSelect_1']='R'

sql_2019_1['WorkToolsSelect_1']='SQL'



python_r_sql_2019 = pd.concat([python_2019_1,r_2019_1,sql_2019_1]).rename(columns = {'Q5' : 'CurrentJobTitleSelect'})

python_r_sql_2019 = python_r_sql_2019['WorkToolsSelect_1'].groupby(python_r_sql_2019['CurrentJobTitleSelect']).value_counts(normalize = True).rename ('Prop').reset_index()





#2018

python_2018_1 = python_2018.copy()

r_2018_1 = R_2018.copy()

sql_2018_1 = SQL_2018.copy()



python_2018_1['WorkToolsSelect_1'] = 'Python'

r_2018_1['WorkToolsSelect_1']='R'

sql_2018_1['WorkToolsSelect_1']='SQL'



python_r_sql_2018 = pd.concat([python_2018_1,r_2018_1,sql_2018_1]).rename(columns = {'Q6' : 'CurrentJobTitleSelect'})

python_r_sql_2018 = python_r_sql_2018['WorkToolsSelect_1'].groupby(python_r_sql_2018['CurrentJobTitleSelect']).value_counts(normalize = True).rename ('Prop').reset_index()



#2017

python_2017_1 = python_2017.copy()

r_2017_1 = R_2017.copy()

sql_2017_1 = SQL_2017.copy()



python_2017_1['WorkToolsSelect_1'] = 'Python'

r_2017_1['WorkToolsSelect_1']='R'

sql_2017_1['WorkToolsSelect_1']='SQL'



python_r_sql_2017 = pd.concat([python_2017_1,r_2017_1,sql_2017_1])

python_r_sql_2017 = python_r_sql_2017['WorkToolsSelect_1'].groupby(python_r_sql_2017['CurrentJobTitleSelect']).value_counts(normalize = True).rename ('Prop').reset_index()





#plot

f, ax = plt.subplots(1,3, figsize = (25,15))

python_r_sql_2019.pivot('CurrentJobTitleSelect','WorkToolsSelect_1','Prop').plot.barh(width=0.8, ax = ax[0])

ax[0].set_title('Percent Programmin Per Current Job 2019')

ax[0].axhspan(1.5,2.5 ,facecolor='Orange', alpha=0.25) # hilight space

ax[0].axhspan(9.5,10.5 ,facecolor='Orange', alpha=0.25) # hilight space



python_r_sql_2018.pivot('CurrentJobTitleSelect','WorkToolsSelect_1','Prop').plot.barh(width=0.8, ax = ax[1])

ax[1].set_title('Percent Programmin Per Current Job 2018')

ax[1].set_ylabel('')

ax[1].axhspan(3.5,4.5 ,facecolor='Orange', alpha=0.25) # hilight space

ax[1].axhspan(18.5,19.5 ,facecolor='Orange', alpha=0.25) # hilight space



python_r_sql_2017.pivot('CurrentJobTitleSelect','WorkToolsSelect_1','Prop').plot.barh(width=0.8, ax = ax[2])

ax[2].set_title('Percent Programmin Per Current Job 2017')

ax[2].set_ylabel('')

ax[2].axhspan(2.5,3.5 ,facecolor='Orange', alpha=0.25) # hilight space

ax[2].axhspan(14.5,15.5 ,facecolor='Orange', alpha=0.25) # hilight space



plt.subplots_adjust(wspace=1.0)

plt.show()
ds_data = survey_2019[survey_2019['Q5'] == 'Data Scientist'].iloc[1:,]
question = 'Q9' # Select any activities that make up an important part of your role at work: (Select all that apply) 

columns_list_2019 = qa_multiple_2019[question]

ds_data ['Activites'] = ds_data[columns_list_2019].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)

activities_2019 = ds_data['Activites'].str.split('?')



at_2019 = []

for i in activities_2019.dropna():

    at_2019.extend(i)



plt.figure(figsize = (15,10))

pd.Series(at_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('inferno_r',15))

plt.title('Data Scientist Importance Role', size = 15)



plt.show()
question = 'Q12' # Select any activities that make up an important part of your role at work: (Select all that apply) 

columns_list_2019 = qa_multiple_2019[question]

ds_data ['Media_Source'] = ds_data[columns_list_2019].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)

media_2019 = ds_data['Media_Source'].str.split('?')



md_2019 = []

for i in media_2019.dropna():

    md_2019.extend(i)



plt.figure(figsize = (15,10))

pd.Series(md_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('inferno',15))

plt.title('DataScientist Media Source', size = 15)



plt.show()
plt.figure(figsize=(10,8))

code_2019 = ds_data['Q15'].value_counts(normalize = True).rename ('Prop').plot.pie(autopct='%1.1f%%',explode=[0.1,0,0,0,0,0,0], shadow=True,)

plt.title('DataScientist Code Time', size = 15)

plt.show()
plt.figure(figsize=(10,8))

code_2019 = ds_data['Q23'].value_counts(normalize = True).rename ('Prop').plot.pie(autopct='%1.1f%%',explode=[0.2,0.1,0,0,0,0,0,0], shadow=True,)

plt.title('DataScientist Code Time', size = 15)

plt.show()
money_2019 = ds_data['Q11'].groupby(ds_data['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending=False).reset_index()

# plot

plt.figure(figsize=(25,15))

sns.barplot('Prop','Q11', data=money_2019 , palette=sns.color_palette('viridis',15))

plt.title('DataScientist Spent Money', size = 15)

plt.show()