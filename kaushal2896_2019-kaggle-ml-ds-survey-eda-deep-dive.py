# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # For plotting

import seaborn as sns # For plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sns.set(rc={'figure.figsize':(12,8)})

sns.set(style="darkgrid")
multiple_choice_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

questions_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

survey_schema_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')

text_res_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')
print (f'Shape of multiple choice responses: {multiple_choice_df.shape}')

print (f'Shape of questions only: {questions_df.shape}')

print (f'Shape of survey schema: {survey_schema_df.shape}')

print (f'Shape of text responses: {text_res_df.shape}')
multiple_choice_df.head()
questions_df.head()
survey_schema_df.head()
text_res_df.head()
from IPython.display import Markdown, display

for index, value in questions_df.iloc[0].items():

    display(Markdown(f'**{index}:** {value}'))
sns.set(rc={'figure.figsize':(20,8)})

sns.set(style="darkgrid")

ax = sns.barplot(survey_schema_df.columns.values[1:][:-1], survey_schema_df.iloc[1][1:].astype('int64')[:-1])

ax.set(xlabel='Questions', ylabel='# of responses', title='Questions vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.show()
total = int(survey_schema_df.iloc[1]['Q1'])

ax = sns.barplot(multiple_choice_df.groupby(['Q1']).size().reset_index(name='counts')['Q1'][:-1], multiple_choice_df.groupby(['Q1']).size().reset_index(name='counts')['counts'][:-1])

ax.set(xlabel='Age groups', ylabel='# of responses', title='Age groups vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")



for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 



plt.show()
sns.set(rc={'figure.figsize':(9,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q2'])

ax = sns.barplot(multiple_choice_df.groupby(['Q2']).size().reset_index(name='counts')['Q2'][:-1], multiple_choice_df.groupby(['Q2']).size().reset_index(name='counts')['counts'][:-1])

ax.set(xlabel='Gender', ylabel='# of responses', title='Gender group vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
sns.set(rc={'figure.figsize':(25,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q3'])

ax = sns.barplot(multiple_choice_df.groupby(['Q3']).size().reset_index(name='counts')['Q3'], multiple_choice_df.groupby(['Q3']).size().reset_index(name='counts')['counts'], color='b')

ax.set(xlabel='Country', ylabel='# of responses', title='Country vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")



for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    if height/total*100 > 2: # Only plot the % if it's more than 2% in this case

        ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
sns.set(rc={'figure.figsize':(12,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q4'])

ax = sns.barplot(multiple_choice_df.groupby(['Q4']).size().reset_index(name='counts')['Q4'][:-1], multiple_choice_df.groupby(['Q4']).size().reset_index(name='counts')['counts'][:-1])

ax.set(xlabel='Educational Qualification', ylabel='# of responses', title='Educational Qualification vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
sns.set(rc={'figure.figsize':(12,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q5'])

ax = sns.barplot(multiple_choice_df.groupby(['Q5']).size().reset_index(name='counts')['Q5'].drop(9, axis=0), multiple_choice_df.groupby(['Q5']).size().reset_index(name='counts')['counts'].drop(9, axis=0))

ax.set(xlabel='Job Title', ylabel='# of responses', title='Job Title vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
sns.set(rc={'figure.figsize':(12,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q6'])

ax = sns.barplot(multiple_choice_df.groupby(['Q6']).size().reset_index(name='counts')['Q6'][:-1], multiple_choice_df.groupby(['Q6']).size().reset_index(name='counts')['counts'][:-1])

ax.set(xlabel='Is ML used?', ylabel='# of responses', title='ML usage vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
sns.set(rc={'figure.figsize':(12,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q7'])

ax = sns.barplot(multiple_choice_df.groupby(['Q7']).size().reset_index(name='counts')['Q7'][:-1], multiple_choice_df.groupby(['Q7']).size().reset_index(name='counts')['counts'][:-1])

ax.set(xlabel='Is ML used?', ylabel='# of responses', title='ML usage vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
sns.set(rc={'figure.figsize':(12,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q8'])

ax = sns.barplot(multiple_choice_df.groupby(['Q8']).size().reset_index(name='counts')['Q8'][1:], multiple_choice_df.groupby(['Q8']).size().reset_index(name='counts')['counts'][1:])

ax.set(xlabel='Is ML used?', ylabel='# of responses', title='ML usage vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
sns.set(rc={'figure.figsize':(20,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q10'])

ax = sns.barplot(multiple_choice_df.groupby(['Q10']).size().reset_index(name='counts')['Q10'][:-1], multiple_choice_df.groupby(['Q10']).size().reset_index(name='counts')['counts'][:-1])

ax.set(xlabel='Compensation group', ylabel='# of responses', title='Compensation group vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=11) 

plt.show()
sns.set(rc={'figure.figsize':(15,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q11'])

ax = sns.barplot(multiple_choice_df.groupby(['Q11']).size().reset_index(name='counts')['Q11'][:-1], multiple_choice_df.groupby(['Q11']).size().reset_index(name='counts')['counts'][:-1])

ax.set(xlabel='Amount spent', ylabel='# of responses', title='Amount spent vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
sns.set(rc={'figure.figsize':(15,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q15'])

ax = sns.barplot(multiple_choice_df.groupby(['Q15']).size().reset_index(name='counts')['Q15'].drop(6, axis=0), multiple_choice_df.groupby(['Q15']).size().reset_index(name='counts')['counts'].drop(6, axis=0))

ax.set(xlabel='# of years of writing code', ylabel='# of responses', title='# of years of writing code vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
sns.set(rc={'figure.figsize':(15,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q19'])

ax = sns.barplot(multiple_choice_df.groupby(['Q19']).size().reset_index(name='counts')['Q19'][:-1], multiple_choice_df.groupby(['Q19']).size().reset_index(name='counts')['counts'][:-1])

ax.set(xlabel='Recommanded Programming Language', ylabel='# of responses', title='Recommanded Programming Language vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
sns.set(rc={'figure.figsize':(15,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q22'])

ax = sns.barplot(multiple_choice_df.groupby(['Q22']).size().reset_index(name='counts')['Q22'].drop(3, axis=0), multiple_choice_df.groupby(['Q22']).size().reset_index(name='counts')['counts'].drop(3, axis=0))

ax.set(xlabel='TPU usage', ylabel='# of responses', title='TPU usage vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
sns.set(rc={'figure.figsize':(15,8)})

sns.set(style="darkgrid")



total = int(survey_schema_df.iloc[1]['Q23'])

ax = sns.barplot(multiple_choice_df.groupby(['Q23']).size().reset_index(name='counts')['Q23'][:-1], multiple_choice_df.groupby(['Q23']).size().reset_index(name='counts')['counts'][:-1])

ax.set(xlabel='ML methods usage', ylabel='# of responses', title='ML methods vs. # of responses')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

for p in ax.patches: # loop to all objects and plot group wise % distribution

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 5,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

plt.show()
multiple_choice_df.columns.values
def plot_multi_choice_dist(question, number_of_parts, xlabel, ylabel):

    sns.set(rc={'figure.figsize':(15,8)})

    sns.set(style="darkgrid")



    cats, counts = [], []

    total = int(survey_schema_df.iloc[1][question])



    for i in range(number_of_parts):

        cats.append(multiple_choice_df[multiple_choice_df[f'{question}_Part_{i+1}'].notnull()][f'{question}_Part_{i+1}'][1:].unique()[0])

        counts.append(int(multiple_choice_df[multiple_choice_df[f'{question}_Part_{i+1}'].notnull()][f'{question}_Part_{i+1}'].shape[0]))

    ax = sns.barplot(cats, counts)

    ax.set(xlabel=xlabel, ylabel=ylabel, title=f'{xlabel} vs. {ylabel}')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    for p in ax.patches: # loop to all objects and plot group wise % distribution

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 5,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=15) 

    return plt
plot_multi_choice_dist('Q9', 8, 'Responsibilities', '# of responses').show()
plot_multi_choice_dist('Q12', 12, 'ML/DS Learning Source', '# of responses').show()
plot_multi_choice_dist('Q13', 12, 'ML/DS Learning Platform', '# of responses').show()
plot_multi_choice_dist('Q16', 12, 'IDE\'s used', '# of responses').show()
plot_multi_choice_dist('Q17', 12, 'Hosted notebook platform', '# of responses').show()
plot_multi_choice_dist('Q18', 12, 'Programming languages', '# of responses').show()
plot_multi_choice_dist('Q20', 12, 'Visualization tools/libraries', '# of responses').show()
plot_multi_choice_dist('Q21', 5, 'Specialized Hardwares', '# of responses').show()
plot_multi_choice_dist('Q24', 12, 'ML algorithms', '# of responses').show()
plot_multi_choice_dist('Q25', 8, 'ML tools', '# of responses').show()
plot_multi_choice_dist('Q26', 7, 'Computer Vision methods', '# of responses').show()
plot_multi_choice_dist('Q27', 6, 'NLP methods', '# of responses').show()
plot_multi_choice_dist('Q28', 12, 'ML Frameworks', '# of responses').show()
plot_multi_choice_dist('Q29', 12, 'Cloud computing platforms', '# of responses').show()
plot_multi_choice_dist('Q30', 12, 'Cloud computing products', '# of responses').show()
plot_multi_choice_dist('Q31', 12, 'Big data / analytics products', '# of responses').show()
plot_multi_choice_dist('Q32', 12, 'Machine Learning products', '# of responses').show()
plot_multi_choice_dist('Q33', 12, 'Automated ML Tools', '# of responses').show()
plot_multi_choice_dist('Q34', 12, 'Relational Database products', '# of responses').show()