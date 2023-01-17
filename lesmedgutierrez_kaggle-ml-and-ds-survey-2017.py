# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3
%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)
from matplotlib.colors import ListedColormap
import operator
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
import re
from subprocess import check_output
import os
#First read data
# I'm analying only multiple choise responses because de data is more consistent than free from responses, however I'll be
# reading the rest as I need them
mcr = pd.read_csv('../input/multipleChoiceResponses.csv', encoding='ISO-8859-1', low_memory=False)
#Demographic analysis
gender = mcr['GenderSelect'].value_counts()
total_gender = gender.sum()
f, ax = plt.subplots(1, 1,  figsize=(12, 5))
ax = sns.barplot(x=gender.values, y=gender.index, alpha=0.7, ax=ax)
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x = p.get_x()
    y = p.get_y()
    ax.text(width, y+height/2,'%0.2f%%' %(100*width/total_gender) , ha="left",)
plt.yticks(range(len(gender.index)), ['Male', 'Female','Different identity','No-binary\n Genderqueer\n Gender Non-confirming'], fontsize=14)    
plt.xlabel("Participants", fontsize=14)
plt.ylabel("Gender ", fontsize=14)
plt.show()
#Country
country = mcr.Country.value_counts().head(15)
total_country = country.sum()
f, ax = plt.subplots(1, 1,  figsize=(12, 8))
ax = sns.barplot(x=country.values, y=country.index, alpha=0.7, ax=ax, palette="Set2")
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x = p.get_x()
    y = p.get_y()
    ax.text(width, y+height/2,'%0.2f%%' %(100*width/total_country) , ha="left",)
plt.xlabel("Participants", fontsize=14)
plt.ylabel("Country ", fontsize=14)
plt.title("Top 15 countries contestants")
plt.show()
#Age 
age = mcr.loc[(mcr.Age > 16) & (mcr.Age <= 70),'Age'].astype(int).to_frame()
f, ax = plt.subplots(1, 1,  figsize=(20, 8))
ax = sns.countplot(x=age.Age, data=age, palette=sns.cubehelix_palette(54, start=2, rot=0, dark=0.25, light=.75, reverse=True), ax=ax)
ax.axvline(age.Age.mean()-16.2, linestyle='dashed', color='Black')
plt.title('Age of contestants')
plt.show()
#Employment status
employment = mcr['EmploymentStatus'].value_counts()
total_employment = employment.sum()
f, ax = plt.subplots(1, 1,  figsize=(10, 6))
ax = sns.barplot(x=employment.values, y=employment.index, ax=ax)
plt.yticks(range(len(employment.index)), 
           ['Employment full-time', 'Not employed\n but looking for work',
            'Independent contractor\nFreelancer\nSelf-employed',
            'Not employed\n Not looking for work',
            'Employed part-time', 'Retired', 'Prefer not to say'], fontsize=12)
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x = p.get_x()
    y = p.get_y()
    ax.text(width, y+height/2,'%0.2f%%' %(100*width/total_employment), ha="left",)
plt.show()
currentJob = mcr.CurrentJobTitleSelect.dropna().value_counts(ascending=False).head(10)
labels = list(currentJob.index)
sizes = list(currentJob.values)
f, ax = plt.subplots(1, 1, figsize=(5, 5))
patches, texts, perc = plt.pie(sizes, shadow=True, labels=labels, startangle=25, autopct="%1.2f%%")
plt.axis('equal')
plt.tight_layout()
plt.title('Current Job Title', fontsize=14)
plt.show()
group = mcr.groupby('CurrentJobTitleSelect').TitleFit.value_counts().to_frame()
cjt = mcr.CurrentJobTitleSelect.dropna().unique()
jtf = mcr.TitleFit.dropna().unique()
jtf = pd.DataFrame(index=['Perfectly', 'Fine', 'Poorly'], columns=cjt)
for c in jtf.columns:
    jtf = jtf.combine_first(group.loc[c].rename({'TitleFit': c}, axis=1))
jtf = jtf.astype(int).transpose().reindex(['Perfectly', 'Fine', 'Poorly', 'Sum'], axis=1)
jtf['Sum'] = jtf.sum(axis=1)
jtf = jtf.sort_values(by='Sum',ascending=True)

f, ax = plt.subplots(1,1, figsize=(12, 7))
colormap = ListedColormap(sns.color_palette("colorblind", n_colors=3 ))
ax = jtf[['Perfectly', 'Fine', 'Poorly']].plot(kind='barh', stacked=True, colormap=colormap, ax=ax)
plt.yticks(fontsize=11)
plt.title('Title fit')
plt.show()
#### Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(12, 7))

jtf.index
sns.set(style="whitegrid")
# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="Perfectly", y=list(jtf.index), data=jtf,
            label="Total", color="b")
# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(x="Fine", y=jtf.index, data=jtf,
             label="Alcohol-involved", color="b", ax=ax)


# Add a legend and informative axis label
# ax.legend(ncol=2, loc="lower right", frameon=True)
# ax.set(xlim=(0, 24), ylabel="",
#        xlabel="Automobile collisions per billion miles")
# sns.despine(left=True, bottom=True)
# load conversion rates and merge it with the corresponding matches
conver = pd.read_csv('../input/conversionRates.csv')
conver.drop('Unnamed: 0', axis=1, inplace=True)
salary = mcr[['CompensationAmount','CompensationCurrency','GenderSelect','Country','CurrentJobTitleSelect']].dropna()
salary = salary.merge(conver, left_on='CompensationCurrency', right_on='originCountry', how='left')
salary.CompensationAmount = salary.CompensationAmount.str.replace(',','').str.replace('-', '')
salary['Salary'] = (pd.to_numeric(salary['CompensationAmount'])*salary['exchangeRate'])
salary.drop(salary[salary.Salary.isnull()].index, inplace=True)
salary.Salary = salary.Salary.astype(int)
sal_clean = salary[(salary.Salary < 500000) & (salary.Salary > 1000)]
f, ax = plt.subplots(1, 2,  figsize=(18, 8))
sal_coun = sal_clean.groupby('Country')['Salary'].mean().sort_values(ascending=False)[:15].to_frame()
sal_coun.Salary = sal_coun.Salary.astype(int)
colormap = ListedColormap(sns.color_palette("colorblind", n_colors=15))
sns.barplot('Salary', sal_coun.index, data=sal_coun, palette='RdYlGn', ax=ax[0])
ax[0].set_title('Highest Salary Paying Countries')
ax[0].set_xlabel('')
ax[0].set_ylabel('Countries')
ax[0].axvline(sal_clean.Salary.mean(), linestyle='dashed')
resp_count = mcr.Country.value_counts()[:15].to_frame()
max_coun = sal_clean.groupby('Country')['Salary'].mean().to_frame()
max_coun = max_coun[max_coun.index.isin(resp_count.index)]
max_coun = max_coun.astype(int)
max_coun.sort_values(by='Salary', ascending=True).plot(kind='barh', width=0.8, ax=ax[1], colormap=colormap)
ax[1].axvline(sal_clean.Salary.mean(), linestyle='dashed')
ax[1].set_title('Top 15 Respondent Countries Salary')
ax[1].set_xlabel('')
ax[1].set_ylabel('Countries')
for p in ax[0].patches:
    width = p.get_width()
    height = p.get_height()
    x = p.get_x()
    y = p.get_y()
    ax[0].text(width, y+height/2, ("{:,}".format(width)), ha="left",)
for p in ax[1].patches:
    width = p.get_width()
    height = p.get_height()
    x = p.get_x()
    y = p.get_y()
    ax[1].text(width, y+height/2, ("{:,}".format(width)), ha="left",)
plt.subplots_adjust(wspace=0.8)
plt.show()
# Don't pay attention to this, it was an experiment
sal_norm = sal_clean.copy()
sal_norm.drop(sal_norm[sal_norm.Salary==0].index, inplace=True)
countries = sal_norm.Country.unique()
for country in countries:
    count = sal_norm[sal_norm.Country == country]
    min = count.Salary.min()
    max = count.Salary.max()
    count.Salary = count.Salary.apply(lambda x: (((x-min)*(100000))/(max - min)))
    sal_norm.loc[count.index, 'Salary'] = count.Salary
sal_norm.Salary = sal_norm.Salary.astype(int)
# Don't pay attention to this, it was an experiment and I dont want to delete it
f, ax = plt.subplots(1, 2,  figsize=(18, 8))
sal_coun = sal_norm.groupby('Country')['Salary'].mean().sort_values(ascending=False)[:15].to_frame()
sal_coun.Salary = sal_coun.Salary.astype(int)
sns.barplot('Salary', sal_coun.index, data=sal_coun, palette='RdYlGn', ax=ax[0])
ax[0].set_title('Highest Salary Paying Countries')
ax[0].set_xlabel('')
ax[0].set_ylabel('Country', fontsize=13)
resp_count = mcr.Country.value_counts()[:15].to_frame()
max_coun = sal_norm.groupby('Country')['Salary'].mean().to_frame()
max_coun = max_coun[max_coun.index.isin(resp_count.index)]
max_coun = max_coun.astype(int)
max_coun.sort_values(by='Salary', ascending=True).plot.barh(width=0.8, ax=ax[1], color=sns.color_palette('RdYlGn'))
ax[1].axvline(sal_norm.Salary.mean(), linestyle='dashed')
ax[1].set_title('Compensation of Top 15 Respondent Countries')
ax[1].set_xlabel('')
ax[1].set_ylabel('Country', fontsize=13)
for p in ax[0].patches:
    width = p.get_width()
    height = p.get_height()
    x = p.get_x()
    y = p.get_y()
    ax[0].text(width, y+height/2, ("{:,}".format(width)), ha="left",)
for p in ax[1].patches:
    width = p.get_width()
    height = p.get_height()
    x = p.get_x()
    y = p.get_y()
    ax[1].text(width, y+height/2, ("{:,}".format(width)), ha="left",)
plt.subplots_adjust(wspace=0.8)
plt.show()
sal_job = sal_clean.groupby('CurrentJobTitleSelect')['Salary'].mean().sort_values(ascending=False).to_frame()
sal_job.Salary = sal_job.Salary.astype(int)
f, ax = plt.subplots(1, 1,  figsize=(12, 7))
sns.barplot('Salary', sal_job.index, data=sal_job, palette=sns.color_palette("RdPu_r", 16), ax=ax)
ax.set_title('Highest Salary Paying by Job Title')
ax.set_xlabel('')
ax.set_ylabel('')
plt.show()
f, ax = plt.subplots(1,1, figsize=(9,5))
sns.barplot( x='GenderSelect', y='Salary', data=sal_clean, palette='rainbow', ax=ax)
plt.xticks([0,1,2,3],  ['Male', 'Female', 'Different Identity', 'No-binary\nGenderqueer\nGender Non-confirming'])
plt.title('Salary by gender')
plt.show()
employtype = mcr.copy()
emp = employtype.CurrentEmployerType.str.split(',').dropna()
emps=[e for em in emp for e in em]
#for e in emp:
#    emps.extend(e)
emps = pd.Series(emps).value_counts().to_frame()
f, ax = plt.subplots(1, 1,  figsize=(12, 7))
ax = emps.plot(kind='barh', ax=ax)
plt.yticks(fontsize=13)
plt.show()
from wordcloud import (WordCloud, get_single_color_func)

#PS : Credits to Andreas Mueller, creator of wordcloud, for the following code of the class 'GroupedColorFunc'.
#He made the code fully public for people who want to use specific color for specific words and made an example.
#Source link : https://amueller.github.io/word_cloud/auto_examples/colored_by_group.html#sphx-glr-auto-examples-colored-by-group-py

class GroupedColorFunc(object):

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)
#############################################################
# Get text data from the freeform
freeForm = pd.read_csv('../input/freeformResponses.csv')
text = freeForm[pd.notnull(freeForm["KaggleMotivationFreeForm"])]["KaggleMotivationFreeForm"]
wc = WordCloud(collocations=False,height=500, width=800,  relative_scaling=0.2,random_state=74364).generate(" ".join(text))
color_to_words = {
    # words below will be colored with a green single color function
    '#151fa5': ['data', 'science', 'mining', 'big',
                'bigdata', 'machine', 'learning']
}

# Words that are not in any of the color_to_words values will be colored with grey
default_color = 'grey'
# Create a color function with multiple tones
grouped_color_func = GroupedColorFunc(color_to_words, default_color)
# Apply our color function
wc.recolor(color_func=grouped_color_func)
# Plot
plt.figure(figsize=(12,12))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off");
dataset = mcr.PublicDatasetsSelect.dropna().str.split(',')
datasets = [d for ds in dataset for d in ds]
ds = pd.Series(datasets).value_counts().sort_values(ascending=True).to_frame().rename({0: 'Public Datasets'}, axis='columns')

f, ax = plt.subplots(1, 1,  figsize=(10, 7))
ax = ds.plot(kind='barh', ax=ax, color="g")
plt.show()

learningplat = mcr.LearningPlatformSelect.dropna().str.split(',')
learningplats = [l for lp in learningplat for l in lp]
lp = pd.Series(learningplats).value_counts(ascending=True).to_frame().rename({0: 'Learning Platform'}, axis='columns')
f, ax = plt.subplots(1, 1,  figsize=(10, 8))
ax = lp.plot(kind='barh', ax=ax, color="g")
ax.set_title('Learning Platform')
plt.show()
language = mcr.LanguageRecommendationSelect.dropna().value_counts().sort_values().to_frame()
f, ax = plt.subplots(1, 1,  figsize=(12, 6))
ax = language.plot(kind='barh', ax=ax)
plt.yticks(fontsize=13)
plt.show()
match = 'WorkToolsFrequency\w*'
worktools = dict()
for c in mcr.columns:
    if re.match(match, c):
        worktools[c] = c.replace('WorkToolsFrequency', '')
wt = mcr.loc[:, worktools.keys()].fillna(0)
wt = wt.rename(worktools, axis='columns')
wt.replace(to_replace=['Rarely', 'Sometimes', 'Often', 'Most of the time'], value=[1, 2, 3, 4], inplace=True)
tools = wt.apply(np.count_nonzero).sort_values(ascending=False).to_frame().rename({0:'Count'}, axis='columns')
pyr = mcr.loc[:,["WorkToolsFrequencyR","WorkToolsFrequencyPython", "WorkToolsFrequencySQL"]].fillna(0)
pyr.replace(to_replace=['Rarely', 'Sometimes', 'Often', 'Most of the time'], value=[1, 2, 3, 4], inplace=True)

pyr['PythonVsR'] = ['Python' if p>2 and p>r else
                    'R' if r>2 and r>p else
                    'Both'if p==r and p>2 else
                    'None' for p, r in zip(pyr.WorkToolsFrequencyPython, pyr.WorkToolsFrequencyR)
                   ]
pyr['PythonVsSQL'] = ['Python' if p>2 and p>s else
                      'SQL' if s>2 and s>p else
                      'Both'if p==s and p>2 else
                      'None' for p, s in zip(pyr.WorkToolsFrequencyPython, pyr.WorkToolsFrequencySQL)
                     ]
pyr['PythonVsRVsSQL'] = ['Python' if p>2 and p>r and p>s else 
                         'R' if r>2 and r>p and r>s else
                         'Python & R'if p==r and p>2 else
                         'SQL' if s>2 and s>p and s>r else
                         'Python & SQL' if p==s and p>2 else
                         'R & SQL' if r==s and r>2 else
                         'All' if p==r and p==s and p>1 else
                         'None' for p, r, s in zip(pyr.WorkToolsFrequencyPython, pyr.WorkToolsFrequencyR, pyr.WorkToolsFrequencySQL)
                        ]
new_index = ['Python',  'R', 'Python & R', 'SQL', 'Python & SQL', 'R & SQL', 'All']
valuesPR = pyr.PythonVsR.value_counts().to_frame().drop('None')
valuesPSQL = pyr.PythonVsSQL.value_counts().to_frame().drop('None')
valuesPRSQL = pyr.PythonVsRVsSQL.value_counts().to_frame().reindex(new_index)
valuesPRSQL.loc['All'] = 0
f, ax = plt.subplots(1, 3,  figsize=(20, 10))
venn2(subsets=valuesPR.loc['Python':].PythonVsR.values, set_labels=valuesPR.loc['Python':].index, ax=ax[0])
venn2(subsets=valuesPSQL.loc['Python':].PythonVsSQL.values, set_labels=valuesPSQL.loc['Python':].index, ax=ax[1])
venn3(subsets=valuesPRSQL.loc['Python':].PythonVsRVsSQL.values, set_labels=['Python', 'R', 'SQL'], ax=ax[2])
plt.show()
match = 'LearningPlatformUsefulness'
useful = dict()
for c in mcr.columns:
    if re.match(match, c):
        useful[c] = c.replace(match, '')
learningplat = mcr.loc[:, useful.keys()].dropna(axis=0, how='all')
learningplat = learningplat.rename(useful, axis='columns')
lp = pd.DataFrame(index=['Very useful', 'Somewhat useful', 'Not Useful' ] , columns=useful.values())
for c in learningplat.columns:
    lp = lp.combine_first(learningplat[c].value_counts().to_frame())
lp = lp.reindex(['Very useful', 'Somewhat useful', 'Not Useful' ]).astype(int)
lp = lp.rename({'SO': 'Stack Overflow',
                'Courses': 'Online Courses'}, axis=1)
lptrans = lp.transpose()
lptrans['Sum'] = lptrans.sum(axis=1)
lptrans = lptrans.sort_values(by='Sum', ascending=True)
f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = lptrans.iloc[:, 0:3].plot(kind='barh', stacked=True,colormap='summer', ax=ax)
plt.yticks(fontsize=13)
plt.show()
match = 'WorkMethodsFrequency'
workmethod = dict()
for c in mcr.columns:
    if re.match(match, c):
        workmethod[c] = c.replace(match, '')
workmethods = mcr.loc[:, workmethod.keys()].dropna(axis=0, how='all').rename(workmethod, axis='columns')
wm = workmethods.DataVisualization.unique() #[nan, 'Sometimes', 'Most of the time', 'Often', 'Rarely']
wm = pd.DataFrame(index=wm, columns=workmethod.values())
for c in workmethods.columns:
    wm = wm.combine_first(workmethods[c].value_counts().to_frame())
wmtrans = wm.dropna(axis=0, how='all').astype(int).transpose()
wmtrans['Sum'] = wmtrans.sum(axis=1)
wmtrans = wmtrans.sort_values(by='Sum', ascending=True).reindex(['Most of the time', 'Often', 'Sometimes',  'Rarely', 'Sum'], axis='columns')
f, ax = plt.subplots(1,1, figsize=(15, 10))
ax = wmtrans.iloc[10:, :4].plot(kind='barh', stacked=True, colormap='tab20c',ax=ax)
plt.yticks(fontsize=13)
plt.show()

courseplat = list(mcr.CoursePlatformSelect.dropna().str.split(','))
courseplat  = pd.Series([c for course in courseplat for c in course ], name='Course Platform').value_counts().to_frame()

f, ax = plt.subplots(1,1, figsize=(10, 5))
ax = courseplat.plot(kind='bar', color='y', ax=ax)
plt.show()
personalhardware = mcr.HardwarePersonalProjectsSelect.dropna().str.split(',')
personalhardware = [ph for perhard in personalhardware for ph in perhard ]
personalhardware = pd.Series(personalhardware, name='Personal Hardware').value_counts().to_frame()
f, ax = plt.subplots(1,1, figsize=(12, 6))
ax = personalhardware.plot(kind='barh', color='r', ax=ax)
plt.yticks(fontsize=13)
plt.show()
timespend = mcr.TimeSpentStudying.dropna().value_counts().to_frame()
timespend = timespend.reindex(['0 - 1 hour', '2 - 10 hours', '11 - 39 hours', '40+'])
f, ax = plt.subplots(1,1, figsize=(12, 5))
ax = timespend.plot(kind='barh', color='c', ax=ax)
plt.show()

proven = mcr.ProveKnowledgeSelect.dropna().value_counts(ascending=True).to_frame()
f, ax = plt.subplots(1,1, figsize=(10, 8))
ax = proven.plot(kind='barh', color='k', ax=ax)
plt.show()
formaled = mcr.FormalEducation.dropna().value_counts(ascending=True).to_frame()
f, ax = plt.subplots(1,1, figsize=(10, 8))
ax = formaled.plot(kind='barh', color='grey', ax=ax)
plt.show()
major = mcr.MajorSelect.dropna().value_counts(ascending=True).to_frame()
f, ax = plt.subplots(1,1, figsize=(10, 8))
ax = major.plot(kind='barh', color='k', ax=ax)
plt.title('Major')
plt.show()
tenure = mcr.Tenure.dropna().value_counts().to_frame()
tenure = tenure.reindex([ 'Less than a year', '1 to 2 years', '3 to 5 years', 
                         '6 to 10 years',  'More than 10 years', "I don't write code to analyze data"])
f, ax = plt.subplots(1,1, figsize=(10, 8))
ax = tenure.plot(kind='barh', color='c', ax=ax)
plt.show()
match = 'LearningCategory\w*'
learningCategory = dict()
for c in mcr.columns:
    if re.match(match, c):
        learningCategory[c] = c.replace(match, '')
learningCategory['LearningCategorySelftTaught'] = 'SelfTaught'
learncat = mcr.loc[:, learningCategory.keys()].dropna(how='all').fillna(0).rename(learningCategory, axis=1).astype(int)
f, ax = plt.subplots(1,1, figsize=(15, 8))
sns.set(style="whitegrid")
ax = sns.boxplot(data=learncat, palette='Set3', ax=ax)
sns.despine(offset=10, trim=True)
plt.show()
