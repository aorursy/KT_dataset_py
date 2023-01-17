# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import glob
import shutil
import altair as alt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
color = sns.color_palette()
%matplotlib inline
sns.set(font_scale=2)
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
# Squarify for treemaps
import squarify
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
print(os.listdir("../input"))
np.random.seed(111)
# Any results you write to the current directory are saved as output.
# Read the CSV file
data = pd.read_csv('../input/survey_results_public.csv')
survey_results = pd.read_csv('../input/survey_results_schema.csv')
color_brewer = ['#41B5A3','#FFAF87','#FF8E72','#ED6A5E','#377771','#E89005','#C6000D','#000000','#05668D','#028090','#9FD35C','#02C39A','#F0F3BD','#41B5A3','#FF6F59','#254441','#B2B09B','#EF3054','#9D9CE8','#0F4777','#5F67DD','#235077','#CCE4F9','#1748D1','#8BB3D6','#467196','#F2C4A2','#F2B1A4','#C42746','#330C25']
def voteSimplifier(v,o):
    d = {}
    for type in v:
        type = str(type).split(';')
        for i in type: 
            if i in d:
                d[i] = d[i] + 1
            else:
                d[i] = 0
    if o == 'v':
        Y = list(d.values())
        X = list(d.keys())
    else: 
        Y = list(d.keys())
        X = list(d.values())
    trace = [go.Bar(
                y=Y,
                x=X,
                orientation = o,
                marker=dict(color=color_brewer),
    )]
    layout = go.Layout(
        margin = go.Margin(
            l = 600 if o == 'h' else 50
        )
    )

    fig = go.Figure(data=trace, layout=layout)
    iplot(fig, filename='horizontal-bar')
data.head()
print(data.isnull().sum())
msno.matrix(data)
plt.show()
# how many total missing values do we have?
missing_values_count = data.isnull().sum()
total_cells = np.product(data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
print((total_missing/total_cells) * 100, '% of Missing Values in Survey Results Public')
# checking missing data in each survey results public column
total_missing = data.isnull().sum().sort_values(ascending = False)
percentage = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
missing_survey_results_public = pd.concat([total_missing, percentage], axis=1, keys=['Total Missing (Column-wise)', 'Percentage (%)'])
missing_survey_results_public.head()
f,ax=plt.subplots(2,2,figsize=(25,20))

val_survey_country = data.Country.value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[0][0])
val_cod_hobby = data.Hobby.value_counts().plot.bar(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[0][1])
val_contri_open_source = data.OpenSource.value_counts().plot.bar(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[1][0])
val_current_student = data.Student.value_counts().plot.bar(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[1][1])


plt.subplots_adjust(wspace=0.1)
ax[0][0].set_title('Popular country among respondent')
ax[0][1].set_title('Do you code as a hobby?')
ax[1][0].set_title('Contribute to open source project')
ax[1][1].set_title('Currently enrolled as student')
# plt.savefig('stat.png')
plt.show()
f,ax=plt.subplots(2,2,figsize=(25,25))


val_current_student = data.Employment.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15),ax=ax[0][1])
val_current_student = data.FormalEducation.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15),ax=ax[0][0])
val_current_student = data.UndergradMajor.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15),ax=ax[1][0])
val_current_student = data.CompanySize.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15),ax=ax[1][1])



plt.subplots_adjust(wspace=0.8)
ax[0][1].set_title('Popular employment status')
ax[0][0].set_title('Popular formal education')
ax[1][0].set_title('Popular Undergrad major')
ax[1][1].set_title('How many people employed by your company')
plt.show()
f,ax=plt.subplots(2,2,figsize=(25,25))

val_survey_country = data.DevType.value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[0][0])
val_cod_hobby = data.YearsCoding.value_counts().plot.bar(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[0][1])
val_contri_open_source = data.YearsCodingProf.value_counts().plot.bar(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[1][0])
val_current_student = data.JobSatisfaction.value_counts().plot.bar(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[1][1])


plt.subplots_adjust(wspace=0.1)
ax[0][0].set_title('Popular Dev Type among respondent')
ax[0][1].set_title('Popular years of coding experience including education among developers')
ax[1][0].set_title('Popular years of professional coding experience among developers')
ax[1][1].set_title('Currently enrolled as student')
plt.show()
# How many entries are there?
print("Total number of responses: ", data.shape[0])

# How many columns are there?
print("Number of columns in the dataset: ", data.shape[1])

# What are the column names?
print("Columns are: ")
print(list(data.columns))
# A handy dandy function for making a bar plot. You can make it as flexible as much as you want!!
def do_barplot(df, figsize=(20,8), plt_title=None, xlabel=None, ylabel=None, title_fontsize=20, fontsize=16, orient='v', clr=None, max_counts=None):
    # Get the value counts 
    df_counts = df.value_counts()
    total = df.shape[0]
    
    # If there are too many values, limit the amount of information for display purpose
    if max_counts:
        df_counts = df_counts[:max_counts]
    
    # Print the values along with their counts and overall %age
    for i, idx in enumerate(df_counts.index):
        val = df_counts.values[i]
        percentage = (val/total)*100
        print("{:<20s}    {}  or roughly {:.2f}% ".format(idx, val, percentage))
    
    # Plot the results 
    plt.figure(figsize=figsize)
    if orient=='h':
        if clr:
            sns.barplot(y=df_counts.index, x=df_counts.values, orient='h', color=color[clr])
        else:
            sns.barplot(y=df_counts.index, x=df_counts.values, orient='h')
    else:
        if color:
            sns.barplot(x=df_counts.index, y=df_counts.values, orient='v', color=color[clr])
        else:
            sns.barplot(x=df_counts.index, y=df_counts.values, orient='v')
            
    plt.title(plt_title, fontsize=title_fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    
    if orient=='h':
        plt.yticks(range(len(df_counts.index)), df_counts.index)
    else:
        plt.xticks(range(len(df_counts.index)), df_counts.index)
    plt.show()
    del df_counts
# A handy dandy function for countplot. (I may or may not use it very often but let's define it anyways)
def do_countplot(df, yval=None, xval=None, hueval=None, axs=[0,0], hue_ord=None):
    if df is None or (xval is None and yval is None):
        print("Either data or the axis values is missing")
        return
    if yval:
        sns.countplot(y=yval, data=df, hue=hueval, ax=axs,hue_order=hue_ord)
    else:
        sns.countplot(x=xval, data=df, hue=hueval, ax=axs,hue_order=hue_ord)
# Get the hobby column 
hobby = data['Hobby'].dropna()

# Visualize the results. (You see, how handy our function is!!)
do_barplot(df=hobby, figsize=(10,8), 
           fontsize=16, title_fontsize=20, 
           xlabel='Hobby?', ylabel='Count', 
           plt_title="Coding as a hobby",
           orient='v', clr=3)
del hobby
# Get the corresponding column and drop the null values
opensource = data['OpenSource'].dropna()

#visualize
do_barplot(df=opensource, figsize=(10,8), 
           fontsize=16, title_fontsize=20, 
           xlabel="Type", ylabel='Count', 
           plt_title="Contribution to OpenSource",
           orient='v', clr=2)
del opensource
# Get the country column and do a value counts
country = data['Country'].dropna()
country_counts = country.value_counts()

# Get the countries with maximum and minimum number of developers
max_count = country_counts.max()
min_count = country_counts.min()

print("Total number of countries: ", len(country_counts))
print("Country with maximum number of developers: {}     #Developers: {}".format(country_counts.index[country_counts.values==max_count][0], max_count))
print("")
print("Country with least number of developers: {}     #Developers: {}".format(list(country_counts.index[country_counts.values==min_count]), min_count))
print("*********************************************************************************\n")

# As there are developers from 183 countries(woahh...), for the sake of plotting we will choose the top 50 countries
max_counts = 50

# visualize(check the max counts argument this time)
do_barplot(df=country, figsize=(30,30), 
           fontsize=16, title_fontsize=20, 
           xlabel="Count", ylabel="Country", 
           plt_title="Country where the developers reside",
           orient='h', max_counts=max_counts)

del country_counts
del country    
aidanger = data['AIDangerous'].dropna()
do_barplot(df=aidanger, figsize=(20,10), 
           fontsize=16, title_fontsize=20, 
           xlabel="Count", ylabel="Reasoning", 
           plt_title="What do developers fear about AI?",
           orient='h')

del aidanger
# Get the column data
aiinterest = data['AIInteresting'].dropna()

# Visualize
do_barplot(df=aiinterest, figsize=(20,10), 
           fontsize=16, title_fontsize=20, 
           xlabel="Count", ylabel="Reasoning", 
           plt_title="What's interesting about AI?",
           orient='h')

del aiinterest
# Same thing..get the column and just use our handy dandy function. Life is easy!!
airesp = data['AIResponsible'].dropna()
do_barplot(df=airesp, figsize=(20,10), 
           fontsize=16, title_fontsize=20, 
           xlabel="Count", ylabel="Who?", 
           plt_title="Who should bear the burden of responsibilites in AI?",
           orient='h', clr=8)
del airesp
# Get the column
aifuture = data['AIFuture'].dropna()

#visualize 
do_barplot(df=aifuture,figsize=(20,8), 
           fontsize=16, title_fontsize=20, 
           xlabel='Count', ylabel='What?', 
           plt_title="Opininon about future of AI?",
           orient='h', clr=5)
del aifuture
plt.figure(figsize=(15,10))
lang = data[data.LanguageWorkedWith.notnull()]['LanguageWorkedWith']

lang_stacked = pd.DataFrame(lang.str.split(';').tolist()).stack()
lang_stacked.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15))
#plt.savefig('sta.png')
plt.show()
temp = data['LanguageDesireNextYear'].value_counts().head(25).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Language Desire Next Year', 
       title = "Top Language Desire Next Year")
plt.show()
random.shuffle(color_brewer)
voteSimplifier(data["DatabaseWorkedWith"].dropna(),'v')
temp = data['JobSatisfaction'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'JobSatisfaction', 
       title = "Most popular JobSatisfaction")
plt.show()
temp = data['PlatformWorkedWith'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Platform Worked With', 
       title = "Top Platform Worked With")
plt.show()
temp = data['PlatformDesireNextYear'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Platform Desire Next Year', 
       title = "Top Platform Desire Next Year")
plt.show()
temp = data['FrameworkWorkedWith'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Framework Worked With', 
       title = "Top Framework Worked With")
plt.show()
temp = data['IDE'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'IDE', 
       title = "Top Top IDE Used by Developers")
plt.show()
temp = data['OperatingSystem'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Operating System', 
       title = "Top Operating Systems Used by Developers")
plt.show()
temp = data['Methodology'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Methodology', 
       title = "Top Methodologies Used by Developers")
plt.show()
temp = data['UpdateCV'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'UpdateCV', 
       title = "Top UpdateCV of Developers")
plt.show()
temp = data['HackathonReasons'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(90,20))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'HackathonReasons', 
       title = "Top HackathonReasons")
plt.show()
# Get the corresponding column and drop the null values
AdBlocker= data['AdBlocker'].dropna()

#visualize
do_barplot(df=AdBlocker, figsize=(10,8), 
           fontsize=16, title_fontsize=20, 
           xlabel="Type", ylabel='Count', 
           plt_title="AdBlocker",
           orient='v', clr=2)
del AdBlocker
# Get the corresponding column and drop the null values
EthicsChoice = data['EthicsChoice'].dropna()

#visualize
do_barplot(df=EthicsChoice, figsize=(10,8), 
           fontsize=16, title_fontsize=20, 
           xlabel="Type", ylabel='Count', 
           plt_title="EthicsChoice",
           orient='v', clr=2)
del EthicsChoice
# Get the corresponding column and drop the null values
EthicsReport = data['EthicsReport'].dropna()

#visualize
do_barplot(df=EthicsReport, figsize=(30,10), 
           fontsize=16, title_fontsize=20, 
           xlabel="Type", ylabel='Count', 
           plt_title="EthicsReport",
           orient='v', clr=5)
del EthicsReport

fig = {
  "data": [
    {
      "values": data["CompanySize"].value_counts().values,
      "labels": data["CompanySize"].value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Company size distribution",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Company size distribution",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Company Size",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')

random.shuffle(color_brewer)
voteSimplifier(data["SelfTaughtTypes"].dropna(),'h')
random.shuffle(color_brewer)
voteSimplifier(data["Currency"].dropna(),'v')
# A handy dandy function for returning a grouby object 
def return_grouped_data(df, group_by =None, group=None):
    if group_by is None or group is None:
        print("ValueError: You mist provide the groupby and group name")
        return
    
    grouped_data = df.groupby(group_by).get_group(group).reset_index(drop=True)
    return grouped_data
# Get the revelvant columns and drop null values
country_gender = data[['Country', 'Gender', 'Employment']].dropna()

# Do some cleaning on the gender column. People fill multiple values for this columns. I never get the logic  behind that. It's a survey
# You fill it up the wrong way, thigs are never gonna improve then.
country_gender['Gender'] = country_gender['Gender'].apply(lambda x: x if x in ['Male', 'Female'] else 'Other')

# Get the groupby object for the top foir countries
country_US = return_grouped_data(df=country_gender,group='United States', group_by='Country')
country_India = return_grouped_data(df=country_gender,group='India', group_by='Country')
country_Germany = return_grouped_data(df=country_gender,group='Germany', group_by='Country')
country_UK = return_grouped_data(df=country_gender,group='United Kingdom', group_by='Country')
country_Bangladesh = return_grouped_data(df=country_gender,group='Bangladesh', group_by='Country')

# Plot the results
f, axs = plt.subplots(2,2, figsize=(40,25), sharey=True, sharex=True)
sns.countplot(y=country_US['Employment'], data=country_US, hue='Gender', ax=axs[0,0], hue_order=['Male', 'Female', 'Other'])
axs[0,0].set_title('US', fontsize=20)

sns.countplot(y=country_India['Employment'], data=country_India, hue='Gender', ax=axs[0,1], hue_order=['Male', 'Female', 'Other'])
axs[0,1].set_title('India', fontsize=20)

sns.countplot(y=country_Germany['Employment'], data=country_Germany, hue='Gender', ax=axs[1,0], hue_order=['Male', 'Female', 'Other'])
axs[1,0].set_title('Germany', fontsize=20)

sns.countplot(y=country_UK['Employment'], data=country_UK, hue='Gender', ax=axs[1,1], hue_order=['Male', 'Female', 'Other'])
axs[1,1].set_title('Bangladesh', fontsize=20)
plt.show()

del country_gender, country_US, country_Germany, country_India, country_UK, country_Bangladesh
from textblob import TextBlob, Word
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

text = " ".join((data['AIDangerous']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('What do developers fear about AI?');
from textblob import TextBlob, Word
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

text = " ".join((data['AIInteresting']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title("What's interesting about AI?");
text = " ".join((data['UpdateCV']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Purpose of the Last CV Update');
text = " ".join((data['HackathonReasons']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Reason for Participating in Hackathons');
text = " ".join((data['LanguageWorkedWith']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Which programming language is popular?');
text = " ".join((data['IDE']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Top IDE Used by Developers');
text = " ".join((data['Country']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Country: In which country do you currently reside?');
text = " ".join((data['AIFuture']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Opininon about future of AI?');
text = " ".join((data['UndergradMajor']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Purpose of the UnderGradMajor');
text = " ".join((data['DevType']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Purpose of the DevType');
text = " ".join((data['JobSatisfaction']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Purpose of the JobSatisfaction');
text = " ".join((data['FormalEducation']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Types of FormalEducation involved');
text = " ".join((data['Employment']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Types of Employment');
text = " ".join((data['Currency']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Top Currency');
text = " ".join((data['SelfTaughtTypes']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Purpose of the SelfTaughtTypes');
text = " ".join((data['DatabaseWorkedWith']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title(' Top DatabaseWorkedWith by Developers');
text = " ".join((data['EthicsResponsible']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Purpose of the EthicsResponsible');