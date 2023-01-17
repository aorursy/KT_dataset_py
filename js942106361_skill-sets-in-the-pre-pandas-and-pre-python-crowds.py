# import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



#plt.rcParams["figure.figsize"]=24,20



# Basic chart

# initialize list of lists 



# Some guesses had to be made (exact dates are not critical)

# University - 1800?

# Trade Journals - 1800?

# Since I'm going to plot them at 1982, this error will not matter for now. 

# I will fix this later.





tech_timeline = [['learn','University Courses', 1800, 40], \

                 ['media','Trade Journals', 1800, 10], \

                 ['tool','SAS (1960)', 1960, 110], \

                 ['tool','SPSS (1968)', 1968, 120], \

                 

                 ['tool','vim(vi)/emacs', 1980, 130], \

                 ['tool','MATLAB', 1980, 140], \

                 

                 ['tool','python', 1985, 150], \

                 ['tool','MS Excel', 1985, 160], \

                 

                 ['tool','R', 1993, 100],\



                 ['tool','Spotfire', 1996, 80], \

                 ['tool','Salesforce', 1999, 100], \



                 ['tool','IPython', 2001, 110], \

                 

                 ['tool','matplotlib', 2002, 120], \

                 



                 ['tool','notepad++', 2003, 130], \

                 ['tool','Tableau', 2003, 90], \

                 

                 ['media','podcasts', 2004, 10], \

                 

                 ['media','Reddit', 2005, 20], \

                 ['media','YouTube', 2005, 30], \

                 ['tool','ggplot', 2005, 80], \

                 

                 ['tool','Twitter', 2006, 100], \

                 ['tool','Google Sheets', 2006, 120], \

                 ['tool','AWS', 2006, 140], \

                 ['tool','numpy', 2006, 160],\

                 

                 ['media','Hacker News', 2007, 10], \

                 

                 ['tool','pandas', 2008, 100], \

                 ['tool','Sublime text', 2008, 110], \

                 ['tool','Spyder', 2009, 140], \

                 

                 ['term','\"Data Science\""', 2010, 200],\

                 ['tool','Azure', 2010, 160], \

                 ['tool','PyCharm', 2010, 180], \

                 ['tool','scikit-learn',2010,150],\

                 ['tool','Kaggle', 2010, 130],\



                 ['learn','Udacity', 2011, 30], \

                 ['learn','Coursera', 2011, 50], \

                 ['tool','GCP', 2011, 190], \

                 

                 ['learn','edX', 2012, 40], \

                 ['tool','seaborn', 2012, 160], \

                 

                 ['tool','bokeh', 2013, 180], \

                 ['media','Slack', 2013, 10], \



                 ['learn','DataCamp', 2014, 30], \

                 ['tool','xgboost',2014,80],\

                 

                 ['learn','fast.ai', 2015, 20], \

                 ['learn','DataQuest', 2015, 40], \

                 ['tool','RStudio', 2015, 100], \

                 ['tool','JupyterLab', 2015, 120], \

                 ['tool','Visual Studio Code', 2015, 140], \

                 ['tool', 'TensorFlow', 2015, 160],\

                 ['tool', 'Keras', 2015, 180],\

                 

                 ['media','Towards Data Science', 2016, 10], \

                 ['tool','geoplotlib', 2016, 150], \

                 ['tool','FloydHub', 2016, 170], \



                 ['tool','Atom', 2017, 70], \

                 ['tool','lightgbm', 2017, 90],\

                 ['tool','Google Colab', 2017, 110], \

                 

                 ['tool','BERT', 2018, 185],\



                 ['term','2019 Survey', 2019, 215],\

                 ['none','',2023, 0] ]



# Create the pandas DataFrame 

tech_df = pd.DataFrame(tech_timeline, columns = ['TechType','TechName', 'Year', 'y-coord']) 



# pYear is "plot Year" -- anything before 1982 is going to be plotted at 1982

tech_df['pYear'] = tech_df.apply(lambda row: row['Year'] if (row['Year'] > 1982) else 1982, axis=1)



# tech_df.dtypes  # for error checking
plt.figure(figsize=(16,8))



plt.xlim = (1980, 2025)

plt.ylim = (0, 100)

plt.yticks([])



#plt.plot('Year', data=tech_df, linestyle='none', marker='o')

plt.plot('pYear','y-coord',data=tech_df[tech_df['TechType']=='tool'][['pYear','y-coord']], linestyle='none', marker='o', color='b')

plt.plot('pYear','y-coord',data=tech_df[tech_df['TechType']=='term'][['pYear','y-coord']], linestyle='none', marker='x',color='r')

plt.plot('pYear','y-coord',data=tech_df[tech_df['TechType']=='learn'][['pYear','y-coord']], linestyle='none', marker='+',color='g')

plt.plot('pYear','y-coord',data=tech_df[tech_df['TechType']=='media'][['pYear','y-coord']], linestyle='none', marker='*',color='black')

plt.plot('pYear','y-coord',data=tech_df[tech_df['TechType']=='none'][['pYear','y-coord']], linestyle='none', marker='.',color='black')



plt.plot([1984.8,1984.8],[0,200],linestyle="--",linewidth=1,color='y')

plt.plot([2007.8,2007.8],[0,200],linestyle="--",linewidth=1,color='y')  





title_font = {'size':'16', 'color':'black'} # Bottom vertical alignment for more space

plt.title("Data Science Technology Timeline**", **title_font)

# add tech labels

for i, row in tech_df.iterrows():

    year = row['pYear']

    plt.annotate(row['TechName'], xy=(year, row['y-coord']), xytext=(year, row['y-coord']+3))        
## Cleaning and Organizing the Data -- hidden



# don't print the file names any more

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# A note about coding style in this notebook. My coding habits are to slip in sanity checks frequently

# in order to catch problems before they get complicated. I've left them in just in case my next notebook

# update needs a little debugging.



# Using skiprows to take out the lengthy questions will prevent the warning about dtypes and low memory



df_raw = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv",skiprows=[1])

# df_other = pd.read_csv("/kaggle/input/kaggle-survey-2019/other_text_responses.csv",skiprows=[1])

# df_longq = pd.read_csv("/kaggle/input/kaggle-survey-2019/questions_only.csv",skiprows=[1])

    

# df_raw.shape    # initial shape was 19717, 246



# df_raw.columns   # 
# Drop unused columns.

# 1. All of the "*_OTHER_TEXT" data is in the file other_text_responses.csv. Since this data cannot

#     be aligned with the main data, the columns that indicate that other text was entered cannot be easily included. 

#     Drop those columns.

df_raw.drop(columns=['Q2_OTHER_TEXT','Q5_OTHER_TEXT','Q9_OTHER_TEXT',\

                     'Q12_OTHER_TEXT','Q13_OTHER_TEXT','Q14_Part_1_TEXT','Q14_Part_2_TEXT','Q14_Part_3_TEXT',\

                     'Q14_Part_4_TEXT','Q14_Part_5_TEXT','Q14_OTHER_TEXT',\

                     'Q16_OTHER_TEXT',\

                     'Q32_OTHER_TEXT','Q33_OTHER_TEXT','Q34_OTHER_TEXT'],inplace=True)

# df_raw.shape

# df_raw.head(3)

# Rename columns

df_raw.rename(columns = {'Time from Start to Finish (seconds)':'SurveyTime'}, inplace = True) 

df_raw.rename(columns = {'Q3':'Country'}, inplace = True) 

df_raw.rename(columns = {'Q15':'Coding Experience'}, inplace = True) 

# df_longq.rename(columns = {'Time from Start to Finish (seconds)':'SurveyTime'}, inplace = True) 

# Replace lengthy values

#  Some of these made for unwieldy labels on plots



# df_raw.loc[10,'Q4']

df_raw = df_raw.replace(to_replace=r"Some college/university study without earning a bachelor’s degree",\

                        value="Some college or uni",regex=True)

# df_raw.loc[10,'Q4']

df_raw = df_raw.replace(to_replace=r"No formal education past high school",\

                        value="High School",regex=True)

df_raw = df_raw.replace(to_replace=r"United Kingdom of Great Britain and Northern Ireland",\

                        value="UK",regex=True)

df_raw = df_raw.replace(to_replace=r"Iran, Islamic Republic of...",\

                        value="Iran",regex=True)

df_raw = df_raw.replace(to_replace=r"United States of America",\

                        value="USA",regex=True)



# df_raw.head(3)

# df_raw.dtypes

df_raw['TimeInMinutes']=df_raw['SurveyTime'].apply(lambda row: pd.to_numeric(row, errors='coerce')/60)
fig,ax = plt.subplots(1,1,figsize=(10,4))

ax = sns.countplot(x='Q1',data=df_raw,order=['18-21','22-24','25-29','30-34','35-39',\

                                               '40-44','45-49','50-54','55-59','60-69','70+'])

ax.set_title("Ages of the survey respondents")



ax.set_xlabel("Survey age groups")

age_list = df_raw['Q1'].value_counts().sort_index()

type(age_list)



i = 0

for p, label in zip(ax.patches, df_raw['Q1'].value_counts().index):

    a_count = age_list[i]

    ax.annotate(a_count, (p.get_x()+0.2, p.get_height()+0.15))

    i = i+1



# Divide into 4 groups, relative to major developments in tools as described in the EDA Plan



def agroup(x):

    group = 0

    

    if (x=='18-21'):

        group = 1

    elif (x=='22-24' or x=='25-29'):

        group = 2

    elif (x=='30-34' or x=='35-39' or x=='40-44'):

        group = 3

    else:

        group = 4

         # '45-49', 50-54','55-59','60-69','70+'



    #print('x was {} and group is {}'.format(x,group))

    return group



df_raw['AgeGroup'] = df_raw.apply(lambda x: agroup(x['Q1']), axis=1  )
fig,ax = plt.subplots(1,1,figsize=(8,4))

ax = sns.countplot(x='AgeGroup',data=df_raw)

ax.set_xticklabels(labels=["1 - Uni","2 - Recent grads","3 - Pre-Pandas","4 - Pre-Python" ])

ax.set_title("Comparison of respondents grouped by age relative to introduction of new tools")



# sanity check -- should not be any values other than 1,2,3,4

df_raw['AgeGroup'].value_counts().sort_index()



# make some easy views based on these groupings

df_age1 = df_raw[df_raw['AgeGroup'] == 1]

df_age2 = df_raw[df_raw['AgeGroup'] == 2]

df_age3 = df_raw[df_raw['AgeGroup'] == 3]

df_age4 = df_raw[df_raw['AgeGroup'] == 4]



df_age12 = df_age1 + df_age2



# Uncomment to double check and watch for errors

# df_age1.shape   # 2502, 233



# df_age2.shape   # 8068, 233

# df_age3.shape     # 6646, 233

# df_age3['Q1'].value_counts()

# df_age4.shape     # 2501, 233

# df_age12.shape    # 10570, 233
topN = df_raw['Country'].value_counts().iloc[:30]

top_countries = list(topN.index)

# top15_countries



df_plot = df_raw[df_raw['Country'].isin(top_countries)].groupby(['AgeGroup', \

                        'Country']).size().reset_index().pivot(columns='AgeGroup',\

                        index='Country', values=0)

refignore = df_plot.plot(kind='barh', stacked=True,figsize=(12,8),\

                         title="Comparison of age groups in each country")

# Group by years of coding experience, which was Q15

exp_list = ['I have never written code', '< 1 years', '1-2 years', '3-5 years',\

           '5-10 years','10-20 years', '20+ years']

df_plot = df_raw.groupby(['AgeGroup', \

                           'Coding Experience']).size().reset_index().pivot(columns='AgeGroup',\

                            index='Coding Experience', values=0)

refignore = df_plot.reindex(exp_list).plot(kind='barh', stacked=True,\

                               figsize=(12,8),\

                               title="Comparison of age groups and coding experience")
# Group by use of python, which was Q18_Part_1

df_raw['usesPython'] = df_raw.apply(lambda x: True if x['Q18_Part_1']=='Python' else False, axis=1)

# df_raw[['Q18_Part_1','usesPython','AgeGroup']].head(10)  # error check
df_plot = df_raw.groupby(['AgeGroup', \

                           'usesPython']).size().reset_index().pivot(columns='AgeGroup',\

                                                                      index='usesPython', values=0)

refignore = df_plot.plot(kind='barh', stacked=True, figsize=(12,4),title="Comparison of age groups and use of python")
df_raw['usesSciKitLearn'] = df_raw.apply(lambda x: True if x['Q28_Part_1']=='  Scikit-learn ' else False, axis=1)

df_raw['usesTensorFlow']  = df_raw.apply(lambda x: True if x['Q28_Part_2']=='  TensorFlow ' else False, axis=1)

df_raw['usesKeras']       = df_raw.apply(lambda x: True if x['Q28_Part_3']==' Keras ' else False, axis=1)

df_raw['usesXgboost']     = df_raw.apply(lambda x: True if x['Q28_Part_5']==' Xgboost ' else False, axis=1)
df_plot = df_raw.groupby(['AgeGroup', \

                           'usesSciKitLearn']).size().reset_index().pivot(columns='AgeGroup',\

                                                                      index='usesSciKitLearn', values=0)

df_plot.plot(kind='barh', stacked=True, figsize=(12,4),title="Comparison of age groups and use of scikit-learn")





df_plot = df_raw.groupby(['AgeGroup', \

                           'usesTensorFlow']).size().reset_index().pivot(columns='AgeGroup',\

                                                                      index='usesTensorFlow', values=0)

df_plot.plot(kind='barh', stacked=True, figsize=(12,4),title="Comparison of age groups and use of TensorFlow")



df_plot = df_raw.groupby(['AgeGroup', \

                           'usesKeras']).size().reset_index().pivot(columns='AgeGroup',\

                                                                      index='usesKeras', values=0)

df_plot.plot(kind='barh', stacked=True, figsize=(12,4),title="Comparison of age groups and use of Keras")



df_plot = df_raw.groupby(['AgeGroup', \

                           'usesXgboost']).size().reset_index().pivot(columns='AgeGroup',\

                                                                      index='usesXgboost', values=0)



refignore = df_plot.plot(kind='barh', stacked=True, figsize=(12,4),title="Comparison of age groups and use of Xgboost")
df_raw['onUdacity'] = df_raw.apply(lambda x: 1 if x['Q13_Part_1']=='Udacity' else 0, axis=1)

# df_raw[['AgeGroup','onUdacity','Q13_Part_1']].head(5)  # check for operation

df_raw['onCoursera']  = df_raw.apply(lambda x: 1 if x['Q13_Part_2']=='Coursera' else 0, axis=1)

df_raw['onedX']       = df_raw.apply(lambda x: 1 if x['Q13_Part_3']=='edX' else 0, axis=1)

df_raw['onDataCamp']     = df_raw.apply(lambda x: 1 if x['Q13_Part_4']=='DataCamp' else 0, axis=1)

df_raw['onDataQuest']       = df_raw.apply(lambda x: 1 if x['Q13_Part_5']=='DataQuest' else 0, axis=1)

df_raw['onKaggle']     = df_raw.apply(lambda x: 1 if x['Q13_Part_6']=='Kaggle Courses (i.e. Kaggle Learn)' else 0, axis=1)

df_raw['onFastAI']       = df_raw.apply(lambda x: 1 if x['Q13_Part_7']=='Fast.ai' else 0, axis=1)

df_raw['onUdemy']       = df_raw.apply(lambda x: 1 if x['Q13_Part_8']=='Udemy' else 0, axis=1)

df_raw['onLinkedIn']     = df_raw.apply(lambda x: 1 if x['Q13_Part_9']=='LinkedIn Learning' else 0, axis=1)



df_raw['numLearning'] = df_raw.apply(lambda x: x['onUdacity']+x['onCoursera']\

                                     +x['onedX']+x['onDataCamp']\

                                     +x['onDataQuest']+x['onKaggle']\

                                     +x['onFastAI']+x['onUdemy']\

                                     +x['onLinkedIn'],axis=1)

                                                        

# df_raw[['AgeGroup','onUdacity','Q13_Part_1','onCoursera','onedX','onDataCamp',\

#        'onDataQuest','onFastAI','onKaggle','onUdemy','onLinkedIn','numLearning']].head(5)  

# check for operation  









## plots start here



df_plot = df_raw.groupby(['AgeGroup', \

                           'numLearning']).size().reset_index().pivot(columns='AgeGroup', index='numLearning', values=0)



df_plot.plot(kind='barh', stacked=True, figsize=(12,4),\

             title="Comparison of age groups and number of learning platforms used")







df_raw[['AgeGroup','onUdacity','Q13_Part_1','onCoursera','onedX','onDataCamp',\

        'onDataQuest','onFastAI','onKaggle','onUdemy','onLinkedIn']].head(5)  # check for operation



# type(df_raw.loc[1,'onUdacity'])



     



#

df_plot = df_raw.groupby(['AgeGroup', \

                           'onUdacity']).size().reset_index().pivot(columns='AgeGroup', index='onUdacity', values=0)

df_plot.plot(kind='barh', stacked=True, figsize=(12,3),title="Comparison of age groups and use of Udacity")





df_plot = df_raw.groupby(['AgeGroup', \

                           'onCoursera']).size().reset_index().pivot(columns='AgeGroup', index='onCoursera', values=0)

df_plot.plot(kind='barh', stacked=True, figsize=(12,3),title="Comparison of age groups and use of Coursera")



df_plot = df_raw.groupby(['AgeGroup', \

                           'onedX']).size().reset_index().pivot(columns='AgeGroup', index='onedX', values=0)

df_plot.plot(kind='barh', stacked=True, figsize=(12,3),title="Comparison of age groups and use of edX")



df_plot = df_raw.groupby(['AgeGroup', \

                           'onDataCamp']).size().reset_index().pivot(columns='AgeGroup', index='onDataCamp', values=0)

df_plot.plot(kind='barh', stacked=True, figsize=(12,3),title="Comparison of age groups and use of DataCamp")



df_plot = df_raw.groupby(['AgeGroup', \

                           'onFastAI']).size().reset_index().pivot(columns='AgeGroup', index='onFastAI', values=0)

df_plot.plot(kind='barh', stacked=True, figsize=(12,3),title="Comparison of age groups and use of FastAI")



df_plot = df_raw.groupby(['AgeGroup', \

                           'onKaggle']).size().reset_index().pivot(columns='AgeGroup', index='onKaggle', values=0)

df_plot.plot(kind='barh', stacked=True, figsize=(12,3),title="Comparison of age groups and use of Kaggle")



df_plot = df_raw.groupby(['AgeGroup', \

                           'onUdemy']).size().reset_index().pivot(columns='AgeGroup', index='onUdemy', values=0)

df_plot.plot(kind='barh', stacked=True, figsize=(12,3),title="Comparison of age groups and use of Udemy")



df_plot = df_raw.groupby(['AgeGroup', \

                           'onLinkedIn']).size().reset_index().pivot(columns='AgeGroup', index='onLinkedIn', values=0)

df_plot.plot(kind='barh', stacked=True, figsize=(12,3),title="Comparison of age groups and use of LinkedIn")







total = df_raw['numLearning'].value_counts().sum()  # check that it adds up

# Sanity checks (looking for errors and suspicious results)

numlearnlist = df_raw['numLearning'].value_counts()

#print("The number of learning platforms used by the {} survey resondents break down as follows: \n{}".format(total,numlearnlist))

#numlearnlist



using0 = numlearnlist[0]/total # percent using 2 or fewer

using1or2 = df_raw['numLearning'].value_counts()[1:3].sum()/total # percent using 2 or fewer

using3ormore = df_raw['numLearning'].value_counts()[3:].sum()/total # percent using 2 or fewer



print("The breakdown of the {} respondents utilizing non-University learning platforms is:".format(total))

print("    Using 0 online learning platforms: {0:.0f}%".format(100*using0))

print("    Using 1 or 2 learning platforms: {0:.0f}%".format(100*using1or2))

print("    Using 3 or more learning platforms: {0:.0f}%".format(100*using3ormore))



# The columns of interest were:

# Years of coding experience - Q15

# Python -- usesPython

# Other tools used -- usesTensorFlow, usesKeras, usesXgboost, usesSciKitLearn

# Learning -- Q13_Part_1 through Q13_Part_9, a.k.a. onCoursera, onUdacity, etc.



# One more look...gather the columns of interest

useCols = ['AgeGroup','usesPython',\

           'usesTensorFlow','usesKeras','usesXgboost','usesSciKitLearn',\

           'onUdacity','onCoursera','onedX','onDataCamp',\

           'onDataQuest','onFastAI','onKaggle','onUdemy','onLinkedIn','numLearning']

agecorr = df_raw[useCols].corr()

f, ax = plt.subplots(figsize=(12,12))

refignore = sns.heatmap(agecorr,annot=True)
