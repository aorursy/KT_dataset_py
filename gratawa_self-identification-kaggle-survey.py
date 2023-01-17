# Included the infographic of the most common attributes of a data science
# Main reason is to get attention of reader so that it increases the chance of viewer
# to continue reading. 
# Used this format of infographic as I expect a lot of viewers are already familiar with
# format so already in their frame in a positive light


# Note as have limited experience with python there will be many cases where my code is inefficient and will need to clean it up
# A lot of issues trying to get formating right as committed notebook is different to interactive version

#Not able to access image in following method so used Ipython.display
# ****<img  src="../input/imagefilesforkagglesurvey/anatomyofkaggler.png" >



from IPython.display import Image
from IPython.core.display import HTML 
PATH = "../input/imagefilesforkagglesurvey/"
Image(filename = PATH + "anatomyofkaggler2.png")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline  

dfmain = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')
dfmain = dfmain.drop(dfmain.index[0]) # Remove the first line of text which has question text
dfmain['genderid'] = dfmain.groupby('Q1').ngroup().add(1)
dfmain['formaleducationid'] = dfmain.groupby('Q4').ngroup().add(1)
# Start with a wordcloud of most of information from multipleChoiceResponses.csv
# Main idea to be communicated is main function of data scientist is to get meaning from
# data using tools and process that can reproduce similar results. The main different from
# a data scientist with a normal scientist is that we are not trying to reproduce same results
# but acquire more meaning which in result able us to use the data more effectively i.e predictions


# So using story is start off with looking at all the data and the then acquired meaning to it in a systematic way


#Using wordcloud as gives reader understanding of type of responses in survey and not until
# able to drill down into questions and domain knowledge of what is a data scientist able to 
# get more meaning from data the finally be able to differentiate kagglers in terms of 
# what most data scientists uses compared to what is a good differntiater for data scientist


# Using very ineffeciant way to get data for wordcloud to get around outputs 
# was getting from from the function that didnt corrolate with data it was working with

import wordcloud
from wordcloud import WordCloud, STOPWORDS 
from collections import Counter
from os import path
from PIL import Image
import numpy as np
import os
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
thoughtbubble_mask = np.array(Image.open(path.join(d, "../input/imagefilesforkagglesurvey/Thought-Bubble-PNG2.png")))
explosion_mask = np.array(Image.open(path.join(d, "../input/imagefilesforkagglesurvey/explosion.png")))

comment_words = ' '
stopwords = set(STOPWORDS) 


# Identified what questions go in four main categories will be working with and adding all
# responses in different list. This occurs first as main wordcloud will use all responses bas
# on categories to be used.




stopwords = stopwords.union({'NAN', '0','-1', 'nan','1','etc.)','years', 'years)','and/or','I'})
dfwordmain = dfmain.copy()
dfwordmain['Q2'] = dfwordmain.apply(lambda row: 'age(' + row['Q2'] +')', axis = 1)
dfwordmain['Q8'] = dfwordmain.apply(lambda row: 'experienceRole(' + str(row['Q8']) +')', axis = 1)
dfwordmain['Q9'] = dfwordmain.apply(lambda row: 'income(' +str(row['Q9']) +')', axis = 1)
dfwordmain['Q25'] = dfwordmain.apply(lambda row: 'experienceML(' +str(row['Q25']) +')', axis = 1)
dfcloud = pd.DataFrame()
dfcloud['cloudtext'] = dfwordmain.iloc[:,[1,3,4,5,6,7,9,11,12,125,126,127,128,291,292,293,
        294,295,296,297,298,299,300,301,302,303,305,307,308,309,310,311,312,313,314,315,316,317,
                                          318,319,320,321,32,323,324,325,326]].stack().values


dfcloud['cloudtext'] = dfcloud.apply(lambda row: str(row['cloudtext']) , axis = 1)
my_list=dfcloud[dfcloud['cloudtext'] != '-1']['cloudtext'].tolist()
my_list2  = [model for word in my_list for model in word.split(' ')]
my_listPersonal = [x for x in my_list2 if x not in stopwords]  

dfwordmain = dfmain.copy()

dfcloud2 = pd.DataFrame()
dfcloud3 = pd.DataFrame()
dfcloud2['cloudtext'] = dfwordmain.iloc[:,21:108].stack().values
dfcloud3['cloudtext'] = dfwordmain.iloc[:,129:193].stack().values
dfcloud= pd.concat([dfcloud2, dfcloud3], ignore_index=True)
dfcloud['cloudtext'] = dfcloud.apply(lambda row: str(row['cloudtext']) , axis = 1)
my_list=dfcloud[dfcloud['cloudtext'] != '-1']['cloudtext'].tolist()
# Thought-Bubble-PNG2
my_list2  = [model for word in my_list for model in word.split(' ')]

# Stopword works randomly in wordcloud function so created on method of removing common words
my_listTools = [x for x in my_list2 if x not in stopwords]

dfwordmain = dfmain.copy()
dfcloud2 = pd.DataFrame()
dfcloud3 = pd.DataFrame()
dfcloud2['cloudtext'] = dfwordmain.iloc[:,109:124].stack().values
dfcloud3['cloudtext'] = dfwordmain.iloc[:,194:275].stack().values
dfcloud= pd.concat([dfcloud2, dfcloud3], ignore_index=True)

dfcloud['cloudtext'] = dfcloud.apply(lambda row: str(row['cloudtext']) , axis = 1)
my_list=dfcloud[dfcloud['cloudtext'] != '-1']['cloudtext'].tolist()

my_list2  = [model for word in my_list for model in word.split(' ')]
my_listData = [x for x in my_list2 if x not in stopwords]

dfwordmain = dfmain.copy()
dfcloud2 = pd.DataFrame()
dfcloud3 = pd.DataFrame()
dfcloud2['cloudtext'] = dfwordmain.iloc[:,[12,13,14,15,16,17,18,19,20,335,336,337,338,338,340,341,342,343,344,345,346,347,348,349,350,
                                          355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371]].stack().values
dfcloud3['cloudtext'] = dfwordmain.iloc[:,385:393].stack().values
dfcloud= pd.concat([dfcloud2, dfcloud3], ignore_index=True)

dfcloud['cloudtext'] = dfcloud.apply(lambda row: str(row['cloudtext']) , axis = 1)
my_list=dfcloud[dfcloud['cloudtext'] != '-1']['cloudtext'].tolist()

my_list2  = [model for word in my_list for model in word.split(' ')]
my_listModel = [x for x in my_list2 if x not in stopwords]

#Will put the above in a fuction if have enought time


my_listall = my_listPersonal.copy()
my_listall.extend(my_listTools)
my_listall.extend(my_listData)
my_listall.extend(my_listModel)

# Uses WordCould to create a good visual of what are the main words from the survey

word_could_dict=Counter(my_listall)


wordcloud = WordCloud(
                          background_color='white',                         
                          max_words=400,
                          max_font_size=None, 
                          repeat=False,
                          mask=explosion_mask ,
                          random_state=42,
                          stopwords=['that','the']
                         ).fit_words(word_could_dict)
# Had to use fit_words otherwise would only include a small subset of the words

plt.figure(figsize=(14,14))

# Thought carl jung quote reflects an important part data science in how using bayesion methods 
# and known priors enables us to make more accurate understanding/meaning/predictions 
# with the data at hand
plt.title('The least of things with a meaning is worth more in life than the greatest of things without it.– Carl Jung')
plt.imshow(wordcloud) #, interpolation="bilinear")
plt.axis('off')
plt.show()


# Nearly 3 million words that the world cloud is working with found using len(my_listall)


# Shaped the wordcloud as an explosion to represent the chaos of first  looking at data 
# with no real frame of reference
from IPython.display import Image
Image(filename = PATH + "predictdatascientist3.png")

# I consider the following infographic the most important story in this notebook


# Used the particular infographic as it included both a male and female which represent
# the results from analsys of the differences. 
# Used same format as first infographic to be consistent
# Main issue I have with this notebook is when to display this infographic 
#Five main categories based on my understanding of short term memory and presentations
# So grouped main areas in the five categories and looked for a base venn diagram using
# Google images. Found the one below and just edited out words and included categories
# and title for this notebook
 
Image(filename = PATH + "surveycategories.png", width=600)


# Converted main Q26 into values as original place was to create notebook the predicts if kaggler is a data scientist
# Using the most important features as the story. 
def update_typeofds(selfidentifytext):
   
  selfid = 0
  if selfidentifytext == 'Definitely not':
    selfid = 1
  if selfidentifytext == 'Probably not':
    selfid = 2
  if selfidentifytext == 'Maybe':
    selfid = 3
  if selfidentifytext == 'Probably yes':
    selfid = 4
  if selfidentifytext == 'Definitely yes':
    selfid = 5  
 
  
  return selfid

# After going through each of the question and dislaying a graph found it to be easier to use a function
# This is mainly for question broken up into parts and being able to include all question in one dataframe

def groupbydataforgraph(groupbyonetext, groupbytwotext,graphcolumnnames):
  
  dfreturn = df.groupby([groupbyonetext, groupbytwotext]).size().reset_index()
  dfreturn.columns = graphcolumnnames
  
  return dfreturn

#Key Fucntion to determine if identified as Data Scientist and what data can work with
def update_isdatascientist(selfidentifytext):
    selfid = 'Not a Data Scientist'
  
    if selfidentifytext == 'Probably yes':
        selfid = 'Identify as Data Scientist'
    if selfidentifytext == 'Definitely yes':
        selfid = 'Identify as Data Scientist'  
    if selfidentifytext == 'Probably not':
        selfid = 'Not a Data Scientist'
    if selfidentifytext == 'Definitely not':
        selfid = 'Not a Data Scientist'  
 
  
    return selfid


# Used for question 34 and 35. Feel like my limited knowledge of python
# was a big drawback in plotting responses to those questions. Hopefully later versions of 
# this notebook reflect my growth

def update_percentcat(percenttouse):
#     print(percenttouse)
    categorycheck = ''
    if not pd.isnull(percenttouse):
        checknumber = float(percenttouse)
#         print('checknumber',checknumber)
        categorycheck = '0'
        if (checknumber > 0) and (checknumber <= 20):
             categorycheck = '1-20'     
        if (checknumber > 20) and (checknumber <= 40):
             categorycheck = '21-40'   
        if (checknumber > 40) and (checknumber <= 60):
             categorycheck = '41-60'    
        if (checknumber > 61) and (checknumber <= 80):
             categorycheck = '61-80'               
        if (checknumber > 80) and (checknumber <= 100):
             categorycheck = '81-100' 
#         if (checknumber > 50) and (checknumber <= 75):
#              categorycheck = '51-75'     
#         if (checknumber > 75) and (checknumber <= 100):
#              categorycheck = '76-100'  
    return categorycheck



dfmain['selfidentifyid'] = dfmain.apply(lambda row: update_typeofds(row['Q26']), axis = 1)
dfmain['selfgroupid'] = dfmain.apply(lambda row: update_isdatascientist(row['Q26']), axis = 1)
dfmain['isstudentid'] = dfmain.apply(lambda x: 'Student' if x['Q6'] == 'Student' else 'Non Student', axis = 1)

dfmain['Q26'].fillna(value='No response', inplace = True)
dfidentify = dfmain.groupby(['Q26']).size().sort_values(ascending = False).reset_index()
dfidentify.columns = ['Identify', 'Count']
plt.figure(figsize=(9,6))
colors = ['grey','yellowgreen', 'gold', 'lightcoral', 'lightskyblue','blue']
explode = (0,0.12,0.2, 0, 0, 0)
ax = plt.pie(dfidentify['Count'], labels = dfidentify['Identify'], colors = colors,explode = explode,
            autopct='%1.1f%%', shadow=True, startangle=105)

df = dfmain[(dfmain['selfgroupid'] != 'Maybe')]
dfidentify = df.groupby(['selfgroupid']).size().sort_values(ascending = False).reset_index()
dfidentify.columns = ['Identify', 'Count']
plt.figure(figsize=(9,6))
colors = [ 'gold', 'lightskyblue']
explode = (0,0.12,0.2, 0, 0, 0)
ax = plt.pie(dfidentify['Count'], labels = dfidentify['Identify'], colors = colors,
            autopct='%1.1f%%', shadow=True, startangle=105)


# Make data in a readable format for plots. 
# As length of answers may be to long to display without overlapping need a way to wrap text, 
# I was not able to find a wrap function so wrapped the answers
# This occurrs for a lot of questions so the following code will be replaced once I find method for doing it all in one go

df['Q4'] = df['Q4'].str.wrap(20)
df['Q9'] = df['Q9'].str.wrap(30)
df['Q10'] = df['Q10'].str.wrap(30)

df['Q42_Part_4'] = df['Q42_Part_4'].str.wrap(30)


# word_could_dict=Counter(my_list)
word_could_dict=Counter(my_listPersonal)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=400,
                          max_font_size=None, 
                          repeat=False,
                          mask=thoughtbubble_mask ,
                          random_state=42
                         ).fit_words(word_could_dict)


plt.figure(figsize=(14,11))
plt.title('If you haven’t found it yet, keep looking. – Steve Jobs')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Wrapped number groupings aroud text i.e age(25-29 so reader can derive more meaning from the word cloud
# Shaped as a thought bubble over water

# Questions used for Personal information are 

# What is your gender? - Selected Choice
# What is your age (# years)?
# In which country do you currently reside?
# What is the highest level of formal education that you have attained or plan to attain within the next 2 years?
# Which best describes your undergraduate major? - Selected Choice
# How many years of experience do you have in your current role?
# What is your current yearly compensation (approximate $USD)?
# How long have you been writing code to analyze data?
# For how many years have you used machine learning methods (at work or in school)?
# On which online platforms have you begun or completed data science courses?
# On which online platform have you spent the most amount of time? - Selected Choice
# Who/what are your favorite media sources that report on data science topics?
# How do you perceive the quality of online learning platforms and in-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions? - Online learning platforms and MOOCs:
# How do you perceive the quality of online learning platforms and in-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions? - In-person bootcamps:
# Which better demonstrates expertise in data science: academic achievements or independent projects? - Your views:
# What barriers prevent you from making your work even easier to reuse and reproduce?


#Questions and variable names may not match categories as first exercies was go through each question and chart it
# Due to time constraits went straight from charting to adding information into an excel spreadsheet about main
# attributes found from each chart which is show in Results section
# Kaggle notebook is much harder to move cells around the jupyter notebook
# Actually started writing in Google Colab than moved to Jupyter notebook and than Kaggle, all good learning experience but time consuming


total = float(len(df))
#Just showing results between males and females to be more concise
dfdsbygender =  df[(df.genderid < 3)].groupby(['Q1','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfdsbygender.columns = ['Gender','Identify', 'Count']
dfdsbygender.head(20)
# dfdsbygender.groupby(level=[0]).apply(lambda x: x / x.sum())

plt.figure(figsize=(9,6))
plt.title('Self Identification by Gender')
ax = sns.barplot(y = dfdsbygender['Count'], x = dfdsbygender['Gender'], hue = dfdsbygender['Identify'])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 

# Chosen the top six so plot is more readable
dfdsbyeducation =  df[(df['Q4'] != 'I prefer not to answer')].groupby(['Q4', 'selfgroupid']).size().sort_values(ascending = False).reset_index().head(6)
dfdsbyeducation.columns = ['Education','Identify', 'Count']


plt.figure(figsize=(12,6))
plt.title('Self Identification by Education')
ax = sns.barplot(y = dfdsbyeducation['Count'], x = dfdsbyeducation['Education'], hue = dfdsbyeducation['Identify'])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 



dfdsbyeducation =  df[(df['Q4'] != 'I prefer not to answer')].groupby(['Q4', 'selfgroupid']).size().sort_values(ascending = False).head(6)
dfplot = dfdsbyeducation.groupby(level=[0]).apply(lambda x: x / x.sum()).reset_index()
dfplot.columns = ['Education','Identify', 'Percent']


plt.figure(figsize=(9,6))
plt.title('Self Identification as Percent of Type of Education')
ax = sns.barplot(y = dfplot['Percent'], x = dfplot['Education'], hue = dfplot['Identify'])
dfdoctor = df[(df['Q4'] == 'Doctoral degree') & (df.genderid < 3)].groupby(['selfgroupid','Q1']).size()
dfplot  = dfdoctor.groupby(level=[0]).apply(lambda x: x / x.sum()).reset_index()
dfplot.columns = ['Identify','Gender', 'Percent']
plt.figure(figsize=(9,6))
plt.title('Self Identification by Gender with Doctorates')
ax = sns.barplot(y = dfplot['Percent'], x = dfplot['Gender'], hue = dfplot['Identify'])
dfcountry = df.groupby(['Q3','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfcountry.columns = ['Country','Identify', 'Count']


plt.figure(figsize=(12,16))
plt.title('Self Identification by Country')
ax = sns.barplot(x = dfcountry['Count'], y = dfcountry['Country'], hue = dfcountry['Identify'])


dfcountry = df.groupby(['Q3','selfgroupid']).size().sort_values(ascending = False)
dfplot  = dfcountry.groupby(level=[0]).apply(lambda x: x / x.sum()).reset_index()
countriestouse = ['United States of America','India','Australia','China','Brazil','Canada', 'Japan','Singapore','Romania']

dfplot = dfplot[(dfplot['selfgroupid'] == 'Identify as Data Scientist') & (dfplot['Q3'].isin(countriestouse))]
dfplot.columns = ['Country','Identify', 'Percent']


plt.figure(figsize=(12,8))
plt.title('Percentage of Kagglers by Country that identify as Data Scientist')
ax = sns.barplot(y = dfplot['Percent'], x = dfplot['Country'])



employmentgroups = ['Data Scientist','Software Engineer','Student',
                    'Not employed','Consultant','Business Analyst',
                   'Data Engineer','Research Assistant','Manager ']

dfemployment = df.groupby(['Q6','selfgroupid']).size()
dfemployment   = dfemployment.groupby(level=[0]).apply(lambda x: x / x.sum()).rename('Percent').reset_index().sort_values('Percent', ascending = False)
dfplot  = dfemployment[(dfemployment['selfgroupid'] == 'Identify as Data Scientist')].head(5)
dfplot.columns = ['Employment', 'Identify', 'Percent']
plt.figure(figsize=(12,8))
plt.title('Percentage of Kagglers by Occupation that identify as Data Scientist')
ax = sns.barplot(y = dfplot['Percent'], x = dfplot['Employment'])
dfemployment = df.groupby(['Q6','selfgroupid']).size()
dfemployment   = dfemployment.groupby(level=[0]).apply(lambda x: x / x.sum()).rename('Percent').reset_index()
dfplot  = dfemployment[(dfemployment['selfgroupid'] == 'Identify as Data Scientist')].sort_values('Percent', ascending = True).head(5)
dfplot.columns = ['Employment', 'Identify', 'Percent']
plt.figure(figsize=(12,7))
plt.title('Percentage of Kagglers by Occupation that identify as Data Scientist')
ax = sns.barplot(y = dfplot['Percent'], x = dfplot['Employment'])
dfemployment = df.groupby(['Q2','selfgroupid']).size()
dfemployment   = dfemployment.groupby(level=[0]).apply(lambda x: x / x.sum()).rename('Percent').reset_index()
dfplot  = dfemployment[(dfemployment['selfgroupid'] == 'Identify as Data Scientist')].sort_values('Percent', ascending = False).head(5)
dfplot.columns = ['Employment', 'Identify', 'Percent']
plt.figure(figsize=(12,7))
plt.title('Percentage of Kagglers by Age that identify as Data Scientist')
ax = sns.barplot(y = dfplot['Percent'], x = dfplot['Employment'])
dfplot = df.groupby(['Q2','selfgroupid']).size().sort_values(ascending = False).reset_index()
# dfemployment   = dfemployment.groupby(level=[0]).apply(lambda x: x / x.sum()).rename('Percent').reset_index()
# dfplot  = dfemployment[(dfemployment['selfgroupid'] == 'Identify as Data Scientist')].sort_values(ascending = True).reset_index().head(5)
dfplot.columns = ['Age Group', 'Identify', 'Percent']
plt.figure(figsize=(12,7))
plt.title('Count of Kagglers by Age Group that identify as Data Scientist')
ax = sns.barplot(y = dfplot['Percent'], x = dfplot['Age Group'], hue = dfplot['Identify'])

dfemployment = df.groupby(['Q2','selfgroupid']).size()
dfemployment   = dfemployment.groupby(level=[0]).apply(lambda x: x / x.sum()).rename('Percent').reset_index()
dfplot  = dfemployment[(dfemployment['selfgroupid'] == 'Identify as Data Scientist')].sort_values('Percent', ascending = True).head(5)
dfplot.columns = ['Age Group', 'Identify', 'Percent']
plt.figure(figsize=(12,7))
plt.title('Percentage of Kagglers by Age Group that identify as Data Scientist')
ax = sns.barplot(y = dfplot['Percent'], x = dfplot['Age Group'])
dfrecommendedlanguage = df.groupby(['Q23','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['Percent Group', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('Approximately what percent of your time at work or school is spent actively coding?')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['Percent Group'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
dfrecommendedlanguage = df.groupby(['Q39_Part_1','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('Perceive the quality of online learning platforms and in-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions? ')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
dfplot = df.groupby(['Q9','selfgroupid']).size().sort_values(ascending = False).reset_index().head(28)
# dfemployment   = dfemployment.groupby(level=[0]).apply(lambda x: x / x.sum()).rename('Percent').reset_index()
# dfplot  = dfemployment[(dfemployment['selfgroupid'] == 'Identify as Data Scientist')].sort_values(ascending = True).reset_index().head(5)
dfplot.columns = ['Yearly Compensation', 'Identify', 'Percent']
plt.figure(figsize=(12,7))
plt.title('Count of Kagglers by Yearly Income that identify as Data Scientist')
ax = sns.barplot(x = dfplot['Percent'], y = dfplot['Yearly Compensation'], hue = dfplot['Identify'])
dfplot = df.groupby(['Q10','selfgroupid']).size().sort_values(ascending = False).reset_index().head(28)
# dfemployment   = dfemployment.groupby(level=[0]).apply(lambda x: x / x.sum()).rename('Percent').reset_index()
# dfplot  = dfemployment[(dfemployment['selfgroupid'] == 'Identify as Data Scientist')].sort_values(ascending = True).reset_index().head(5)
dfplot.columns = ['', 'Identify', 'Percent']
plt.figure(figsize=(12,7))
plt.title('Does your current employer incorporate machine learning methods into their business?')
ax = sns.barplot(x = dfplot['Percent'], y = dfplot[''], hue = dfplot['Identify'])
dfrecommendedlanguage = df.groupby(['Q37','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['Online Platform', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('On which online platform have you spent the most amount of time? ')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['Online Platform'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
workcolumns = ['', 'Identify', 'Count']
dfIDE = pd.concat([groupbydataforgraph('Q38_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q38_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q38_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q38_Part_4','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q38_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q38_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q38_Part_7','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q38_Part_8','selfgroupid',workcolumns),
                   groupbydataforgraph('Q38_Part_9','selfgroupid',workcolumns),
                   groupbydataforgraph('Q38_Part_10','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q38_Part_11','selfgroupid',workcolumns),
                   groupbydataforgraph('Q38_Part_12','selfgroupid',workcolumns),
                   groupbydataforgraph('Q38_Part_13','selfgroupid',workcolumns),
                   groupbydataforgraph('Q38_Part_14','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q38_Part_15','selfgroupid',workcolumns),
                   groupbydataforgraph('Q38_Part_16','selfgroupid',workcolumns),
                   groupbydataforgraph('Q38_Part_17','selfgroupid',workcolumns),
                   groupbydataforgraph('Q38_Part_18','selfgroupid',workcolumns),                   
#                    groupbydataforgraph('Q38_Part_19','selfgroupid',workcolumns),
#                    groupbydataforgraph('Q38_Part_20','selfgroupid',workcolumns),
#                    groupbydataforgraph('Q38_Part_21','selfgroupid',workcolumns)              
                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('Who/what are your favorite media sources that report on data science topics?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE[''], hue = dfIDE['Identify'])
dfrecommendedlanguage = df.groupby(['Q39_Part_2','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('How do you perceive the quality of online learning platforms and in-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions?')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage[''], hue = dfrecommendedlanguage['Identify'])
df['Q24'] = df['Q24'].str.wrap(20)

dfrecommendedlanguage = df.groupby(['Q24','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('How long have you been writing code to analyze data?')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
df['Q40'] = df['Q40'].str.wrap(30)
dfrecommendedlanguage = df.groupby(['Q40','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('Which better demonstrates expertise in data science: academic achievements or independent projects?')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
df['Q35_Part_1cat'] = df.apply(lambda row: update_percentcat(row['Q35_Part_1']), axis = 1)
df['Q35_Part_2cat'] = df.apply(lambda row: update_percentcat(row['Q35_Part_2']), axis = 1)
df['Q35_Part_3cat'] = df.apply(lambda row: update_percentcat(row['Q35_Part_3']), axis = 1)
df['Q35_Part_4cat'] = df.apply(lambda row: update_percentcat(row['Q35_Part_4']), axis = 1)
df['Q35_Part_5cat'] = df.apply(lambda row: update_percentcat(row['Q35_Part_5']), axis = 1)
df['Q35_Part_6cat'] = df.apply(lambda row: update_percentcat(row['Q35_Part_6']), axis = 1)

dfInfluenceProducts = df.groupby(['Q35_Part_1cat','selfgroupid']).size().reset_index()
dfInfluenceProducts.columns = ['Percent', 'Identify', 'Count']
dfInfluenceProducts['Category'] = 'Self-Taught'
dfWorkflows = df.groupby(['Q35_Part_2cat','selfgroupid']).size().reset_index()
dfWorkflows.columns = ['Percent', 'Identify', 'Count']
dfWorkflows['Category'] = 'Online'
dfinfrastructure = df.groupby(['Q35_Part_3cat','selfgroupid']).size().reset_index()
dfinfrastructure.columns = ['Percent', 'Identify', 'Count']
dfinfrastructure['Category'] = 'Work'
dfprototype = df.groupby(['Q35_Part_4cat','selfgroupid']).size().reset_index()
dfprototype.columns = ['Percent', 'Identify', 'Count']
dfprototype['Category'] = 'University'
dfresearch = df.groupby(['Q35_Part_5cat','selfgroupid']).size().reset_index()
dfresearch.columns = ['Percent', 'Identify', 'Count']
dfresearch['Category'] = 'Kaggle Competitions'
dfnoneactivities = df.groupby(['Q35_Part_6cat','selfgroupid']).size().reset_index()
dfnoneactivities.columns = ['Percent', 'Identify', 'Count']
dfnoneactivities['Category'] = 'Other'


dfrole = pd.concat([dfWorkflows,  dfprototype, dfInfluenceProducts,  dfresearch, dfinfrastructure,  dfnoneactivities], ignore_index=True)
dfrole2 = dfrole[(dfrole['Percent'] != '')]


plt.figure(figsize=(11,7))
plt.title('What percentage of your current machine learning/data science training falls under each category?')


ax = sns.lineplot(y = dfrole2['Count'], x = dfrole2['Percent'], hue = dfrole2['Category'])
plt.figure(figsize=(11,7))
# plt.title('Select any activities that make up an important part of your role at work')
g = sns.catplot(x="Category", y="Count",
                hue="Percent", col="Identify",
                 data=dfrole2, kind="point",
                height=6,  aspect=1.6);

# Questions relating to modelling are
# Select any activities that make up an important part of your role at work:
# Approximately what percent of your time at work or school is spent actively coding?
# During a typical data science project at work or school, approximately what proportion of your time is devoted to the following?
# What percentage of your current machine learning/data science training falls under each category?
# What metrics do you or your organization use to determine whether or not your models were successful?
# In what circumstances would you explore model insights and interpret your model's predictions? 
# Approximately what percent of your data projects involve exploring model insights?
# What methods do you prefer for explaining and/or interpreting decisions that are made by ML models?
# Do you consider ML models to be "black boxes" with outputs that are difficult or impossible to explain?


word_could_dict=Counter(my_listModel)


wordcloud = WordCloud(
                          background_color='white',
                          
                          max_words=250,
                          max_font_size=None, 
                          repeat=False,
                          mask=thoughtbubble_mask ,
                          random_state=42,
                          stopwords=['that','the']
                         ).fit_words(word_could_dict)


plt.figure(figsize=(11,11))
plt.imshow(wordcloud) #, interpolation="bilinear")
plt.axis('off')
plt.show()


df['Q11_Part_1'] = df['Q11_Part_1'].str.wrap(30)
df['Q11_Part_2'] = df['Q11_Part_2'].str.wrap(30)
df['Q11_Part_3'] = df['Q11_Part_3'].str.wrap(30)
df['Q11_Part_4'] = df['Q11_Part_4'].str.wrap(30)
df['Q11_Part_5'] = df['Q11_Part_5'].str.wrap(30)
df['Q11_Part_6'] = df['Q11_Part_6'].str.wrap(30)

dfInfluenceProducts = df.groupby(['Q11_Part_1','selfgroupid']).size().reset_index()
dfInfluenceProducts.columns = ['Role', 'Identify', 'Count']
dfWorkflows = df.groupby(['Q11_Part_2','selfgroupid']).size().reset_index()
dfWorkflows.columns = ['Role', 'Identify', 'Count']
dfinfrastructure = df.groupby(['Q11_Part_3','selfgroupid']).size().reset_index()
dfinfrastructure.columns = ['Role', 'Identify', 'Count']
dfprototype = df.groupby(['Q11_Part_4','selfgroupid']).size().reset_index()
dfprototype.columns = ['Role', 'Identify', 'Count']
dfresearch = df.groupby(['Q11_Part_5','selfgroupid']).size().reset_index()
dfresearch.columns = ['Role', 'Identify', 'Count']
dfnoneactivities = df.groupby(['Q11_Part_6','selfgroupid']).size().reset_index()
dfnoneactivities.columns = ['Role', 'Identify', 'Count']


dfrole = pd.concat([dfWorkflows,  dfprototype, dfInfluenceProducts,  dfresearch, dfinfrastructure,  dfnoneactivities], ignore_index=True)

plt.figure(figsize=(11,9))
plt.title('Select any activities that make up an important part of your role at work')


ax = sns.barplot(x = dfrole['Count'], y = dfrole['Role'], hue = dfrole['Identify'])


workcolumns = ['default', 'Identify', 'Count']
dfIDE = pd.concat([groupbydataforgraph('Q42_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q42_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q42_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q42_Part_4','selfgroupid',workcolumns)            
                   ], ignore_index=True)


plt.figure(figsize=(12,8))
plt.title('What metrics do you or your organization use to determine whether or not your models were successful?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])
dfrecommendedlanguage = df.groupby(['Q41_Part_3','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']


plt.figure(figsize=(8,7))
plt.title('Importance of Reproducibility in data science')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)

df['Q49_Part_1'] = df['Q49_Part_1'].str.wrap(30)
df['Q49_Part_2'] = df['Q49_Part_2'].str.wrap(30)
df['Q49_Part_3'] = df['Q49_Part_3'].str.wrap(30)
df['Q49_Part_4'] = df['Q49_Part_4'].str.wrap(30)
df['Q49_Part_5'] = df['Q49_Part_5'].str.wrap(30)
df['Q49_Part_11'] = df['Q49_Part_11'].str.wrap(30)






workcolumns = ['default', 'Identify', 'Count']
dfIDE = pd.concat([groupbydataforgraph('Q49_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q49_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q49_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q49_Part_4','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q49_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q49_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q49_Part_7','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q49_Part_8','selfgroupid',workcolumns),
                   groupbydataforgraph('Q49_Part_9','selfgroupid',workcolumns),
                   groupbydataforgraph('Q49_Part_10','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q49_Part_11','selfgroupid',workcolumns)                  

                   ], ignore_index=True)



plt.figure(figsize=(12,12))
plt.title('What tools and methods do you use to make your work easy to reproduce? ')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])
dfrecommendedlanguage = df.groupby(['Q46','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('Approximately what percent of your data projects involve exploring model insights?')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
df['Q45_Part_1'] = df['Q45_Part_1'].str.wrap(30)
df['Q45_Part_2'] = df['Q45_Part_2'].str.wrap(30)
df['Q45_Part_3'] = df['Q45_Part_3'].str.wrap(30)
df['Q45_Part_4'] = df['Q45_Part_4'].str.wrap(30)
df['Q45_Part_5'] = df['Q45_Part_5'].str.wrap(30)
df['Q45_Part_6'] = df['Q45_Part_6'].str.wrap(30)



workcolumns = ['default', 'Identify', 'Count']
dfIDE = pd.concat([groupbydataforgraph('Q45_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q45_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q45_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q45_Part_4','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q45_Part_5','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q45_Part_6','selfgroupid',workcolumns) 
                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('In what circumstances would you explore model insights and interpret your models predictions? ')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])
df['Q50_Part_3'] = df['Q50_Part_3'].str.wrap(30)
df['Q50_Part_4'] = df['Q50_Part_4'].str.wrap(30)
df['Q50_Part_5'] = df['Q50_Part_5'].str.wrap(30)
df['Q50_Part_6'] = df['Q50_Part_6'].str.wrap(30)

workcolumns = ['default', 'Identify', 'Count']
dfIDE = pd.concat([groupbydataforgraph('Q50_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q50_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q50_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q50_Part_4','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q50_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q50_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q50_Part_7','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q50_Part_8','selfgroupid',workcolumns)                 

                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('What barriers prevent you from making your work even easier to reuse and reproduce?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])
df['Q34_Part_1cat'] = df.apply(lambda row: update_percentcat(row['Q34_Part_1']), axis = 1)
df['Q34_Part_2cat'] = df.apply(lambda row: update_percentcat(row['Q34_Part_2']), axis = 1)
df['Q34_Part_3cat'] = df.apply(lambda row: update_percentcat(row['Q34_Part_3']), axis = 1)
df['Q34_Part_4cat'] = df.apply(lambda row: update_percentcat(row['Q34_Part_4']), axis = 1)
df['Q34_Part_5cat'] = df.apply(lambda row: update_percentcat(row['Q34_Part_5']), axis = 1)
df['Q34_Part_6cat'] = df.apply(lambda row: update_percentcat(row['Q34_Part_6']), axis = 1)


dfInfluenceProducts = df.groupby(['Q34_Part_1cat','selfgroupid']).size().reset_index()
dfInfluenceProducts.columns = ['Percent', 'Identify', 'Count']
dfInfluenceProducts['Category'] = 'Gathering Data'
dfWorkflows = df.groupby(['Q34_Part_2cat','selfgroupid']).size().reset_index()
dfWorkflows.columns = ['Percent', 'Identify', 'Count']
dfWorkflows['Category'] = 'Cleaning Data'
dfinfrastructure = df.groupby(['Q34_Part_3cat','selfgroupid']).size().reset_index()
dfinfrastructure.columns = ['Percent', 'Identify', 'Count']
dfinfrastructure['Category'] = 'Visualizing data'
dfprototype = df.groupby(['Q34_Part_4cat','selfgroupid']).size().reset_index()
dfprototype.columns = ['Percent', 'Identify', 'Count']
dfprototype['Category'] = 'Model Selection'
dfresearch = df.groupby(['Q34_Part_5cat','selfgroupid']).size().reset_index()
dfresearch.columns = ['Percent', 'Identify', 'Count']
dfresearch['Category'] = 'Model into production'
dfnoneactivities = df.groupby(['Q34_Part_6cat','selfgroupid']).size().reset_index()
dfnoneactivities.columns = ['Percent', 'Identify', 'Count']
dfnoneactivities['Category'] = 'Finding Insights'


dfrole = pd.concat([dfWorkflows,  dfprototype, dfInfluenceProducts,  dfresearch, dfinfrastructure,  dfnoneactivities], ignore_index=True)
dfrole2 = dfrole[(dfrole['Percent'] != '')]


plt.figure(figsize=(11,7))
plt.title('Select any activities that make up an important part of your role at work')


ax = sns.lineplot(y = dfrole2['Count'], x = dfrole2['Percent'], hue = dfrole2['Category'])
dfplot = df.groupby(['Q34_Part_1','selfgroupid']).size().reset_index()
dfplot.columns = ['Percent', 'Identify', 'Count']

plt.figure(figsize=(11,7))
# plt.title('Select any activities that make up an important part of your role at work')
g = sns.catplot(x="Percent", y="Count",
                hue="Category", col="Identify",
                 data=dfrole2, kind="bar",
                height=6,  aspect=1.6);

# What is the primary tool that you use at work or school to analyze data? 
# Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? 
# Which of the following hosted notebooks have you used at work or school in the last 5 years? 
# Which of the following cloud computing services have you used at work or school in the last 5 years?
# Which of the following cloud computing services have you used at work or school in the last 5 years?
# What specific programming language do you use most often? - Selected Choice
# What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice
# Which of the following cloud computing products have you used at work or school in the last 5 years 
# What tools and methods do you use to make your work easy to reproduce? 


word_could_dict=Counter(my_listTools)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=250,
                          max_font_size=None, 
                          repeat=False,
                          mask=thoughtbubble_mask ,
                          random_state=42
                         ).fit_words(word_could_dict)


plt.figure(figsize=(11,14))
plt.title('Word Cloud of Answers relating to Tools/Skills Questions')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
df['Q12_MULTIPLE_CHOICE'] = df['Q12_MULTIPLE_CHOICE'].str.wrap(30)
dfprimarytool = df[(df['Q12_MULTIPLE_CHOICE'] != 'Other')].groupby(['Q12_MULTIPLE_CHOICE','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfprimarytool.columns = ['PrimaryTool', 'Identify', 'Count']

plt.figure(figsize=(17,7))
plt.title('What is the primary tool that you use at work or school to analyze data?')
ax = sns.barplot(y = dfprimarytool['Count'], x = dfprimarytool['PrimaryTool'], hue = dfprimarytool['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)

workcolumns = ['IDE', 'Identify', 'Count']

dfIDE = pd.concat([groupbydataforgraph('Q13_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q13_Part_2','selfgroupid',workcolumns),
                   groupbydataforgraph('Q13_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q13_Part_13','selfgroupid',workcolumns),               
                   groupbydataforgraph('Q13_Part_6','selfgroupid',workcolumns),
                   groupbydataforgraph('Q13_Part_7','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q13_Part_9','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q13_Part_10','selfgroupid',workcolumns),
                   groupbydataforgraph('Q13_Part_11','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q13_Part_12','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q13_Part_4','selfgroupid',workcolumns),
                   groupbydataforgraph('Q13_Part_8','selfgroupid',workcolumns) 
                    
                   
                   
                   ], ignore_index=True)



plt.figure(figsize=(11,9))
plt.title('Which of the following integrated development environments (IDEs) have you used at work or school in the last 5 years?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['IDE'], hue = dfIDE['Identify'])
workcolumns = ['Hosted Notebooks', 'Identify', 'Count']

dfIDE = pd.concat([groupbydataforgraph('Q14_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q14_Part_2','selfgroupid',workcolumns),
                   groupbydataforgraph('Q14_Part_3','selfgroupid',workcolumns), 
#                    groupbydataforgraph('Q14_Part_4','selfgroupid',workcolumns),               
                   groupbydataforgraph('Q14_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q14_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q14_Part_7','selfgroupid',workcolumns),                      
#                    groupbydataforgraph('Q14_Part_8','selfgroupid',workcolumns),
                   groupbydataforgraph('Q14_Part_9','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q14_Part_10','selfgroupid',workcolumns)
                   
                   
                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('Which of the following hosted notebooks have you used at work or school in the last 5 years? ')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['Hosted Notebooks'], hue = dfIDE['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=33)

workcolumns = ['Cloud', 'Identify', 'Count']

dfIDE = pd.concat([groupbydataforgraph('Q15_Part_2','selfgroupid',workcolumns),
                   groupbydataforgraph('Q15_Part_1','selfgroupid',workcolumns), 
                   
                   groupbydataforgraph('Q15_Part_3','selfgroupid',workcolumns), 
#                    groupbydataforgraph('Q15_Part_4','selfgroupid',workcolumns),               
                   groupbydataforgraph('Q15_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q15_Part_6','selfgroupid',workcolumns) 

                   
                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('Which of the following cloud computing services have you used at work or school in the last 5 years?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['Cloud'], hue = dfIDE['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=33)
workcolumns = ['default', 'Identify', 'Count']

dfIDE = pd.concat([                   groupbydataforgraph('Q27_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q27_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q27_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q27_Part_4','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q27_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q27_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q27_Part_7','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q27_Part_8','selfgroupid',workcolumns),
                   groupbydataforgraph('Q27_Part_9','selfgroupid',workcolumns),
                   groupbydataforgraph('Q27_Part_10','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q27_Part_11','selfgroupid',workcolumns),
                   groupbydataforgraph('Q27_Part_12','selfgroupid',workcolumns),
                   groupbydataforgraph('Q27_Part_13','selfgroupid',workcolumns),
                   groupbydataforgraph('Q27_Part_14','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q27_Part_15','selfgroupid',workcolumns),
                   groupbydataforgraph('Q27_Part_16','selfgroupid',workcolumns),
                   groupbydataforgraph('Q27_Part_17','selfgroupid',workcolumns),
                   groupbydataforgraph('Q27_Part_18','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q27_Part_19','selfgroupid',workcolumns),
                   groupbydataforgraph('Q27_Part_20','selfgroupid',workcolumns)                
                   ], ignore_index=True)



plt.figure(figsize=(12,14))
plt.title('Which of the following cloud computing products have you used at work or school in the last 5 years')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])
workcolumns = ['Programming', 'Identify', 'Count']

dfIDE = pd.concat([groupbydataforgraph('Q16_Part_2','selfgroupid',workcolumns),
                   groupbydataforgraph('Q16_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q16_Part_4','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q16_Part_3','selfgroupid',workcolumns), 

                   groupbydataforgraph('Q16_Part_9','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q16_Part_10','selfgroupid',workcolumns),
#                    groupbydataforgraph('Q16_Part_11','selfgroupid',workcolumns), 
#                    groupbydataforgraph('Q16_Part_12','selfgroupid',workcolumns),                     
#                    groupbydataforgraph('Q16_Part_15','selfgroupid',workcolumns),
                   groupbydataforgraph('Q16_Part_14','selfgroupid',workcolumns),
                   groupbydataforgraph('Q16_Part_7','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q16_Part_16','selfgroupid',workcolumns),
                   groupbydataforgraph('Q16_Part_13','selfgroupid',workcolumns),
                   groupbydataforgraph('Q16_Part_8','selfgroupid',workcolumns),    
               
                   groupbydataforgraph('Q16_Part_6','selfgroupid',workcolumns),                    
               
                   groupbydataforgraph('Q16_Part_5','selfgroupid',workcolumns)                    
                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('What programming languages do you use on a regular basis?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['Programming'], hue = dfIDE['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=43)
dfprimarylanguage = df.groupby(['Q17','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfprimarylanguage.columns = ['Primarylanguage', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('What specific programming language do you use most often? ')
ax = sns.barplot(x = dfprimarylanguage['Count'], y = dfprimarylanguage['Primarylanguage'], hue = dfprimarylanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)

dfrecommendedlanguage = df.groupby(['Q18','selfgroupid']).size().sort_values(ascending = False).reset_index().head(8)
dfrecommendedlanguage.columns = ['Recommendedlanguage', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('What programming language would you recommend an aspiring data scientist to learn first?')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['Recommendedlanguage'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)


# Questions relating to Machine Learning
# What machine learning frameworks have you used in the past 5 years?
# Of the choices that you selected in the previous question, which ML library have you used the most?
# Does your current employer incorporate machine learning methods into their business?
# Which of the following machine learning products have you used at work or school in the last 5 years? 
# What percentage of your current machine learning/data science training falls under each category? 
# How do you perceive the importance of the following topics? 
# What do you find most difficult about ensuring that your algorithms are fair and unbiased? 

workcolumns = ['Framework', 'Identify', 'Count']

dfIDE = pd.concat([groupbydataforgraph('Q19_Part_2','selfgroupid',workcolumns),
                   groupbydataforgraph('Q19_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q19_Part_4','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q19_Part_3','selfgroupid',workcolumns), 

                   groupbydataforgraph('Q19_Part_9','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q19_Part_10','selfgroupid',workcolumns),
                   groupbydataforgraph('Q19_Part_11','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q19_Part_12','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q19_Part_15','selfgroupid',workcolumns),
                   groupbydataforgraph('Q19_Part_14','selfgroupid',workcolumns),
                   groupbydataforgraph('Q19_Part_7','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q19_Part_16','selfgroupid',workcolumns),
                   groupbydataforgraph('Q19_Part_13','selfgroupid',workcolumns),
                   groupbydataforgraph('Q19_Part_8','selfgroupid',workcolumns),    
               
                   groupbydataforgraph('Q19_Part_6','selfgroupid',workcolumns),                    
               
                   groupbydataforgraph('Q19_Part_5','selfgroupid',workcolumns)                    
                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('What machine learning frameworks have you used in the past 5 years?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['Framework'], hue = dfIDE['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=43)
dfrecommendedlanguage = df.groupby(['Q20','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('Of the choices that you selected in the previous question, which ML library have you used the most?')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)

# Group up the answers into different questions types



dfrecommendedlanguage = df.groupby(['Q41_Part_1','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']


plt.figure(figsize=(8,7))
plt.title('Importance of Fairness and bias in ML algorithms')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
df['Q25'] = df['Q25'].str.wrap(20)
dfrecommendedlanguage = df.groupby(['Q25','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('For how many years have you used machine learning methods (at work or in school)?')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
workcolumns = ['default', 'Identify', 'Count']
dfIDE = pd.concat([ groupbydataforgraph('Q28_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q28_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q28_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q28_Part_4','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q28_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q28_Part_7','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q28_Part_8','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_9','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_10','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q28_Part_11','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_12','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_13','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_14','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q28_Part_15','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_16','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_17','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_18','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q28_Part_19','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_20','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_21','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q28_Part_22','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q28_Part_23','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q28_Part_24','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q28_Part_25','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_26','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q28_Part_27','selfgroupid',workcolumns),  
                   groupbydataforgraph('Q28_Part_28','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q28_Part_29','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q28_Part_30','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_31','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q28_Part_32','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q28_Part_33','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q28_Part_34','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q28_Part_35','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_36','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q28_Part_37','selfgroupid',workcolumns),  
                   groupbydataforgraph('Q28_Part_38','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_39','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_40','selfgroupid',workcolumns),
                   groupbydataforgraph('Q28_Part_41','selfgroupid',workcolumns),                 
                   ], ignore_index=True)



plt.figure(figsize=(12,14))
plt.title('Which of the following machine learning products have you used at work or school in the last 5 years?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])
dfrecommendedlanguage = df.groupby(['Q43','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('Approximately what percent of your data projects involved exploring unfair bias in the dataset and/or algorithm?')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
df['Q44_Part_1'] = df['Q44_Part_1'].str.wrap(30)
df['Q44_Part_2'] = df['Q44_Part_2'].str.wrap(30)
df['Q44_Part_3'] = df['Q44_Part_3'].str.wrap(30)
df['Q44_Part_4'] = df['Q44_Part_4'].str.wrap(30)
df['Q44_Part_5'] = df['Q44_Part_5'].str.wrap(30)
df['Q44_Part_6'] = df['Q44_Part_6'].str.wrap(30)


workcolumns = ['default', 'Identify', 'Count']
dfIDE = pd.concat([groupbydataforgraph('Q44_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q44_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q44_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q44_Part_4','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q44_Part_5','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q44_Part_6','selfgroupid',workcolumns) 
                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('What do you find most difficult about ensuring that your algorithms are fair and unbiased? ')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])

df['Q48'] = df['Q48'].str.wrap(30)

dfrecommendedlanguage = df.groupby(['Q48','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('Do you consider ML models to be "black boxes" with outputs that are difficult or impossible to explain?')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
dfrecommendedlanguage = df.groupby(['Q41_Part_2','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']

plt.figure(figsize=(8,7))
plt.title('Importance of Being able to explain ML model outputs and/or predictions')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
workcolumns = ['default', 'Identify', 'Count']
dfIDE = pd.concat([groupbydataforgraph('Q47_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q47_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q47_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q47_Part_4','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q47_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q47_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q47_Part_7','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q47_Part_8','selfgroupid',workcolumns),
                   groupbydataforgraph('Q47_Part_9','selfgroupid',workcolumns),
                   groupbydataforgraph('Q47_Part_10','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q47_Part_11','selfgroupid',workcolumns),
                   groupbydataforgraph('Q47_Part_12','selfgroupid',workcolumns),
                   groupbydataforgraph('Q47_Part_13','selfgroupid',workcolumns),
                   groupbydataforgraph('Q47_Part_14','selfgroupid',workcolumns)                   

                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('What methods do you prefer for explaining and/or interpreting decisions that are made by ML models? ')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])
word_could_dict=Counter(my_listData)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=250,
                          max_font_size=None, 
                          repeat=False,
                          mask=thoughtbubble_mask ,
                          random_state=42
                         ).fit_words(word_could_dict)


plt.figure(figsize=(11,11))
plt.title('If we have data, let’s look at data. If all we have are opinions, let’s go with mine. – Jim Barksdale,')

plt.imshow(wordcloud) #, interpolation="bilinear")
plt.axis('off')
plt.show()
# Questions relating to Data
# What data visualization libraries or tools have you used in the past 5 years?
# Which of the following relational database products have you used at work or school in the last 5 years? 
# Which of the following big data and analytics products have you used at work or school in the last 5 years?
# Which types of data do you currently interact with most often at work or school?
# Where do you find public datasets?
# Approximately what percent of your data projects involved exploring unfair bias in the dataset and/or algorithm?

workcolumns = ['default', 'Identify', 'Count']

dfIDE = pd.concat([groupbydataforgraph('Q21_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q21_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q21_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q21_Part_4','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q21_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q21_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q21_Part_7','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q21_Part_8','selfgroupid',workcolumns),
                   groupbydataforgraph('Q21_Part_9','selfgroupid',workcolumns),
                   groupbydataforgraph('Q21_Part_10','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q21_Part_11','selfgroupid',workcolumns),
                   groupbydataforgraph('Q21_Part_12','selfgroupid',workcolumns),
                   groupbydataforgraph('Q21_Part_13','selfgroupid',workcolumns)                    
                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('What Data visualization libraries or tools have you used in the past 5 years? ')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])

dfrecommendedlanguage = df.groupby(['Q22','selfgroupid']).size().sort_values(ascending = False).reset_index()
dfrecommendedlanguage.columns = ['default', 'Identify', 'Count']

plt.figure(figsize=(11,9))
plt.title('Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most?')
ax = sns.barplot(x = dfrecommendedlanguage['Count'], y = dfrecommendedlanguage['default'], hue = dfrecommendedlanguage['Identify'])
# plt.setp(ax.get_xticklabels(), rotation=75)
workcolumns = ['default', 'Identify', 'Count']
dfIDE = pd.concat([groupbydataforgraph('Q29_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q29_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q29_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q29_Part_4','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q29_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q29_Part_7','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q29_Part_8','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_9','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_10','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q29_Part_11','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_12','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_13','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_14','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q29_Part_15','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_16','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_17','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_18','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q29_Part_19','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_20','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_21','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q29_Part_22','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q29_Part_23','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q29_Part_24','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q29_Part_25','selfgroupid',workcolumns),
                   groupbydataforgraph('Q29_Part_26','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q29_Part_27','selfgroupid',workcolumns)                  
                   ], ignore_index=True)



plt.figure(figsize=(12,14))
plt.title('Which of the following relational database products have you used at work or school in the last 5 years?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])

workcolumns = ['default', 'Identify', 'Count']
dfIDE = pd.concat([groupbydataforgraph('Q30_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q30_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q30_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q30_Part_4','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q30_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q30_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q30_Part_7','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q30_Part_8','selfgroupid',workcolumns),
                   groupbydataforgraph('Q30_Part_9','selfgroupid',workcolumns),
                   groupbydataforgraph('Q30_Part_10','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q30_Part_11','selfgroupid',workcolumns),
                   groupbydataforgraph('Q30_Part_12','selfgroupid',workcolumns),
                   groupbydataforgraph('Q30_Part_13','selfgroupid',workcolumns),
                   groupbydataforgraph('Q30_Part_14','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q30_Part_15','selfgroupid',workcolumns),
                   groupbydataforgraph('Q30_Part_16','selfgroupid',workcolumns),
                   groupbydataforgraph('Q30_Part_17','selfgroupid',workcolumns),
                   groupbydataforgraph('Q30_Part_18','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q30_Part_19','selfgroupid',workcolumns),
                   groupbydataforgraph('Q30_Part_20','selfgroupid',workcolumns),
                   groupbydataforgraph('Q30_Part_21','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q30_Part_22','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q30_Part_23','selfgroupid',workcolumns)                 
                   ], ignore_index=True)



plt.figure(figsize=(12,14))
plt.title('Which of the following big data and analytics products have you used at work or school in the last 5 years?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])

workcolumns = ['default', 'Identify', 'Count']
dfIDE = pd.concat([groupbydataforgraph('Q31_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q31_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q31_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q31_Part_4','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q31_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q31_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q31_Part_7','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q31_Part_8','selfgroupid',workcolumns),
                   groupbydataforgraph('Q31_Part_9','selfgroupid',workcolumns),
                   groupbydataforgraph('Q31_Part_10','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q31_Part_11','selfgroupid',workcolumns)                 
                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('Which types of data do you currently interact with most often at work or school?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])

df['Q33_Part_4'] = df['Q33_Part_4'].str.wrap(30)
df['Q33_Part_5'] = df['Q33_Part_5'].str.wrap(30)
df['Q33_Part_6'] = df['Q33_Part_6'].str.wrap(30)

workcolumns = ['default', 'Identify', 'Count']
dfIDE = pd.concat([groupbydataforgraph('Q33_Part_1','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q33_Part_2','selfgroupid',workcolumns),                   
                   groupbydataforgraph('Q33_Part_3','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q33_Part_4','selfgroupid',workcolumns),                      
                   groupbydataforgraph('Q33_Part_5','selfgroupid',workcolumns),
                   groupbydataforgraph('Q33_Part_6','selfgroupid',workcolumns), 
                   groupbydataforgraph('Q33_Part_7','selfgroupid',workcolumns),                     
                   groupbydataforgraph('Q33_Part_8','selfgroupid',workcolumns),
                   groupbydataforgraph('Q33_Part_9','selfgroupid',workcolumns),
                   groupbydataforgraph('Q33_Part_10','selfgroupid',workcolumns)               
                   ], ignore_index=True)



plt.figure(figsize=(12,8))
plt.title('Where do you find public datasets?')
ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])


# maybe do a violin for 34 and 35


workcolumns = ['percentwork', 'Identify', 'Count']

dfgatheringdata =  groupbydataforgraph('Q34_Part_1','selfgroupid',workcolumns)
dfgatheringdata['typework'] = 'GatheringData'
dfcleaningdata =  groupbydataforgraph('Q34_Part_2','selfgroupid',workcolumns)
dfcleaningdata['typework'] = 'CleaningData'

dfgatheringdata.head(20)
dfIDE = pd.concat([ dfgatheringdata, dfcleaningdata     ], ignore_index=True)

# with sns.axes_style(style=None):
#     sns.violinplot("age_dec", "split_frac", hue="gender", data=data,
#                    split=True, inner="quartile",
#                    palette=["lightblue", "lightpink"]);

# plt.figure(figsize=(12,8))
# ax = sns.barplot(x = dfIDE['Count'], y = dfIDE['default'], hue = dfIDE['Identify'])

  

dfGender = df[((df['Q1'] == 'Male') & (df['selfgroupid'] == 'Identify as Data Scientist'))].groupby(['Q1','selfgroupid']).size().reset_index()
dfGender.columns = ['default', 'Identify', 'Count']
dfGender['Category'] = 'Personal'
dfAge = df[((df['Q2'] == '25-29') & (df['selfgroupid'] == 'Identify as Data Scientist'))].groupby(['Q2','selfgroupid']).size().reset_index()
dfAge.columns = ['default', 'Identify', 'Count']
dfAge['Category'] = 'Personal'
dfAge.iloc[0,0] = 'Age 25-29'
dfCountry = df[((df['Q3'] == 'United States of America') & (df['selfgroupid'] == 'Identify as Data Scientist'))].groupby(['Q3','selfgroupid']).size().reset_index()
dfCountry.columns = ['default', 'Identify', 'Count']
dfCountry['Category'] = 'Personal'
dfeducation = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q4','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfeducation.columns = ['default', 'Identify', 'Count']
dfeducation['Category'] = 'Personal'
dfemployed = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q6','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfemployed.columns = ['default', 'Identify', 'Count']
dfemployed['Category'] = 'Personal'
dfemployed.iloc[0,0] = 'Employed as Data Scientist'
dfcoding = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q23','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfcoding.columns = ['default', 'Identify', 'Count']
dfcoding['Category'] = 'Personal'
dfcoding.iloc[0,0] = '50% to 70% Coding'
dfonline = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q38_Part_4','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfonline.columns = ['default', 'Identify', 'Count']
dfonline['Category'] = 'Personal'
dftimeml = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q37','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dftimeml.columns = ['default', 'Identify', 'Count']
dftimeml['Category'] = 'Personal'
dfuseml = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q10','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfuseml.columns = ['default', 'Identify', 'Count']
dfuseml['Category'] = 'Personal'
dfonlineplatform = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q24','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfonlineplatform.columns = ['default', 'Identify', 'Count']
dfonlineplatform['Category'] = 'Personal'



dfPersonal = pd.concat([dfGender,dfAge,dfCountry,dfeducation,dfemployed,dfcoding,dfonline,dftimeml,dfuseml,dfonlineplatform], ignore_index=True)
dfplot = dfPersonal.sort_values(by=['Count'], ascending = False)
plt.figure(figsize=(11,9))
plt.title('Comparison of most common personal attributes that consider themselves as Data Scientists')


ax = sns.barplot(x = dfplot['Count'], y = dfplot['default'])
dfQ12_MULTIPLE_CHOICE = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q12_MULTIPLE_CHOICE','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ12_MULTIPLE_CHOICE.columns = ['default', 'Identify', 'Count']
dfQ12_MULTIPLE_CHOICE['Category'] = 'Tools'
dfQ13_Part_1 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q13_Part_1','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ13_Part_1.columns = ['default', 'Identify', 'Count']
dfQ13_Part_1['Category'] = 'Tools'
dfQ14_Part_1 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q14_Part_1','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ14_Part_1.columns = ['default', 'Identify', 'Count']
dfQ14_Part_1['Category'] = 'Tools'
dfQ15_Part_2 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q15_Part_2','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ15_Part_2.columns = ['default', 'Identify', 'Count']
dfQ15_Part_2['Category'] = 'Tools'
dfQ17 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q17','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ17.columns = ['default', 'Identify', 'Count']
dfQ17['Category'] = 'Tools'
dfrecommendlanguage = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q18','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfrecommendlanguage.columns = ['default', 'Identify', 'Count']
dfrecommendlanguage['Category'] = 'Tools'
dfrecommendlanguage.iloc[0,0] = 'Recommneded Python'

dfTools = pd.concat([dfQ15_Part_2,dfQ12_MULTIPLE_CHOICE,dfQ13_Part_1,dfQ14_Part_1,dfQ17,dfrecommendlanguage], ignore_index=True)
dfplot = dfTools.sort_values(by=['Count'], ascending = False)
plt.figure(figsize=(11,9))
plt.title('Comparison of most common Languages and Software attributes that consider themselves as Data Scientists')

ax = sns.barplot(x = dfplot['Count'], y = dfplot['default'])

dfQ31_Part_6 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q31_Part_6','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ31_Part_6.columns = ['default', 'Identify', 'Count']
dfQ31_Part_6['Category'] = 'Data'
dfQ21_Part_2 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q21_Part_2','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ21_Part_2.columns = ['default', 'Identify', 'Count']
dfQ21_Part_2['Category'] = 'Data'
dfQ29_Part_10 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q29_Part_10','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ29_Part_10.columns = ['default', 'Identify', 'Count']
dfQ29_Part_10['Category'] = 'Data'
dfQ30_Part_10 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q30_Part_10','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ30_Part_10.columns = ['default', 'Identify', 'Count']
dfQ30_Part_10['Category'] = 'Data'
# dfQ17 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q17','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
# dfQ17.columns = ['default', 'Identify', 'Count']
# dfQ17['Category'] = 'Personal'
dfQ33_Part_4 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q33_Part_4','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ33_Part_4.columns = ['default', 'Identify', 'Count']
dfQ33_Part_4['Category'] = 'Data'
dfQ33_Part_4.iloc[0,0] = 'Use Data Aggregrator'

dfData = pd.concat([dfQ31_Part_6,dfQ21_Part_2,dfQ29_Part_10,dfQ30_Part_10,dfQ33_Part_4], ignore_index=True)
dfplot = dfData.sort_values(by=['Count'], ascending = False)
plt.figure(figsize=(11,9))
plt.title('Most common Data attributes that consider themselves as Data Scientists')
plt.ylabel('')
ax = sns.barplot(x = dfplot['Count'], y = dfplot['default'])
dfreproducityimportance = df[((df['Q41_Part_3'] == 'Very important') & (df['selfgroupid'] == 'Identify as Data Scientist'))].groupby(['Q41_Part_3','selfgroupid']).size().reset_index()
dfreproducityimportance.columns = ['default', 'Identify', 'Count']
dfreproducityimportance.iloc[0,0] = 'Reproducity is very important'
dfreproducityimportance['Category'] = 'Model'
dfQ49_Part_6 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q49_Part_6','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ49_Part_6.columns = ['default', 'Identify', 'Count']
dfQ49_Part_6['Category'] = 'Model'
dfpercentmodelinsight = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q46','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfpercentmodelinsight.columns = ['default', 'Identify', 'Count']
dfpercentmodelinsight.iloc[0,0] = '10-20% Project is Model Insights'
dfpercentmodelinsight['Category'] = 'Model'
dfbarriertoreproceduce = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q50_Part_2','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfbarriertoreproceduce.columns = ['default', 'Identify', 'Count']
dfbarriertoreproceduce.iloc[0,0] = 'Barrier to reproduce is its time consuming'
dfbarriertoreproceduce['Category'] = 'Model'
dfmetrics = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q42_Part_1','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfmetrics.columns = ['default', 'Identify', 'Count']
dfmetrics['Category'] = 'Model'
dfmlwork = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q11_Part_3','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfmlwork.columns = ['Model', 'Identify', 'Count']
dfmlwork['Category'] = 'Personal'



dfModel = pd.concat([dfreproducityimportance,dfpercentmodelinsight,dfQ49_Part_6,dfbarriertoreproceduce,dfmetrics,dfmlwork], ignore_index=True)
dfplot = dfModel.sort_values(by=['Count'], ascending = False)
plt.figure(figsize=(11,9))
plt.title('Comparison of most common Model attributes that consider themselves as Data Scientists')


ax = sns.barplot(x = dfplot['Count'], y = dfplot['default'])
dfQ28_Part_19 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q28_Part_19','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ28_Part_19.columns = ['default', 'Identify', 'Count']
dfQ28_Part_19['Category'] = 'Machine Learning'
dfQ43 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q43','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ43.columns = ['default', 'Identify', 'Count']
dfQ43.iloc[0,0] = '0-10% exploring unfair bias'
dfQ43['Category'] = 'Machine Learning'
dfQ48 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q48','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ48.columns = ['default', 'Identify', 'Count']
dfQ48['Category'] = 'Machine Learning'
dfQ47_Part_8 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q47_Part_8','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ47_Part_8.columns = ['default', 'Identify', 'Count']
dfQ47_Part_8['Category'] = 'Machine Learning'
dfQ25 = df[((df['Q25'] == '1-2 years') & (df['selfgroupid'] == 'Identify as Data Scientist'))].groupby(['Q25','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ25.columns = ['default', 'Identify', 'Count']
dfQ25.iloc[0,0] = '1-2 years using ML'
dfQ25['Category'] = 'Machine Learning'
dfQ19_Part_1 = df[(df['selfgroupid'] == 'Identify as Data Scientist')].groupby(['Q19_Part_1','selfgroupid']).size().sort_values(ascending = False).head(1).reset_index()
dfQ19_Part_1.columns = ['default', 'Identify', 'Count']
dfQ19_Part_1['Category'] = 'Machine Learningl'


dfMachineLearning = pd.concat([dfQ28_Part_19,dfQ19_Part_1,dfQ47_Part_8,dfQ48,dfQ43,dfQ25], ignore_index=True)
dfplot = dfMachineLearning.sort_values(by=['Count'], ascending = False)
plt.figure(figsize=(11,9))
plt.title('Most common Machine Learning attributes that consider themselves as Data Scientists')

ax = sns.barplot(x = dfplot['Count'], y = dfplot['default'])
dfall = pd.concat([dfModel.head(3),dfData.head(3),dfPersonal.head(5), dfMachineLearning.head(3),dfTools.head(4)], ignore_index=True)
dfplot = dfall.sort_values(by=['Count'], ascending = False)
plt.figure(figsize=(11,9))
plt.title('Most common attributes for self identified Data Scientists')


ax = sns.barplot(x = dfplot['Count'],  y = dfplot['default'])



from IPython.display import Image
from IPython.core.display import HTML 
PATH = "../input/imagefilesforkagglesurvey/"
Image(filename = PATH + "anatomyofkaggler2.png")

from IPython.display import Image
Image(filename = PATH + "predictdatascientist3.png")



