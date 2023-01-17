import pandas as pd 

import seaborn as sns

sns.set()

giving_data = [ # from https://www.philanthropy.com/article/Gifts-to-Charity-Dropped-17/246511

    [ 2018, 292_090 ],

    [ 2017, 302_510 ],

    [ 2016, 292_300 ],

    [ 2015, 280_430 ],

    [ 2014, 267_560 ],

    [ 2013, 261_320 ],

    [ 2012, 267_280 ],

    [ 2011, 238_790 ],

    [ 2010, 239_520 ],

    [ 2009, 235_000 ],

    [ 2008, 249_310 ],

    [ 2007, 282_240 ]

]

giving_df = pd.DataFrame(giving_data, columns=['Year','Individual Giving'])

giving_df = giving_df[::-1]  # reverse data 

ax = giving_df.plot.barh(x=0,y=1,rot=0,figsize=(12,6),legend=False, width=.8)

ax.set_title('Individual Giving for U.S. Charities', fontsize=18)

for p in ax.patches:

    #print(p)

    ax.annotate('$'+format(p.get_width()*1_000_000,',d'), (p.get_width() - 64_000, p.get_y()+.25), color='white', weight='bold')  

ax.axes.get_xaxis().set_ticklabels([])

_ = 0 # be quiet, matplotlib
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas.plotting as pp

from IPython.display import display, HTML

import os

import warnings

warnings.filterwarnings('always')

def print_files():

    files = [] 

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            files.append(os.path.join(dirname, filename))

    files.sort()

    for file in files:

        print(file)

def print_full(x):

    pd.set_option('display.max_rows', len(x))

    pd.set_option('display.max_columns', None)

    pd.set_option('display.width', 2000)

    pd.set_option('display.float_format', '{:20,.2f}'.format)

    pd.set_option('display.max_colwidth', -1)

    x = x.style.set_properties(**{'text-align': 'left'})

    display(x) # print(x)

    pd.reset_option('display.max_rows')

    pd.reset_option('display.max_columns')

    pd.reset_option('display.width')

    pd.reset_option('display.float_format')

    pd.reset_option('display.max_colwidth')

# kaggle ML and DS survey - 2019

k19mr = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv",skiprows=[1])

k19mh = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv',nrows=1)

k19mh = pd.Series(k19mh.transpose()[0]) # questions keyed by column 

k19tr = pd.read_csv("/kaggle/input/kaggle-survey-2019/other_text_responses.csv",skiprows=[1])

k19th = pd.read_csv("/kaggle/input/kaggle-survey-2019/other_text_responses.csv",nrows=1)

k19th = pd.Series(k19th.transpose()[0]) # questions keyed by column 

k19sr = pd.read_csv("/kaggle/input/kaggle-survey-2019/survey_schema.csv",skiprows=[1])

k19sh = pd.read_csv("/kaggle/input/kaggle-survey-2019/survey_schema.csv",nrows=1)

k19sh = pd.Series(k19sh.transpose()[0]) # questions keyed by column 

k19qh = pd.read_csv("/kaggle/input/kaggle-survey-2019/questions_only.csv")

k19qh = pd.Series(k19qh.transpose()[0]) # questions keyed by column 

k19mr_usa = (k19mr['Q3'] == 'United States of America')

k19mr_mid = (k19mr['Q6'] == '50-249 employees')

sp = k19mr[['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q10','Q23']]

orders = {

    1: None,

    2: None,

    3: None,

    4: ["No formal education past high school", "Professional degree", "Some college/university study without earning a bachelor’s degree", "Bachelor’s degree", "Master’s degree", "Doctoral degree", "I prefer not to answer"],

    5: None,

    6: ["0-49 employees", "50-249 employees", "250-999 employees", "1000-9,999 employees", "> 10,000 employees"],

    7: ["0", "1-2", "3-4", "5-9", "10-14", "15-19", "20+"],

    8: ["No (we do not use ML methods)", "We are exploring ML methods (and may one day put a model into production)", "We use ML methods for generating insights (but do not put working models into production)", "We recently started using ML methods (i.e., models in production for less than 2 years)", "We have well established ML methods (i.e., models in production for more than 2 years)", "I do not know"],

    10: ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "200,000-249,999", "250,000-299,999", "300,000-500,000", "> $500,000"],

    11: ["$0 (USD)", "$1-$99", "$100-$999", "$1000-$9,999", "$10,000-$99,999", "> $100,000 ($USD)"],

    15: ["I have never written code", "< 1 years", "1-2 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"],

    19: None,

    22: ["Never", "Once", "2-5 times", "6-24 times", "> 25 times"],

    23: ["< 1 years", "1-2 years", "2-3 years", "3-4 years", "4-5 years", "5-10 years", "10-15 years", "20+ years"]    

}
### Nathan starts here 

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

%matplotlib inline

import seaborn as sns

sns.set()



# Graphics in retina format are more sharp and legible

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

plt.rcParams['image.cmap'] = 'viridis'





import plotly.offline as py

import pycountry



py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from plotly.offline import init_notebook_mode, iplot 

init_notebook_mode(connected=True)



import folium 

from folium import plugins



import re



colors = ["steelblue","dodgerblue","lightskyblue","powderblue","deepskyblue","cyan","darkturquoise","paleturquoise","turquoise"]



#Importing the 2019 Dataset

df_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

df_2019.columns = df_2019.iloc[0]

df_2019=df_2019.drop([0])

pd.options.display.max_columns = None



#Importing the 2018 Dataset

df_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')

df_2018.columns = df_2018.iloc[0]

df_2018=df_2018.drop([0])



#Importing the 2017 Dataset

df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')



#Removing everyone that took less than 4 minutes

less3 = df_2019[round(df_2019.iloc[:,0].astype(int) / 60) <= 4].index

df_2019 = df_2019.drop(less3, axis=0)



less3 = df_2018[round(df_2018.iloc[:,0].astype(int) / 60) <= 4].index

df_2018 = df_2018.drop(less3, axis=0)

display(df_2017)

df_2017.columns.tolist().index('Tenure')







#Creating a smaller subset of the data

companyInfo17 = df_2017[df_2017['Country'] == 'United States'].iloc[:,[1,54,8,56]]

companyInfo18 = df_2018[df_2018['In which country do you currently reside?'] == 'United States of America'].iloc[:,[4,5,7,127]]

companyInfo17.columns = companyInfo18.columns = ['Country', 'Degree', 'Title','Experience']

USA_2019 = df_2019[df_2019['In which country do you currently reside?'] == 'United States of America']

companyInfo19 = USA_2019.iloc[:,[4,5,6,8,9,10,11,12,13,14,15,16,17,18,20,21,55]].copy()

#companyInfo18 = df_2018.iloc[:,[4,5,7,10,11,12,13,14,15,16,17,18,20,21]].copy()

#Renaming Columns

cols = ['Country', 'Degree', 'Title', 'Size of Company', 'Size of Team', 'Machine Learning Methods', 'Analyze and understand data to influence product or business decisions', 'Build and_or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data','Build prototypes to explore applying machine learning to new areas', 'Build and/or run a machine learning service that operationally improves my product or workflows', 'Experimentation and iteration to improve existing ML models', 'Do research that advances the state of the art of machine learning', 'None', 'Other', 'Compensation', 'Money Spent on Product','Experience']

companyInfo19.columns = cols

med_companyInfo19 = companyInfo19[companyInfo19['Size of Company']== '250-999 employees']



#Help 2017 Titles Match

changeF = ['Software Developer/Software Engineer', 'Scientist/Researcher', 'Researcher']

changeT = ['Software Engineer', 'Research Scientist', 'Research Scientist']

companyInfo17 = companyInfo17.replace(changeF,changeT)



display(companyInfo19)

##AVG TEAM SIZE BY COMPANY SIZE



numericalTeamSizes = []

for size in companyInfo19['Size of Team']:

    if size == '20+':

        numericalTeamSizes.append(20)

    elif size == '15-19':

        numericalTeamSizes.append(17)

    elif size == '10-14':

        numericalTeamSizes.append(12)

    elif size == '5-9':

        numericalTeamSizes.append(7)

    elif size == '3-4':

        numericalTeamSizes.append(3.5)

    elif size == '1-2':

        numericalTeamSizes.append(1.5)

    elif size == '0':

        numericalTeamSizes.append(0)

    else:

        numericalTeamSizes.append(np.nan)

        

companyInfo19['numericalTeamSizes'] = numericalTeamSizes

meanTmSz=[companyInfo19[companyInfo19['Size of Company'] == companySize].numericalTeamSizes.mean() for companySize in companyInfo19['Size of Company'].unique()]



companySizes = companyInfo19['Size of Company'].unique()

correctOrder = [5, 1, 2, 4, 0]

chartData = np.vstack(([meanTmSz[i] for i in correctOrder],[companySizes[i] for i in correctOrder]))



fig = go.Figure([go.Bar(x=chartData[1,:], y=chartData[0,:], hovertemplate = '<i>Company Size: %{x} </i> <br> Mean Team Size: %{y} <extra></extra>')])

fig.update_xaxes(title_text='Company Size')

fig.update_yaxes(title_text='Mean Data Science Team Size')

fig.update_layout(

    hoverlabel_align = 'right', 

    title = "Data Science Team Size vs. Company Size")



fig.show()
import pandas as pd

import matplotlib.pyplot as plt

#import seaborn as sns

cdata = pd.DataFrame(sp[k19mr_usa])

company_size_category = pd.api.types.CategoricalDtype(categories=orders[6][::-1], ordered=True)

ml_team_size_category = pd.api.types.CategoricalDtype(categories=orders[7], ordered=True)

cdata['Organization Size'] = cdata['Q6'].astype(company_size_category)

cdata['ML Team Size'] = cdata['Q7'].astype(ml_team_size_category)

del cdata['Q6']

del cdata['Q7']

df2 = cdata.groupby(['Organization Size', 'ML Team Size'])['Organization Size'].count().unstack('ML Team Size').fillna(0)

df2.plot(kind='barh', stacked=True, figsize=(14,5));
## Nathan continued

##TITLE BY TEAMZISE SUNBURST PLOT

#Get titles for each teamsize

titles = companyInfo19.iloc[:,2].dropna().unique()

teamSizes = companyInfo19.iloc[:,4].dropna().unique()

teamCounts = []

i = 0

titleCounts = np.zeros((teamSizes.__len__(), titles.__len__()))

#titlesByTeamSize

for team in teamSizes:

    tempSubset = companyInfo19[companyInfo19['Size of Team'] == team]

    teamCounts.append(tempSubset.iloc[:,4].count())

    tempList = []

    for title in titles:

        tempList.append(tempSubset[tempSubset['Title'] == title].iloc[:,2].count())  

    titleCounts[i,:] = tempList

    i += 1

#Get data in the correct format for a sunburst plot

import plotly.graph_objects as go

centerText = 'North American Companies'

labels1 = np.concatenate(('Team Size: ' + teamSizes, titles, titles, titles, titles, titles, titles, titles), axis=0)

parents1 = np.concatenate((np.repeat(centerText,7),np.repeat("20+",12),np.repeat("1-2",12),\

                            np.repeat("10-14",12),np.repeat("3-4",12),np.repeat("5-9",12),np.repeat("15-19",12),np.repeat("0",12)), axis=0)



values1 = np.concatenate((np.sum(titleCounts,axis=1), titleCounts[0,:], titleCounts[1,:], titleCounts[2,:], titleCounts[3,:], titleCounts[4,:], titleCounts[5,:], titleCounts[6,:]), axis=0)



ids1 = np.concatenate((teamSizes, ['20+' + title for title in titles], ['1-2' + title for title in titles], ['10-14' + title for title in titles],\

                       ['3-4' + title for title in titles], ['5-9' + title for title in titles], ['15-19' + title for title in titles], ['0' + title for title in titles]), axis=0)





sunburst = pd.DataFrame({'Ids': np.insert(ids1,0,centerText),

                        'Labels': np.insert(labels1,0,centerText),

                        'Parents': np.insert(parents1,0,""),

                       'Values': np.insert(values1,0,np.sum(titleCounts))})

sunburst['Percents'] = Percents = sunburst.Values/[sunburst[sunburst.Ids == parent].Values for parent in sunburst.Parents]*100



#RemoveSlices with zero values

sunburst = sunburst[sunburst['Values']!=0]





#Plot Data

fig =go.Figure(go.Sunburst(

    ids = sunburst.Ids,

    labels = sunburst.Labels,

    parents = sunburst.Parents,

    values = sunburst.Values,

    branchvalues = "total",

    hovertemplate='<b>%{label} </b> <br> Responses: %{value}<br> <extra></extra>',

))

fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))

fig.update_layout(showlegend=True)

fig.show()
##TITLE BY TEAMZISE SUNBURST PLOT FOR MEDIUM COMPANIES ONLY

#Get titles for each teamsize

titles = med_companyInfo19.iloc[:,2].dropna().unique()

teamSizes = med_companyInfo19.iloc[:,4].dropna().unique()

teamCounts = []

i = 0

titleCounts = np.zeros((teamSizes.__len__(), titles.__len__()))

#titlesByTeamSize

for team in teamSizes:

    tempSubset = med_companyInfo19[med_companyInfo19['Size of Team'] == team]

    teamCounts.append(tempSubset.iloc[:,4].count())

    tempList = []

    for title in titles:

        tempList.append(tempSubset[tempSubset['Title'] == title].iloc[:,2].count())  

    titleCounts[i,:] = tempList

    i += 1





#Get data in the correct format for a sunburst plot

import plotly.graph_objects as go

centerText = 'Mid-Size<br>North American Companies'

titleNum = len(titles)

teamSizeNum = len(teamSizes)



labels1 = np.concatenate(('Team Size: ' + teamSizes, 'team size' + titles, titles, titles, titles, titles, titles, titles), axis=0)



parents1 = np.concatenate((np.repeat(centerText,teamSizeNum),np.repeat("10-14",titleNum),np.repeat("3-4",titleNum),\

                            np.repeat("20+",titleNum),np.repeat("5-9",titleNum),np.repeat("1-2",titleNum),np.repeat("15-19",titleNum),np.repeat("0",titleNum)), axis=0)



values1 = np.concatenate((np.sum(titleCounts,axis=1), titleCounts[0,:], titleCounts[1,:], titleCounts[2,:], titleCounts[3,:], titleCounts[4,:], titleCounts[5,:], titleCounts[6,:]), axis=0)



ids1 = np.concatenate((teamSizes, ['20+' + title for title in titles], ['1-2' + title for title in titles], ['10-14' + title for title in titles],\

                       ['3-4' + title for title in titles], ['5-9' + title for title in titles], ['15-19' + title for title in titles], ['0' + title for title in titles]), axis=0)





sunburst = pd.DataFrame({'Ids': np.insert(ids1,0,centerText),

                        'Labels': np.insert(labels1,0,centerText),

                        'Parents': np.insert(parents1,0,""),

                       'Values': np.insert(values1,0,np.sum(titleCounts))})

sunburst['Percents'] = Percents = sunburst.Values/[sunburst[sunburst.Ids == parent].Values for parent in sunburst.Parents]*100



#RemoveSlices with zero values

sunburst = sunburst[sunburst['Values']!=0]





#Plot Data

fig =go.Figure(go.Sunburst(

    ids = sunburst.Ids,

    labels = sunburst.Labels,

    parents = sunburst.Parents,

    values = sunburst.Values,

    branchvalues = "total",

    hovertemplate='<b>%{label} </b> <br> Responses: %{value}<br> <extra></extra>',

))

fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))

fig.update_layout(showlegend=True)

fig.show()
titlesToCount = ['DBA/Database Engineer', 'Statistician', 'Data Scientist', 'Software Engineer', 'Data Analyst', 'Research Scientist', 'Business Analyst']





titleCount19 = [companyInfo19[companyInfo19.Title == title].Title.count()/companyInfo19.Title.count()*100 for title in titlesToCount]

titleCount18 = [companyInfo18[companyInfo18.Title == title].Title.count()/companyInfo18.Title.count()*100 for title in titlesToCount]

titleCount17 = [companyInfo17[companyInfo17.Title == title].Title.count()/companyInfo17.Title.count()*100 for title in titlesToCount]



titleCountByYear = pd.DataFrame([titleCount17,titleCount18, titleCount19], columns = titlesToCount)

titleCountByYear.index = [2017,2018,2019]



    

fig = go.Figure()

for title in titlesToCount:

    fig.add_trace(go.Scatter(x=[2017, 2018, 2019], y=titleCountByYear[title],

                             mode='lines',

                             name=title,

                            ))

fig.update_xaxes(title_text='Survey Year', dtick=1)

fig.update_yaxes(title_text='Response Frequency (Percent)')

fig.update_layout(

    hoverlabel_align = 'right', 

    title = "Title Response Frequency by Year")

   



fig.show()
##TITLE BY TEAMZISE SUNBURST PLOT

#Get titles for each teamsize

titles = companyInfo19.iloc[:,2].dropna().unique()

companySizes = companyInfo19['Size of Company'].dropna().unique()



correctOrder = [4,1,2,3,0]

companySizes = [companySizes[i] for i in correctOrder]



teamCounts = []

i = 0

titleCounts = np.zeros((companySizes.__len__(), titles.__len__()))

#titlesByTeamSize

for size in companySizes:

    tempSubset = companyInfo19[companyInfo19['Size of Company'] == size]

    teamCounts.append(tempSubset.iloc[:,3].count())

    tempList = []

    for title in titles:

        tempList.append(tempSubset[tempSubset['Title'] == title].iloc[:,2].count())  

    titleCounts[i,:] = tempList

    i += 1





#Get data in the correct format for a sunburst plot

import plotly.graph_objects as go

centerText = 'North American Companies'

rows = len(companySizes)

cols = len(titles)



labels1 = np.concatenate((companySizes, np.tile(titles,rows)), axis=0)



parents1 = np.concatenate((np.repeat(centerText,rows),np.repeat(companySizes,cols)), axis=0)



values1 = np.concatenate((np.sum(titleCounts,axis=1), np.asarray(titleCounts).reshape(-1)), axis=0)



ids1 = np.concatenate((companySizes, np.asarray([[company + title for title in titles]for company in companySizes]).reshape(-1)), axis=0)







sunburst = pd.DataFrame({'Ids': np.insert(ids1,0,centerText),

                        'Labels': np.insert(labels1,0,centerText),

                        'Parents': np.insert(parents1,0,""),

                       'Values': np.insert(values1,0,np.sum(titleCounts))})

#sunburst['Percents'] = sunburst.Values/[sunburst[sunburst.Ids == parent].Values for parent in sunburst.Parents]*100



#RemoveSlices with zero values

sunburst = sunburst[sunburst['Values']!=0]





#Plot Data

fig =go.Figure(go.Sunburst(

    ids = sunburst.Ids,

    labels = sunburst.Labels,

    parents = sunburst.Parents,

    values = sunburst.Values,

    branchvalues = "total",

    hovertemplate = '<b>%{label}</b><br><br>Responses: %{value}<extra></extra>'

   # hovertemplate =

    #'<b>%{label}</b>'+

    #'<br>Percent: %{hovertext}<br>'+ 

    #'Responses: %{value}<br> <extra></extra>',

    #hovertext = ['{}%'.format(i) for i in sunburst.Percents]

))

fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))

fig.update_layout(title = 'Role Prevalence by Company Size')

fig.show()
#AVG TEAM SIZE BY COMPANY SIZE



numericalYearsExperience = []

for years in companyInfo19.Experience:

    if years == 'I have never written code':

        numericalYearsExperience.append(0)

    elif years == '3-5 years':

        numericalYearsExperience.append(4)

    elif years == '< 1 years':

        numericalYearsExperience.append(0.5)

    elif years == '1-2 years':

        numericalYearsExperience.append(1.5)

    elif years == '5-10 years':

        numericalYearsExperience.append(7.5)

    elif years == '10-20 years':

        numericalYearsExperience.append(15)

    elif years == '20+ years':

        numericalYearsExperience.append(20)

    else:

        numericalYearsExperience.append(np.nan)

        

companyInfo19['numericalYearsExperience'] = numericalYearsExperience

meanYearsExperience=[companyInfo19[companyInfo19['Size of Company'] == companySize].numericalYearsExperience.mean() for companySize in companyInfo19['Size of Company'].unique()]



companySizes = companyInfo19['Size of Company'].unique()

correctOrder = [5, 1, 2, 4, 0]

chartData = np.vstack(([meanYearsExperience[i] for i in correctOrder],[companySizes[i] for i in correctOrder]))



fig = go.Figure([go.Bar(x=chartData[1,:], y=chartData[0,:], hovertemplate = '<i>Company Size: %{x} </i> <br> Mean Data Science Experience: %{y} <extra></extra>')])

fig.update_xaxes(title_text='Company Size')

fig.update_yaxes(title_text='Mean Data Science Experience (years)')

fig.update_layout(

    hoverlabel_align = 'right', 

    title = "Data Science Experience vs. Company Size")



fig.show()
import pandas as pd

import matplotlib.pyplot as plt

#import seaborn as sns

cdata = pd.DataFrame(sp[k19mr_usa])

company_size_category = pd.api.types.CategoricalDtype(categories=orders[6][::-1], ordered=True)

education_category = pd.api.types.CategoricalDtype(categories=orders[23], ordered=True)

cdata['Company Size'] = cdata['Q6'].astype(company_size_category)

cdata['Experience'] = cdata['Q23'].astype(education_category)

del cdata['Q6']

del cdata['Q23']

df2 = cdata.groupby(['Company Size', 'Experience'])['Company Size'].count().unstack('Experience').fillna(0)

df2.plot(kind='barh', stacked=True, figsize=(14,5));
companyInfo19['numericalYearsExperienceCompat'] = companyInfo19.numericalYearsExperience.replace([15,20],10)



numericalYearsExperience = []

for years in companyInfo18.Experience:

    if (years == 'I have never written code but I want to learn')|(years == 'I have never written code and I do not want to learn'):

        numericalYearsExperience.append(0)

    elif years == '3-5 years':

        numericalYearsExperience.append(4)

    elif years == '< 1 years':

        numericalYearsExperience.append(0.5)

    elif years == '1-2 years':

        numericalYearsExperience.append(1.5)

    elif years == '5-10 years':

        numericalYearsExperience.append(7.5)

    elif years == '10-20 years':

        numericalYearsExperience.append(10)

    elif (years == '20-30 years')|(years == '30-40 years')|(years == '40+ years'):

        numericalYearsExperience.append(10)

    else:

        numericalYearsExperience.append(np.nan)        

companyInfo18['numericalYearsExperience'] = numericalYearsExperience



numericalYearsExperience = []

for years in companyInfo17.Experience:

    if years == 'I don\'t write code to analyze data':

        numericalYearsExperience.append(0)

    elif years == '3 to 5 years':

        numericalYearsExperience.append(4)

    elif years == 'Less than a year':

        numericalYearsExperience.append(0.5)

    elif years == '1 to 2 years':

        numericalYearsExperience.append(1.5)

    elif years == '6 to 10 years':

        numericalYearsExperience.append(7.5)

    elif years == 'More than 10 years':

        numericalYearsExperience.append(10)

    else:

        numericalYearsExperience.append(np.nan)        

companyInfo17['numericalYearsExperience'] = numericalYearsExperience



    

fig = go.Figure()

fig.add_trace(go.Scatter(x=[2017, 2018, 2019], y=[companyInfo17['numericalYearsExperience'].mean(),companyInfo18['numericalYearsExperience'].mean(),companyInfo19['numericalYearsExperienceCompat'].mean()], mode='lines', name=title))

fig.update_xaxes(title_text='Survey Year', dtick=1)

fig.update_yaxes(title_text='Mean Data Science Experience (Years)', range = [0.1, 10])

fig.update_layout(

    hoverlabel_align = 'right', 

    title = "Data Science Experience by Survey Year")

   



fig.show()
#Avg Year of Education BY COMPANY SIZE

education = []

for degree in companyInfo19.Degree:

    if degree == 'No formal education past high school':

        education.append(0)

    elif degree == 'Some college/university study without earning a bachelor’s degree':

        education.append(2)

    elif degree == 'Professional degree':

        education.append(7)

    elif degree == 'Bachelor’s degree':

        education.append(4)

    elif degree == 'Master’s degree':

        education.append(6)

    elif degree == 'Doctoral degree':

        education.append(11)

    else:

        education.append(np.nan)

        

companyInfo19['Education'] = education

meanYearsEducation=[companyInfo19[companyInfo19['Size of Company'] == companySize].Education.mean() for companySize in companyInfo19['Size of Company'].unique()]



companySizes = companyInfo19['Size of Company'].unique()

correctOrder = [5, 1, 2, 4, 0]

chartData = np.vstack(([meanYearsEducation[i] for i in correctOrder],[companySizes[i] for i in correctOrder]))



fig = go.Figure([go.Bar(x=chartData[1,:], y=chartData[0,:], hovertemplate = '<i>Company Size: %{x} </i> <br> Mean Years of Education: %{y} <extra></extra>')])

fig.update_xaxes(title_text='Company Size')

fig.update_yaxes(title_text='Mean Years of Secondary Education')

fig.update_layout(

    hoverlabel_align = 'right', 

    title = "Data Science Employee Education by Company Size")



fig.show()
import pandas as pd

import matplotlib.pyplot as plt

#import seaborn as sns

cdata = pd.DataFrame(sp[k19mr_usa])

company_size_category = pd.api.types.CategoricalDtype(categories=orders[6][::-1], ordered=True)

education_category = pd.api.types.CategoricalDtype(categories=orders[4], ordered=True)

cdata['Company Size'] = cdata['Q6'].astype(company_size_category)

cdata['Education'] = cdata['Q4'].astype(education_category)

del cdata['Q6']

del cdata['Q4']

df2 = cdata.groupby(['Company Size', 'Education'])['Company Size'].count().unstack('Education').fillna(0)

df2.plot(kind='barh', stacked=True, figsize=(14,5));
import pandas as pd

import matplotlib.pyplot as plt

#import seaborn as sns

cdata = pd.DataFrame(sp[k19mr_usa])

company_size_category = pd.api.types.CategoricalDtype(categories=orders[6][::-1], ordered=True)

salary_category = pd.api.types.CategoricalDtype(categories=orders[10], ordered=True)

cdata['Company Size'] = cdata['Q6'].astype(company_size_category)

cdata['Salary'] = cdata['Q10'].astype(salary_category)

del cdata['Q6']

del cdata['Q10']

df2 = cdata.groupby(['Company Size', 'Salary'])['Company Size'].count().unstack('Salary').fillna(0)

df2.plot(kind='barh', stacked=True, figsize=(14,5)).legend(loc='center left', bbox_to_anchor=(1.0, 0.5));

#plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))