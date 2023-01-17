import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt 
data=pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv")

data.head()
df=pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv")

df.head()
data.columns   #Seeing the available Features.
data.info()   #Getting an info about the columns to see instances of missing data.
#Removing the features that we are not going to use for our analysis.

data.drop(['Extraversion', 'ScreenName', 'SOVisit1st',

       'SOVisitFreq', 'SOVisitTo', 'SOFindAnswer', 'SOTimeSaved',

       'SOHowMuchTime', 'SOAccount', 'SOPartFreq', 'SOJobs', 'EntTeams',

       'SOComm', 'WelcomeChange', 'SONewContent','CodeRevHrs', 'UnitTests', 'PurchaseHow', 'PurchaseWhat'

       , 'DatabaseWorkedWith',

       'DatabaseDesireNextYear', 'PlatformWorkedWith',

       'PlatformDesireNextYear', 'WebFrameWorkedWith',

       'WebFrameDesireNextYear', 'MiscTechWorkedWith',

       'MiscTechDesireNextYear', 'DevEnviron', 'OpSys', 'Containers',

       'BlockchainOrg', 'BlockchainIs', 'BetterLife', 'ITperson', 'OffOn','CurrencyDesc',

       'CompTotal', 'CompFreq', 'ConvertedComp', 'WorkPlan',

       'WorkChallenge', 'WorkRemote', 'WorkLoc', 'ImpSyn', 'CodeRev', 'UndergradMajor',

       'EduOther', 'OrgSize', 'DevType', 'Age1stCode',

       'YearsCodePro', 'CareerSat', 'JobSat', 'MgrIdiot', 'MgrMoney',

       'MgrWant', 'JobSeek', 'LastHireDate', 'LastInt', 'FizzBuzz',

       'JobFactors', 'ResumeUpdate'],axis=1,inplace=True)

data.columns

data['Employment'].value_counts()
#Pie Chart for the different data of employment records(Freq. depends on type of employment)

data['Employment'].value_counts().plot(kind='pie',autopct="%.02f")
data['Gender'].value_counts().head(5)  #Freq. Dist. Table for the different Gender types
#Pie chart plot for the Different Genders hyst to give ys an idea of the male dominated nature ofthe data.

data['Gender'].value_counts().plot(kind='pie',autopct="%.02f")
#Freq. Dist. Table of the Languages the developers have worked with(TOP 10 MOST POPULAR)

data['LanguageWorkedWith'].value_counts().head(10)
#Pie chart to display the freq. cap of all the TOP 10 most famous languages / lang. combninations

#preffered by the coders of today.

data['LanguageWorkedWith'].value_counts().head(10).plot(kind='pie',autopct="%.02f")
#language desired by these people to work with next year

data['LanguageDesireNextYear'].value_counts().head(10)
#Pie plot of language desired next year

data['LanguageDesireNextYear'].value_counts().head(10).plot(kind='pie',autopct='%.02f')
#The most preffered medium of social media for these developers.

data['SocialMedia'].value_counts().head(10).plot(kind='bar')

#We get a better understanding of the people we are dealing with from this graph. Developers top 3 preffered Social Media are

#Reddit(dubbed the smart peoples social media) , Youtube(For getting info,learning as well as entertainment), & 

#Whatsapp(For both personal and proffesional Communication). NOTE: Theres also a good chunk of these people who dont use any social media.
#Freq Dist. Table for the ease of the survey

data['SurveyEase'].value_counts()
#We Can conclude that the survey was considered easy by majority of the survey takers.

data['SurveyEase'].value_counts().plot(kind='pie',autopct="%.02f")
#How many years these developers who took the survey have been coding.

data['YearsCode'].value_counts().head(10).plot(kind='bar')
#Education Level of the Developers

data['EdLevel'].value_counts().plot(kind='bar')
#Frequency Distribution of the currencies used by our developers.

data['CurrencySymbol'].value_counts().head(10)
data['CurrencySymbol'].value_counts().head(10).plot(kind='pie',autopct="%.02f")

#Seeing the % of the currencies we get a fair idea about the countries where StackOverflow is widely used, not everyone takes the survey.

#However we can get a rough idea that in USA , Europe and India and UK . These countries host the most no. of the devs who took the survey.
