import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm

from wordcloud import WordCloud, STOPWORDS

import collections as cl

from datetime import datetime

import os
try:

    data = pd.read_csv('/kaggle/input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')

except:

    data = pd.read_csv('C:/Users/Shakib/CampusX-Files/survey_results_public.csv')
data.head()
data.columns
data.drop(columns={'Respondent','Hobbyist','Employment','Student','EdLevel','UndergradMajor','EduOther','OrgSize','DevType','YearsCode','Age1stCode','YearsCodePro','JobSat','MgrIdiot','MgrMoney','MgrWant','JobSeek','LastHireDate','LastInt','FizzBuzz','JobFactors','ResumeUpdate','CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','WorkPlan','WorkChallenge','WorkRemote','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','MiscTechWorkedWith','MiscTechDesireNextYear','DevEnviron','OpSys','Containers','BlockchainOrg','BlockchainIs','BetterLife','ITperson','OffOn','Extraversion','ScreenName','SOVisit1st','SOVisitFreq','SOVisitTo','SOFindAnswer','SOTimeSaved','SOHowMuchTime','SOAccount','SOPartFreq','SOJobs','EntTeams','SOComm','WelcomeChange','OpenSource','SONewContent','Trans','Sexuality','Ethnicity','Dependents','SurveyLength','SurveyEase'},inplace = True)
data.head()
data.shape
data.info()
data.describe()
data['MainBranch'].value_counts()
MainBranch = []

for i in data['MainBranch']:

    if i == 'I am a developer by profession':

        MainBranch.append('Developer')

    elif i == 'I am a student who is learning to code':

        MainBranch.append('Student')

    elif i == 'I am not primarily a developer, but I write code sometimes as part of my work':

        MainBranch.append('Semi_Developer')

    elif i == 'I code primarily as a hobby':

        MainBranch.append('Hobby') 

    elif i == 'I used to be a developer by profession, but no longer am':

        MainBranch.append('Ex_Developer')

    else:

        MainBranch.append('Student')

data['MainBranch'] = MainBranch

data['MainBranch'] = data['MainBranch'].astype('category',inplace=True)
data['OpenSourcer'].value_counts()
OpenSourcer = []

for i in data['OpenSourcer']:

    if (i == 'Never' ) or (i=='Less than once per year') or (i==''):

        OpenSourcer.append('No')

    else:

        OpenSourcer.append('Yes')

data['OpenSourcer'] = OpenSourcer

data['OpenSourcer'] = data['OpenSourcer'].astype('category')
data['CareerSat'].value_counts()
data['CareerSat'] = data['CareerSat'].astype('category')
data['WorkLoc'].value_counts()
WorkLoc = []

for i in data['WorkLoc']:

    if i == 'Office':

        WorkLoc.append(i)

    else:

        WorkLoc.append('Home')

data['WorkLoc'] = WorkLoc

data['WorkLoc'] = data['WorkLoc'].astype('category')
Age = []

count = 0

for i in data['Age']:

    try:

        Age.append(int(i))

    except:

        Age.append(i)

        count += 1

data['Age'] = Age
count
data['Gender'].value_counts()
Gender = []

for i in tqdm(data['Gender']):

    if (i=='Man') or (i== 'Man;Non-binary, genderqueer, or gender non-conforming'):

        Gender.append('Male')

    elif (i=='Woman') or (i=='Woman;Non-binary, genderqueer, or gender non-conforming') or (i=='Woman;Man;Non-binary, genderqueer, or gender non-conforming'):

        Gender.append('Female')

    else:

        Gender.append('Transgender')

data['Gender'] = Gender

data['Gender'] = data['Gender'].astype('category')
Country = data['Country'].value_counts().index

filters = []

for i in tqdm(data['Country']):

    filters.append(i in Country)

    

data = data[filters]
data.info()
data.shape
try:

    os.makedirs("Output_Graphs")

except FileExistsError:

    print("Folder alrady present")
haha = data.groupby('MainBranch')['OpenSourcer'].value_counts()

haha = haha.to_frame('Number_of_Developers')

haha = haha.reset_index()

sns.barplot(x='MainBranch',y='Number_of_Developers',hue='OpenSourcer',data=haha).get_figure().savefig('Output_Graphs/Developers_vs_Opensourcer.jpg',dpi=1200,bbox_inches = 'tight')
data_dev = data[data['MainBranch']=='Developer']

data = data_dev

data_plot = data['Country'].value_counts()[:25].reset_index().rename(columns={"index":"Country Name","Country":"Number of Developers"})

sns.barplot(y='Country Name',x='Number of Developers',data = data_plot).get_figure().savefig('Output_Graphs/Country_vs_number_of_developers.jpg',dpi=1200,bbox_inches = 'tight')
num = data['Country'].value_counts()[:30]

total = data.groupby('Country')['ConvertedComp'].sum()

data_plot1 = (total/num).sort_values(ascending=False)[:30]

data_plot = data_plot1.reset_index()

data_plot.rename(columns={0:'Average_income_in_USD','index':'Country_Name'},inplace=True)

sns.barplot(y='Country_Name',x='Average_income_in_USD',data = data_plot).get_figure().savefig('Output_Graphs/Salary_of_the_developers.jpg',dpi=1200,bbox_inches = 'tight')
plot_data=data_dev['CareerSat'].value_counts().sort_index().reset_index()

plot_data.rename(columns={'index':'','CareerSat':'Number of Developers'},inplace=True)

sns.barplot(y='',x='Number of Developers',data=plot_data).get_figure().savefig('Output_Graphs/job_satis.jpg',dpi=1200,bbox_inches = 'tight')
plot_data = data_dev['Age'].value_counts().sort_index().reset_index()

sns.lineplot(x='index',y='Age',data=plot_data).get_figure().savefig('Output_Graphs/Age_of_developers.jpg',dpi=1200,bbox_inches = 'tight')
data_plot = data_dev['SocialMedia'].value_counts().reset_index()

data_plot.rename(columns={'index':'Name of SocialMedia','SocialMedia':'Number of Users'},inplace=True)

sns.barplot(x='Number of Users',y='Name of SocialMedia',data=data_plot).get_figure().savefig('Output_Graphs/Social.jpg',dpi=1200,bbox_inches = 'tight')
def generate_word_column_for_the_column_of(column,color):

    column_name = column

    os_now_all_word = ''

    for i in data_dev[column_name]:

        try:

            a=i.split(';')

            for j in a:

                os_now_all_word+=' '+ j

        except:

            a=5



    cloud = WordCloud(background_color=color,max_font_size=250,width=960, height=1080).generate(os_now_all_word)



    cloud.to_file("Output_Graphs/" + column_name + '.png')

    plt.imshow(cloud)
generate_word_column_for_the_column_of('LanguageWorkedWith','white')
generate_word_column_for_the_column_of('LanguageDesireNextYear','black')
generate_word_column_for_the_column_of('DatabaseWorkedWith','white')
generate_word_column_for_the_column_of('DatabaseDesireNextYear','black')
generate_word_column_for_the_column_of('PlatformWorkedWith','white')
generate_word_column_for_the_column_of('PlatformDesireNextYear','black')
generate_word_column_for_the_column_of('WebFrameWorkedWith','white')
generate_word_column_for_the_column_of('WebFrameDesireNextYear','black')
data_plot_number = data_dev['WorkLoc'].value_counts().values

data_plot_name = data_dev['WorkLoc'].value_counts().index 

s = plt.pie(data_plot_number,labels=data_plot_name,autopct='%1.1f%%',)

plt.savefig('Output_Graphs/Location_to_work.jpg',dpi=1200,bbox_inches = 'tight')
print(datetime.now())