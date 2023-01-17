# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# reading dataset
survey = pd.read_csv("../input/survey_results_public.csv")
survey.head()
# plotting User count according to country
plt.subplots(figsize=(15,35))
sns.countplot(y = survey.Country, order = survey.Country.value_counts().index)
# ploting employment status count of Users 
survey.EmploymentStatus.unique()
sns.countplot(y=survey.EmploymentStatus, order = survey.EmploymentStatus.value_counts().index)
sns.countplot(y = survey.ProgramHobby, order = survey.ProgramHobby.value_counts().index)
# programHobby vs satisfaction level
# lets check the satisfaction level of people working in startups
# survey.CompanyType == "Venture-funded startup"
dev_in_startup = survey.loc[(survey.CompanyType == "Venture-funded startup") | (survey.CompanyType == "Pre-series A startup")]
fig,ax=plt.subplots(1,2,figsize=(25,12))

#### first figure
dev_venture_funded = survey.loc[survey.CompanyType == "Venture-funded startup"]
sns.countplot(y = dev_venture_funded.JobSatisfaction, ax=ax[0])
ax[0].set_title('Job Satisfaction for Users in a venture-Funded startup')

#### second figure
dev_pre_startup = survey.loc[survey.CompanyType == "Pre-series A startup"]
sns.countplot(y = dev_pre_startup.JobSatisfaction, ax=ax[1])
ax[1].set_title('Job Satisfaction for Users in a Pre-series Startup')

plt.subplots_adjust(hspace=0.1,wspace=0.6)
ax[0].tick_params(labelsize=15)
ax[1].tick_params(labelsize=15)
plt.show()
from collections import Counter
developerType = []
temp = survey.DeveloperType.drop(survey.loc[survey.DeveloperType.isnull()].index)
for i in temp:
    if i is not None:
        types = i.replace(' ', '').split(";")
        developerType.extend(types)
print(Counter(developerType))


plt.subplots(figsize=(15,15))   
sns.countplot(y = developerType)
survey.Professional.unique()
sns.countplot(y = survey.Professional, order = survey.Professional.value_counts().index)
plt.subplots(figsize=(7,7))  
plt.pie(dict(Counter(survey.Professional)).values(),
        labels = dict(Counter(survey.Professional)).keys(),
        shadow = True,
        startangle = 0);
survey.MajorUndergrad.unique()
sns.countplot(y = survey.MajorUndergrad, order = survey.MajorUndergrad.value_counts().index)
plt.subplots(figsize=(15,5))  
survey.WorkStart.unique()
sns.countplot(y = survey.WorkStart, order = survey.WorkStart.value_counts().index)
temp = survey.HaveWorkedLanguage.drop(survey.loc[survey.HaveWorkedLanguage.isnull()].index)
languages = []
row = {}
for i in temp:
    if i is not None: 
        types = i.split(";")
        languages.extend(types)
popularLanguages = Counter(languages).most_common(20)

languages = [i[0] for i in popularLanguages]
lang_count = [i[1] for i in popularLanguages]

df = pd.DataFrame()
df['Languages'] = languages
df['Number_of_Users'] = lang_count
df.head(20)                   
plt.subplots(figsize=(10,10))  
plt.pie(df.Number_of_Users,
        labels = df.Languages,
        shadow = True,
        startangle = 0);
# Most Famous Languages By Country
def get_Most_Used_Language(countryName):
    df = survey.loc[(survey.Country == countryName)]
    # print(df)
    temp = df.HaveWorkedLanguage.drop(df.loc[df.HaveWorkedLanguage.isnull()].index)
    languages = []
    for i in temp:
        if i is not None and type(i) is not float: 
            types = i.split(";")
            languages.extend(types)
    return Counter(languages).most_common(7)
countriesList = [ c[0] for c in Counter(survey.Country).most_common(20)]
country_name = []
lang = [[], [], [], [], [], [], []]
row = {}
Popular_Languages = pd.DataFrame()
for ind, country in enumerate(countriesList):
    pLangs = get_Most_Used_Language(country)
    for i, l in enumerate(lang):
        l.append(pLangs[i][0])

Popular_Languages['Country'] = countriesList
Popular_Languages['1st Language'] = lang[0]
Popular_Languages['2nd Language'] = lang[1]
Popular_Languages['3rd Language'] = lang[2]
Popular_Languages['4th Language'] = lang[3]
Popular_Languages['5th Language'] = lang[4]

Popular_Languages.head(20)
    
# compare salaries with prof
saltemp = survey.drop(survey.loc[survey.Salary.isnull()].index)
saltemp = saltemp.drop(saltemp.loc[saltemp.DeveloperType.isnull()].index)
developerType = list(set(developerType))

devDict = {}
for index, dev in enumerate(developerType):
    devDict[dev] = index
    
devSalaries = [[] for i in range(14)]
for index in saltemp.index:
    devlist = saltemp.DeveloperType[index].replace(" ", "").split(";")
    for d in devlist:
        devSalaries[devDict[d]].append(saltemp.Salary[index])

Salaries = []
for sal in devSalaries:
    Salaries.append(np.mean(sal))
    
devSalaries = pd.DataFrame()
devSalaries["developerType"] = developerType
devSalaries["AverageSalary"] = Salaries
devSalaries.head(14)
plt.subplots(figsize=(15,7))
sns.set_style("whitegrid")
sal = sns.barplot(x=devSalaries.developerType, y=devSalaries.AverageSalary, orient = 1);
sal.set_xticklabels(devSalaries.developerType, rotation=90);
# remote working vs satisfaction level, company type, JobSatisfaction
temp = survey.drop(survey.loc[survey.HomeRemote.isnull()].index)
temp[['HomeRemote', 'JobSatisfaction']].groupby('HomeRemote').describe()
temp[['HomeRemote', 'CareerSatisfaction']].groupby('HomeRemote').describe()
#salary vs years coded
salYears = survey.drop(survey.loc[survey.Salary.isnull()].index)
salYears = salYears.drop(salYears.loc[salYears.YearsCodedJob.isnull()].index)
salYears = salYears[['Salary', 'YearsCodedJob']].groupby('YearsCodedJob').describe()
salYears = salYears.Salary
salYears.drop(['count', '25%', '50%', '75%'], axis = 1)
# dev type vs job satisfaction
devSatisfaction = survey.drop(survey.loc[survey.JobSatisfaction.isnull()].index)
devSatisfaction = devSatisfaction.drop(devSatisfaction.loc[devSatisfaction.DeveloperType.isnull()].index)
developerType = list(set(developerType))

devDict = {}
for index, dev in enumerate(developerType):
    devDict[dev] = index
    
devSat = [[] for i in range(14)]
for index in devSatisfaction.index:
    devlist = devSatisfaction.DeveloperType[index].replace(" ", "").split(";")
    for d in devlist:
        devSat[devDict[d]].append(devSatisfaction.JobSatisfaction[index])

jobSatisfaction = []
JobSatSTD = []
JobSatMAX = []
JobSatMIN = []
for sat in devSat:
    jobSatisfaction.append(np.mean(sat))
    JobSatSTD.append(np.std(sat))
    JobSatMIN.append(np.min(sat))
    JobSatMAX.append(np.max(sat))
    
devSatisfaction = pd.DataFrame()
devSatisfaction["Developer_Type"] = developerType
devSatisfaction["Average_JobSatisfaction"] = jobSatisfaction
devSatisfaction["STD_JobSatisfaction"] = JobSatSTD
devSatisfaction["MIN_JobSatisfaction"] = JobSatMIN
devSatisfaction["MAX_JobSatisfaction"] = JobSatMAX
devSatisfaction.head(14)
# Expected Salary of Students
students = survey.loc[(survey.Professional == "Student").index]
students = students.ExpectedSalary.drop(students.loc[students.ExpectedSalary.isnull()].index)
plt.subplots(figsize=(7,7))
sns.distplot(students)
AverageSalary = np.mean(students)
print("Average Expected Salary = " + str(AverageSalary))

WebDevs = survey.drop(survey.loc[survey.WebDeveloperType.isnull()].index)
# formal Education of WebDevs
survey.Professional.unique()
sns.countplot(y = survey.FormalEducation, order = survey.FormalEducation.value_counts().index);
plt.subplots(figsize=(6,6))  
plt.pie(dict(Counter(WebDevs.FormalEducation)).values(),
        shadow = True,
        startangle = 0);
plt.legend(list(WebDevs.FormalEducation.unique()),loc = 2, bbox_to_anchor=(1.1, 1))
sns.countplot(y = WebDevs.EmploymentStatus, order = survey.EmploymentStatus.value_counts().index)
fig,ax=plt.subplots(1,2,figsize=(30,12))
sns.set(font_scale=2)
#### first figure
temp = WebDevs.drop(WebDevs.loc[WebDevs.ChallengeMyself.isnull()].index)
sns.countplot(y = temp.ChallengeMyself, ax=ax[0], label = 'big')
ax[0].set_title('How Do Devs feel about Challenging Themselves?')

#### second figure
temp = WebDevs.drop(WebDevs.loc[WebDevs.ProblemSolving.isnull()].index)
sns.countplot(y = temp.ProblemSolving, ax=ax[1])
ax[1].set_title('How Do Devs feel about Solving Great Problems?')

plt.subplots_adjust(hspace=0.1,wspace=0.6)
ax[0].tick_params(labelsize=20)
ax[1].tick_params(labelsize=20)
plt.show()

fig,ax=plt.subplots(1,2,figsize=(30,12))
sns.set(font_scale=2)
#### first figure
temp = WebDevs.drop(WebDevs.loc[WebDevs.BuildingThings.isnull()].index)
sns.countplot(y = temp.BuildingThings, ax=ax[0], label = 'big')
ax[0].set_title('How Do Devs feel about Building Things?')

#### second figure
temp = WebDevs.drop(WebDevs.loc[WebDevs.LearningNewTech.isnull()].index)
sns.countplot(y = temp.LearningNewTech, ax=ax[1])
ax[1].set_title('How Do Devs feel about Learning New Technoogies?')

plt.subplots_adjust(hspace=0.1,wspace=0.6)
ax[0].tick_params(labelsize=20)
ax[1].tick_params(labelsize=20)
plt.show()
sns.set(font_scale=1)
sns.countplot(y = survey.WebDeveloperType, order = survey.WebDeveloperType.value_counts().index);
plt.subplots(figsize=(6,6))  
plt.pie(dict(Counter(WebDevs.WebDeveloperType)).values(),
        labels = list(WebDevs.WebDeveloperType.unique()),
        shadow = True,
        startangle = 0);
sal = WebDevs.drop(WebDevs.loc[survey.Salary.isnull()].index)
sal = sal[['Salary', 'WebDeveloperType']].groupby('WebDeveloperType').describe()
sal = sal.Salary
sal.drop(['count', 'min', 'std', '25%', '50%', '75%'], axis = 1)
# compare salaries with prof
saltemp = survey.drop(survey.loc[survey.Salary.isnull()].index)
saltemp = saltemp.drop(saltemp.loc[saltemp.DeveloperType.isnull()].index)
developerType = list(set(developerType))

devDict = {}
for index, dev in enumerate(developerType):
    devDict[dev] = index
    
devSalaries = [[] for i in range(14)]
for index in saltemp.index:
    devlist = saltemp.DeveloperType[index].replace(" ", "").split(";")
    for d in devlist:
        devSalaries[devDict[d]].append(saltemp.Salary[index])

Salaries = []
for sal in devSalaries:
    Salaries.append(np.mean(sal))
    
devSalaries = pd.DataFrame()
devSalaries["developerType"] = developerType
devSalaries["AverageSalary"] = Salaries
devSalaries.head(14)
# what are tools used by web devs
# what are tools used by mobdevs
# degrees obtained by data scientists, mob devs, web devs, devops etc
# dev type vs FormalEducation
# Number of Years Web Developers and others Have Coded For
# Web Developer's preferred Frameworks