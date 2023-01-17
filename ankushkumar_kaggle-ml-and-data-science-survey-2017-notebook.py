# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)

import operator

import warnings

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
#Loading below csv to pandas dataframe

convRates = pd.read_csv('../input/conversionRates.csv', encoding="ISO-8859-1")

data = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")

print(data.shape)
print(data.head())
country=data['Country'].value_counts()

print("Total Respondents : {}".format(data.shape[0]))

print("Total Countries : {}".format(len(country)))
gender=data['GenderSelect'].value_counts()

fig, ax = plt.subplots(figsize=(10,6))



y_pos = np.arange(len(gender.index))



ax.barh(y_pos, gender,align='center',

        color='RGB', ecolor='black',alpha=0.6)

ax.set_yticks(y_pos)

ax.set_yticklabels(gender.index)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('No. of Respondents')

ax.set_ylabel('Gender')



plt.show()
print("Women % in Survey : {}".format(((gender[1]*100.0)/sum(gender))))

print("Men % in Survey: {}".format(((gender[0]*100.0)/sum(gender))))
country=data['Country'].value_counts().head(10)

plt.figure(figsize=(12,8))

sns.barplot(y=country.index, x=country.values,palette='Set1', alpha=.60)

plt.title("Distribution", fontsize=20)

plt.xlabel("Number of respondents", fontsize=16)

plt.ylabel("Country", fontsize=16)

plt.show();
plt.figure(figsize=(12,8))

age=data.Age[data.Age>0]

plt.hist(age,bins=25,normed=True)

plt.title('Age Distribution',size=20)

plt.xlabel('Age in Year.',size=16)

plt.show()
print("Median Age of Respondents : %d year"%(np.median(age)))
data['CompensationAmount']=data['CompensationAmount'].str.replace(',','')

data['CompensationAmount']=data['CompensationAmount'].str.replace('-','')

salary=data[['CompensationAmount','CompensationCurrency','GenderSelect','Country','CurrentJobTitleSelect']].dropna()

salary=salary.merge(convRates,left_on='CompensationCurrency',right_on='originCountry',how='left')

salary['Salary']=pd.to_numeric(salary['CompensationAmount'])*salary['exchangeRate']

print('Maximum Salary is USD $',salary['Salary'].dropna().astype(int).max())

print('Minimum Salary is USD $',salary['Salary'].dropna().astype(int).min())

print('Median Salary is USD $',salary['Salary'].dropna().astype(int).median())



plt.figure(figsize=(12,8))

salary=salary[salary['Salary']<1000000]

sns.distplot(salary['Salary'],color='g')

plt.title('Salary Distribution',size=20)

plt.show()
plt.figure(figsize=(15,12))

edu=data['FormalEducation'].value_counts()

sns.barplot(y=edu.index, x=edu.values, palette='Set2',alpha=1.0)

plt.title("", fontsize=20)

plt.xlabel("Number of respondents", fontsize=16)

plt.ylabel("Degree", fontsize=16)

plt.show();
print("Master's degree  : %0.2f %%"%((edu[0]*100)/sum(edu)))

print("Bachelor's degree  : %0.2f %%"%((edu[1]*100)/sum(edu)))

print("Doctoral degree  : %0.2f %%"%((edu[2]*100)/sum(edu)))
data['MajorSelect']=data['MajorSelect'].replace(to_replace ='Information technology, networking, or system administration',

                                                       value = 'Info. tech / Sys. admin', axis=0)

major=data['MajorSelect'].value_counts()

plt.figure(figsize=(15,12))

sns.barplot(y=major.index, x=major.values, palette="Set3",alpha=1.0)

plt.title("", fontsize=20)

plt.xlabel("Number of respondents", fontsize=16)

plt.ylabel("Major", fontsize=16)

plt.show();

print("Computer Science Major : %0.2f %%"%((major[0]*100)/sum(major)))

print("Mathematics or Statistics Major  : %0.2f %%"%((major[1]*100)/sum(major)))

print("Engineering(non-computer focused)  : %0.2f %%"%((major[2]*100)/sum(major)))
plt.figure(figsize=(15,10))

job=data['EmploymentStatus'].value_counts()

plt.pie(job, labels=job.index,autopct='%1.1f%%')

        

#draw a circle at the center of pie to make it look like a donut

centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)





# Set aspect ratio to be equal so that pie is drawn as a circle.

plt.axis('equal')

plt.show();
plt.figure(figsize=(15,10))

platform=data['LearningPlatformSelect'].value_counts()[:15]

sns.barplot(y=platform.index,x=platform,palette='Set2')

plt.xlabel("No. of respondents",size=16)

plt.ylabel("Platform",size=16)

plt.title("Top 15 Most Frequently used Platform",size=20)

plt.plot()

plt.figure(figsize=(15,10))

skills=data['MLSkillsSelect'].str.split(',')

skills_set=[]

for i in skills.dropna():

    skills_set.extend(i)

plt1=pd.Series(skills_set).value_counts().sort_values(ascending=False).to_frame()

sns.barplot(plt1[0],plt1.index,palette='BuGn_d')

plt.xlabel('No. of respondents',size=16)

plt.ylabel('ML Skills',size=16)

plt.show()
plt.figure(figsize=(15,10))

time=data['LearningDataScienceTime'].value_counts()

sns.barplot(time.index,time,palette='Set1')

plt.xlabel("Time in years",size=16)

plt.ylabel("No. of respondents",size=16)

plt.title("Time Distribution for learning Data Science",size=20)

plt.plot()
time=data['Tenure'].value_counts()

s = [400*(len(time)-n) for n in range(len(time))]

plt.figure(figsize=(15,10))

plt.scatter(time.index,time,s=s,color='rgby',alpha=0.6)

plt.title("Tenure Distribution",size=20)

plt.plot()
#print(schema.to_string())

plt.figure(figsize=(15,10))

js=data.copy()

js['JobSatisfaction'].replace({'10 - Highly Satisfied':'10','1 - Highly Dissatisfied':'1','I prefer not to share':np.NaN},inplace=True)

js.dropna(subset=['JobSatisfaction'],inplace=True)

js['JobSatisfaction']=js['JobSatisfaction'].astype(int)

js=js.groupby(['CurrentJobTitleSelect'])['JobSatisfaction'].mean().sort_values(ascending=False).to_frame()

sns.barplot(js.JobSatisfaction,js.index,palette=sns.color_palette('inferno',16))

plt.plot()


lang=data['WorkToolsSelect'].dropna().to_frame()

lang1={'Python':len(lang[lang.WorkToolsSelect.str.contains('Python')]),'R':len(lang[lang.WorkToolsSelect.str.contains('R')]),'Both':len(lang[lang.WorkToolsSelect.str.contains('Python')&lang.WorkToolsSelect.str.contains('R')])}

plt.figure(figsize=(15,10))

from matplotlib_venn import venn2, venn2_circles



v = venn2(subsets=(2,1,1))

v.get_label_by_id('10').set_text(lang1['Python'])

v.get_label_by_id('01').set_text(lang1['R'])

v.get_label_by_id('11').set_text(lang1['Both'])

plt.title("Venn diagram",size=20)

v.get_label_by_id('A').set_text('Python')

v.get_label_by_id('B').set_text('R')

plt.show()
plt.figure(figsize=(15,10))

recomm=data['LanguageRecommendationSelect'].dropna().value_counts()

sns.barplot(recomm.index,recomm,palette='Set2')

plt.xlabel('Language',size=16)

plt.ylabel('No. of Respondents',size=16)

plt.title("Language Recommendation Distribution ",size=20)

plt.show()
plt.figure(figsize=(15,10))

pk=data.ProveKnowledgeSelect.dropna().value_counts()

s=[500*(len(pk)-i) for i in range(len(pk))]

plt.scatter(pk,pk.index,s=s)

plt.plot()
plt.figure(figsize=(15,10))

ws=data['BlogsPodcastsNewslettersSelect'].dropna().apply(lambda x: pd.value_counts(x.split(","))).sum(axis = 0).sort_values(ascending=False)[:10]

sns.barplot(ws,ws.index,palette='Set1')

plt.title('Top 10 Famous BlogsPodcastsNewsletters',size=20)

plt.plot()


ds=data['DataScienceIdentitySelect'].value_counts()

plt.figure(figsize=(15,10))

plt.pie(ds,explode=(0.05,0,0), labels=ds.index, autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Surveyor's Answer")

plt.show()