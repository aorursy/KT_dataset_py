%matplotlib inline



import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
question = pd.read_csv("../input/schema.csv")

question.shape
mcq = pd.read_csv("../input/multipleChoiceResponses.csv", encoding="ISO-8859-1", low_memory = False)

print(mcq.shape)

mcq.head()
import missingno as msno

msno.matrix(mcq, figsize = (12,5))
# 1. 성별

sns.countplot(y="GenderSelect", data=mcq)
# 2. 국가

con_df = pd.DataFrame(mcq["Country"].value_counts())

con_df["국가"] = con_df.index

con_df.columns = ["응답 수", "국가"]

con_df = con_df.reset_index().drop("index", axis=1)

con_df.index +=1 

con_df.head(20)
# 3. 연령

mcq["Age"].describe()
sns.distplot(mcq[mcq["Age"]>0]["Age"])

# NaN 데이터는 표현하지 않도록 
# 4. 학력

sns.countplot(y = "FormalEducation", data=mcq)
# 5. 전공

mcq_major_count = pd.DataFrame(

    mcq["MajorSelect"].value_counts())

mcq_major_percent = pd.DataFrame(

    mcq["MajorSelect"].value_counts(normalize=True))

mcq_major_df = mcq_major_count.merge(

    mcq_major_percent, left_index = True, right_index = True)

mcq_major_df.columns = ["응답 수", "비율"]

mcq_major_df
plt.figure(figsize=(6,8))

sns.countplot(y="MajorSelect", data=mcq)
# 6. 취업 여부 

sns.countplot(y='EmploymentStatus', data=mcq)
employdata = mcq['EmploymentStatus'].value_counts()

employdata
employdata.sum(axis=0)
sns.barplot(y=employdata.index, x=employdata)
sns.barplot(x=mcq["EmploymentStatus"].unique(),

           y=mcq["EmploymentStatus"].value_counts()/employdata.sum(axis=0))

plt.xticks(rotation=30, ha="right")

plt.title("Employment Status")

plt.ylabel("")

plt.show()
# 7. 프로그래밍 경험

sns.countplot(y="Tenure", data=mcq)
korea = mcq.loc[mcq["Country"]=="South Korea"]

print('The number of interviewees in Korea: ' + str(korea.shape[0]))



sns.distplot(korea["Age"].dropna())

plt.title("Korean")

plt.show()
pd.DataFrame(korea["GenderSelect"].value_counts())
sns.countplot(x="GenderSelect", data=korea)
figure, (ax1, ax2) = plt.subplots(ncols=2)

figure.set_size_inches(12,5)



sns.distplot(korea["Age"].loc[korea["GenderSelect"]=="Female"].dropna(),

             norm_hist = False, color=sns.color_palette("Paired")[4], ax=ax1)

plt.title("Korean Female")



sns.distplot(korea["Age"].loc[korea["GenderSelect"]=="Male"].dropna(),

             norm_hist = False, color=sns.color_palette("Paired")[0], ax=ax2)

plt.title("Korean Male")
sns.barplot(x=korea["EmploymentStatus"].unique(), 

            y=korea["EmploymentStatus"].value_counts()/len(korea))

plt.xticks(rotation=30, ha="right")

plt.title("Employment Status of Korea")

plt.ylabel("")

plt.show()
korea["StudentStatus"] = korea["StudentStatus"].fillna("No")

sns.countplot(x="StudentStatus", data=korea)

plt.title("Korean Student Status")

plt.show()
full_time = mcq.loc[mcq["EmploymentStatus"] == 'Employed full-time']

print(full_time.shape)

looking_for_job = mcq.loc[

    mcq["EmploymentStatus"] == 'Not employed, but looking for work']

print(looking_for_job.shape)
# 1. 선호 언어

sns.countplot(y="LanguageRecommendationSelect", data=mcq)
# 2. 현재 하는 일

sns.countplot(y=mcq["CurrentJobTitleSelect"])
mcq["CurrentJobTitleSelect"].describe()
# 3. 현재 하는 일에 대해 응답을 해 준 사람 중 Python과 R을 사용하는 사람

data = mcq[(mcq['CurrentJobTitleSelect'].notnull()) & (

 (mcq['LanguageRecommendationSelect'] == 'Python') | (

 mcq['LanguageRecommendationSelect'] == 'R'))]

print(data.shape)

plt.figure(figsize=(8, 10))

sns.countplot(y='CurrentJobTitleSelect',

 hue='LanguageRecommendationSelect',

 data=data)
# 4. 데이터 사이언스 툴

mcq_tool_count = pd.DataFrame(

mcq['MLToolNextYearSelect'].value_counts())

mcq_tool_percent = pd.DataFrame(

mcq['MLToolNextYearSelect'].value_counts(normalize = True))

mcq_tool_df = mcq_tool_count.merge(

mcq_tool_percent, left_index = True, right_index = True).head(20)

mcq_tool_df.columns = ["응답 수", "비율"]

mcq_tool_df
data_tool = mcq['MLToolNextYearSelect'].value_counts().head(20)

sns.barplot(y=data_tool.index , x=data_tool)
# 5. data science method

sns.countplot(y=mcq["MLMethodNextYearSelect"])
data = mcq["MLMethodNextYearSelect"].value_counts().head(15)

sns.barplot(y=data.index, x=data)
mcq["LearningPlatformSelect"] = mcq["LearningPlatformSelect"].astype("str").apply(lambda x:x.split(","))

s = mcq.apply(lambda x: pd.Series(x["LearningPlatformSelect"]),axis=1).stack().reset_index(level=1, drop=True)

s.name = "platform"
plt.figure(figsize=(6,8))

data = s[s != "nan"].value_counts().head(15)

sns.barplot(y=data.index, x=data)
s
qc = question.loc[question["Column"].str.contains("LearningCategory")]

print(qc)
# 6. 학습플랫폼과 유용함에 대한 연관성 살펴보기

use_features = [x for x in mcq.columns if x.find("LearningPlatformUsefulness") != -1]
use_features
fdf = {}

for feature in use_features: 

    a = mcq[feature].value_counts()

    a = a/a.sum()

    fdf[feature[len("LearningPlatformUsefulness"):]] = a



fdf = pd.DataFrame(fdf).transpose().sort_values(

"Very useful", ascending=False)
fdf
plt.figure(figsize=(10,10))

sns.heatmap(

fdf.sort_values("Very useful", ascending = False), annot=True)
fdf.plot(kind="bar", figsize=(20,8), title = "Usefulness of Learning Platforms")
cat_features = [x for x in mcq.columns if x.find("LearningCategory") != -1]

cat_features
cdf = {}

for feature in cat_features:

    cdf[feature[len("LearningCategory"):]] = mcq[feature].mean()



cdf = pd.Series(cdf)

cdf
plt.pie(cdf, labels=cdf.index, autopct='%1.1f%%', startangle=0)

plt.title("Contribution of each Platform to Learning")

plt.show()
# 7. 데이터 사이언스를 위해 높은 사양의 컴퓨터가 필요한지

qc = question.loc[question["Column"].str.contains("HardwarePersonalProjectsSelect")]

print(qc)
mcq[mcq["HardwarePersonalProjectsSelect"].notnull()][

    "HardwarePersonalProjectsSelect"].shape
mcq["HardwarePersonalProjectsSelect"] = mcq["HardwarePersonalProjectsSelect"].astype("str").apply(lambda x: x.split(","))

s = mcq.apply(lambda x:pd.Series(x["HardwarePersonalProjectsSelect"]), axis=1).stack().reset_index(level=1, drop=True)

s.name = "hardware"
s = s[s != "nan"]

pd.DataFrame(s.value_counts())
plt.figure(figsize=(6,8))

data = s[s != "nan"].value_counts().head(15)

sns.barplot(y=data.index, x=data)
plt.pie(data, labels = data.index,autopct='%1.1f%%')
mcq["TimeSpentStudying"].value_counts()
mcq["TimeSpentStudying"] = pd.Categorical(mcq["TimeSpentStudying"],

                                categories = ["0 - 1 hour", "2 - 10 hours",

                                             "11 - 39 hours", "40+"], ordered = True)
# 8. 데이터 사이언스 공부에 얼마나 많은 시간을 할애하는지

plt.figure(figsize = (6,8))

sns.countplot(y = "TimeSpentStudying", data=mcq, hue = "EmploymentStatus").legend(loc='center left',

 bbox_to_anchor=(1, 0.5))
figure, (ax1, ax2) = plt.subplots(ncols=2)



figure.set_size_inches(12,5)

sns.countplot(x="TimeSpentStudying", data=full_time, hue="EmploymentStatus",ax = ax1)

sns.countplot(x="TimeSpentStudying", data=looking_for_job, hue="EmploymentStatus",ax = ax2,

             order = ["0 - 1 hour", "2 - 10 hours", "11 - 39 hours", "40+"])
timespent = mcq.pivot_table(

index = "TimeSpentStudying", columns = "EmploymentStatus", aggfunc="size").fillna(0).astype("int")

timespent
plt.figure(figsize=(10,10))

sns.heatmap(timespent,cmap=sns.light_palette("gray"), annot=True, fmt="d")

plt.title("Heatmap")

plt.show()
# 1. 플랫폼 추천

mcq["BlogsPodcastsNewslettersSelect"]=mcq["BlogsPodcastsNewslettersSelect"].astype("str").apply(lambda x: x.split(","))

mcq["BlogsPodcastsNewslettersSelect"].head()
s = mcq.apply(lambda x : pd.Series(x["BlogsPodcastsNewslettersSelect"]), 

             axis=1).stack().reset_index(level=1, drop=True)

s.name="platforms"

s=s[s!="nan"].value_counts().head(20)
plt.figure(figsize=(6,8))

plt.title("Most popular Blogs and Podcasts")

sns.barplot(y=s.index, x=s)
mcq['CoursePlatformSelect']= mcq['CoursePlatformSelect'].astype("str").apply(lambda x : x.split(","))

t = mcq.apply(lambda x: pd.Series(x['CoursePlatformSelect']), axis=1).stack().reset_index(level=1, drop=True)

t.name = "courses"

t.head()
t = t[t != "nan"].value_counts()
plt.title("Most Popular Course Platforms")

sns.barplot(y=t.index, x=t)
# 2. 데이터 사이언스 직무에서 가장 중요하다고 생각되는 스킬

job_features = [

    x for x in mcq.columns if x.find(

    "JobSkillImportance") != -1 

    and x.find("JobSkillImportanceOther") == -1]

job_features
jdf = {}

for feature in job_features:

    a = mcq[feature].value_counts()

    a = a/a.sum()

    jdf[feature[len("JobSkillImportance"):]] = a
jdf = pd.DataFrame(jdf).transpose()

jdf
plt.figure(figsize = (10,6))

sns.heatmap(jdf.sort_values("Necessary", ascending = False), annot=True)
jdf.plot(kind="bar", figsize = (12,6),

         title = "Skill Importance in Data Science Jobs")
# 3. data scientists들의 평균 급여

mcq["CompensationAmount"].describe()
mcq["CompensationAmount"] = mcq["CompensationAmount"].str.replace(",", "")

mcq["CompensationAmount"] = mcq["CompensationAmount"].str.replace("-", "")



rates = pd.read_csv("../input/conversionRates.csv")

rates.drop("Unnamed: 0", axis=1, inplace = True)



salary = mcq[["CompensationAmount", "CompensationCurrency", "GenderSelect",

            "Country", "CurrentJobTitleSelect"]].dropna()

salary = salary.merge(rates, left_on="CompensationCurrency", right_on = "originCountry", how="left")

salary["Salary"] = pd.to_numeric(salary["CompensationAmount"], errors='coerce') * salary["exchangeRate"]

print(salary.head())
print('Maximum Salary is USD $',

 salary['Salary'].dropna().astype(int).max())

print('Minimum Salary is USD $',

 salary['Salary'].dropna().astype(int).min())

print('Median Salary is USD $',

 salary['Salary'].dropna().astype(int).median())
plt.subplots(figsize = (15,8))

salary = salary[salary['Salary']<1000000]

sns.distplot(salary['Salary'])

plt.axvline(salary["Salary"].median(), linestyle = "dashed")

plt.title("Salary Distribution", size=15)

plt.show()
plt.subplots(figsize = (8,12))



sal_coun = salary.groupby("Country")["Salary"].median().sort_values(

ascending=False)[:30].to_frame()



sns.barplot("Salary", sal_coun.index, data=sal_coun, palette="RdYlGn")



plt.axvline(salary["Salary"].median(), linestyle= "dashed")

plt.title("Highest Salary Paying Countries")
plt.subplots(figsize=(8,4))

sns.boxplot(y="GenderSelect", x="Salary", data=salary)
salary_korea = salary.loc[(salary["Country"]=="South Korea")]

plt.subplots(figsize =(8,4))

sns.boxplot(y="GenderSelect", x="Salary", data=salary_korea)
salary_korea_male = salary_korea[

    salary_korea["GenderSelect"] == "Male"]

salary_korea_male["Salary"].describe()
# 1. Job Satisfaction

mcq["JobSatisfaction"].value_counts()
mcq["JobSatisfaction"].replace(

{"10 - Highly Satisfied" : "10", 

"1 - Highly Dissatisfied" : "1", 

"I prefer not to share": np.NaN}, inplace = True)



mcq.dropna(subset=["JobSatisfaction"], inplace = True)
mcq["JobSatisfaction"] = mcq["JobSatisfaction"].astype(int)

satisfy_job = mcq.groupby(["CurrentJobTitleSelect"])["JobSatisfaction"].mean().sort_values(ascending = False).to_frame()

satisfy_job
ax = sns.barplot(y=satisfy_job.index, x=satisfy_job.JobSatisfaction, 

                palette=sns.color_palette("inferno", 20))

fig = plt.gcf()

fig.set_size_inches(8,10)

for i, v in enumerate(satisfy_job.JobSatisfaction):

    ax.text(.1, i, round(v,3), fontsize=10, color="white", weight="bold")

plt.title("Job Satisfaction out of 10")

plt.show()
import plotly.offline as py

py.init_notebook_mode(connected=True)
data = [dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = 'Viridis',

        reversescale = True,

        showscale = True,

        locations = satisfy_job.index,

        z = satisfy_job['JobSatisfaction'],

        locationmode = 'country names',

        text = satisfy_job['JobSatisfaction'],

        marker = dict(

            line = dict(color = 'rgb(200,200,200)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Satisfaction')

            )

       ]



layout = dict(

    title = 'Job Satisfaction By Country',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(0,0,255)',

        projection = dict(

        type = 'chloropleth',

            

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap2018')