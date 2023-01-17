# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_columns', 1000)

plt.rcParams['figure.figsize'] = (17.0, 6.0)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', 1000)

plt.rcParams['figure.figsize'] = (17.0, 6.0)

plt.rcParams['figure.titlesize'] = 16

plt.rcParams['figure.titleweight'] = 'bold'

plt.rcParams['axes.titlesize'] = 16

plt.rcParams['axes.titleweight'] = 'bold'

plt.rcParams["axes.labelsize"] = 13

plt.rcParams["axes.labelweight"] = 'bold'

plt.rcParams["xtick.labelsize"] = 12

plt.rcParams["ytick.labelsize"] = 12
data = pd.read_csv("../input/2016-FCC-New-Coders-Survey-Data.csv")
gen_inc_labels = ["{0}-{1}".format(i,i+15000) for i in range(6000,200000,15000)]

male_inc = pd.cut(data[data.Gender == "male"]["Income"], range(6000,210000,15000),right=False,labels=gen_inc_labels)

g = sns.barplot(male_inc.value_counts().index,(male_inc.value_counts().values/(len(male_inc)*1.0)))

g.set_xticklabels(g.get_xticklabels(),rotation=90)

g.set_title("Income Distribution of Male programmers")

plt.xlabel("Income")

plt.ylabel("Male population(%age)")

plt.show()
gen_inc_labels = ["{0}-{1}".format(i,i+15000) for i in range(6000,200000,15000)]

female_inc = pd.cut(data[data.Gender == "female"]["Income"], range(6000,210000,15000),right=False,labels=gen_inc_labels)

g = sns.barplot(female_inc.value_counts().index,(female_inc.value_counts().values/(len(female_inc)*1.0)))

g.set_xticklabels(g.get_xticklabels(),rotation=90)

g.set_title("Income Distribution of Female programmers")

plt.xlabel("Income")

plt.ylabel("Female population(%age)")

plt.show()
code_event_df = pd.DataFrame()

for i in data.columns.tolist():

    if "CodeEvent" in i:

        temp_df = pd.DataFrame([[i,data[i].value_counts()[1]]])

        code_event_df = code_event_df.append(temp_df)

code_event_df.columns = ["CodeEventName","Count"]

g = sns.barplot(code_event_df.CodeEventName,code_event_df.Count)

g.set_xticklabels(g.get_xticklabels(),rotation=55)

g.set_title("Popularity of Code Events amongsts Programmers")

plt.ylabel("Count")

plt.show()
sns.countplot(data["Gender"])

g = plt.title("Distribution of New Programmers over different Gender ")
g = sns.violinplot(y=data["Age"],x=data["EmploymentStatus"],jitter=True,hue=data.Gender.map(lambda x : x if x == "male" or x == "female" else "other"),alpha=1,bw=1.1)

g.set_xticklabels(g.get_xticklabels(),rotation=90)

g = plt.title("Age Distribution of different genders over possible values of Employment Status")
temp = data[["ExpectedEarning","MoneyForLearning"]].dropna()

g = sns.jointplot(y=temp["ExpectedEarning"],x=temp["MoneyForLearning"],xlim = (-100,10500),ylim = (0,109000),kind="kde",size=10,ratio=8)

g.fig.suptitle("Money spent on learning v/s Expected earning ")

g.fig.subplots_adjust(top=0.95)
colors = matplotlib.cm.Paired(np.linspace(0, 1, 200))

s = [n/80 for n in range(len(temp))]

g = plt.figure(figsize=(17,17))

g = plt.scatter(y=temp["ExpectedEarning"],x=temp["MoneyForLearning"],s=s,color=colors,alpha=0.2)

g = plt.xlim(-500,10500)

g = plt.ylim(0,109000)

g = plt.xlabel("Money spent on Learning")

g = plt.ylabel("Expected Earning")

g = plt.title("Money spent on learning v/s Expected earning ")
g = sns.factorplot("EmploymentField", data=data, aspect=4, kind="count")

g.set_xticklabels(rotation=90)

plt.xlabel("Employment Field")

g = plt.title("Distribution of coders among various Employment Fields")
t = pd.DataFrame(data.groupby(["EmploymentField","AttendedBootcamp"])["EmploymentField","AttendedBootcamp"].size())

t = t.unstack(1)

t.columns = ["Attended Bootcamp","Not Attended Bootcamp"]

t.plot.bar(stacked=True)

plt.ylabel("Employment Field")

plt.ylabel("Count")

plt.title("Distribution of Coders over different Employment Field")
colors = matplotlib.cm.Paired(np.linspace(0, 1, 23))

temp = pd.DataFrame(data.BootcampName.value_counts())

temp.reset_index(inplace=True)

temp.columns = ["BootcampName","Count"]

temp = temp[(temp.BootcampName != 'Free Code Camp is not a bootcamp - please scroll up and change answer to "no"') & (temp.Count > 9)]

g = temp.plot.bar(color = colors)

g = g.set_xticklabels(temp.BootcampName,rotation=90)

plt.xlabel("Boot Camp Name")

plt.ylabel("Count of Participants")

plt.title("Popularity of Boot Camps")
job = data.groupby(["AttendedBootcamp","BootcampRecommend"])[["AttendedBootcamp","BootcampRecommend"]].size()

g = job.plot.bar(stacked=True)

g.set_xticklabels(["False","True"])

plt.xlabel("IsRecommended")

plt.ylabel("Count")

g = plt.title("If Boot campers reccommend bootcamps or not")
job = data.groupby(["AttendedBootcamp","BootcampFullJobAfter"])[["AttendedBootcamp","BootcampFullJobAfter"]].size()

g = job.plot.bar(stacked=True)

g.set_xticklabels(["False","True"])

plt.xlabel("Has Full-Time Job")

plt.ylabel("Count")

g = plt.title("If the Bootcamp Attendees has Full time job or not")
g = sns.factorplot("Age", data=data, aspect=4, kind="count")

g.set_xticklabels(rotation=90)

g = plt.title("Distribution of New Programmers over different Ages")
labels = [ "{0} - {1}".format(i, i + 5) for i in range(10, 90, 5) ]

data['group'] = pd.cut(data.Age, range(10, 92, 5), right=False, labels=labels)

age_att = pd.pivot_table(data=data,index="group",columns=["AttendedBootcamp"],aggfunc={"AttendedBootcamp":np.size})

age_att.columns = age_att.columns.droplevel()

age_att.columns = ['0','1']

labels = [ "{0} - {1}".format(i, i + 5) for i in range(10, 90, 5) ]

data['group'] = pd.cut(data.Age, range(10, 92, 5), right=False, labels=labels)

colors = matplotlib.cm.Paired(np.linspace(0, 1, 60))

age_att['1'].plot.bar(stacked=True,color = colors)

plt.title("Age distribution of programmers who have attended boot camps")

plt.xlabel("Age Group")

g = plt.ylabel("Count")
age_att['0'].plot.bar(stacked=True,color = colors)

plt.title("Age distribution of programmers who haven't attended any boot camps")

plt.xlabel("Age Group")

g = plt.ylabel("Count")
con_boot = pd.pivot_table(data=data,index="CountryLive", columns=["AttendedBootcamp"],aggfunc={"CountryLive":np.size})

con_boot.columns = con_boot.columns.droplevel()

con_boot.columns = ['0','1']

con_boot.sort("1",inplace=True,axis=0,ascending=False)

colors = matplotlib.cm.Paired(np.linspace(0, 1, 30))

g = con_boot['1'][:10].plot.bar(color = colors)

plt.ylabel("Count")

plt.xlabel("Country")

g = plt.title("Top 10 countries with highest Bootcamp attendees")
online_learning_col = []

for i in data.columns.values:

    if "Resource" in i:

        online_learning_col.append(i)

online_learning_col.append("CountryLive")

con_on = data[online_learning_col]

con_on['Sum'] = con_on.sum(axis=1)

tes = con_on.groupby(['CountryLive'])['Sum'].sum().sort_values(ascending=False)

colors = matplotlib.cm.Paired(np.linspace(0, 1, 60))

tes[:10].plot.bar(color=colors)

plt.title("Top 10 countries with highest count of programmers enrolled in Online courses")

plt.xlabel("Country")

plt.ylim(0,21000)

g = plt.ylabel("Count")
bootcamp = data[data.AttendedBootcamp == 1][['AttendedBootcamp','BootcampFinish','CommuteTime']]

bootcamp = bootcamp.groupby(['AttendedBootcamp','BootcampFinish','CommuteTime']).size()

a = bootcamp.unstack(0).unstack(0).dropna().plot.bar(stacked=True,title="Relation between Commute time and completing/not completing Bootcamp")
bootcamp = data[data.AttendedBootcamp == 1][['AttendedBootcamp','BootcampFinish','HasDebt']]

bootcamp = bootcamp.groupby(['AttendedBootcamp','BootcampFinish','HasDebt']).size()

g = bootcamp.unstack(0).unstack(0).dropna().plot.bar(stacked=True,title="Relation between Debt and completing/not completing Bootcamp")

g.set_xticklabels(["False","True"])

g = plt.ylabel("Count")