

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding = "ISO-8859-1")
sal = pd.read_csv('/kaggle/input/kaggle-survey-2017/conversionRates.csv')
w = df['BlogsPodcastsNewslettersSelect'].value_counts().head(10)

sns.barplot(y=w.index, x= w)
def kdeplot(feature):

    plt.figure(figsize=(9, 4))

    plt.title("KDE for {}".format(feature))

    ax0 = sns.kdeplot(df[df['EmploymentStatus'] == 'Employed full-time'][feature].dropna(), color= 'navy', label= 'Employed full-time')

    ax1 = sns.kdeplot(df[df['EmploymentStatus'] == 'Not employed, but looking for work'][feature].dropna(), color= 'orange', label= 'Not employed, but looking for work')

kdeplot('Age')
job_factors = [

    x for x in df.columns if x.find('JobFactor') != -1]
jfdf = {}

for feature in job_factors:

    a = df[feature].value_counts()

    a = a/a.sum()

    jfdf[feature[len('JobFactor'):]] = a
jfdf = pd.DataFrame(jfdf).transpose()



jfdf.plot(kind='bar', figsize=(18,6), 

          title="Things to look for while considering Data Science Jobs")

plt.show()

w = df[(df['EmploymentStatus'] == 'Employed full-time')]

plt.title("Languages used by full-time workers")

sns.countplot(y='LanguageRecommendationSelect', data = w, order=w['LanguageRecommendationSelect'].value_counts().head(15).index)
sns.countplot(y='EmploymentStatus', data=df, order=df['EmploymentStatus'].value_counts().head(15).index)
q = df['LanguageRecommendationSelect']

sns.countplot(y='LanguageRecommendationSelect', data=df, order=df['LanguageRecommendationSelect'].value_counts().head(15).index)
features = [x for x in df.columns if x.find('Learning') != -1]

features
df['LearningPlatformSelect'] = df['LearningPlatformSelect'].astype('str').apply(lambda x : x.split(','))



s = df.apply(lambda x : pd.Series(x['LearningPlatformSelect']),axis =1).stack().reset_index(level = 1, drop=True)

s = s[s != 'nan'].value_counts().head(10)

sns.barplot(y=s.index, x=s)
df['HardwarePersonalProjectsSelect'] = df['HardwarePersonalProjectsSelect'].astype('str').apply(lambda x: x.split(','))



d = df.apply(lambda x: pd.Series(x['HardwarePersonalProjectsSelect']),axis=1).stack().reset_index(level=1, drop=True)

d = d[d != 'nan'].value_counts().head(15)

sns.barplot(y=d.index, x=d)
sns.countplot(y='CurrentJobTitleSelect', data=df, order=df['CurrentJobTitleSelect'].value_counts().head(15).index)
j = df[df['EmploymentStatus'] == 'Employed full-time']

plt.title('LearningDataScienceTime of Employed full-time')

sns.countplot(y=j.LearningDataScienceTime, order=j['LearningDataScienceTime'].value_counts().head(10).index)
g = df[(df['EmploymentStatus'] == 'Employed full-time') & 

      (df['LearningDataScienceTime'] == '< 1 year')]

plt.figure(figsize=(20,12))

sns.countplot(y=g.Country, order=g['Country'].value_counts().head(20).index)
df['PublicDatasetsSelect'] = df['PublicDatasetsSelect'].astype('str').apply(lambda x : x.split(','))

f = df.apply(lambda x : pd.Series(x['PublicDatasetsSelect']), axis=1).stack().reset_index(level=1, drop=True)

f = f[f != 'nan'].value_counts().head(15)

pd.DataFrame(f)

sns.barplot(y=f.index, x=f)
s = df['EmploymentStatus'].value_counts().head(10)

sns.barplot(y=s.index, x=s)
korea = df[df['Country'] == 'South Korea']



a = korea.apply(lambda x : pd.Series(x['CurrentEmployerType']), axis=1).stack().reset_index(level = 1, drop=True)

a = a[a != 'nan'].value_counts().head(10)



plt.title('Employer type of Korean Kagglers')

sns.barplot(y=a.index, x=a)
korean = df[df['Country'] == 'South Korea']



korean['PastJobTitlesSelect'] = korean['PastJobTitlesSelect'].astype('str').apply(lambda x : x.split(','))

q = korean.apply(lambda x : pd.Series(x['PastJobTitlesSelect']), axis=1).stack().reset_index(level=1, drop=True)

q = q[q != 'nan'].value_counts().head(10)



plt.title('Past jobs of Korean Kagglers')

sns.barplot(y=q.index, x=q)
def kdeplot(feature):

    plt.figure(figsize=(9, 4))

    plt.title("KDE for {}".format(feature))



    ax0 = sns.kdeplot(df[df['Country'] == 'United States'][feature].dropna(), label= 'United States')

    ax1 = sns.kdeplot(df[df['Country'] == 'South Korea'][feature].dropna(), label= 'South Korea')

    ax2 = sns.kdeplot(df[df['Country'] == 'India'][feature].dropna(), label= 'India')

    

kdeplot('TimeGatheringData')

kdeplot('TimeProduction')

kdeplot('TimeVisualizing')
a = [x for x in df.columns if x.find('Time') != -1]

a
i = {}

for feature in a:

    s = df[feature].value_counts().head(5)

    s = s/s.sum()

    i[feature[len('Time'):]] = s

i
df['WorkCodeSharing'] = df['WorkCodeSharing'].astype('str').apply(lambda x : x.split(','))



s = df.apply(lambda x : pd.Series(x['WorkCodeSharing']), axis=1).stack().reset_index(level=1, drop=True)

s = s[s != 'nan']

s.name = 'codeshare'

s = pd.DataFrame(s)
q = s['codeshare'].value_counts().head(10)

sns.barplot(y=q.index, x=q)
sns.countplot(y=df.WorkMLTeamSeatSelect, hue =df.GenderSelect).legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
f = df[df['GenderSelect'] == 'Female']

f = f['WorkMLTeamSeatSelect'].value_counts().head(10)



sns.barplot(y=f.index, x=f)
sal = sal.drop(['Unnamed: 0'], axis=1)
df['CompensationAmount'] = df['CompensationAmount'].str.replace(',','')

df['CompensationAmount'] = df['CompensationAmount'].str.replace('-','')



salary = df[['CompensationAmount','CompensationCurrency']].dropna()
salary = salary.merge(sal, left_on ='CompensationCurrency', right_on='originCountry', how='left')
salary['salary'] = pd.to_numeric(salary['CompensationAmount'])* salary['exchangeRate']
def sal(x):

    if x <= 200000:

        return '~200,000'

    elif 200000 < x and x <= 400000:

        return '200,000 ~ 400,000'

    else:

        return '400,000 ~ 500,000'
salary['salary_cat'] = salary['salary'].map(sal)
n = df[(df['GenderSelect']== 'Male') &

      (df['EmploymentStatus'] == 'Employed full-time')]

n = n['WorkMLTeamSeatSelect'].value_counts().head()



sns.barplot(y=n.index, x=n)