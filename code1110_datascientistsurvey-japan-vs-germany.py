# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# ignore warning

import warnings

warnings.filterwarnings('ignore')



# for visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

import plotly.graph_objs as go



# Any results you write to the current directory are saved as output.
# load data

cvRates = pd.read_csv('../input/conversionRates.csv', encoding="ISO-8859-1")

freeForm = pd.read_csv('../input/freeformResponses.csv', encoding="ISO-8859-1")

data = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")

schema = pd.read_csv('../input/schema.csv', encoding="ISO-8859-1")
data.head()
data.shape
# list of column names

print(list(data))
# Let's just take these two countries for now

japan = data.loc[(data['Country']=='Japan')]

germany = data.loc[(data['Country']=='Germany')]



print('The number of interviewees in Japan: ' + str(japan.shape[0]))

print('The number of interviewees in Germany: ' + str(germany.shape[0]))
# define functions for visualization

def goodax(ax):

    ax.spines['right'].set_color('none')

    ax.spines['top'].set_color('none')

    ax.spines['left'].set_smart_bounds(True)

    ax.spines['bottom'].set_smart_bounds(True)

    

def putPercent(ax, data):

    for p in ax.patches:

        x = p.get_bbox().get_points()[:,0]

        y = p.get_bbox().get_points()[1,1]

        ax.annotate('{:.1f}%'.format(100.*y/len(data)), (x.mean(), y), 

                ha='center', va='bottom')  

        

def putMedian(xx, yy, med):

    plt.plot(med*np.array([1,1]),yy,'-r')

    plt.text(med+(xx[1]-xx[0])*0.05,yy[1]*0.8,'median = ' + str(np.round(med).astype(int)),color='r')

    plt.xlim(xx)

    plt.ylim(yy)
# Age

plt.figure(figsize=(10,5))

plt.subplot(121)

ax = sns.distplot(japan['Age'].dropna(),color='k')

putMedian(np.array([0,80]),np.array([0,0.08]),np.median(japan['Age'].dropna()))

plt.title('Japanese')

goodax(ax)



plt.subplot(122)

ax = sns.distplot(germany['Age'].dropna(),color='k')

putMedian(np.array([0,80]),np.array([0,0.08]),np.median(germany['Age'].dropna()))

plt.title('German')

goodax(ax)



plt.tight_layout()

plt.show()
# How many male and females?

plt.figure(figsize=(10,5))

plt.subplot(121)

ax = sns.countplot(x='GenderSelect', data=japan)

putPercent(ax,japan)

plt.xticks(rotation=30, ha='right')

plt.ylabel('count (Japan)')

plt.xlabel('')

goodax(ax)



plt.subplot(122)

ax = sns.countplot(x='GenderSelect', data=germany)

putPercent(ax,germany)

plt.xticks(rotation=30, ha='right')

plt.ylabel('count (Germany)')

plt.xlabel('')

goodax(ax)



plt.tight_layout()

plt.show()
# Age in each gender?

plt.figure(figsize=(10,5))



plt.subplot(221)

ax = sns.distplot(japan['Age'].loc[japan['GenderSelect']=='Female'].dropna(), 

             norm_hist=False, color=sns.color_palette("Paired")[4])

putMedian(np.array([0,80]),np.array([0, 0.08]),np.median(japan['Age'].loc[japan['GenderSelect']=='Female'].dropna()))

plt.title('Japanese Female')

goodax(ax)



plt.subplot(222)

ax = sns.distplot(japan['Age'].loc[japan['GenderSelect']=='Male'].dropna(), 

             norm_hist=False, color=sns.color_palette("Paired")[0])

putMedian(np.array([0,80]),np.array([0, 0.08]),np.median(japan['Age'].loc[japan['GenderSelect']=='Male'].dropna()))

plt.title('Japanese Male')

goodax(ax)



plt.subplot(223)

ax = sns.distplot(germany['Age'].loc[germany['GenderSelect']=='Female'].dropna(), 

             norm_hist=False, color=sns.color_palette("Paired")[4])

putMedian(np.array([0,80]),np.array([0, 0.08]),np.median(germany['Age'].loc[germany['GenderSelect']=='Female'].dropna()))

plt.title('German Female')

goodax(ax)



plt.subplot(224)

ax = sns.distplot(germany['Age'].loc[germany['GenderSelect']=='Male'].dropna(), 

             norm_hist=False, color=sns.color_palette("Paired")[0])

putMedian(np.array([0,80]),np.array([0, 0.08]),np.median(germany['Age'].loc[germany['GenderSelect']=='Male'].dropna()))

plt.title('German Male')

goodax(ax)



plt.tight_layout()

plt.show()
# EmploymentStatus?

order = japan['EmploymentStatus'].unique()



plt.figure(figsize=(10,5))



plt.subplot(121)

ax = sns.countplot(x='EmploymentStatus', data=japan, order=order)

putPercent(ax, japan)

goodax(ax)

plt.xticks(rotation=30, ha='right')

plt.title('Japan')

plt.xlabel('')

plt.ylabel('')



plt.subplot(122)

ax = sns.countplot(x='EmploymentStatus', data=germany, order=order)

putPercent(ax, germany)

goodax(ax)

plt.xticks(rotation=30, ha='right')

plt.title('Germany')

plt.xlabel('')

plt.ylabel('')



# plt.tight_layout()

plt.show()
# student status?

plt.figure(figsize=(8,3))



japan['StudentStatus'] = japan['StudentStatus'].fillna('No')

plt.subplot(121)

ax = sns.countplot(x='StudentStatus', data=japan)

putPercent(ax, japan)

goodax(ax)

plt.title('Japan')



germany['StudentStatus'] = germany['StudentStatus'].fillna('No')

plt.subplot(122)

ax = sns.countplot(x='StudentStatus', data=germany)

putPercent(ax, germany)

goodax(ax)

plt.title('Germany')



plt.tight_layout()

plt.show()
# Current Job Title

# print(japan['CurrentJobTitleSelect'].dropna(axis=0).value_counts())

# print(germany['CurrentJobTitleSelect'].dropna(axis=0).value_counts())



japan['CurrentJobTitleSelect'] = japan['CurrentJobTitleSelect'].fillna('no comment')

germany['CurrentJobTitleSelect'] = germany['CurrentJobTitleSelect'].fillna('no comment')

order = japan['CurrentJobTitleSelect'].unique()



plt.figure(figsize=(15,8))



plt.subplot(121)

ax = sns.countplot(x='CurrentJobTitleSelect', data=japan, order=order)

putPercent(ax, japan)

goodax(ax)

plt.xticks(rotation=45, ha='right')

plt.title('Japan')

plt.xlabel('')

plt.ylabel('count')



plt.subplot(122)

ax = sns.countplot(x='CurrentJobTitleSelect', data=germany, order=order)

putPercent(ax, germany)

goodax(ax)

plt.xticks(rotation=45, ha='right')

plt.title('Germany')

plt.xlabel('')

plt.ylabel('')

# Job skill importance

col = [col for col in list(japan) if 'JobSkillImportance' in col]

lab = [c[18:] for c in col]

# print(japan[col[0]].value_counts())



# data retrieval (but removing "other select" columns)

jsi_japan = []

jsi_germany = []

for c in col[:-3]:

    jsi_japan.append(japan[c].map({'Necessary':2, 'Nice to have':1, 'Unnecessary':0}).sum())

    jsi_germany.append(germany[c].map({'Necessary':2, 'Nice to have':1, 'Unnecessary':0}).sum())



plt.figure(figsize=(10,5))

ax = sns.barplot(x=lab[:-3], y=jsi_japan/np.nanmean(jsi_japan) - jsi_germany/np.nanmean(jsi_germany))

goodax(ax)

plt.xticks(rotation=45, ha='right')

plt.title('Japan - Germany')

plt.ylabel('importance')



plt.tight_layout()

plt.show()
# University importance?

# print(japan['UniversityImportance'].value_counts())

# print(germany['UniversityImportance'].value_counts())

lab = japan['UniversityImportance'].dropna(axis=0).unique()

plt.figure(figsize=(10,5))



plt.subplot(121)

ax = sns.countplot(x='UniversityImportance', data=japan, order=lab)

putPercent(ax, japan)

goodax(ax)

plt.xticks(rotation=30, ha='right')

plt.title('Japan')

plt.xlabel('')



plt.subplot(122)

ax = sns.countplot(x='UniversityImportance', data=germany, order=lab)

putPercent(ax, germany)

goodax(ax)

plt.xticks(rotation=30, ha='right')

plt.title('Germany')

plt.xlabel('')

plt.ylabel('')



plt.tight_layout()

plt.show()
# Current Employer Type

# print(japan['CurrentEmployerType'].value_counts())

lab = ["Employed by a company that doesn't perform advanced analytics",

      "Employed by a company that performs advanced analytics",

      "Employed by professional services/consulting firm",

      "Employed by college or university",

      "Employed by company that makes advanced analytic software",

      "Employed by non-profit or NGO","Self-employed"]

cet_jp = np.zeros(7)

jp = japan['CurrentEmployerType'].dropna(axis=0)

for i in np.arange(len(jp)):

    for n in np.arange(7):

        if lab[n] in jp.iloc[i]:

            cet_jp[n] += 1

cet_de = np.zeros(7)

de = germany['CurrentEmployerType'].dropna(axis=0)

for i in np.arange(len(de)):

    for n in np.arange(7):

        if lab[n] in de.iloc[i]:

            cet_de[n] += 1



plt.figure(figsize=(15,8))

ax = sns.barplot(x=lab, y=100*(cet_jp/len(jp) - cet_de/len(de)))

goodax(ax)

plt.xticks(rotation=30, ha='right')

plt.title('Japan - Germany')

plt.ylabel('difference (%)')

plt.xlabel('')



plt.tight_layout()

plt.show()
# LearningDataSicence?

plt.figure(figsize=(8,3))



plt.subplot(121)

japan['LearningDataScience'] = japan['LearningDataScience'].fillna('no comment')

ax = sns.countplot(x='LearningDataScience', data=japan)

putPercent(ax, japan)

goodax(ax)

plt.xticks(rotation=30, ha='right')

plt.title('Japanese')

plt.xlabel('')



plt.subplot(122)

germany['LearningDataScience'] = germany['LearningDataScience'].fillna('no comment')

ax = sns.countplot(x='LearningDataScience', data=germany)

putPercent(ax, germany)

goodax(ax)

plt.xticks(rotation=30, ha='right')

plt.title('German')

plt.xlabel('')



# plt.tight_layout()

plt.show()
# Job Factor?

col = [col for col in list(japan) if 'JobFactor' in col]

lab = [c[9:] for c in col]

# print(japan[col[0]].value_counts())



jf_japan = []

jf_germany = []

for c in col:

    jf_japan.append(japan[c].map({'Very Important':2, 'Somewhat important':1, 'Not important':0}).sum())

    jf_germany.append(germany[c].map({'Very Important':2, 'Somewhat important':1, 'Not important':0}).sum())



plt.figure(figsize=(10,5))

ax = sns.barplot(x=lab, y=jf_japan/np.nanmean(jf_japan) - jf_germany/np.nanmean(jf_germany))

goodax(ax)

plt.xticks(rotation=45, ha='right')

plt.title('Japan - Germany')

plt.ylabel('importance')



plt.tight_layout()

plt.show()
# salary (compensation amount) --- convert to USD based on the conversion rate (Nov 15, 2017)

japan['CompensationAmount']= japan['CompensationAmount'].str.replace(',','')

germany['CompensationAmount'] = germany['CompensationAmount'].str.replace('-','')

japan['CompensationAmount'] = 0.008857*pd.to_numeric(japan['CompensationAmount'],errors='coerse')

germany['CompensationAmount'] = 1.17905*pd.to_numeric(germany['CompensationAmount'],errors='coerse')
plt.figure(figsize=(10,5))



plt.subplot(121)

ax = sns.distplot(japan['CompensationAmount'].dropna(), 

             norm_hist=False, color='k')

putMedian(np.array([0,300000]),np.array([0, 0.000014]),

          np.median(japan['CompensationAmount'].dropna()))

plt.title('Japan')

goodax(ax)

plt.xlabel('USD / Year')



plt.subplot(122)

ax = sns.distplot(germany['CompensationAmount'].dropna(), 

             norm_hist=False, color='k')

putMedian(np.array([0,300000]),np.array([0, 0.000014]),

          np.median(germany['CompensationAmount'].dropna()))

plt.title('Germany')

goodax(ax)

plt.xlabel('USD / Year')
# gender difference in salary

plt.figure(figsize=(10,5))



plt.subplot(221)

ax = sns.distplot(japan['CompensationAmount'].loc[japan['GenderSelect']=='Female'].dropna(), 

             norm_hist=False, color=sns.color_palette("Paired")[4])

putMedian(np.array([0,300000]),np.array([0, 0.00004]),

          np.median(japan['CompensationAmount'].loc[japan['GenderSelect']=='Female'].dropna()))

plt.title('Japanese Female')

goodax(ax)

plt.xlabel('')



plt.subplot(222)

ax = sns.distplot(japan['CompensationAmount'].loc[japan['GenderSelect']=='Male'].dropna(), 

             norm_hist=False, color=sns.color_palette("Paired")[0])

putMedian(np.array([0,300000]),np.array([0, 0.00002]),

          np.median(japan['CompensationAmount'].loc[japan['GenderSelect']=='Male'].dropna()))

plt.title('Japanese Male')

goodax(ax)

plt.xlabel('')



plt.subplot(223)

ax = sns.distplot(germany['CompensationAmount'].loc[germany['GenderSelect']=='Female'].dropna(), 

             norm_hist=False, color=sns.color_palette("Paired")[4])

putMedian(np.array([0,300000]),np.array([0, 0.00002]),

          np.median(germany['CompensationAmount'].loc[germany['GenderSelect']=='Female'].dropna()))

plt.title('German Female')

goodax(ax)

plt.xlabel('USD / Year')



plt.subplot(224)

ax = sns.distplot(germany['CompensationAmount'].loc[germany['GenderSelect']=='Male'].dropna(), 

             norm_hist=False, color=sns.color_palette("Paired")[0])

putMedian(np.array([0,300000]),np.array([0, 0.00002]),

          np.median(germany['CompensationAmount'].loc[germany['GenderSelect']=='Male'].dropna()))

plt.title('German Male')

goodax(ax)

plt.xlabel('USD / Year')



plt.tight_layout()

plt.show()
# Let's see what people wrote in the freeForm 

freeForm.head(10)
# What do people say?

for col in list(freeForm):

    print('----------------------')

    print(col + ':')

    print(freeForm[col].unique())