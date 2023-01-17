import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import shapiro

from scipy.stats import anderson

from scipy.stats import normaltest

from scipy.stats import norm

import seaborn as sns

import numpy as np

from sklearn.preprocessing import StandardScaler

from scipy import stats

import re

import warnings

from pandas.api.types import is_string_dtype

from pandas.api.types import is_numeric_dtype

from wordcloud import WordCloud, STOPWORDS 

warnings.filterwarnings('ignore')

%matplotlib inline



data = pd.read_csv('../input/data-scientist-jobs/DataScientist.csv')

data.head()



data = data.drop('Unnamed: 0', 1)

data = data.drop('index', 1)

data = data.drop('Competitors', 1)

data = data.drop('Easy Apply', 1)



data = data.replace(-1, np.nan)

data["Rating"].interpolate(method='linear', direction = 'forward', inplace=True) 



data.drop(data[data['Headquarters'] == "-1"].index, inplace=True)

data.drop(data[data['Size'].str.contains("-1")].index, inplace=True)

data.drop(data[data['Type of ownership'].str.contains("-1")].index, inplace=True)

data.drop(data[data['Revenue'].str.contains("-1")].index, inplace=True)

data.drop(data[data['Sector'].str.contains("-1")].index, inplace=True)

data.drop(data[data['Industry'].str.contains("-1")].index, inplace=True)
HOURS_PER_WEEK = 40

WEEKS_PER_YEAR = 52

THOUSAND = 1000



def return_digits(x):

    result = re.findall(r'\d+', str(x))

    result = int(result[0]) if result else 0

    return result



def return_salary(string, isFrom):

    patternMain = None

    patternPerHour = None

    if(isFrom):

        patternMain = r'^\$\d+K';

        patternPerHour = r'^\$\d+';

    else:

        patternMain = r'-\$\d+K';

        patternPerHour = r'-\$\d+';

    

    result = None

    if('Per Hour' in string):

        result = re.findall(patternPerHour, str(string))

        result = return_digits(result[0]) if result else 0

        result = result * HOURS_PER_WEEK * WEEKS_PER_YEAR

    else:

        result = re.findall(patternMain, str(string))

        result = return_digits(result[0]) if result else 0

        result = result * THOUSAND

    return result



def return_average_salary(x):

    from_salary = return_salary(x, True)

    to_salary = return_salary(x, False)

    result = (from_salary+to_salary)/2

    return result



data['SalaryAverage'] =  data['Salary Estimate'].apply(return_average_salary)
print(data.shape)

print(data.columns)



def count_missing_values():

    for column in data:

        nullAmount = None

        if (is_numeric_dtype(data[column])):

            nullAmount = data[data[column] == -1].shape[0]

        else:

            nullAmount = data[data[column] == "-1"].shape[0]

        print('{}{},  \t{:2.1f}%'.format(column.ljust(20),nullAmount, nullAmount*100/data[column].shape[0]))

    

count_missing_values()
seniorData =  data[data['Job Title'].str.contains("Senior")|data['Job Title'].str.contains("Sr.")]

print(len(seniorData))
juniorData =  data[data['Job Title'].str.contains("Junior")|data['Job Title'].str.contains("Jr.")]

print(len(juniorData))
print(sns.distplot(juniorData['SalaryAverage'], fit=norm))

fig = plt.figure()

res = stats.probplot(juniorData['SalaryAverage'], plot=plt)
juniorData.boxplot(column=['SalaryAverage'])
stat, p = shapiro(juniorData['SalaryAverage'])

print('Statistics=%.3f, p=%.3f' % (stat, p))
juniorData["SalaryAverage"].mean()
import statsmodels.stats.api as sms



print(stats.ttest_1samp(juniorData['SalaryAverage'], popmean=100000))



bounds = sms.DescrStatsW(juniorData['SalaryAverage']).tconfint_mean()

print(bounds)
bestData = juniorData[(juniorData['SalaryAverage']>92077) & (juniorData['SalaryAverage']<113022)]

print(bestData.shape)
print(sns.countplot(y='Company Name',data=bestData, order = bestData['Company Name'].value_counts().index))
companyData = bestData[bestData['Company Name'].str.contains("Staffigo")]

print(companyData.shape)

print(sns.catplot(x="Location", y="SalaryAverage", hue = "Job Title", s = 20, data=companyData, aspect=1.5))
print(sns.countplot(y='Location',data=bestData, order = bestData['Location'].value_counts().index))
print(sns.catplot(x="Size", y="SalaryAverage", hue = "Sector", s = 10, data=bestData))
print(sns.countplot(y='Sector',data=bestData, order = bestData['Sector'].value_counts().index))
stopwords = set(STOPWORDS) 

wordcloud = WordCloud(width = 500, height = 500, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(' '.join(bestData["Job Description"])) 

                         

plt.figure(figsize = (10, 10), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

  

plt.tight_layout(pad = 0) 

plt.show() 