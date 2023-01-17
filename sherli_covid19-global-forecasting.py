# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from scipy.stats import skew

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("../input/covid19-global-forecasting-week-5/train.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-5/test.csv")
ts_confirmed_df = pd.read_csv("../input/coronavirus-covid19-cases-jhu-data/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
ts_confirmed_us_df = pd.read_csv("../input/coronavirus-covid19-cases-jhu-data/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
ts_deaths_df = pd.read_csv("../input/coronavirus-covid19-cases-jhu-data/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
ts_deaths_us_df = pd.read_csv("../input/coronavirus-covid19-cases-jhu-data/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv")
ts_recovered_df = pd.read_csv("../input/coronavirus-covid19-cases-jhu-data/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
useful_features_df = pd.read_csv("../input/covid19-useful-features-by-country/Countries_usefulFeatures.csv")
train_df.describe()

# Finding missing values and Imputation
train_df.isnull().sum()
train_df["County"].fillna("Unknown",inplace=True)
train_df["Province_State"].fillna("Unknown",inplace=True)

test_df.isnull().sum()
test_df["County"].fillna("Unknown",inplace=True)
test_df["Province_State"].fillna("Unknown",inplace=True)

ts_confirmed_df.isnull().sum()
ts_confirmed_df["Province/State"].fillna("Unknown",inplace=True)

ts_deaths_df.isnull().sum()
ts_deaths_df["Province/State"].fillna("Unknown",inplace=True)

ts_confirmed_us_df.isnull().sum()
ts_confirmed_us_df["FIPS"].fillna("0",inplace=True)
ts_confirmed_us_df["Admin2"].fillna("Unknown",inplace=True)

ts_deaths_us_df.isnull().sum()
ts_deaths_us_df["FIPS"].fillna("0",inplace=True)
ts_deaths_us_df["Admin2"].fillna("Unknown",inplace=True)

ts_recovered_df.isnull().sum()
ts_recovered_df["Province/State"].fillna("Unknown",inplace=True)


train_df.columns


#Outlier Detection

def get_lower_upper_bound(train_df):
    q1 = np.percentile(train_df,25)
    q3 = np.percentile(train_df,75)
    iqr = q3 - q1
    #compute lower and upper bound
    lower_bound = q1-(iqr*1.5)
    upper_bound = q3+(iqr*1.5)
    return lower_bound,upper_bound

def get_outliers_iqr(train_df):
    lower_bound,upper_bound=get_lower_upper_bound(train_df)
    return train_df[np.where((train_df>upper_bound)|(train_df<lower_bound))]
print(get_outliers_iqr(train_df['TargetValue'].values))
train_df.boxplot(column='TargetValue')


# Plot to find the mean ages of Covid19 cases
plt.figure()
plt.title('Mean ages of Covid19 cases in all regions')
useful_features_df['Mean_Age'].hist(density=True,bins=10)
plt.xlabel('Mean Age')
plt.ylabel('Distribution')
plt.grid(False)
plt.show()


#Skewness in TargetValue count of Covid19 patients across the world
count_df = pd.to_numeric(train_df['TargetValue'],errors = 'coerce')
print("Skewness is {}".format(skew(count_df)))
print("Mean is {}".format(np.mean(count_df)))
print("Median is {}".format(np.median(count_df)))



#Density distribution of Mean age
sns.distplot(useful_features_df['Mean_Age'], hist=True, kde=True, 
             bins=int(180/5), color = 'blue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4}).set_title('Density distribution of mean age')


# Unique countries with Covid19 cases
train_df['Country_Region'].unique()
# Tourism in the Population

x=useful_features_df[['Country_Region','Population_Size','Tourism']]
y=x.set_index('Country_Region')
z=y.groupby('Country_Region').mean()
#ind = [country for country in x]
ax=z.plot.bar(stacked=True)
tick_idx = plt.xticks()[0]
country_labels=useful_features_df.Country_Region[tick_idx].values
population_labels=useful_features_df.Population_Size[tick_idx].values
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.xaxis.set_ticklabels(country_labels)
ax.yaxis.set_ticklabels(population_labels)
plt.rcParams['figure.figsize'] = (20, 8)
#plt.xlabel('Country-Region',fontsize=20)
plt.tight_layout()
ax.legend(labels=['Population_Size', 'Tourism'])
plt.show()

 # Increase in Covid19 cases around the world from Jan till date
 
province_cases = train_df[['County','Province_State','Country_Region','Date','Target','TargetValue']]

confirmed_df = train_df.loc[train_df['Target'].isin(['ConfirmedCases'])]
fatalities_df = train_df.loc[train_df['Target'].isin(['Fatalities'])]
sum_cases = confirmed_df.groupby(by=['Date'])['TargetValue'].sum()
print(sum_cases)
plt.rcParams['figure.figsize'] = (10, 8)
plt.title("Increase in Covid19 cases around the world from Jan till date")
sum_cases.plot()


#Increase in Covid19 cases from Jan till date in US
region_cases = confirmed_df[confirmed_df['Country_Region'].str.contains('US')]
us_cases = region_cases.groupby(by=['Date'])['TargetValue'].sum()
plt.rcParams['figure.figsize'] = (10, 8)
plt.title("Increase in Covid19 cases in US from Jan till date")
us_cases.plot()

# Confirmed cases in various provinces in US
us_confirmed = confirmed_df[confirmed_df['Country_Region'].str.contains('US')]
ax = sns.countplot(data=us_confirmed,x='Province_State')
ax.set_xticklabels(ax.get_xticklabels(), rotation=80, ha="right")
plt.tight_layout()
plt.show()

#Fatalities in various provinces in US
us_fatalities = fatalities_df[fatalities_df['Country_Region'].str.contains('US')]
ax = sns.countplot(data=us_confirmed,x='Province_State')
ax.set_xticklabels(ax.get_xticklabels(), rotation=80, ha="right")
plt.tight_layout()
plt.show()
# Confirmed cases in various provinces in India
india_confirmed = confirmed_df[confirmed_df['Country_Region'].str.contains('India')]
sns.boxplot(data=india_confirmed,x='Province_State',y='TargetValue')

# Confirmed cases in various provinces in India
china_confirmed = confirmed_df[confirmed_df['Country_Region'].str.contains('China')]
ax=sns.boxplot(data=china_confirmed,x='Province_State',y='TargetValue')
ax.set_xticklabels(ax.get_xticklabels(), rotation=80, ha="right")
plt.tight_layout()
plt.show()
train_df['Target'].value_counts().plot.bar(title='Frequence distribution of overall Fatal and Confirmed cases')
