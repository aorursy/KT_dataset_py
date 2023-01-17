# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
import pylab 
import scipy.stats as stats
import datetime
%matplotlib inline

pd.set_option('display.max_columns', None)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# ignore warnings

import warnings
warnings.filterwarnings('ignore')
# load the dataset
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')
# make a copy of titanic dataset
data1 = titanic.copy()
 # check the percentage of missing values per variable

data1.isnull().mean()
# check how many observations we would drop
print('total passengers with values in all variables: ', data1.dropna().shape[0])
print('total passengers in the Titanic: ', data1.shape[0])
print('percentage of data without missing values: ', data1.dropna().shape[0]/ np.float(data1.shape[0]))
# make a copy of titanic dataset
data2 = titanic.copy()
# check the percentage of NA values in dataset

data2.isnull().mean()
# plot the distribution of age to find out if they are Gaussian or skewed.

plt.figure(figsize=(12,8))
fig = data2.Age.hist(bins=10)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Age')
# separate dataset into training and testing set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data2, data2.Survived, test_size=0.3, 
                                                    random_state=0)
X_train.shape, X_test.shape
# calculate median of Age
median = X_train.Age.median()
median
# impute missing values in age in train and test set

for df in [X_train, X_test]:
    df['Age'].fillna(median, inplace=True)
X_train['Age'].isnull().sum()
X_test['Age'].isnull().sum()
# make a copy of titanic dataset

data3 = titanic.copy()
# check the percentage of NA values

data3.isnull().mean()
# separate dataset into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data3, data3.Survived, test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
# write a function to create 3 variables from Age:

def impute_na(df, variable, median):
    
    df[variable+'_median'] = df[variable].fillna(median)
    df[variable+'_zero'] = df[variable].fillna(0)
    
    # random sampling
    df[variable+'_random'] = df[variable]
    
    # extract the random sample to fill the na
    random_sample = X_train[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)
    
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable+'_random'] = random_sample
    
    # fill with random-sample
    df[variable+'_random_sample'] = df[variable].fillna(random_sample)
impute_na(X_train, 'Age', median)
impute_na(X_test, 'Age', median)
# make a copy of titanic dataset

data4 = titanic.copy()
# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data4, data4.Survived, test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
def impute_na(df, variable):
    df[variable+'_zero'] = df[variable].fillna(0)
    df[variable+'_hundred']= df[variable].fillna(100)
# replace NA with the median value in the training and test set
impute_na(X_train, 'Age')
impute_na(X_test, 'Age')

# make a copy of titanic dataset

data5 = titanic.copy()
# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data5, data5.Survived, test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
plt.figure(figsize=(12,8))
X_train.Age.hist(bins=50)
# at far end of the distribution
X_train.Age.mean()+3*X_train.Age.std()
# we can see that there are a few outliers for Age
# according to its distribution, these outliers will be masked when we replace NA by values at the far end 

plt.figure(figsize=(12,8))
sns.boxplot('Age', data=data5)
def impute_na(df, variable, median, extreme):
    df[variable+'_far_end'] = df[variable].fillna(extreme)
    df[variable].fillna(median, inplace=True)
# let's replace the NA with the median value in the training and testing sets
impute_na(X_train, 'Age', X_train.Age.median(), X_train.Age.mean()+3*X_train.Age.std())
impute_na(X_test, 'Age', X_train.Age.median(), X_train.Age.mean()+3*X_train.Age.std())
# make a copy of titanic dataset

data6 = titanic.copy()
# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data6, data6.Survived, test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
# create variable indicating missingness

X_train['Age_NA'] = np.where(X_train['Age'].isnull(), 1, 0)
X_test['Age_NA'] = np.where(X_test['Age'].isnull(), 1, 0)

X_train.head()
# we can see that mean and median are similar. So I will replace with the median

X_train.Age.mean(), X_train.Age.median()
# let's replace the NA with the median value in the training set
X_train['Age'].fillna(X_train.Age.median(), inplace=True)
X_test['Age'].fillna(X_train.Age.median(), inplace=True)

X_train.head(10)
# make a copy of titanic dataset

data7 = titanic.copy()
data7['Sex'].head()
# one hot encoding

pd.get_dummies(data7['Sex']).head()
# for better visualisation
pd.concat([data7['Sex'], pd.get_dummies(data7['Sex'])], axis=1).head()
# obtaining k-1 labels
pd.get_dummies(data7['Sex'], drop_first=True).head()
# Let's now look at an example with more than 2 labels

data7['Embarked'].head()
# check the number of different labels
data7.Embarked.unique()
# get whole set of dummy variables

pd.get_dummies(data7['Embarked']).head()
# get k-1 dummy variables

pd.get_dummies(data7['Embarked'], drop_first=True).head()
#import dataset
df_train = pd.read_csv('/kaggle/input/mercedesbenz-greener-manufacturing/train.csv')
                       

df_test = pd.read_csv('/kaggle/input/mercedesbenz-greener-manufacturing/test.csv') 
                      
df_train.head()
# let's have a look at how many labels

for col in df_train.columns[3:9]:
    print(col, ': ', len(df_train[col].unique()), ' labels')
X_train, X_test, y_train, y_test = train_test_split(df_train[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']], df_train.y,
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
# let's obtain the counts for each one of the labels in variable X2
# let's capture this in a dictionary that we can use to re-map the labels

X_train.X2.value_counts().to_dict()
# lets look at X_train so we can compare then the variable re-coding

X_train.head()
# now let's replace each label in X2 by its count

# first we make a dictionary that maps each label to the counts
X_frequency_map = X_train.X2.value_counts().to_dict()

# and now we replace X2 labels both in train and test set with the same map
X_train.X2 = X_train.X2.map(X_frequency_map)
X_test.X2 = X_test.X2.map(X_frequency_map)

X_train.head()
# let's load again the titanic dataset

data = pd.read_csv('/kaggle/input/titanic/train.csv', usecols=['Cabin', 'Survived'])
data.head()
# let's fill NA values with an additional label

data.Cabin.fillna('Missing', inplace=True)
data.head()
# check number of different labels in Cabin

len(data.Cabin.unique())
# Now we extract the first letter of the cabin

data['Cabin'] = data['Cabin'].astype(str).str[0]
data.head()
# check the labels
data.Cabin.unique()
# Let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data[['Cabin', 'Survived']], data.Survived, test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
# let's calculate the target frequency for each label

X_train.groupby(['Cabin'])['Survived'].mean()
# and now let's do the same but capturing the result in a dictionary

ordered_labels = X_train.groupby(['Cabin'])['Survived'].mean().to_dict()
ordered_labels
# replace the labels with the 'risk' (target frequency)
# note that we calculated the frequencies based on the training set only

X_train['Cabin_ordered'] = X_train.Cabin.map(ordered_labels)
X_test['Cabin_ordered'] = X_test.Cabin.map(ordered_labels)
# view results

X_train.head()
# plot the original variable

fig = plt.figure(figsize=(8,6))
fig = X_train.groupby(['Cabin'])['Survived'].mean().plot()
fig.set_title('Normal relationship between variable and target')
fig.set_ylabel('Survived')
# plot the transformed result: the monotonic variable

fig = plt.figure(figsize=(8,6))
fig = X_train.groupby(['Cabin_ordered'])['Survived'].mean().plot()
fig.set_title('Monotonic relationship between variable and target')
fig.set_ylabel('Survived')
# preview X_train

X_train.head()
# now we calculate the probability of target=1 
X_train.groupby(['Cabin'])['Survived'].mean()
# let's make a dataframe with the above calculation

prob_df = X_train.groupby(['Cabin'])['Survived'].mean()
prob_df = pd.DataFrame(prob_df)
prob_df
# and now the probability of target = 0 
# and we add it to the dataframe

prob_df = X_train.groupby(['Cabin'])['Survived'].mean()
prob_df = pd.DataFrame(prob_df)
prob_df['Died'] = 1-prob_df.Survived
prob_df
# since the log of zero is not defined, let's set this number to something small and non-zero

prob_df.loc[prob_df.Survived == 0, 'Survived'] = 0.00001
prob_df
# now we calculate the WoE

prob_df['WoE'] = np.log(prob_df.Survived/prob_df.Died)
prob_df
# and we create a dictionary to re-map the variable

prob_df['WoE'].to_dict()
# and we make a dictionary to map the orignal variable to the WoE
# same as above but we capture the dictionary in a variable

ordered_labels = prob_df['WoE'].to_dict()
# replace the labels with the WoE

X_train['Cabin_ordered'] = X_train.Cabin.map(ordered_labels)
X_test['Cabin_ordered'] = X_test.Cabin.map(ordered_labels)
# check the results

X_train.head()
# plot the original variable

fig = plt.figure(figsize=(8,6))
fig = X_train.groupby(['Cabin'])['Survived'].mean().plot()
fig.set_title('Normal relationship between variable and target')
fig.set_ylabel('Survived')
# plot the transformed result: the monotonic variable

fig = plt.figure(figsize=(8,6))
fig = X_train.groupby(['Cabin_ordered'])['Survived'].mean().plot()
fig.set_title('Monotonic relationship between variable and target')
fig.set_ylabel('Survived')
# load the numerical variables of the Titanic Dataset

data = pd.read_csv('/kaggle/input/titanic/train.csv', usecols = ['Age', 'Fare', 'Survived'])
data.head()
# first I will fill the missing data of the variable age, with a random sample of the variable

def impute_na(data, variable):
    # function to fill na with a random sample
    df = data.copy()
    
    # random sampling
    df[variable+'_random'] = df[variable]
    
    # extract the random sample to fill the na
    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)
    
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable+'_random'] = random_sample
    
    return df[variable+'_random']
# fill na
data['Age'] = impute_na(data, 'Age')
# plot the histograms to have a quick look at the distributions
# we can plot Q-Q plots to visualise if the variable is normally distributed

def diagnostic_plots(df, variable):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=pylab)

    plt.show()
    
diagnostic_plots(data, 'Age')
### Logarithmic transformation
data['Age_log'] = np.log(data.Age)

diagnostic_plots(data, 'Age_log')
### Reciprocal transformation
data['Age_reciprocal'] = 1 / data.Age

diagnostic_plots(data, 'Age_reciprocal')
data['Age_sqr'] =data.Age**(1/2)

diagnostic_plots(data, 'Age_sqr')
data['Age_exp'] = data.Age**(1/1.2) 

diagnostic_plots(data, 'Age_exp')
data['Age_boxcox'], param = stats.boxcox(data.Age) 

print('Optimal λ: ', param)

diagnostic_plots(data, 'Age_boxcox')
# define x
x = np.array([24,  7,  2, 25, 22, 29])
x    
# equal width discretisation with cut 
pd.cut(x, bins = 3, labels = ["bad", "medium", "good"]).value_counts() #Bins size has equal interval of 9   
# define x
x = np.array([24,  7,  2, 25, 22, 29])
x    
# equal frequency discretisation with qcut 
pd.qcut(x, q = 3, labels = ["bad", "medium", "good"]).value_counts() #Equal frequency of 2 in each bins
# load the numerical variables of the Titanic Dataset
data = pd.read_csv('/kaggle/input/titanic/train.csv', usecols = ['Age', 'Survived'])
data.head()
def impute_na(data, variable):
    df = data.copy()
    
    # random sampling
    df[variable+'_random'] = df[variable]
    
    # extract the random sample to fill the na
    random_sample = data[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)
    
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable+'_random'] = random_sample
    
    return df[variable+'_random']
# let's fill the missing data
data['Age'] = impute_na(data, 'Age')
data['Age'].isnull().sum()
# let's divide age into the buckets 

# bucket boundaries
buckets = [0,20,40,60,100]

# bucket labels
labels = ['0-20', '20-40', '40-60', '>60']

# discretisation
pd.cut(data.Age, bins = buckets, labels = labels, include_lowest=True).value_counts()
# create two new columns after discretisation

data['Age_buckets_labels'] = pd.cut(data.Age, bins=buckets, labels = labels, include_lowest=True)
data['Age_buckets'] = pd.cut(data.Age, bins=buckets, include_lowest=True)

data.head()
data.tail()
# number of passengers per age bucket

plt.figure(figsize=(12,8))
data.groupby('Age_buckets_labels')['Age'].count().plot.bar()
# load the numerical variables of the Titanic Dataset
data = pd.read_csv('/kaggle/input/titanic/train.csv', usecols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived'])
data.head()
# divide dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(data, data.Survived,
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
# let's make boxplots to visualise outliers in the continuous variables 
# Age and Fare

plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.boxplot(column='Age')
fig.set_title('')
fig.set_ylabel('Age')

plt.subplot(1, 2, 2)
fig = data.boxplot(column='Fare')
fig.set_title('')
fig.set_ylabel('Fare')
# first we plot the distributions to find out if they are Gaussian or skewed.
# Depending on the distribution, we will use the normal assumption or the interquantile
# range to find outliers

plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.Age.hist(bins=20)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Age')

plt.subplot(1, 2, 2)
fig = data.Fare.hist(bins=20)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Fare')
# find outliers

# Age
Upper_boundary = data.Age.mean() + 3* data.Age.std()
Lower_boundary = data.Age.mean() - 3* data.Age.std()
print('Age outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))

# Fare
IQR = data.Fare.quantile(0.75) - data.Fare.quantile(0.25)
Lower_fence = data.Fare.quantile(0.25) - (IQR * 3)
Upper_fence = data.Fare.quantile(0.75) + (IQR * 3)
print('Fare outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
# view the statistical summary of Age
data.Age.describe()
# Assuming normality

Upper_boundary = X_train.Age.mean() + 3* X_train.Age.std()
Upper_boundary
# top-coding the Age variable

X_train.loc[X_train.Age>73, 'Age'] = 73
X_test.loc[X_test.Age>73, 'Age'] = 73

X_train.Age.max(), X_test.Age.max()
# view statistical properties of Fare

X_train.Fare.describe()
# top coding: upper boundary for outliers according to interquantile proximity rule

IQR = data.Fare.quantile(0.75) - data.Fare.quantile(0.25)

Upper_fence = X_train.Fare.quantile(0.75) + (IQR * 3)

Upper_fence
# top-coding: capping the variable Fare at 100
X_train.loc[X_train.Fare>100, 'Fare'] = 100
X_test.loc[X_test.Fare>100, 'Fare'] = 100
X_train.Fare.max(), X_test.Fare.max()
# let's load the Lending Club dataset with selected columns and rows

use_cols = ['issue_d', 'last_pymnt_d']
data = pd.read_csv('/kaggle/input/lending-club-loan-data/loan.csv', usecols=use_cols, nrows=10000)
data.head()
# now let's parse the dates, currently coded as strings, into datetime format

data['issue_dt'] = pd.to_datetime(data.issue_d)
data['last_pymnt_dt'] = pd.to_datetime(data.last_pymnt_d)

data[['issue_d','issue_dt','last_pymnt_d', 'last_pymnt_dt']].head()
# Extracting Month from date

data['issue_dt_month'] = data['issue_dt'].dt.month

data[['issue_dt', 'issue_dt_month']].head()
data[['issue_dt', 'issue_dt_month']].tail()
# Extract quarter from date variable

data['issue_dt_quarter'] = data['issue_dt'].dt.quarter

data[['issue_dt', 'issue_dt_quarter']].head()
data[['issue_dt', 'issue_dt_quarter']].tail()
# We could also extract semester

data['issue_dt_semester'] = np.where(data.issue_dt_quarter.isin([1,2]),1,2)
data.head()
# day - numeric from 1-31

data['issue_dt_day'] = data['issue_dt'].dt.day

data[['issue_dt', 'issue_dt_day']].head()
# day of the week - from 0 to 6

data['issue_dt_dayofweek'] = data['issue_dt'].dt.dayofweek

data[['issue_dt', 'issue_dt_dayofweek']].head()
data[['issue_dt', 'issue_dt_dayofweek']].tail()
# day of the week - name

data['issue_dt_dayofweek'] = data['issue_dt'].dt.weekday_name

data[['issue_dt', 'issue_dt_dayofweek']].head()
data[['issue_dt', 'issue_dt_dayofweek']].tail()
# was the application done on the weekend?

data['issue_dt_is_weekend'] = np.where(data['issue_dt_dayofweek'].isin(['Sunday', 'Saturday']), 1,0)
data[['issue_dt', 'issue_dt_dayofweek','issue_dt_is_weekend']].head()
data[data.issue_dt_is_weekend==1][['issue_dt', 'issue_dt_dayofweek','issue_dt_is_weekend']].head()
# extract year 

data['issue_dt_year'] = data['issue_dt'].dt.year

data[['issue_dt', 'issue_dt_year']].head()
# extract the date difference between 2 dates

data['issue_dt'] - data['last_pymnt_dt']