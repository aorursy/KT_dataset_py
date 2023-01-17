import pandas as pd
import numpy as np
df = pd.read_csv("../input/titanic/train.csv")
df.head()
df.isnull().sum()
df[df['Embarked'].isnull()]
df[df['Cabin'].isnull()]
# convert "Cabin" 1 or 0
df['cabin_null']  = np.where(df['Cabin'].isnull(), 1, 0)
df['cabin_null'].mean()
df.head()
df.columns
#compare cabin null with survived
df.groupby(['Survived'])['cabin_null'].mean()

df = pd.read_csv("../input/titanic/train.csv", usecols = ['Age', 'Fare', 'Survived'])
df.head()
df.isnull()
df.isnull().sum()
df.isnull().mean()    #percentage of missing values
def impute_nan(df, variable, median):
    """fill the nan values with median"""
    
    df[variable+"_median"] = df[variable].fillna(median)
    
median = df.Age.median()
print("median of age is : "+str(median))

impute_nan(df, 'Age', median)
df.head()
df.isnull().sum()
print("standard deviation of age: {}".format(df['Age'].std()))
print("standard deviation of Age_median: {}".format(df['Age_median'].std()))
import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_median.plot(kind='kde', ax=ax, color = 'red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
df = pd.read_csv("../input/titanic/train.csv", usecols = ['Age', 'Fare', 'Survived'])
df.head()
df.isnull().sum()
df['Age'].dropna().sample()
#this will replace NA with randomvalue
#but this will replace only one value ....we need 
#177 values to replace
#for this we can use
"""This will replace all the 177 values with random values"""
df['Age'].dropna().sample(df['Age'].isnull().sum(), random_state=0)


df[df['Age'].isnull()].index
def impute_nan(df, variable, median):
    """fill the nan values with median"""
    
    df[variable+"_median"] = df[variable].fillna(median)
    df[variable+"_random"] = df[variable]
    ##It will have the random sample to fill the na
    random_sample=df[variable].dropna().sample(df['Age'].isnull().sum(), random_state=0)
    
    #pandas need to have some index in order to merge the data set
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable+'_random'] = random_sample
median = df.Age.median()
median
impute_nan(df, 'Age', median)
df.head()
import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_random.plot(kind = 'kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')
fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_median.plot(kind = 'kde', ax=ax, color='red')
df.Age_random.plot(kind = 'kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines,labels,loc='best')
df = pd.read_csv("../input/titanic/train.csv", usecols = ['Age', 'Fare', 'Survived'])
# Create a new feature and wherever there is null than replace with 
# 1 else replace with 0

df['Age_NAN'] = np.where(df['Age'].isnull() , 1,0)
df.head(20)
df = pd.read_csv("../input/titanic/train.csv", usecols = ['Age', 'Fare', 'Survived'])
df.Age.hist(bins=50)
extreme=df.Age.mean() + 3* df.Age.std()
extreme
import seaborn as sns
sns.boxplot('Age', data = df)
def impute_nan(df, var, median, extreme):
    """Create a new Feature and fill with exteme value"""
    
    df[var+"_end_distribution"] = df[var].fillna(extreme)
    df[var].fillna(median, inplace=True) #replace the age NaN with median
impute_nan(df, 'Age',df.Age.median(), extreme)
df.head()
df['Age'].hist(bins=50)

df['Age_end_distribution'].hist(bins=50)
sns.boxplot('Age_end_distribution', data=df)
df = pd.read_csv("../input/titanic/train.csv", usecols = ['Age', 'Fare', 'Survived'])
df.head()
df.Age.hist(bins=50)
def impute_nan(df, var):
    df[var+'_hundred'] = df[var].fillna(100)      #fill the NAN with 100 and create a new feature
    df[var +'_zero']= df[var].fillna(0)        #fill NAN withh 0 and create a new feature.
    
impute_nan(df, 'Age')
df.head(20)