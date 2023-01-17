import pandas as pd
df=pd.read_csv('titanic.csv', usecols=['Age','Fare','Survived'])
df.head()
df.isnull().sum()
df.isnull().mean()
df['Age'].isnull().sum()
df['Age'].dropna().sample(df['Age'].isnull().sum(),random_state=0)
df[df['Age'].isnull()].index
def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)
    df[variable+"_random"]=df[variable]
    ##It will have the random sample to fill the na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random']=random_sample
median=df.Age.median()
median
impute_nan(df,"Age",median)
df.head()
import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_median.plot(kind='kde', ax=ax, color='red')
df.Age_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
df=pd.read_csv('titanic.csv', usecols=['Age','Fare','Survived'])
df.head()
import numpy as np
df['Age_NAN']=np.where(df['Age'].isnull(),1,0)
df.head()
df.Age.median()
df['Age'].fillna(df.Age.median(),inplace=True)
df.head(10)
df=pd.read_csv('titanic.csv', usecols=['Age','Fare','Survived'])
df.head()
df.Age.hist(bins=50)
extreme=df.Age.mean()+3*df.Age.std()
import seaborn as sns
sns.boxplot('Age',data=df)
def impute_nan(df,variable,median,extreme):
    df[variable+"_end_distribution"]=df[variable].fillna(extreme)
    df[variable].fillna(median,inplace=True)
impute_nan(df,'Age',df.Age.median(),extreme)
df.head()
df['Age'].hist(bins=50)
df['Age_end_distribution'].hist(bins=50)
sns.boxplot('Age_end_distribution',data=df)
