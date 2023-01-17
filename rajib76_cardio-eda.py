# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Import the necessary packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.gridspec as gridspec

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load the CSV file

data = pd.read_csv('/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv')
# Let us take a look at few rows of the data

data.head()
# Let us add the AgeRange to the dataframe

bins = [0,18, 20, 25,30,35,40,45,50,np.inf]

names = ['<=18','18-20', '20-25','25-30','30-35','35-40','40-45','45-50','50+']



data['AgeRange'] = pd.cut(data['Age'], bins, labels=names)

data.head(50)
# Let us add the IncomeRange to the dataframe

bins = [0,10000, 30000, 50000,70000,90000,110000,np.inf]

names = ['<=10k','10-30k', '30-50k','50-70k','70-90k','90-110k','110k+']



data['IncomeRange'] = pd.cut(data['Income'], bins, labels=names)
# Let us add the fitness range also

bins=[0,3,np.inf]

names=['<=3','4+']

data['FitRange']=pd.cut(data['Fitness'],bins,labels=names)
# How many rows and columns do we have

data.shape
# What is the 5 point data summary for the dataset. I love to see this in a horizontal view, so transposing

data.describe().T
# What is the median of the numerical values

# I am more interested in the median income

# The median income in US is around 33KUSD. So, this group of customers look like are fairly rich

data.median()
# Do we have any missing value?

# We do not have any missing value in the data

data.isnull().sum()

# data.info() - This would also give you the information if there are any missing values
# What is the distribution of each feature?

dist=data.hist(figsize=(10,15))

print('Age has a skew of {} and Kurtosis of {}'.format(data['Age'].skew(),data['Age'].kurt()))

print('Education has a skew of {} and Kurtosis of {}'.format(data['Education'].skew(),data['Education'].kurt()))

print('Fitness has a skew of {} and Kurtosis of {}'.format(data['Fitness'].skew(),data['Fitness'].kurt()))

print('Income has a skew of {} and Kurtosis of {}'.format(data['Income'].skew(),data['Income'].kurt()))

print('Miles has a skew of {} and Kurtosis of {}'.format(data['Miles'].skew(),data['Miles'].kurt()))

print('Usage has a skew of {} and Kurtosis of {}'.format(data['Usage'].skew(),data['Usage'].kurt()))
#Lets look at the gender distribution

sns.axes_style('whitegrid')

g=sns.catplot("Gender",data=data,aspect=2,kind="count",legend=True,palette=sns.color_palette(['blue','pink']))

for i,bar in enumerate(g.ax.patches):

    h=bar.get_height()

    g.ax.text(i,h+2,'{},{}%'.format(int(h),round(int(h)/180*100)),ha='center',va='center',fontweight='bold',size=14)
#Lets look at the marital status distribution

sns.axes_style('whitegrid')

g=sns.catplot("MaritalStatus",data=data,aspect=2,kind="count",legend=True,palette=sns.color_palette(['blue','green']))

for i,bar in enumerate(g.ax.patches):

    h=bar.get_height()

    g.ax.text(i,h+2,'{},{}%'.format(int(h),round(int(h)/180*100)),ha='center',va='center',fontweight='bold',size=14)
# Let us understand the customer distribution across age range

g=sns.catplot("AgeRange",data=data,aspect=2,kind="count",color="steelblue",legend=True)

for i,bar in enumerate(g.ax.patches):

    h=bar.get_height()

    if h>0:

        g.ax.text(i,h+2,'{},{}%'.format(int(h),round(int(h)/180*100)),ha='center',va='center',fontweight='bold',size=14)

    else:

        h=0

        g.ax.text(i,h+2,'{},{}%'.format(int(h),round(int(h)/180*100)),ha='center',va='center',fontweight='bold',size=14)

# Let us understand the customer distribution across Income range

g=sns.catplot("IncomeRange",data=data,aspect=2,kind="count",color="steelblue",legend=True)

for i,bar in enumerate(g.ax.patches):

    h=bar.get_height()

    if h>0:

        g.ax.text(i,h+2,'{},{}%'.format(int(h),round(int(h)/180*100)),ha='center',va='center',fontweight='bold',size=14)

    else:

        h=0

        g.ax.text(i,h+2,'{},{}%'.format(int(h),round(int(h)/180*100)),ha='center',va='center',fontweight='bold',size=14)

# This also shows that majority of the customers are earning between 40-70KUSD higher that the US median salary

# There are some customers in the higher end of the salary bracket(but mostly males)

ax = sns.stripplot(x=data["Gender"],y=data["Income"], jitter=True)
#What is the education level of my clients?

sns.axes_style('whitegrid')

g=sns.catplot("Education",data=data,aspect=2,kind="count",legend=True)

for i,bar in enumerate(g.ax.patches):

    h=bar.get_height()

    g.ax.text(i,h+2,'{},{}%'.format(int(h),round(int(h)/180*100)),ha='center',va='center',fontweight='bold',size=14)
#Lets look at which product is selling more

sns.axes_style('whitegrid')

g=sns.catplot("Product",data=data,aspect=2,kind="count",legend=True)

for i,bar in enumerate(g.ax.patches):

    h=bar.get_height()

    g.ax.text(i,h+2,'{},{}%'.format(int(h),round(int(h)/180*100)),ha='center',va='center',fontweight='bold',size=14)
ax=sns.boxplot(data['Income'])
ax=sns.boxplot(data['Age'])
ax=sns.boxplot(data['Education'])
ax=sns.boxplot(data['Usage'])
ax=sns.boxplot(data['Miles'])
ax=sns.catplot(x='Product',kind='count',hue='IncomeRange',data=data,aspect=2)
ax=sns.catplot(x='Product',kind='count',hue='FitRange',data=data,aspect=2)
ax=sns.catplot(x='FitRange',kind='count',hue='AgeRange',data=data,aspect=2)
ax=sns.catplot(x='Product',kind='count',hue='Gender',data=data,aspect=2)
plot=sns.catplot(x='Product',kind='count',hue='MaritalStatus',data=data)
ax=sns.boxplot(x='Income', y='Gender', data=data)
ax=sns.boxplot(x='Income', y='Product', data=data)
ax=sns.boxplot(x='Education', y='Product', data=data)
ax=sns.boxplot(x='Usage', y='Product', data=data)
ax=sns.boxplot(x='Age', y='Product', data=data)
ax=sns.boxplot(x='Miles', y='Product', data=data)
ax=sns.boxplot(x='Education', y='Income', data=data)
pd.pivot_table(data,'Income', index=['Product', 'Gender'],

                     columns=[ 'MaritalStatus'],aggfunc=[np.median,np.mean,len])
pd.pivot_table(data,'Age', index=['Product', 'Gender'],

                     columns=[ 'MaritalStatus'],aggfunc=[np.median,np.mean])
plt.figure(dpi=120,figsize=(5,4))

mask=np.triu(np.ones_like(data.corr(),dtype=bool),0)

sns.heatmap(data.corr(),mask=mask,fmt=".1f",annot=True,lw=1,cmap='plasma')

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.title('Correlation Heatmap')

plt.show()
ax=sns.pairplot(data=data)
import pandas_profiling as pp
data_profile = pd.read_csv('/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv')

pp.ProfileReport(data_profile)