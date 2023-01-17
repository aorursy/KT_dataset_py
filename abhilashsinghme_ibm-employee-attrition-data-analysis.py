# Importing the necessary libraries.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

df.head()
# Some columns have same values and are purely redundant in this case. Hence we can drop or remove them.



cols_to_remove = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'PerformanceRating', 'StandardHours']



df = df[[col for col in df.columns if col not in cols_to_remove]]
# Lets check for any missing value within the data.



df.isnull().sum()
# Lets inspect our dataset a little.



df.info()
df.shape
# Mapping the Attrition Column



mapping = {'Yes': True, 'No': False}



df.Attrition = df.Attrition.map(mapping)

df.Attrition
sns.countplot(df.Attrition)
# Lets find out our dataset's naive attrition rate



print(len(df[df.Attrition==True])/len(df)*100)
# Sampling mean with confidence interval --- Defining our function



def conf_sample(data, a, b, func ,size):

    

    replicates = np.empty(size)

    

    for i in range(size):

        

        replicate = np.random.choice(data, len(data))

        

        """ This can be replaced with np.std as well"""

        replicates[i] = func(replicate)          

        

        x,y = np.percentile(replicates, [a,b])

        

    return (x,y)

    


conf_sample(df['Attrition'], 2.5, 97.5, np.mean, 10000)
# Before running any analysis, it is very important to understand how each variable (column) relates to other variables.



# Additionally, it is important to check whether the values in the column make any sense and are they practically possible.



# The more time we spend during this stage, the better would be our understanding of the dataset.



""" Keep experimenting and researching as you play around the dataset. Be an intrigued child """

# Lets look at the Probability Mass Function of the employees age.





def PMF(data):

    

    return((data.value_counts().sort_index()/len(data)))





age_pmf = PMF(df['Age'])

plt.figure(figsize=(10,5))

plt.title('Age Distribution')

age_pmf.plot.bar()
df.Age.describe()
sns.kdeplot(df.Age)
# Lets look at the employees who left the company.



age_pmf = PMF(df[df['Attrition']==True].Age)

plt.figure(figsize=(10,5))

age_pmf.plot.bar(color = 'r')

plt.title('Age PMF of Employees Who Left')
# Lets compare the empirical cumulative distributions of ages of two groups. (Attrited and Non Attrited Employees)





# Defining a function  for ecdf



def ecdf(data):

    

    y = (np.arange(1, len(data) + 1))/len(data)

    x = np.sort(data)

    return x,y



# PLotting the ECDFS



x_yes, y_yes = ecdf(df[df['Attrition']==True].Age)

x_no, y_no = ecdf(df[df['Attrition']==False].Age)

plt.figure(figsize=(10,5))

plt.plot(x_yes, y_yes, linestyle = 'none', marker = '.', color = 'r')

plt.plot(x_no, y_no, linestyle = 'none', marker = '.', color = 'b')

plt.ylabel('PROPORTION')

plt.title('ECDFS')

plt.legend(['Yes','No'], title = 'Attrition')



plt.annotate('Higher Difference',

             xy = (35, 0.5),

             xytext = (45, 0.4),

             arrowprops = {'arrowstyle':'->', 'color':'gray'})
df.groupby('Gender').Attrition.mean()
df.groupby('BusinessTravel')['Attrition'].mean()
df.groupby('Department')['Attrition'].mean()



plt.figure(figsize=(10,5))

sns.barplot('Department','Attrition', data = df, hue = 'Gender', ci = None)
cols = ['Education', 'EducationField']

fig, ax = plt.subplots(len(cols),1, figsize= (10,5), constrained_layout=True)



for i, col in enumerate(cols):

    

    sns.barplot(col,'Attrition', data = df, ax = ax[i])

df.groupby('EducationField').MonthlyRate.mean()
df.groupby('Education').MonthlyRate.mean()
cols = ['JobInvolvement', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction']

fig, ax = plt.subplots(len(cols),1, figsize= (10,8), constrained_layout=True)



for i, col in enumerate(cols):

    

    sns.barplot(col,'Attrition', data = df, ax = ax[i], ci = None)

cols = ['JobLevel', 'JobRole']

fig, ax = plt.subplots(len(cols),1, figsize= (17,8), constrained_layout=True)



for i, col in enumerate(cols):

    

    sns.barplot(col,'Attrition', data = df, ax = ax[i], ci = None)
Attrition_Y = df[df['Attrition']==True]

Attrition_N = df[df['Attrition']==False]

sns.kdeplot(Attrition_Y.DistanceFromHome)

sns.kdeplot(Attrition_N.DistanceFromHome)

plt.legend(('Yes', 'No'))


# Lets look at distance from home and attrition levels among various job roles.



df.groupby(['JobRole','Attrition']).DistanceFromHome.mean().unstack()
df[df['DistanceFromHome']>10].groupby('BusinessTravel').Attrition.mean()
sns.kdeplot(Attrition_Y.MonthlyIncome)

sns.kdeplot(Attrition_N.MonthlyIncome)

plt.legend(('Yes', 'No'))
plt.figure(figsize=(20,8))

sns.boxplot('JobRole', 'MonthlyIncome',data = df)
plt.figure(figsize=(20,8))

sns.boxplot('JobRole', 'MonthlyIncome', hue = 'Gender',data = df)
df_c = df.select_dtypes('int64')

plt.figure(figsize=(15,15))



sns.heatmap(df_c.corr(), annot = True, fmt = '.2f')