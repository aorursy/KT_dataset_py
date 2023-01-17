# This Python 3 environment comes with many helpful analytics libraries installed# libraries for data wrangling

import pandas as pd

import numpy as np

# libraries for plotting

import matplotlib.pyplot as plt

%matplotlib inline  

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

sns.set(style="whitegrid")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# reading longitudinal data

df_long = pd.read_csv('../input/mri-and-alzheimers/oasis_longitudinal.csv')
# lets see first few entries of the dataset

df_long.head()
# lets see the summary stats of numerical columns

df_long.describe(include=[np.number])
# lets see the summary of categorical columns

df_long.describe(include=[np.object])
# dropping irrelevant columns

df_long=df_long.drop(['Subject ID','MRI ID','Hand'],axis=1)



df_long.head()
# checking missing values in each column

df_long.isna().sum()
# for better understanding lets check the percentage of missing values in each column

round(df_long.isnull().sum()/len(df_long.index), 2)*100
# Plotting distribution of SES

def univariate_mul(var):

    fig = plt.figure(figsize=(16,12))

    cmap=plt.cm.Blues

    cmap1=plt.cm.coolwarm_r

    ax1 = fig.add_subplot(221)

    ax2 = fig.add_subplot(212)

    df_long[var].plot(kind='hist',ax=ax1, grid=True)

    ax1.set_title('Histogram of '+var, fontsize=14)

    

    ax2=sns.distplot(df_long[[var]],hist=False)

    ax2.set_title('Distribution of '+ var)

    plt.show()
# lets see the distribution of SES to decide which value we can impute in place of missing values.

univariate_mul('SES')

df_long['SES'].describe()
# imputing missing value in SES with median

df_long['SES'].fillna((df_long['SES'].median()), inplace=True)
univariate_mul('MMSE')

df_long['MMSE'].describe()
# imputing MMSE with median values

df_long['MMSE'].fillna((df_long['MMSE'].median()), inplace=True)
# Now, lets check the percentage of missing values in each column

round(df_long.isnull().sum()/len(df_long.index), 2)*100
# Defining function to create pie chart and bar plot as subplots

def plot_piechart(var):

  plt.figure(figsize=(14,7))

  plt.subplot(121)

  label_list = df_long[var].unique().tolist()

  df_long[var].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",7),startangle = 60,labels=label_list,

  wedgeprops={"linewidth":2,"edgecolor":"k"},shadow =True)

  plt.title("Distribution of "+ var +"  variable")



  plt.subplot(122)

  ax = df_long[var].value_counts().plot(kind="barh")



  for i,j in enumerate(df_long[var].value_counts().values):

    ax.text(.7,i,j,weight = "bold",fontsize=20)



  plt.title("Count of "+ var +" cases")

  plt.show()



plot_piechart('Group')
df_long['CDR'].describe()
# Plotting CDR with other variable

def univariate_percent_plot(cat):

    fig = plt.figure(figsize=(18,12))

    cmap=plt.cm.Blues

    cmap1=plt.cm.coolwarm_r

    ax1 = fig.add_subplot(221)

    ax2 = fig.add_subplot(222)

    

    result = df_long.groupby(cat).apply (lambda group: (group.CDR == 'Normal').sum() / float(group.CDR.count())

         ).to_frame('Normal')

    result['Dementia'] = 1 -result.Normal

    result.plot(kind='bar', stacked = True,colormap=cmap1, ax=ax1, grid=True)

    ax1.set_title('stacked Bar Plot of '+ cat +' (in %)', fontsize=14)

    ax1.set_ylabel('% Dementia status (Normal vs Dementia)')

    ax1.legend(loc="lower right")

    group_by_stat = df_long.groupby([cat, 'CDR']).size()

    group_by_stat.unstack().plot(kind='bar', stacked=True,ax=ax2,grid=True)

    ax2.set_title('stacked Bar Plot of '+ cat +' (in %)', fontsize=14)

    ax2.set_ylabel('Number of Cases')

    plt.show()







# Categorizing feature CDR

def cat_CDR(n):

    if n == 0:

        return 'Normal'

    

    else:                                         # As we have no cases of sever dementia CDR score=3

        return 'Dementia'



df_long['CDR'] = df_long['CDR'].apply(lambda x: cat_CDR(x))
plot_piechart('CDR')
df_long['MMSE'].describe()
# Categorizing feature MMSE

def cat_MMSE(n):

    if n >= 24:

        return 'Normal'

    elif n <= 9:

        return 'Severe'

    elif n >= 10 and n <= 18:

        return 'Moderate'

    elif n >= 19 and n <= 23:                                        # As we have no cases of sever dementia CDR score=3

        return 'Mild'



df_long['MMSE'] = df_long['MMSE'].apply(lambda x: cat_MMSE(x))
plot_piechart('MMSE')
univariate_percent_plot('MMSE')
univariate_mul('Age')

df_long['Age'].describe()
df_long['age_group'] = pd.cut(df_long['Age'], [60, 70, 80,90, 100], labels=['60-70', '70-80', '80-90','90-100'])

df_long['age_group'].value_counts()
# Now plotting age group to see dementia distribution

univariate_percent_plot('age_group')
plt.figure(figsize=(12, 8))

ax = sns.violinplot(x="M/F", y="Age",hue="CDR",split=True, data=df_long)

plt.show()
df_long['eTIV'].describe()
plt.figure(figsize=(12, 8))

ax = sns.violinplot(x="age_group", y="eTIV",hue="CDR",split=True, data=df_long)

plt.show()
plt.figure(figsize=(12, 8))

ax = sns.violinplot(x="M/F", y="eTIV",hue="CDR",split=True, data=df_long)

plt.show()
plt.figure(figsize=(12, 8))

ax = sns.violinplot(x="M/F", y="nWBV",hue="CDR",split=True, data=df_long)

plt.show()

df_long['EDUC'].describe()
plt.figure(figsize=(12, 8))

ax = sns.violinplot(x="M/F", y="EDUC",hue="CDR",split=True, data=df_long)

plt.show()

df_long['SES'].describe()
# Now plotting socio economic status to see dementia distribution

univariate_percent_plot('SES')
plt.figure(figsize=(12, 8))

ax = sns.violinplot(x="M/F", y="SES",hue="CDR",split=True, data=df_long)

plt.show()

df_long['ASF'].describe()
plt.figure(figsize=(12, 8))

ax = sns.violinplot(x="M/F", y="ASF",hue="CDR",split=True, data=df_long)

plt.show()
plt.figure(figsize=(12, 8))

ax = sns.violinplot(x="MMSE", y="ASF",split=True, data=df_long)

plt.show()
plt.figure(figsize=(12, 8))

ax = sns.violinplot(x="MMSE", y="nWBV",split=True, data=df_long)

plt.show()
plt.figure(figsize=(12, 8))

ax = sns.violinplot(x="MMSE", y="Visit",split=True, data=df_long)

plt.show()
plt.figure(figsize=(14, 8))

sns.heatmap(df_long.corr(), annot=True)

plt.show()