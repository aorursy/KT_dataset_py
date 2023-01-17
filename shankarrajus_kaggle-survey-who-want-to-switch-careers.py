# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
### Jupyter notebook settings

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



### Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



### Interactive visualization

# from plotly import __version__

# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# init_notebook_mode(connected=False)  



# import cufflinks as cf

# cf.go_offline()



### Ignoring warnings

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
multi_choice_raw = pd.read_csv('../input/multipleChoiceResponses.csv', header=0, encoding='latin-1', low_memory=False)
### Quick data glimpse

multi_choice_raw.head()

multi_choice_raw.info()

multi_choice_raw.describe()
### Check missing values

sns.set_style('whitegrid')

fig = plt.figure(figsize=(20,12))

ax = sns.heatmap(multi_choice_raw.isnull(), cbar=False, yticklabels=False, cmap='viridis_r')  #Yellow is value
### Data cleaning



#0) Preserve the original data, beware of shallow copy

multi_choice = multi_choice_raw[:]



#1) GenderSelect  -- Simplifies gender types to Male, Female and Others

multi_choice['GenderSelect'] = multi_choice['GenderSelect'].apply(lambda x: 'Others' if x not in ('Male', 'Female') else x )

# multi_choice['GenderSelect'].isnull().sum()

# multi_choice['GenderSelect'].unique()



#2) Age -- Make NaN to 0

multi_choice['Age'].fillna(value = 0, inplace = True)

# multi_choice['Age'].isnull().sum() 

# multi_choice['Age'].unique()



#3) Employment Status -- Create RespType to categorize respondents as 'Working', 'Learners' or 'Not_Working'

def EmpType( Args ):

    EmpStatus = Args[0]

    StudStatus = Args[1]

    LearningDS = Args[2]

    if (EmpStatus in ["Employed full-time", "Employed part-time", "Independent contractor, freelancer, or self-employed","Retired"]):

        return 'Working'

    elif ((StudStatus in ['Yes']) | (EmpStatus in ['Not employed, but looking for work']) | (LearningDS.startswith('Yes'))):

        return 'Learners'

    elif (EmpStatus in ["Not employed, and not looking for work" ,"I prefer not to say"]):

        return 'Not_Working'

    else:

        return 'Others'



multi_choice['EmploymentStatus'].fillna(value = 'Not Available', inplace = True)

multi_choice['StudentStatus'].fillna(value = 'Not Available', inplace = True)

multi_choice['LearningDataScience'].fillna(value = 'Not Available', inplace = True)



# multi_choice.groupby(by = ['EmploymentStatus', 'StudentStatus', 'LearningDataScience']).agg('size').reset_index()  

multi_choice['RespType'] = multi_choice[['EmploymentStatus', 'StudentStatus', 'LearningDataScience']].apply(EmpType, axis =1)

multi_choice.groupby(by = ['EmploymentStatus', 'StudentStatus', 'LearningDataScience', 'RespType']).agg('size').reset_index()  



#4) Fixing Nans

multi_choice['CurrentJobTitleSelect'].fillna(value = 'Not Available', inplace = True)

multi_choice['CareerSwitcher'].fillna(value = 'Not Available', inplace = True)

multi_choice['CodeWriter'].fillna(value = 'Not Available', inplace = True)



# multi_choice['CurrentJobTitleSelect'].unique()

# multi_choice['CareerSwitcher'].unique()

# multi_choice['CodeWriter'].unique()
## Gender and Country analysis

gender_agg = multi_choice.groupby('GenderSelect').agg('size')

t = plt.pie(gender_agg.values, labels=gender_agg.index, autopct='%1.1f%%',shadow=True, 

        startangle=90, colors=(['Orange','LightBlue','Grey']) )

t = plt.title('Gender Split-up', size = 20)



fig = plt.figure(figsize=(12,12))

ax = multi_choice['Country'].value_counts().sort_values().plot(kind='barh' )

t = plt.title('Country survey participation Split-up', size = 20)





# gender_agg = np.sum(pd.get_dummies(multi_choice['GenderSelect']))

# gender_agg.index = ['Others', 'Female', 'Male', 'Others']

# gender_agg

# plt.pie(gender_agg.values, labels=gender_agg.index, labeldistance=1.2)

# sns.countplot(x = 'GenderSelect', data = multi_choice, order = ['Male', 'Female', 'A different identity'], palette='viridis')
### Recommendation for next year



# cols = ['RespType', 'EmploymentStatus']

# temp_grp_w1 = multi_choice.groupby(by = cols , sort=False, as_index=False)

# temp_grp_w1.agg('size').sort_values(ascending=False)



fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10,8))

t = sns.countplot(x = 'RespType', hue = 'EmploymentStatus', data=multi_choice, orient='h', ax=ax1 )

t = ax1.set_title('Respondent working types', fontdict={'fontsize':20})

t = ax1.legend(loc=9)



### Workers learning choice

fig, (ax2, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(20,12))

cols = ['RespType', 'MLToolNextYearSelect']

temp_grp_w2 = multi_choice.loc[multi_choice.RespType == 'Working', :].groupby(by = cols , sort=False, as_index=False)

temp_w2 = temp_grp_w2.agg('size').nlargest(10).sort_values(ascending=False).reset_index()

temp_w2.columns = ['RespType', 'MLToolNextYearSelect', 'Voted_size']

# temp_w2



t = sns.barplot(y = 'MLToolNextYearSelect', x='Voted_size' , data = temp_w2, ax = ax2 , orient='h')

t = plt.xticks(rotation = 17)

t = ax2.set_title('Worker\'s MLtool wish', fontdict={'fontsize':20})



### Non-worker's learning choice

# fig, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(10,8))

cols = ['RespType', 'MLToolNextYearSelect']

temp_grp_w3 = multi_choice.loc[multi_choice.RespType == 'Learners', :].groupby(by = cols , sort=False, as_index=False)

temp_w3 = temp_grp_w3.agg('size').nlargest(10).sort_values(ascending=False).reset_index()

temp_w3.columns = ['RespType', 'MLToolNextYearSelect', 'Voted_size']

# temp_w3



t = sns.barplot(y = 'MLToolNextYearSelect', x='Voted_size' , data = temp_w3, ax = ax3, orient='h' )

tks = plt.xticks(rotation = 17)

t = ax3.set_title('Learner\'s MLtool wish', fontdict={'fontsize':20})



### Which Workers wants to switch jobs ?



# multi_choice['CurrentJobTitleSelect'].unique()

# multi_choice['CareerSwitcher'].unique()

# multi_choice['CodeWriter'].unique()



## JobTitle 

fig = plt.figure(figsize=(20,12))

t = plt.subplot(1,2,1)

t = sns.countplot(x='CurrentJobTitleSelect', hue='CareerSwitcher' , data=multi_choice.loc[(multi_choice['RespType'] == 'Working') & (multi_choice['CareerSwitcher'] != 'Not Available' ), ['CurrentJobTitleSelect', 'CodeWriter', 'CareerSwitcher']], palette='coolwarm')

t = plt.title('JobTitle Vs CareerSwitchers', size = 20)

t = plt.xticks(rotation=90)



temp_agg1 = multi_choice.groupby('CurrentJobTitleSelect').agg('size').reset_index()

temp_agg1.columns = ['CurrentJobTitleSelect' , 'size']

temp_agg1.sort_values(by = 'size', ascending = False, inplace=True)

t = plt.subplot(1,2,2)

t = plt.pie( temp_agg1['size'] , labels = temp_agg1['CurrentJobTitleSelect'], autopct='%1.1f%%',shadow=True, 

        startangle=90, colors=(['Orange','LightBlue','Grey']))

t = plt.title('CurrentJobTitle', size=20)

t = plt.xticks(rotation=90)



## Code writers

fig = plt.figure(figsize=(20,12))

t = plt.subplot(1,2,1)

t = sns.countplot(x='CodeWriter', hue='CareerSwitcher' , data=multi_choice.loc[(multi_choice['RespType'] == 'Working') & (multi_choice['CareerSwitcher'] != 'Not Available' ), ['CurrentJobTitleSelect', 'CodeWriter', 'CareerSwitcher']], palette='coolwarm')

t = plt.title('CodeWriter Vs CareerSwitchers', size = 20)

t = plt.xticks(rotation=90)



# multi_choice.groupby(by = ['CodeWriter', 'CareerSwitcher']).agg('size').reset_index()



temp_agg1 = multi_choice.groupby('CodeWriter').agg('size').reset_index()

temp_agg1.columns = ['CodeWriter' , 'size']

t = plt.subplot(1,2,2)

t = plt.pie( temp_agg1['size'] , labels = temp_agg1['CodeWriter'], autopct='%1.1f%%',shadow=True, 

        startangle=90, colors=(['Orange','LightBlue','Grey']))

t = plt.title('Code Writers', size=20)



## Both JobTitle_CodeWriter

temp_df1  = multi_choice.loc[(multi_choice['RespType'] == 'Working') & (multi_choice['CareerSwitcher'] != 'Not Available' ), ['CurrentJobTitleSelect', 'CodeWriter', 'CareerSwitcher'] ]

temp_df1['JobTitle_CodeWriter'] = temp_df1['CurrentJobTitleSelect'].map(str) + '_' + temp_df1['CodeWriter'].map(str)



fig = plt.figure(figsize=(20,12))

t = plt.subplot(1,2,1)

t = sns.countplot(x='JobTitle_CodeWriter', hue='CareerSwitcher' , data=temp_df1, palette='coolwarm')

t = plt.title('JobTitle_CodeWriter Vs CareerSwitchers', size = 20)

t = plt.xticks(rotation=90)



temp_agg1 = multi_choice.groupby('CareerSwitcher').agg('size').reset_index()

temp_agg1.columns = ['CareerSwitcher' , 'size']

t = plt.subplot(1,2,2)

t = plt.pie( temp_agg1['size'] , labels = temp_agg1['CareerSwitcher'], autopct='%1.1f%%',shadow=True, 

        startangle=90, colors=(['Orange','LightBlue','Grey']))

t = plt.title('Wish to switch Career', size=20)