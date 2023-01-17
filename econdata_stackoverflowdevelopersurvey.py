# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')  #index_col = 'Respondent'
df
df.shape  # dataframe attribute
df.info()  # dataframe  info method gives datatype
# want to see all the column

# need to change the settings

pd.set_option('display.max_columns',85)

pd.set_option('display.max_rows',85)
schema_df =pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv')
schema_df
df.head()   # head method
df.tail() # tail method
# datafraame is a python object
person = {

    "first":"corey",

    "last":"schafer",

    "email":"coreyschafer@gmail.com"

}
person = {

    "first":["corey"],

    "last":["schafer"],

    "email":["coreyschafer@gmail.com"]

}

person = {

    "first":["corey",'joe','jane'],

    "last":["schafer",'doe','ed'],

    "email":["coreyschafer@gmail.com",'joe@hoe','hoe@doe']

}

person['email']
person['email'][0]
for i  in person['email']:

    print(i)
dframe = pd.DataFrame(person)
dframe
# accessing the single column

# equivalent to accessing the single key from dictionary
dframe['email'] # prefernce personal 
type(dframe['email'])
# series is  like a list of data

# daatframe is the container of multiple series

# dataframe has rows and columns
dframe.email #not prefered
# i want to access a list of columns
dframe[['last','email']]
type(dframe[['last','email']])
df.columns
# LOC AND ILOC index
#iloc is integer location
df.iloc[0]
dframe.iloc[0] #selecting fows
dframe.iloc[[0,1]] #selecting column
dframe.iloc[[0,1],2] #selecting column
# loc , searching by labels
dframe.loc[0]==dframe.loc[0]
dframe.loc[[0,1],'email']
df.shape
df['Hobbyist']
# attach valu count methods

df['Hobbyist'].value_counts()
df.loc[0]
df.loc[0,'Hobbyist']
type(df.loc[0,'Hobbyist'])
df.loc[[0,1,2],'Hobbyist']
# indexing and slicing
df.loc[0:2,'Hobbyist']
df.loc[0:2,'Hobbyist':'Employment']
type(df.loc[0:2,'Hobbyist':'Employment'])
type(df.loc[0:2,'Hobbyist'])
# string , series and dataframe
type(df.loc[0:2,'Hobbyist':'OpenSourcer'])
df.loc[0:2,'Hobbyist':'OpenSourcer']
#  dataframe and series objects
# these are two main datatypes
dframe['email']
dframe.set_index('email')   # set index method
# set index option
# pandas doesnt do inplace changes
dframe.set_index('email',inplace=True)   # set index method
dframe
dframe.index
# emakil addrsss gives the nice unique identifier
dframe.loc['joe@hoe']
dframe.loc['joe@hoe','last']
dframe.reset_index(inplace=True)
schema_df.set_index('Column')
schema_df
schema_df.set_index('Column',inplace = True)
schema_df
schema_df.loc['MgrIdiot']
schema_df.loc['MgrIdiot','QuestionText']
schema_df.sort_index()   #sorting the index
schema_df.sort_index(ascending  = False)   #sorting the index
schema_df
schema_df.sort_index(inplace = True)   #sorting the index
schema_df
schema_df.sort_index(ascending  = False,inplace = True)   #sorting the index
schema_df
schema_df.sort_index(ascending  = True,inplace = True)   #sorting the index
schema_df
dframe['last']=='doe' #filter mask
schema_df['QuestionText']
# basic comparison with filter
dframe['last']== 'joe'
type(dframe['last']== 'joe')
FILT  = (dframe['last']== 'doe')
FILT
type(FILT)
dframe[FILT]
high_salary = (df['ConvertedComp']>10000)     # creating filters
df.loc[high_salary]
df.loc[high_salary,['Country','LanguageWorkedWith','ConvertedComp']]
countries = ['Unites States','India','United Kingdom','Germany','Canada']

filt = df['Country'].isin(countries)
df.loc[filt,'Country']
df['LanguageWorkedWith']
filt = df['LanguageWorkedWith'].str.contains('Python',na = False)  # filter , looking up for python lang users
df.loc[filt,'LanguageWorkedWith']
filt
df.loc[filt]
# alter existing rows and columns in our data frame'
dframe.columns
type(df.columns)
dframe.columns = ['firstname','lastname','email']
dframe.columns = dframe.columns.str.replace(' ','-')
dframe
df.rename(columns={'ConvertedComp':'SalaryUSD'})
df.rename(columns={'ConvertedComp':'SalaryUSD'},inplace = True)
df
df['SalaryUSD']
df['Hobbyist'].map({'Yes':True,'No':False})
df['Hobbyist']=df['Hobbyist'].map({'Yes':True,'No':False})
df
# previously update information in rows and columns
# update information in rows and  columns


# combining columns
dframe['firstname'] + ' ' + dframe['lastname']
dframe['fullname']=dframe['firstname'] + ' ' + dframe['lastname']
dframe
dframe.drop(columns=['firstname','lastname'])
dframe['fullname'].str.split(' ',expand = True)
df.sort_values(by="Country",inplace = True)
df
df['Country']
df.columns
# grouping and aggregating
# grouping and aggregating
# mean meadian and mode are aggregate functions
# what is the typical salary of a developer
df['SalaryUSD'].head(5)
df['SalaryUSD'].median()
# ignore nan values
df.median()  # running the aggregate function on the entire data frame
df.describe()   # describe method on data frame
# mean is affected heavily by outliers
df['Respondent'].describe()
df['SalaryUSD'].count()
df.shape
df['Hobbyist'].value_counts()  # value counts methods on series
df['SocialMedia'].value_counts()  # value counts methods on series
df['SocialMedia']
schema_df.loc['SocialMedia']
df['SocialMedia'].value_counts(normalize = True)  # value counts methods on series
# country wise social media popularity
# groupby function
# splitting the object,apply the function and combine those results
df['Country'].value_counts()
country_grp = df.groupby(['Country'])
country_grp.get_group('United States')
country_grp.get_group('India')
filt = df['Country'] == 'United States'
df[filt]
# most popular social media sites broken down by country wise US
df.loc[filt]['SocialMedia'].value_counts()
filt = df['Country'] == 'India'
df.loc[filt]['SocialMedia'].value_counts()
country_grp['SocialMedia'].value_counts()
country_grp['SocialMedia'].value_counts().head(50)
country_grp['SocialMedia'].value_counts().loc['India']
country_grp['SocialMedia'].value_counts().loc['United States']
country_grp['SocialMedia'].value_counts(normalize = True).loc['China']
country_grp['SocialMedia'].value_counts(normalize = True).loc['Russian Federation']
# median salary for all these countries
country_grp['SalaryUSD'].median()
country_grp['SalaryUSD'].median().loc['Germany']
country_grp['SalaryUSD'].agg(['median','mean'])
country_grp['SalaryUSD'].agg(['median','mean']).loc['Canada']
# python as the preference
filt  = df['Country'] == 'India'

df.loc[filt]['LanguageWorkedWith'].str.contains('Python')
df.loc[filt]['LanguageWorkedWith'].str.contains('Python').sum()
# apply method on grop series object

# nice quick easy function -lambda function
country_grp['LanguageWorkedWith'].apply(lambda  x : x.str.contains('Python').sum())
country_grp['LanguageWorkedWith'].apply(lambda  x : x.str.contains('Python').sum()).loc['India']
country_respondent = df['Country'].value_counts()
country_respondent
country_uses_python = country_grp['LanguageWorkedWith'].apply(lambda  x : x.str.contains('Python').sum())
country_uses_python
python_df = pd.concat([country_respondent,country_uses_python],axis = 'columns',sort = False)
python_df
python_df .rename(columns = {'Country':'NumRespondents','LanguageWorkedWith':'NumKnowsPython'},inplace = True)
python_df
# create a new column
python_df['PctknowsPython'] = (python_df['NumKnowsPython']/python_df['NumRespondents'])*100
python_df
python_df.sort_values(by='PctknowsPython',ascending = False,inplace = True)
python_df
python_df.head(70)
python_df.loc['Japan']
import numpy as np
# nan is float under the hood
# calualate the average number of coding experience participated in the survey
df['YearsCode'].head(10)
df['YearsCode'].unique()
df['YearsCode'].replace('Less than 1 year',0,inplace = True)
df['YearsCode'].replace('More than 50 years',0,inplace = True)
df['YearsCode'] = df['YearsCode'].astype(float)
df['YearsCode'].mean()
df['YearsCode'].median()