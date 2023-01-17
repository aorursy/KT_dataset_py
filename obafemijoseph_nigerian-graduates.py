# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#for data processing

import pandas as pd

import numpy as np



#for visualizing

from matplotlib import pyplot as plt

import seaborn as sns

#make use of sns beautiful plots

sns.set()





#for MachineLearning

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
file_path= ('/kaggle/input/nigerian-gradaute-report-2018/Nigerian Graduates Destination Survey (2013 -2017) (Responses) - nigerian graduates survey 2018.csv')

data_original = pd.read_csv(file_path)

#make a copy of data you can always come back to incase of mistakes

data = data_original
#Lets check the statistical summaries of both the categorical and numerical data

data.head()

data.describe(include = 'all')

#renaming columns

old_columns = list(data.columns.values)

#in order not to mess up the workspace, you display present column names, then rename them by slicing the list



new_columns=['Time','Gender','Grad_Y','Course','School','Highest_Q','Current_Stat','No_Of_Jobs','NYSC_cert','NYSC_Year',

             'Through_NYSC','FJob_Level','FJob_Role','FJob_sector','FJ_Income_Level','FJ_Required_HQ','Reason_FJ',

             'PJ_level', 'PJ_Role','PJ_Sector','PJ_Income','PJ_Required_HQ','Reason_PJ',

            'Best_Employer','Reason_Best_Employer','Most_PS','Currency','Job_Hours','Most_Important_Qualification',

             'Findout_Job','Worked_For_Employer',

            'Transport_TW','Rent_Buy','CP_Job','CP_Further_Studies','Skills_Prepared']

columns_dict = dict(zip(old_columns,new_columns))

new_data = data.rename(columns=columns_dict)



#we call data.head() to confirm that our new_columns has been renamed accordingly

new_data.head()
#The columns has been renamed to shorter neater forms, but just in case, lets create a dataframe that can be used as dictionary

#in case we forgot what the columns means

dictionary = pd.DataFrame({'Column Names':[x for x in new_columns],'Meaning':[y for y in old_columns]})
#now before we move forward, lets examine each columns and see what explorations we can do.

#lets start with the gender, to examine the balace in the gender of participant

new_data.Gender.unique()
#to check if they are moderately/evenly distributed

new_data.Gender.value_counts()
def get_percent_age(new_data):    

    l = list(new_data.Gender.value_counts())

    percent_male = l[0]*100/sum(l)

    percent_female = l[1]*100/sum(l)

    others = l[2]*100/sum(l)

    return percent_male,percent_female,others

#percentages = get_percent_age(new_data)

#print(percentages)
#dropping the 'Prefered Not to say'

index = new_data[new_data['Gender']=='Prefer not to say'].index

new_data.drop(index, axis = 0, inplace = True)
#checking

new_data.Gender.unique()
#for Graduation year, lets check if there are any problems

new_data.Grad_Y.unique()
new_data.Course.value_counts()
print(set(new_data.Course[200:500]))
def pick_top(data,interest):

    courses_counts = data[interest].value_counts() 

    low_courses=[]

    low_courses_index=[]

    for course, count in courses_counts.items():

        if count<10:

            low_courses.append(course)

    for i in low_courses:

        index = data[data[interest]==i].index

        for i in index:

            low_courses_index.append(i)

    return low_courses_index
dropped_courses_index= pick_top(new_data, 'Course')

major_courses = new_data.drop(dropped_courses_index, axis=0)

data_copy = major_courses

data_copy.School.value_counts()
dropped_schools_index = pick_top(data_copy, 'School')

major_schools = data_copy.drop(dropped_schools_index, axis=0)
major_schools.Highest_Q.value_counts()
major_schools.Current_Stat.value_counts()
#for currency

major_schools.Currency.value_counts()
major_schools.Job_Hours.isnull().sum()
major_schools['Job_Hours'] = major_schools['Job_Hours'].fillna(0)
major_schools.Most_Important_Qualification.value_counts()
low_qualifications = pick_top(major_schools,'Most_Important_Qualification')

Most_qualifications = major_schools.drop(low_qualifications, axis = 0)
Most_qualifications.Findout_Job.value_counts()
Most_qualifications.CP_Further_Studies.value_counts()
#for option i, we have to do the following:

#1. you pick the ones that are done with NYSC,as only those could be working, so we drop nysc option, using  the major schools

drop_nysc = major_schools[major_schools.Current_Stat !='Youth Corper (NYSC)']

#so now we have a dataframe containig only people that have completed their NYSC
#2. generalise the group into two, employed and unemployed

#next we seperate them into employed and unemployed, we will consider only employed, self employed as well as umemployed

data1=drop_nysc[drop_nysc.Current_Stat == 'Working full time (paid employment)']

data2=drop_nysc[drop_nysc.Current_Stat == 'Unemployed']

data3=drop_nysc[drop_nysc.Current_Stat == 'Self-employed/freelance/entrepreneur']

frames = [data1,data2,data3]

employed_by_school = pd.concat(frames)

employed_by_school
def final_table(data):

    '''Returns the final data frame including students employability stats'''

    school_ = data.groupby('School')['Current_Stat'].value_counts()

    #which returns a key/values pairs, we can get the name of school by



    list_schools = list(school_.keys())

    #then we extract only the names of schools

    names_of_school = []

    for i in range(len(list_schools)):

        names_of_school.append(list_schools[i][0])

    name_of_school = set(names_of_school)

    

    #create dataframe for the name of schools

    name_of_schools_df = pd.DataFrame ({'Name_Of_School':[x for x in name_of_school]})

    #sort it alphabetically

    name_of_school_df = name_of_schools_df.sort_values(by='Name_Of_School', ascending=True)

    name_of_school_df = name_of_school_df.reset_index(drop=True)

    

    

    #Below is a list of the numbers for each school, it was computed manually, but there will research to see how it can be

    #automatically computed

    unemployed = [11,2,19,11,19,2,10,13,26,35,4,3,7,7,4,22,4,3,30,4,4,6,16,5,

                  36,8,6,12,2,27,13,18,6,6,15,6,1,2,5,1,6,41,13,22,8,10,10,10,

                  9,4,20,69,3,24,6,25,5,17,11,2,9,25,4,36,4,38,35,5,96,11,19,34,5,6,5,21]

    working=[8,4,19,8,12,4,12,9,3,37,2,10,3,3,2,33,4,5,154,1,4,1,4,6,24,6,3,12,2,29,18,24,

             0,6,15,3,4,4,4,1,2,24,12,31,6,9,5,6,9,3,21,101,6,21,11,43,17,11,6,7,7,26,1,50,

             6,60,53,5,135,3,43,21,4,1,0,10]

    self_employed=[5,3,0,5,10,1,10,1,8,14,3,3,

                   3,2,2,10,4,0,49,4,0,1,9,2,20,8,1,13,4,13,10,11,2,1,3,5,5,2,4,3,3,16,8,19,2,

                   2,0,2,3,0,6,33,2,7,3,15,1,5,5,9,7,16,4,15,3,20,17,4,62,4,64,15,6,2,2,6]

    

    

    

    #Add another column for unemployed, employed, self employed, sum of graduates, percentage unemployed, percentage employed

    Unemployed_df = pd.DataFrame({'Unemployed':[x for x in unemployed]})

    employed_df = pd.DataFrame({'Employed':[x for x in working]})

    self_employed_df = pd.DataFrame({'Self_Employed':[x for x in self_employed]})

   

    #create a list of the dataframes

    dataframes = [name_of_school_df,Unemployed_df,employed_df,self_employed_df]



    #return a concatenated dataframe

    Name_of_schools_df= pd.concat(dataframes,sort=False, axis = 1)

    return Name_of_schools_df
final_dataframe= final_table(employed_by_school)

final_dataframe
def calculate_summaries(data):

    '''Calculates the percentage as well as the sum total of features'''

    data['Total_Graduates']=data['Unemployed']+data['Employed']+data['Self_Employed']

    data['Percentage_Employed']=(data['Employed']*100)/(data['Unemployed']+data['Employed']+data['Self_Employed'])

    data['Percentage_Unemployed']=(data['Unemployed']*100)/(data['Unemployed']+data['Employed']+data['Self_Employed'])

    data['Percentage_Self_Employed']=(data['Self_Employed']*100)/(data['Unemployed']+data['Employed']+data['Self_Employed'])

    data['Total_Working_Percentage']=data['Percentage_Self_Employed']+data['Percentage_Employed']

    return data
full_data_frame = calculate_summaries(final_dataframe)

full_data_frame
full_data_frame.groupby(['Name_Of_School','Total_Graduates'])['Total_Working_Percentage'].agg(['max'])
copy_for_use = full_data_frame
status = ['Government','Government','Government','Private','Government','Private','Government','Government','Government','Private',

          'Government','Private','Private','Government','Private','Private','Private','Private','Private','Private','Private',

          'Government','Government','Government','Government','Government','Government','Government','Government','Government',

          'Government','Government','Government','Private','Government','Government','Government','Government','Government','Government',

          'Government','Government','Government','Government','Private','Private','Government','Government','Government','Government',

          'Government','Government','Government','Government','Government','Government','Government','Private','Government','Government',

          'Government','Government','Government','Government','Government','Government','Government','Government','Government','Government',

           'Government','Government','Government','Government',

           'Private',

          'Government']
copy_for_use['Government_or_Private']=status

copy_for_use
copy_for_use.groupby(['Government_or_Private','Total_Graduates'])['Total_Working_Percentage'].agg(['max'])


