# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#Visualization libraries 
import seaborn as sns
import matplotlib.pyplot as plt


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

survey = '../input/SurveySchema.csv'
response = '../input/freeFormResponses.csv'
mcq = '../input/multipleChoiceResponses.csv'
df_survey = pd.read_csv(survey)
df_survey.head()
df_survey.columns
print('Below are all the questions that were asked:\n')

# df_survey[['Q1', 'Q10','Q11', 'Q12', 'Q13', 'Q14']]
# df_survey[[]]
# df_survey[['Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q2']]
# for i in df_survey.columns:
#     print(i,':',df_survey[i][0])

df_response = pd.read_csv(response)
df_response.isna().values.any()
# df_response.isna().sum()
# df_response.describe()
# df_response.head()
df_mcq = pd.read_csv(mcq)
# df_mcq.head(5)
gender_age_orig = df_mcq[['Q1','Q2']]
gender_age_orig.columns = ['gender', 'age_range']

# gender_age_orig.sample(10)
gender_age_orig.head()
# gender_age['gender'].describe()
# gender_age.age_range.unique()


gender_age= gender_age_orig
gender_age = gender_age.drop(gender_age.index[[0]])
gender_age.head()
age_unique = gender_age['age_range'].unique()
print('The respondents fell in these age ranges: ')
for age in age_unique:
    print(age)
# gender_age.head()
# gender_age.dtypes
age_code = {'18-21':1, '22-24':2, '25-29':3,'30-34':4,'35-39':5, '40-44':6,'45-49':7, '50-54':8,'55-59':9, '60-69':10, '70-79':11, '80+':12  }
for k, v in age_code.items():
    print('The code for the age group', k, 'is :', age_code[k])
gender_age['gender']= gender_age['gender'].astype('category').cat.codes
# gender_age['age_range'] = gender_age['age_range'].astype('category')
gender_age['age_range'] = gender_age['age_range'].map({'18-21':1, '22-24':2, '25-29':3,'30-34':4,'35-39':5, '40-44':6,'45-49':7, '50-54':8,'55-59':9, '60-69':10, '70-79':11, '80+':12  })#.astype(int)


df_ga = gender_age 
sns.set(style = "darkgrid")
# sns.set(rc={'figure.figsize':(10,10)})
ax_sns = sns.countplot(x = 'age_range', data = df_ga).set_title('Age range distribution for Data Scientists')
df_ga = df_ga[df_ga['gender']<2]
df_ga_m = df_ga.loc[df_ga['gender'] == 1]
df_ga_f = df_ga.loc[df_ga['gender'] == 0]

# Female
sns.set(style= "darkgrid", rc={'figure.figsize':(8,8)})
ax_f = sns.countplot(x = "age_range", data = df_ga_f).set_title('Age distribution of female Data Scientists')
# Male 
sns.set(style= "darkgrid", rc={'figure.figsize':(8,8)})
ax_b = sns.countplot(x = "age_range", data = df_ga_m).set_title('Age distribution of male Data Scientists')
# df_ga = df_ga[df_ga['gender']<2]
# df_ga_m = df_ga.loc[df_ga['gender'] == 1]
# df_ga_f = df_ga.loc[df_ga['gender'] == 0]

males = df_ga_m['gender'].value_counts()
# print(males)
female = df_ga_f['gender'].value_counts()
# print(female)

gender_total = {'gender':[1,0] ,'total': [males.values[0],female.values[0]]}
print(gender_total)
pd_gender_total = pd.DataFrame.from_dict(gender_total)

pd_gender_total

# df_ga[df_ga['gender']==2]
# df_ga['gender'].hist()
sns.set(style="darkgrid", rc={'figure.figsize':(5,5)})
ax_g = sns.countplot(x='gender', data = df_ga).set_title('Total Females(0) vs Males(1) Data Scientists')

#Q1: Q4, Q5,  
df_education = df_mcq[['Q1','Q4','Q5']]
df_education.columns  = ['gender', 'highest education', 'undergrad major']
df_education = df_education.drop(df_education.index[[0]])
# df_education.sample(5)
df_education.head(6)
# df_education.columns
majors =pd.DataFrame(df_education['undergrad major'].value_counts()).reset_index()
# majors = df_education['undergrad major'].unique()
majors.columns = ['undergrad major', 'count']
print('The total number of people answering these questions are:',majors['count'].sum())
# majors.describe() #13 total
majors
plt.figure(figsize=(10,5))
# ax = sns.barplot(majors['undergrad major'].head(5), majors['count'].head(5), alpha = 0.8)
ax = sns.barplot(majors['undergrad major'], majors['count'], alpha = 0.8)
ax.set_title('Undergraduate majors of Data Scientists')
ax.set_xlabel('Majors')
ax.set_ylabel('Number of people with the major.')
ax.set_xticklabels(majors['undergrad major'], rotation='vertical',fontsize= 10)
# ax.set_xticklabels(label_text['labels'], rotation='vertical', fontsize=10)
plt.show()                   
high_edu = df_education[['gender', 'highest education']]
high_edu.head(5)

edu_count = pd.DataFrame(high_edu['highest education'].value_counts()).reset_index()
edu_count.columns = ['degree', 'count']
edu_count
plt.figure(figsize= (10,5))
ax = sns.barplot(edu_count['degree'], edu_count['count'])
ax.set_title('Degrees held by Data Scientists')
ax.set_xlabel('Degree')
ax.set_ylabel('Number of people')
ax.set_xticklabels(edu_count['degree'], rotation='vertical', fontsize=10)
plt.show()
df_edu_money = df_mcq[['Q4','Q9']]
df_edu_money = df_edu_money.drop(df_edu_money.index[[0]])
df_edu_money.columns = ['highest education', 'salary']
df_edu_money= df_edu_money.dropna()
# I am going to drop all the rows where the participants didn't disclose their salary
df_edu_money= df_edu_money[df_edu_money['salary'] !='I do not wish to disclose my approximate yearly compensation']
# df_edu_money.head(7)
df_edu_money.sample(10)
# df_edu_money['salary'].unique()

# df_edu_money['salary'].head(7).unique()
salary_scales = {'0-10,000':0, '10-20,000':1,'20-30,000':2, '30-40,000':3,
                 '40-50,000':4,'50-60,000':5, '60-70,000':6,
                 '70-80,000':7, '80-90,000':8,'90-100,000':9,'100-125,000':10,
                 '125-150,000':11,'150-200,000':12 ,'200-250,000':13, '250-300,000':14 ,
                 '300-400,000':15,'400-500,000':16 , '500,000+':17}

for k, v in salary_scales.items():
    print('For anyone earning $',k,'their pay scale is:', v)
# gender_age['gender']= gender_age['gender'].astype('category').cat.codes
# # gender_age['age_range'] = gender_age['age_range'].astype('category')
# gender_age['age_range'] = gender_age['age_range'].map({'18-21':1, '22-24':2, '25-29':3,'30-34':4,'35-39':5, '40-44':6,'45-49':7, '50-54':8,'55-59':9, '60-69':10, '70-79':11, '80+':12  })#.astype(int)

# df_edu_money['salary'] =df_edu_money['salary'].astype('category').cat.codes
df_edu_money['salary'] = df_edu_money['salary'].map({k:v for k, v in salary_scales.items()})

# df_test = pd.DataFrame(df_edu_money.groupby('highest education'))
# df_test = df_edu_money.groupby(['highest education'])['salary'].aggregate()
# df_test
df_money_avg= df_edu_money.groupby('highest education',as_index= False)['salary'].mean().round()
df_money_avg

#For people making between $200,000 and $300,000
df_2k = df_edu_money[df_edu_money['salary'].isin([13,14])]
# df_2k['salary'].value_counts()
df_2k.sample(10)
df_2k_people = pd.DataFrame(df_2k['highest education'].value_counts()).reset_index()
df_2k_people.columns = ['degree', '200-300']
df_2k_people
# For people making between $300,00 and $400,000
df_3k = df_edu_money[df_edu_money['salary']==15]
# df_3k.sample(10)
# df_3k['highest education'].value_counts()
df_3k_people = pd.DataFrame(df_3k['highest education'].value_counts()).reset_index()
df_3k_people.columns = ['degree', '300-400']
df_3k_people
# For people making between $400,000 and $500,000
df_4k = df_edu_money[df_edu_money['salary']==16]
df_4k.sample(10)
df_4k_people = pd.DataFrame(df_4k['highest education'].value_counts()).reset_index()
df_4k_people.columns = ['degree', '400-500']
df_4k_people
# For people making over $500,000+ 
df_5k = df_edu_money[df_edu_money['salary']==17]
df_5k.sample(10)
df_5k_people = pd.DataFrame(df_5k['highest education'].value_counts().reset_index())
df_5k_people.columns =  ['degree', '500+']
df_5k_people
df_rich =pd.merge(pd.merge(pd.merge(df_2k_people, df_3k_people, on ='degree'), df_4k_people, on= 'degree'), df_5k_people, on='degree')
df_rich
plt.figure(figsize=(8,8))
# ax= sns.barplot(df_rich['degree'], df_rich[['200-300','300-400','400-500','500+']])
df_rich.plot(x= 'degree', y= ['200-300','300-400','400-500','500+'], kind ='bar', figsize = (10,5), title= '$200,000 + earners and their degrees')
#  ML Questions: Q9, Q10, Q25

df_ml = df_mcq[['Q9', 'Q10', 'Q25']]
df_ml.columns = ['salary', 'yes/no','years of ml']
df_ml = df_ml.drop(df_ml.index[[0]])
df_ml = df_ml.dropna()
df_ml= df_ml[df_ml['salary'] !='I do not wish to disclose my approximate yearly compensation']
df_ml = df_ml[df_ml['yes/no']!= 'I do not know']
df_ml['yes/no'].unique()
# df_ml
ml_dict = {'No (we do not use ML methods)': 0,
          'We recently started using ML methods (i.e., models in production for less than 2 years)':1,
           'We have well established ML methods (i.e., models in production for more than 2 years)':1,
           'We are exploring ML methods (and may one day put a model into production)':1,
           'We use ML methods for generating insights (but do not put working models into production)':1           
          }
for k,v in ml_dict.items():
    print(k,':',v)
df_ml_salary = df_ml[['salary', 'yes/no']]
# df_edu_money['salary'] = df_edu_money['salary'].map({k:v for k, v in salary_scales.items()})
df_ml_salary['salary'] = df_ml_salary['salary'].map({k:v for k,v in salary_scales.items()})
df_ml_salary['yes/no'] = df_ml_salary['yes/no'].map({k:v for k,v in ml_dict.items()})
df_ml_salary.sample(10)

df_ml_yes = df_ml_salary[df_ml_salary['yes/no']==1]
df_ml_ycount= pd.DataFrame(df_ml_yes['salary'].value_counts().reset_index())
df_ml_ycount.columns= ['salary scale', 'count']
df_ml_y100K = df_ml_ycount[df_ml_ycount['salary scale']>=10]
df_ml_y100K.plot(x='salary scale', y= 'count', kind= 'bar', title = 'Salary scale for Companies using Machine Learning(only people making over $100K+)', figsize= (10,5))

print('At the comapnies using Some level of machine learning',round(df_ml_y100K['count'].sum()/df_ml_ycount['count'].sum()*100), '% of employees make over $100k+ a year')
print('The mean salary scale at companies using machine learnig is:',df_ml_ycount['salary scale'].mean())
print('An 8.5 scale translates to $80,000 to $100,000 per year as a mean salary for the employees')
print(df_ml_ycount['count'].sum(), 'people who responded to this question said they worked for a company where machine learning was involved at some level')
df_ml_no = df_ml_salary[df_ml_salary['yes/no']==0]
df_ml_ncount= pd.DataFrame(df_ml_no['salary'].value_counts().reset_index())
df_ml_ncount.columns= ['salary scale', 'count']
df_ml_n100K = df_ml_ncount[df_ml_ncount['salary scale']>=10]
df_ml_n100K.plot(x='salary scale', y= 'count', kind= 'bar', title = 'Salary scale for Companies not using Machine Learning(only people making over $100K+)', figsize= (10,5))

print('At the comapnies not using any kind machine learning',round(df_ml_n100K['count'].sum()/df_ml_ncount['count'].sum()*100), '% of employees make over $100k+ a year')
print('The mean salary scale at companies not using machine learnig is:',df_ml_ncount['salary scale'].mean())
print('An 8 scale translates to $80,000 to $90,000 per year as a mean salary for the employees')
print(df_ml_ncount['count'].sum(), 'people who responded to this question said they worked for a company where there was no machine learning used')