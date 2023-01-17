# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import os
nyc_df=pd.read_csv("../input/2016 School Explorer.csv")

nyc_df.head()
def p2f(x):

    return float(x.strip('%'))/100



nyc_df['Percent of Students Chronically Absent']=nyc_df['Percent of Students Chronically Absent'].astype(str).apply(p2f)

nyc_df['Rigorous Instruction %'] = nyc_df['Rigorous Instruction %'].astype(str).apply(p2f)

nyc_df['Collaborative Teachers %'] = nyc_df['Collaborative Teachers %'].astype(str).apply(p2f)

nyc_df['Supportive Environment %'] = nyc_df['Supportive Environment %'].astype(str).apply(p2f)

nyc_df['Effective School Leadership %'] = nyc_df['Effective School Leadership %'].astype(str).apply(p2f)

nyc_df['Strong Family-Community Ties %'] = nyc_df['Strong Family-Community Ties %'].astype(str).apply(p2f)

nyc_df['Trust %'] = nyc_df['Trust %'].astype(str).apply(p2f)

nyc_df['Student Attendance Rate'] = nyc_df['Student Attendance Rate'].astype(str).apply(p2f)
nyc_df['Percent White'] = nyc_df['Percent White'].astype(str).apply(p2f)

nyc_df['Percent Asian'] = nyc_df['Percent Asian'].astype(str).apply(p2f)

nyc_df['Percent Black'] = nyc_df['Percent Black'].astype(str).apply(p2f)

nyc_df['Percent Black / Hispanic'] = nyc_df['Percent Black / Hispanic'].astype(str).apply(p2f)



nyc_df[["Percent White","Percent Asian","Percent Black","Percent Black / Hispanic"]].plot(kind="hist",stacked=True)
#The below scatter plot show the relationship between the percentage of black\hispanic population in schools with the economic need index of respective schools

#There is a positive correlation between the two parameters

#Schools having high black/hispanic population have a higher economic need index



import matplotlib.pyplot as plt

plt.scatter(x = 'Economic Need Index', y = 'Percent Black / Hispanic', data = nyc_df, c = 'g')

plt.grid()

plt.legend()

plt.xlabel('Economic Need Index')

plt.ylabel('Percent Black / Hispanic')

plt.title('Percent Black / Hispanic vs Economic Need Index')

plt.show()
#The below scatter plot show the relationship between the percentage of white population in schools with the economic need index of respective schools

#There is a negative correlation between the two parameters

#Schools having high white population have a lower economic need index



plt.scatter(x = 'Economic Need Index', y = 'Percent White', data = nyc_df, c = 'g')

plt.grid()

plt.legend()

plt.xlabel('Economic Need Index')

plt.ylabel('Percent White')

plt.title('Percent White vs Economic Need Index')

plt.show()
#The below scatter plot show the relationship between the percentage of asian population in schools with the economic need index of respective schools

#There is a negative correlation between the two parameters

#Schools having high white population have a lower economic need index



plt.scatter(x = 'Economic Need Index', y = 'Percent Asian', data = nyc_df, c = 'g')

plt.grid()

plt.legend()

plt.xlabel('Economic Need Index')

plt.ylabel('Percent Asian')

plt.title('Percent Asian vs Economic Need Index')

plt.show()
#Scatter plot to understand the relationship between %black/hispanic population and Avg ELA Proficiency

#The below scatter plot indicates a negative correlation

#Schools having a higher hispanic population have lower Avg ELA Proficiency



plt.scatter(x = 'Percent Black / Hispanic', y = 'Average ELA Proficiency', data = nyc_df, c = 'g')

plt.grid()

plt.legend()

plt.xlabel('Percent Black / Hispanic')

plt.ylabel('Average ELA Proficiency')

plt.title('Percent Black / Hispanic vs Average ELA Proficiency')

plt.show()
#The below scatter plot shows a positive relationship between the two parameters

#The average ELA and Math Proficiency go hand in hand i.e. higher the ELA Proficiency higher is the Math Proficiency



plt.scatter(x = 'Average ELA Proficiency', y = 'Average Math Proficiency', data = nyc_df, c = 'g')

plt.grid()

plt.legend()

plt.xlabel('Average ELA Proficiency')

plt.ylabel('Average Math Proficiency')

plt.title('Average ELA Proficiency vs Average Math Proficiency')

plt.show()
#Correlation Between Various Features of schools

# 3 Features namely - Collaborative Teachers %,Effective School Leadership %,Trust % have high correlation

features_list = ['Rigorous Instruction %',

'Collaborative Teachers %',

'Supportive Environment %',

'Effective School Leadership %',

'Strong Family-Community Ties %',

'Trust %']



nyc_df[features_list].corr()
#Summary of Population and Income between Community School and Not Community Schools

#Community Schools have a higher population of black/hispanic students(93%) and high Economic Need Index



community = nyc_df.groupby('Community School?')

community[['Economic Need Index', 'School Income Estimate', 'Percent Asian','Percent Black / Hispanic','Percent White', 'Average ELA Proficiency', 'Average Math Proficiency']].mean()
#Distribution of Asians in New York City

nyc_df["Percent Asian"].value_counts()

nyc_df["Percent Asian"].sort_values().unique()



nyc_df['Percent Asian'].dtype

school= nyc_df.loc[nyc_df['Community School?']=='Yes']

school['Percent Asian'].dtype



import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



sns.distplot(school['Percent Asian'])

plt.show()   
#The below scatter plot shows the relationship between the percentage of students chronically absent in schools and their Supportive Environment %

#There is a negative correlation between the two parameters with the value -0.191

#This indicates schools with a supportive environment have less absenteeism

#These schools have a supportive environment rating of either 'Meeting Targets' or 'Exceeding Targets'



plt.scatter(x = 'Supportive Environment %', y = 'Percent of Students Chronically Absent', data = nyc_df, c = 'g')

plt.grid()

plt.legend()

plt.xlabel('Supportive Environment %')

plt.ylabel('Percent of Students Chronically Absent')

plt.title('Supportive Environment % vs % Students Chronically Absent')

plt.show()

nyc_df['Supportive Environment %'].corr(nyc_df['Percent of Students Chronically Absent'])

#Percent Black / Hispanic and Supportive Environment % show a negative correlation

#This indicates schools with higher black/hispanic population have less supportive environment for students

#This indicates schools with  higher black/hispanic population lack a supportive environment and hence leads to higher number of students being absent regularly

features_list1 = ['Percent of Students Chronically Absent',

'Percent Black / Hispanic',

'Supportive Environment %']

nyc_df[features_list1].corr()
#Number of American Indian students in Each grade scoring 4s in ELA

AmericanIndianELA=nyc_df[['Grade 3 ELA 4s - American Indian or Alaska Native','Grade 4 ELA 4s - American Indian or Alaska Native',

       'Grade 5 ELA 4s - American Indian or Alaska Native','Grade 6 ELA 4s - American Indian or Alaska Native',

       'Grade 7 ELA 4s - American Indian or Alaska Native']].T 

AmericanIndianELA['Total'] = AmericanIndianELA.sum(axis=1) 



df = pd.DataFrame({'Grades':['Grade 3 ELA 4s - American Indian or Alaska Native','Grade 4 ELA 4s - American Indian or Alaska Native',

       'Grade 5 ELA 4s - American Indian or Alaska Native','Grade 6 ELA 4s - American Indian or Alaska Native',

       'Grade 7 ELA 4s - American Indian or Alaska Native'], 'Number of students':AmericanIndianELA['Total']})

ax = df.plot.barh(x='Grades', y='Number of students', rot=0)

plt.show()
#Number of Black of African American students in Each grade scoring 4s in ELA

BlackELA=nyc_df[['Grade 3 ELA 4s - Black or African American','Grade 4 ELA 4s - Black or African American',

       'Grade 5 ELA 4s - Black or African American','Grade 6 ELA 4s - Black or African American',

       'Grade 7 ELA 4s - Black or African American']].T 

BlackELA['Total'] = BlackELA.sum(axis=1) 



df = pd.DataFrame({'Grades':['Grade 3 ELA 4s - Black or African American','Grade 4 ELA 4s - Black or African American',

       'Grade 5 ELA 4s - Black or African American','Grade 6 ELA 4s - Black or African American',

       'Grade 7 ELA 4s - Black or African American'], 'Number of students':BlackELA['Total']})

ax = df.plot.barh(x='Grades', y='Number of students', rot=0)

plt.show()
#Number of Hispanic or Latino students in Each grade scoring 4s in ELA

HispanicELA=nyc_df[['Grade 3 ELA 4s - Hispanic or Latino','Grade 4 ELA 4s - Hispanic or Latino',

       'Grade 5 ELA 4s - Hispanic or Latino','Grade 6 ELA 4s - Hispanic or Latino',

       'Grade 7 ELA 4s - Hispanic or Latino']].T 

HispanicELA['Total'] = HispanicELA.sum(axis=1) 



df = pd.DataFrame({'Grades':['Grade 3 ELA 4s - Hispanic or Latino','Grade 4 ELA 4s - Hispanic or Latino',

       'Grade 5 ELA 4s - Hispanic or Latino','Grade 6 ELA 4s - Hispanic or Latino',

       'Grade 7 ELA 4s - Hispanic or Latino'], 'Number of students':HispanicELA['Total']})

ax = df.plot.barh(x='Grades', y='Number of students', rot=0)

plt.show()
AsianELA=nyc_df[['Grade 3 ELA 4s - Asian or Pacific Islander','Grade 4 ELA 4s - Asian or Pacific Islander',

       'Grade 5 ELA 4s - Asian or Pacific Islander','Grade 6 ELA 4s - Asian or Pacific Islander',

       'Grade 7 ELA 4s - Asian or Pacific Islander']].T 

AsianELA['Total'] = AsianELA.sum(axis=1) 



df = pd.DataFrame({'Grades':['Grade 3 ELA 4s - Asian or Pacific Islander','Grade 4 ELA 4s - Asian or Pacific Islander',

       'Grade 5 ELA 4s - Asian or Pacific Islander','Grade 6 ELA 4s - Asian or Pacific Islander',

       'Grade 7 ELA 4s - Asian or Pacific Islander'], 'Number of students':AsianELA['Total']})

ax = df.plot.barh(x='Grades', y='Number of students', rot=0)

plt.show()
#Number of Economically Disadvantaged students in Each grade scoring 4s in ELA

DisadvELA=nyc_df[['Grade 3 ELA 4s - Economically Disadvantaged','Grade 4 ELA 4s - Economically Disadvantaged',

       'Grade 5 ELA 4s - Economically Disadvantaged','Grade 6 ELA 4s - Economically Disadvantaged',

       'Grade 7 ELA 4s - Economically Disadvantaged']].T 

DisadvELA['Total'] = DisadvELA.sum(axis=1) 



df = pd.DataFrame({'Grades':['Grade 3 ELA 4s - Economically Disadvantaged','Grade 4 ELA 4s - Economically Disadvantaged',

       'Grade 5 ELA 4s - Economically Disadvantaged','Grade 6 ELA 4s - Economically Disadvantaged',

       'Grade 7 ELA 4s - Economically Disadvantaged'], 'Number of students':DisadvELA['Total']})

ax = df.plot.barh(x='Grades', y='Number of students', rot=0)

plt.show()
#Number of White students in Each grade scoring 4s in ELA

WhiteELA=nyc_df[['Grade 3 ELA 4s - White','Grade 4 ELA 4s - White',

       'Grade 5 ELA 4s - White','Grade 6 ELA 4s - White',

       'Grade 7 ELA 4s - White']].T 

WhiteELA['Total'] =WhiteELA.sum(axis=1) 



df = pd.DataFrame({'Grades':['Grade 3 ELA 4s - White','Grade 4 ELA 4s - White',

       'Grade 5 ELA 4s - White','Grade 6 ELA 4s - White',

       'Grade 7 ELA 4s - White'], 'Number of students':WhiteELA['Total']})

ax = df.plot.barh(x='Grades', y='Number of students', rot=0)

plt.show()
#Number of American Indian students in Each grade scoring 4s in Math

AmericanIndianMath=nyc_df[['Grade 3 Math 4s - American Indian or Alaska Native','Grade 4 Math 4s - American Indian or Alaska Native',

       'Grade 5 Math 4s - American Indian or Alaska Native','Grade 6 Math 4s - American Indian or Alaska Native',

       'Grade 7 Math 4s - American Indian or Alaska Native']].T 

AmericanIndianMath['Total'] =AmericanIndianMath.sum(axis=1) 



df = pd.DataFrame({'Grades':['Grade 3 Math 4s - American Indian or Alaska Native','Grade 4 Math 4s - American Indian or Alaska Native',

       'Grade 5 Math 4s - American Indian or Alaska Native','Grade 6 Math 4s - American Indian or Alaska Native',

       'Grade 7 Math 4s - American Indian or Alaska Native'], 'Number of students':AmericanIndianMath['Total']})

ax = df.plot.barh(x='Grades', y='Number of students', rot=0)

plt.show()
#Number of Black or African American students in Each grade scoring 4s in Math

BlackMath=nyc_df[['Grade 3 Math 4s - Black or African American','Grade 4 Math 4s - Black or African American',

       'Grade 5 Math 4s - Black or African American','Grade 6 Math 4s - Black or African American',

       'Grade 7 Math 4s - Black or African American']].T 

BlackMath['Total'] = BlackMath.sum(axis=1) 



df = pd.DataFrame({'Grades':['Grade 3 Math 4s - Black or African American','Grade 4 Math 4s - Black or African American',

       'Grade 5 Math 4s - Black or African American','Grade 6 Math 4s - Black or African American',

       'Grade 7 Math 4s - Black or African American'], 'Number of students':BlackMath['Total']})

ax = df.plot.barh(x='Grades', y='Number of students', rot=0)

plt.show()
#Number of Economically Disadvantaged students in Each grade scoring 4s in Math

DisadvMath=nyc_df[['Grade 3 Math 4s - Economically Disadvantaged','Grade 4 Math 4s - Economically Disadvantaged',

       'Grade 5 Math 4s - Economically Disadvantaged','Grade 6 Math 4s - Economically Disadvantaged',

       'Grade 7 Math 4s - Economically Disadvantaged']].T 

DisadvMath['Total'] = DisadvMath.sum(axis=1) 



df = pd.DataFrame({'Grades':['Grade 3 Math 4s - Economically Disadvantaged','Grade 4 Math 4s - Economically Disadvantaged',

       'Grade 5 Math 4s - Economically Disadvantaged','Grade 6 Math 4s - Economically Disadvantaged',

       'Grade 7 Math 4s - Economically Disadvantaged'], 'Number of students':DisadvMath['Total']})

ax = df.plot.barh(x='Grades', y='Number of students', rot=0)

plt.show()
#Number of White students in Each grade scoring 4s in Math

WhiteMath=nyc_df[['Grade 3 Math 4s - White','Grade 4 Math 4s - White',

       'Grade 5 Math 4s - White','Grade 6 Math 4s - White',

       'Grade 7 Math 4s - White']].T 

WhiteMath['Total'] = WhiteMath.sum(axis=1) 



df = pd.DataFrame({'Grades':['Grade 3 Math 4s - White','Grade 4 Math 4s - White',

       'Grade 5 Math 4s - White','Grade 6 Math 4s - White',

       'Grade 7 Math 4s - White'], 'Number of students':WhiteMath['Total']})

ax = df.plot.barh(x='Grades', y='Number of students', rot=0)

plt.show()
#From the above plots and figures, we observe that the schools with high Economic Need Index have a high percentage of black/ hispanic students and a lower school income.

#Their performance is much lower compared to low Economic Need Index schools.

#This could be because of the lack of financial/ educational resources among the students.

#Schools with higher black/hispanic population also have low Average Math and ELA Proficiency 

#Organisations providing study materials, after school guidance/ mentoring could help boost the performance of the students from these schools.

#Schools with higher black/hispanic population should focus on student counselling as this could help understand the reason behind absenteeism 