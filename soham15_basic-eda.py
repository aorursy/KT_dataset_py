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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import scipy.stats as stats
#import data

school=pd.read_csv("../input/2016 School Explorer.csv")

school.head()

school.isna().any()
#Dropping first three columns as they contain very few values

school.drop(["Adjusted Grade","New?","Other Location Code in LCGMS"],axis=1,inplace=True)

school_race=school.iloc[:,[0,12,16,17,18,20]]

school_race.head()
#Checking for na values

school_race.isna().any()
#Changing column values to numeric

school_race['Percent Asian'] = school_race['Percent Asian'].str.replace('%', '')

school_race['Percent Black'] = school_race['Percent Black'].str.replace('%', '')

school_race['Percent Hispanic'] = school_race['Percent Hispanic'].str.replace('%', '')

school_race['Percent White'] = school_race['Percent White'].str.replace('%', '')

school_race['Percent Asian'] = pd.to_numeric(school_race['Percent Asian'])

school_race['Percent Black'] = pd.to_numeric(school_race['Percent Black'])

school_race['Percent Hispanic'] = pd.to_numeric(school_race['Percent Hispanic'])

school_race['Percent White'] = pd.to_numeric(school_race['Percent White'])

school_race.head()

#Recoding column Community School

school_race.loc[school_race["Community School?"]=="Yes","Community School?"]="Community School"

school_race.loc[school_race["Community School?"]=="No","Community School?"]="Non community school"

school_race.head(3)
#visualizing distribution of Asian students according to community and non-community schools

sns.distplot(school_race['Percent Asian'],norm_hist=True,bins=5)

plt.suptitle('Distribution of Asian students')
#Distribution of asian students in community school

ascom=school_race.loc[school_race["Community School?"]=="Community School",:]

#ascom.head(10)

sns.distplot(ascom['Percent Asian'],bins=5)

plt.suptitle('Distribution of Asian students in Community Colleges')
#Distribution of asian students in non-community school

asncom=school_race.loc[school_race["Community School?"]=="Non community school",:]

#asncom.head(10)

sns.distplot(asncom['Percent Asian'],bins=5)

plt.suptitle('Distribution of Asian students in Non-Community Colleges')
#visualizing distribution of black students according to community and non-community schools

sns.distplot(school_race['Percent Black'],kde=False,bins=5)

plt.suptitle('Distribution of Black students')
#Distribution of black students in community school

sns.distplot(ascom['Percent Black'],kde=False,bins=5)

plt.suptitle('Distribution of Black students in Community Colleges')
#Distribution of black students in non-community school

sns.distplot(asncom['Percent Black'],kde=False,bins=5)

plt.suptitle('Distribution of Black students in non-community Colleges')
#visualizing distribution of Hispanic students according to community and non-community schools

sns.distplot(school_race['Percent Hispanic'],kde=False,bins=5)

plt.suptitle('Distribution of Hispanic students')
#Distribution of hispanic students in community school

sns.distplot(ascom['Percent Hispanic'],kde=False,bins=5)

plt.suptitle('Distribution of Hispanic students in Community Colleges')
#Distribution of non hispanic students in community school

sns.distplot(asncom['Percent Hispanic'],kde=False,bins=5)

plt.suptitle('Distribution of Hispanic students in Non Community Colleges')
#Creating dataframe for attendance and Supportive Environment rating

attendance=school.iloc[:,[0,22,28]]

attendance.head()

len(attendance)
#checking for null values

attendance.isna().any()

attendance["Percent of Students Chronically Absent"].isnull().sum()

attendance["Supportive Environment Rating"].isnull().sum()

#25 and 80 na values out of 1272 values, therfore deleting the rows
#dropping na

attendance=attendance.dropna()

attendance.isna().any()

len(attendance)

attendance.head(3)
#Removing % from absent column

attendance["Percent of Students Chronically Absent"]=attendance["Percent of Students Chronically Absent"].str.replace('%','')
#Converting to numeric

attendance["Percent of Students Chronically Absent"]=pd.to_numeric(attendance["Percent of Students Chronically Absent"])

attendance.head(3)
#bar plot

#bar plot shows average value

plt.figure(figsize=(10,7))

sns.barplot(x="Supportive Environment Rating",y="Percent of Students Chronically Absent",data=attendance)

#plt.xlabel("RATING CATEGORY")

#plt.ylabel("Mean percentage of students being absent")

plt.suptitle("Association between students being absent and rating category",fontsize=16)
#Diving into four data-frames to apply One-way ANOVA test

attendance.head()

approaching=attendance.loc[attendance["Supportive Environment Rating"]=="Approaching Target","Percent of Students Chronically Absent"]

meeting=attendance.loc[attendance["Supportive Environment Rating"]=="Meeting Target","Percent of Students Chronically Absent"]

                       

not_meeting=attendance.loc[attendance["Supportive Environment Rating"]=="Not Meeting Target","Percent of Students Chronically Absent"]

exceeding=attendance.loc[attendance["Supportive Environment Rating"]=="Exceeding Target","Percent of Students Chronically Absent"]



#One-Way ANOVA

result_anova=stats.f_oneway(approaching,meeting,not_meeting,exceeding)

print(result_anova)
#Creating dataframe for student achievement and Effective School Leadership Rating

achievement=school.iloc[:,[0,29,30,35]]

achievement.head()

#len(attendance)
#checking for NA values

achievement.isna().any()

achievement["Effective School Leadership %"].isnull().sum(),achievement["Effective School Leadership Rating"].isnull().sum(),achievement["Student Achievement Rating"].isnull().sum()
#Dropping NA values

achievement=achievement.dropna()

achievement.head(3)
#Changing percentage column to numeric

achievement["Effective School Leadership %"]=achievement["Effective School Leadership %"].str.replace('%','')

achievement["Effective School Leadership %"]=pd.to_numeric(achievement["Effective School Leadership %"])

achievement.head(3)
#bar plot

plt.figure(figsize=(10,7))

sns.barplot(x="Student Achievement Rating",y="Effective School Leadership %",data=achievement)

#plt.xlabel("RATING CATEGORY")

#plt.ylabel("Mean percentage of students being absent")

plt.suptitle("Association between student achievement rating and Effectiive school leadership",fontsize=16)
#Scatter plot to check performance between community and non-community schools

sns.scatterplot(x="Average ELA Proficiency",y="Average Math Proficiency",hue="Community School?",data=school)