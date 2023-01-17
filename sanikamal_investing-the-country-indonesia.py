# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
survey=pd.read_csv("../input/SurveySchema.csv")
survey.head()
survey.tail()
free_responses=pd.read_csv('../input/freeFormResponses.csv',low_memory=False)
free_responses.head()
free_responses.tail()
multiple_responses=pd.read_csv("../input/multipleChoiceResponses.csv",low_memory=False)
multiple_responses.head()
multiple_responses.tail()
Indonesia=multiple_responses.loc[multiple_responses['Q3'] == 'Indonesia']
Indonesia.head()
Indonesia.tail()
TotalIndonesia=len(Indonesia)
TotalIndonesia
MaleIndonesia=len(Indonesia[Indonesia['Q1']=='Male'])
MaleIndonesia
FemaleIndonesia=len(Indonesia[Indonesia['Q1']=='Female'])
FemaleIndonesia
Indonesia.Q1.unique()
Prefersay=len(Indonesia[Indonesia['Q1']=='Prefer not to say'])
Prefersay
Preferself=len(Indonesia[Indonesia['Q1']=='Prefer to self-describe'])
Preferself
names = ['Male', 'Female','Not to Say','self-describe']
values = [MaleIndonesia,FemaleIndonesia,Prefersay,Preferself]
plt.figure(1, figsize=(30,5))
plt.subplot(131)
plt.bar(names, values,color="rgb")
plt.xlabel('Category') 
# naming the y axis 
plt.ylabel('Number of people') 
  
# giving a title to my graph 
plt.title('Distribution of Male and Female') 
plt.show()
Indonesia.Q4.unique()
Masters=Indonesia.loc[Indonesia['Q4'] == 'Master’s degree']
LMast=len(Masters)
LMast
Bachelor=Indonesia.loc[Indonesia['Q4'] == 'Bachelor’s degree']
LBach=len(Bachelor)
LBach
OnlyStudy=Indonesia.loc[Indonesia['Q4'] == 'Some college/university study without earning a bachelor’s degree']
LOn=len(OnlyStudy)
LOn
Proffesional=Indonesia.loc[Indonesia['Q4'] == 'Professional degree']
LProff=len(Proffesional)
LProff
Doctoral=Indonesia.loc[Indonesia['Q4'] == 'Doctoral degree']
LDoct=len(Doctoral)
LDoct
PreferNot=Indonesia.loc[Indonesia['Q4'] == 'I prefer not to answer']
LPre=len(PreferNot)
LPre
NA=Indonesia.Q4.isna().sum()
NA
# labels for bars 
names =['Bachelor','Masters','OnlyStudy','Proffesional','Doctoral','PreferNot','NotAvailable']
# heights of bars 
values =[LBach, LMast,LOn,LProff,LDoct,LPre,NA]
plt.figure(1, figsize=(30,5))
plt.subplot(131)
plt.bar(names, values,color = ['red', 'green','orange','yellow','blue','black','pink'])  
# naming the x-axis 
plt.xlabel('No. of people') 
# naming the y-axis 
plt.ylabel('Degree') 
# plot title 
plt.title('Distribution of degree') 
# function to show the plot 
plt.show() 
labels = 'Bachelor','Masters','OnlyStudy','Proffesional','Doctoral','PreferNot','NotAvailable'
sizes =  [LBach, LMast,LOn,LProff,LDoct,LPre,NA]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','blue','orange','red']
explode = (0, .1,0, 0,0,0,0)  
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=30)

plt.axis('equal')
plt.show()
Indonesia.Q12_MULTIPLE_CHOICE.unique()
Basic=Indonesia.loc[Indonesia['Q12_MULTIPLE_CHOICE'] == 'Basic statistical software (Microsoft Excel, Google Sheets, etc.)']
LBasic=len(Basic)
LBasic
Local=Indonesia.loc[Indonesia['Q12_MULTIPLE_CHOICE'] == 'Local or hosted development environments (RStudio, JupyterLab, etc.)']
LLocal=len(Local)
LLocal
Other=Indonesia.loc[Indonesia['Q12_MULTIPLE_CHOICE'] == 'Other']
LOther=len(Other)
LOther
Cloud=Indonesia.loc[Indonesia['Q12_MULTIPLE_CHOICE'] == 'Cloud-based data software & APIs (AWS, GCP, Azure, etc.)']
LCloud=len(Cloud)
LCloud
Bussiness=Indonesia.loc[Indonesia['Q12_MULTIPLE_CHOICE'] == 'Business intelligence software (Salesforce, Tableau, Spotfire, etc.)']
LBussiness=len(Bussiness)
LBussiness
Advance=Indonesia.loc[Indonesia['Q12_MULTIPLE_CHOICE'] == 'Advanced statistical software (SPSS, SAS, etc.)']
LAdvance=len(Advance)
LAdvance
NAA=Indonesia.Q12_MULTIPLE_CHOICE.isna().sum()
NAA
# labels for bars 
label = 'Advance','Local','Basic','Bussiness','Cloud','Other','NotAvailable'
# heights of bars 
values =[LAdvance, LLocal,LBasic,LBussiness,LCloud,LOther,NAA]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','blue','orange','red']
plt.figure(1, figsize=(30,5))
plt.subplot(131)
plt.bar(label, values,color =colors)  
# naming the x-axis 
plt.xlabel('No. of tools') 
# naming the y-axis 
plt.ylabel('Tools') 
# plot title 
plt.title('Distribution of tools') 
# function to show the plot 
plt.show() 
# Pie Plot
explode = (0, 0, 0, 0,0,0,1)
plt.pie(values, explode=explode, labels=label, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=30)

plt.axis('equal')
plt.show()
Indonesia.Q9.unique()
First=Indonesia.loc[Indonesia['Q9'] == '0-10,000']
LFirst=len(First)
LFirst
Second=Indonesia.loc[Indonesia['Q9'] == '10-20,000']
LSecond=len(Second)
LSecond
Third=Indonesia.loc[Indonesia['Q9'] == '30-40,000']
LThird=len(Third)
LThird
Fourth=Indonesia.loc[Indonesia['Q9']=='40-50,000']
LFourth=len(Fourth)
LFourth
Fifth=Indonesia.loc[Indonesia['Q9']=='50-60,000']
LFifth=len(Fifth)
LFifth
Sixth=Indonesia.loc[Indonesia['Q9']=='60-70,000']
LSixth=len(Sixth)
LSixth
Seventh=Indonesia.loc[Indonesia['Q9']=='90-100,000']
LSeventh=len(Seventh)
LSeventh
Eighth=Indonesia.loc[Indonesia['Q9']=='125-150,000']
LEighth=len(Eighth)
LEighth
Ninth=Indonesia.loc[Indonesia['Q9']=='500,000+']
LNinth=len(Ninth)
LNinth
Tenth=Indonesia.loc[Indonesia['Q9']=='I do not wish to disclose my approximate yearly compensation']
LTenth=len(Tenth)
LTenth
NAA=Indonesia.Q9.isna().sum()
NAA
names = ['0-10,000','10-20,000','30-40,000','40-50,000','50-60,000','60-70,000','90-100,000',
         '125-150,000','500,000+','not disclose','nan']
values = [LFirst,LSecond,LThird,LFourth,LFifth,LSixth,LSeventh,LEighth,LNinth,LTenth,NAA]
plt.figure(1, figsize=(50, 10))
plt.subplot(131)
plt.bar(names, values)
plt.xlabel('Category') 
# naming the y axis 
plt.ylabel('Number of people') 
  
# giving a title to my graph 
plt.title('Distribution of Salaries') 
plt.show()
# Pie Plot
explode = (.2, .3, 0, 0,0,0,0,0,0,0,0)
plt.pie(values, explode=explode, labels=names,autopct='%1.1f%%', shadow=True)

plt.axis('equal')
plt.show()