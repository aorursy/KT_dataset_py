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
Survey=pd.read_csv("../input/SurveySchema.csv")
Survey.head()
FreeResponses=pd.read_csv('../input/freeFormResponses.csv')
FreeResponses.head()
MultipleResponses=pd.read_csv("../input/multipleChoiceResponses.csv")
MultipleResponses.head()
India=MultipleResponses.loc[MultipleResponses['Q3'] == 'India']
India.head()
TotalIndia=len(India)
TotalIndia
MaleIndia=len(India[India['Q1']=='Male'])
MaleIndia
FemaleIndia=len(India[India['Q1']=='Female'])
FemaleIndia
India.Q1.unique()
Prefersay=len(India[India['Q1']=='Prefer not to say'])
Prefersay
Preferself=len(India[India['Q1']=='Prefer to self-describe'])
Preferself
import matplotlib.pyplot as plt 
names = ['Male', 'Female','Not to Say','self-describe']
values = [MaleIndia,FemaleIndia,Prefersay,Preferself]
plt.figure(1, figsize=(19, 7))
plt.subplot(131)
plt.bar(names, values)
plt.xlabel('Category') 
# naming the y axis 
plt.ylabel('Number of people') 
  
# giving a title to my graph 
plt.title('Distribution of Male and Female') 
plt.show()
India.Q4.unique()

Bachelor=India.loc[India['Q4'] == 'Bachelor’s degree']
LBach=len(Bachelor)
LBach
Masters=India.loc[India['Q4'] == 'Master’s degree']
LMast=len(Masters)
LMast
OnlyStudy=India.loc[India['Q4'] == 'Some college/university study without earning a bachelor’s degree']
LOn=len(OnlyStudy)
LOn
Proffesional=India.loc[India['Q4'] == 'Professional degree']
LProff=len(Proffesional)
LProff
Doctoral=India.loc[India['Q4'] == 'Doctoral degree']
LDoct=len(Doctoral)
LDoct
PreferNot=India.loc[India['Q4'] == 'I prefer not to answer']
LPre=len(PreferNot)
LPre
NA=India.Q4.isna().sum()
NA
# x-coordinates of left sides of bars  
left = [1, 2, 3, 4, 5,6,7] 
  
# heights of bars 
height =[LBach, LMast,LOn,LProff,LDoct,LPre,NA]
  
# labels for bars 
tick_label =['Bachelor','Masters','OnlyStudy','Proffesional','Doctoral','PreferNot','NotAvailable']
  
# plotting a bar chart 
plt.bar(left, height, tick_label = tick_label, 
        width = 0.8, color = ['red', 'green','orange','yellow','blue','black','pink']) 
  
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
explode = (0, 0, 0, 0,0,0,0)  
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=30)

plt.axis('equal')
plt.show()


India.Q12_MULTIPLE_CHOICE.unique()
Advance=India.loc[India['Q12_MULTIPLE_CHOICE'] == 'Advanced statistical software (SPSS, SAS, etc.)']
LAdvance=len(Advance)
LAdvance
Local=India.loc[India['Q12_MULTIPLE_CHOICE'] == 'Local or hosted development environments (RStudio, JupyterLab, etc.)']
LLocal=len(Local)
LLocal
Basic=India.loc[India['Q12_MULTIPLE_CHOICE'] == 'Basic statistical software (Microsoft Excel, Google Sheets, etc.)']
LBasic=len(Basic)
LBasic
Bussiness=India.loc[India['Q12_MULTIPLE_CHOICE'] == 'Business intelligence software (Salesforce, Tableau, Spotfire, etc.)']
LBussiness=len(Bussiness)
LBussiness
Cloud=India.loc[India['Q12_MULTIPLE_CHOICE'] == 'Cloud-based data software & APIs (AWS, GCP, Azure, etc.)']
LCloud=len(Cloud)
LCloud
Other=India.loc[India['Q12_MULTIPLE_CHOICE'] == 'Other']
LOther=len(Other)
LOther
NAA=India.Q12_MULTIPLE_CHOICE.isna().sum()
NAA
label = 'Advance','Local','Basic','Bussiness','Cloud','Other','NotAvailable'
s =  [LAdvance, LLocal,LBasic,LBussiness,LCloud,LOther,NAA]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','blue','orange','red']
explode = (0, 0, 0, 0,0,0,0)  
 
# Plot
plt.pie(s, explode=explode, labels=label, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=30)

plt.axis('equal')
plt.show()
India.Q9.unique()
First=India.loc[India['Q9'] == '0-10,000']
LFirst=len(First)
LFirst
Second=India.loc[India['Q9'] == '30-40,000']
LSecond=len(Second)
LSecond
Third=India.loc[India['Q9'] == '20-30,000']
LThird=len(Third)
LThird
Fourth=India.loc[India['Q9']=='10-20,000']
LFourth=len(Fourth)
LFourth
Fifth=India.loc[India['Q9']=='50-60,000']
LFifth=len(Fifth)
LFifth
Sixth=India.loc[India['Q9']=='70-80,000']
LSixth=len(Sixth)
LSixth
Seventh=India.loc[India['Q9']=='40-50,000']
LSeventh=len(Seventh)
LSeventh
Eighth=India.loc[India['Q9']=='150-200,000']
LEighth=len(Eighth)
LEighth

Ninth=India.loc[India['Q9']=='60-70,000']
LNinth=len(Ninth)
LNinth
Tenth=India.loc[India['Q9']=='80-90,000']
LTenth=len(Tenth)
LTenth
Eleventh=India.loc[India['Q9']=='100-125,000']
LEleventh=len(Eleventh)
LEleventh
Twelve=India.loc[India['Q9']=='90-100,000']
LTwelve=len(Twelve)
LTwelve
Thirteen=India.loc[India['Q9']=='125-150,000']
LThirteen=len(Thirteen)
LThirteen
Fourteen=India.loc[India['Q9']=='300-400,000']
LFourteen=len(Fourteen)
LFourteen
Fifteen=India.loc[India['Q9']=='200-250,000']
LFifteen=len(Fifteen)
LFifteen
Sixteen=India.loc[India['Q9']=='500,000+']
LSixteen=len(Sixteen)
LSixteen
Seventeen=India.loc[India['Q9']=='250-300,000']
LSeventeen=len(Seventeen)
LSeventeen
Eighteen=India.loc[India['Q9']=='400-500,000']
LEighteen=len(Eighteen)
LEighteen
names = ['0-10,000','30-40,000', '20-30,000', '10-20,000', '50-60,000', '70-80,000',
       '40-50,000', '150-200,000', '60-70,000', '80-90,000',
       '100-125,000', '90-100,000', '125-150,000', '300-400,000',
       '200-250,000', '500,000+', '250-300,000', '400-500,000']
values = [LFirst,LSecond,LThird,LFourth,LFifth,LSixth,LSeventh,LEighth,LNinth,LTenth,LEleventh,LTwelve,LThirteen,LFourteen,LFifteen,LSixteen,LSeventeen,LEighteen]
plt.figure(1, figsize=(79, 20))
plt.subplot(131)
plt.bar(names, values)
plt.xlabel('Category') 
# naming the y axis 
plt.ylabel('Number of people') 
  
# giving a title to my graph 
plt.title('Distribution of Salaries') 
plt.show()
MaleIndia1=(India[India['Q1']=='Male'])
MaleIndia1.head()
FirstMale=MaleIndia1.loc[MaleIndia1['Q9'] == '0-10,000']
LFirstMale=len(FirstMale)
LFirstMale
SecondMale=MaleIndia1.loc[MaleIndia1['Q9'] == '30-40,000']
LSecondMale=len(SecondMale)
LSecondMale
ThirdMale=MaleIndia1.loc[MaleIndia1['Q9'] == '20-30,000']
LThirdMale=len(ThirdMale)
LThirdMale
FourthMale=MaleIndia1.loc[MaleIndia1['Q9']=='10-20,000']
LFourthMale=len(FourthMale)
LFourthMale
FifthMale=MaleIndia1.loc[MaleIndia1['Q9']=='50-60,000']
LFifthMale=len(FifthMale)
LFifthMale
SixthMale=MaleIndia1.loc[MaleIndia1['Q9']=='70-80,000']
LSixthMale=len(SixthMale)
LSixthMale
SeventhMale=MaleIndia1.loc[MaleIndia1['Q9']=='40-50,000']
LSeventhMale=len(SeventhMale)
LSeventhMale
EighthMale=MaleIndia1.loc[MaleIndia1['Q9']=='150-200,000']
LEighthMale=len(EighthMale)
LEighthMale
NinthMale=MaleIndia1.loc[MaleIndia1['Q9']=='60-70,000']
LNinthMale=len(NinthMale)
LNinthMale
TenthMale=MaleIndia1.loc[MaleIndia1['Q9']=='80-90,000']
LTenthMale=len(TenthMale)
LTenthMale
EleventhMale=MaleIndia1.loc[MaleIndia1['Q9']=='100-125,000']
LEleventhMale=len(EleventhMale)
LEleventhMale
TwelveMale=MaleIndia1.loc[MaleIndia1['Q9']=='90-100,000']
LTwelveMale=len(TwelveMale)
LTwelveMale
ThirteenMale=MaleIndia1.loc[MaleIndia1['Q9']=='125-150,000']
LThirteenMale=len(ThirteenMale)
LThirteenMale
FourteenMale=MaleIndia1.loc[MaleIndia1['Q9']=='300-400,000']
LFourteenMale=len(FourteenMale)
LFourteenMale
FifteenMale=MaleIndia1.loc[MaleIndia1['Q9']=='200-250,000']
LFifteenMale=len(FifteenMale)
LFifteenMale
SixteenMale=MaleIndia1.loc[MaleIndia1['Q9']=='500,000+']
LSixteenMale=len(SixteenMale)
LSixteenMale
SeventeenMale=MaleIndia1.loc[MaleIndia1['Q9']=='250-300,000']
LSeventeenMale=len(SeventeenMale)
LSeventeenMale
EighteenMale=MaleIndia1.loc[MaleIndia1['Q9']=='400-500,000']
LEighteenMale=len(EighteenMale)
LEighteenMale
names = ['0-10,000','30-40,000', '20-30,000', '10-20,000', '50-60,000', '70-80,000',
       '40-50,000', '150-200,000', '60-70,000', '80-90,000',
       '100-125,000', '90-100,000', '125-150,000', '300-400,000',
       '200-250,000', '500,000+', '250-300,000', '400-500,000']
values = [LFirstMale,LSecondMale,LThirdMale,LFourthMale,LFifthMale,LSixthMale,LSeventhMale,LEighthMale,LNinthMale,LTenthMale,LEleventhMale,LTwelveMale,LThirteenMale,LFourteenMale,LFifteenMale,LSixteenMale,LSeventeenMale,LEighteenMale]
plt.figure(1, figsize=(79, 20))
plt.subplot(131)
plt.bar(names, values)
plt.xlabel('Category') 
# naming the y axis 
plt.ylabel('Number of Male') 
  
# giving a title to my graph 
plt.title('Distribution of Salaries') 
plt.show()
FemaleIndia1=(India[India['Q1']=='Female'])
FemaleIndia1.head()
FirstFemale=FemaleIndia1.loc[FemaleIndia1['Q9'] == '0-10,000']
LFirstFemale=len(FirstFemale)
LFirstFemale
SecondFemale=FemaleIndia1.loc[FemaleIndia1['Q9'] == '30-40,000']
LSecondFemale=len(SecondFemale)
LSecondFemale
ThirdFemale=FemaleIndia1.loc[FemaleIndia1['Q9'] == '20-30,000']
LThirdFemale=len(ThirdFemale)
LThirdFemale
FourthFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='10-20,000']
LFourthFemale=len(FourthFemale)
LFourthFemale
FifthFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='50-60,000']
LFifthFemale=len(FifthFemale)
LFifthFemale
SixthFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='70-80,000']
LSixthFemale=len(SixthFemale)
LSixthFemale
SeventhFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='40-50,000']
LSeventhFemale=len(SeventhFemale)
LSeventhFemale
EighthFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='150-200,000']
LEighthFemale=len(EighthFemale)
LEighthFemale
NinthFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='60-70,000']
LNinthFemale=len(NinthFemale)
LNinthFemale
TenthFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='80-90,000']
LTenthFemale=len(TenthFemale)
LTenthFemale
EleventhFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='100-125,000']
LEleventhFemale=len(EleventhFemale)
LEleventhFemale
TwelveFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='90-100,000']
LTwelveFemale=len(TwelveFemale)
LTwelveFemale
ThirteenFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='125-150,000']
LThirteenFemale=len(ThirteenFemale)
LThirteenFemale
FourteenFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='300-400,000']
LFourteenFemale=len(FourteenFemale)
LFourteenFemale
FifteenFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='200-250,000']
LFifteenFemale=len(FifteenFemale)
LFifteenFemale
SixteenFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='500,000+']
LSixteenFemale=len(SixteenFemale)
LSixteenFemale
SeventeenFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='250-300,000']
LSeventeenFemale=len(SeventeenFemale)
LSeventeenFemale
EighteenFemale=FemaleIndia1.loc[FemaleIndia1['Q9']=='400-500,000']
LEighteenFemale=len(EighteenFemale)
LEighteenFemale
n = ['0-10,000','30-40,000', '20-30,000', '10-20,000', '50-60,000', '70-80,000',
       '40-50,000', '150-200,000', '60-70,000', '80-90,000',
       '100-125,000', '90-100,000', '125-150,000', '300-400,000',
       '200-250,000', '500,000+', '250-300,000', '400-500,000']
v = [LFirstFemale,LSecondFemale,LThirdFemale,LFourthFemale,LFifthFemale,LSixthFemale,LSeventhFemale,LEighthFemale,LNinthFemale,LTenthFemale,LEleventhFemale,LTwelveFemale,LThirteenFemale,LFourteenFemale,LFifteenFemale,LSixteenFemale,LSeventeenFemale,LEighteenFemale]
plt.figure(1, figsize=(79, 20))
plt.subplot(131)
plt.bar(n, v)
plt.xlabel('Category') 
# naming the y axis 
plt.ylabel('Number of Female') 
  
# giving a title to my graph 
plt.title('Distribution of Salaries') 
plt.show()
