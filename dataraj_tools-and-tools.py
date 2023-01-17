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

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as sts
path2019 = "/kaggle/input/kaggle-survey-2019/"

mulcho19 = path2019 + "multiple_choice_responses.csv"

survey2019 = pd.read_csv(mulcho19,

                         skiprows=0,

                         header=1

                        )
age19 = survey2019.iloc[:,1]

gender19 = survey2019.iloc[:,2]

country19 = survey2019.iloc[:,4]

education19 = survey2019.iloc[:,5]

jobrole19 = survey2019.iloc[:,6]

df19 = pd.DataFrame({

    "Age" : age19,

    "Gender" : gender19,

    "Country" : country19,

    "Education" : education19,

    "Jobrole" : jobrole19

})
g = sns.catplot(y="Age",

                 data=df19.sort_values("Age"), kind="count",

                height=4, aspect=2,

                edgecolor=sns.color_palette("dark", 3));

plt.title(" Age group wise participants frequency in Survey");

plt.figtext(0.5, 1.1, "Figure 1 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
g = sns.catplot(y="Gender",

                 data=df19.sort_values("Age"), kind="count",

                height=4, aspect=2,

                edgecolor=sns.color_palette("dark", 3));

plt.title("Figure : Genderwise participants frequency");

plt.figtext(0.5, 1.1, "Figure 2 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
temp = df19.Country.value_counts().sort_values(ascending=False)[0:10]

top10Country = temp.reset_index()

top10Country.columns = ["Country","NumberOfParticipants"]

top10Country
g = sns.catplot(y="NumberOfParticipants", x= "Country",

                 data=top10Country, kind="bar",

                height=5, aspect=2,

                edgecolor=sns.color_palette("dark", 3));

plt.title("Participants gender count");

g.set_xticklabels(rotation=90);

plt.figtext(0.5, 1.1, "Figure 3 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
g = sns.catplot(y="Education",

                 data=df19, kind="count",

                height=5, aspect=2,

                edgecolor=sns.color_palette("dark", 3));

plt.title("Education wise participants frequency");

g.set_xticklabels(rotation=90);

plt.figtext(0.5, 1.1, "Figure 4 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
# Which of the following integrated development environments (Done)

jupyter19 = survey2019.iloc[:,56].notnull().astype('int')

rstudio19 = survey2019.iloc[:,57].notnull().astype('int')

pycharm19 = survey2019.iloc[:,58].notnull().astype('int')

atom19 = survey2019.iloc[:,59].notnull().astype('int')

matlab19 = survey2019.iloc[:,60].notnull().astype('int')

visualstudio19 = survey2019.iloc[:,61].notnull().astype('int')

spyder19 = survey2019.iloc[:,62].notnull().astype('int')

vimemacs19 = survey2019.iloc[:,63].notnull().astype('int')

notpadplusplus19 = survey2019.iloc[:,64].notnull().astype('int')

sublime19 = survey2019.iloc[:,65].notnull().astype('int')



idedf19 = pd.DataFrame({

    "Jupyter" : jupyter19,

    "R_Studio": rstudio19,

    "Pycharm": pycharm19,

    "Atom": atom19,

    "Matlab": matlab19,

    "Visual_Studio": visualstudio19,

    "Spyder" : spyder19,

    "Vim_Emac": vimemacs19,

    "Notpad++": notpadplusplus19,

    "Sublime" : sublime19

})

ideused19 = idedf19.sum(axis=0).sort_values(ascending=False)

ideused19 = ideused19.reset_index() 

ideused19.columns = ["IDE","Number of Users"]

sns.catplot(y="IDE", x="Number of Users",data=ideused19, 

            kind='bar', height=6, aspect=1.5);

plt.figtext(0.5, 1.1, "Figure 5 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
combineIDE19 = pd.concat([df19, idedf19], axis = 1)

agewiseIDE2019 = combineIDE19.iloc[:,[0,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Age",as_index=False).sum()

meltedIDE19 = agewiseIDE2019.melt(id_vars="Age",var_name="IDE",value_name="Users")

sns.catplot(y="Age", x="Users",data=meltedIDE19, 

            kind='bar', height=9, aspect=1.5,hue="IDE");

plt.title("Frequency of different IDE users by their age group");

plt.figtext(0.5, 1.1, "Figure 6 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
agewiseIDE2019
resData = sts.chi2_contingency(agewiseIDE2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
genderwiseIDE2019 = combineIDE19.iloc[:,[1,5,6,7,8,9,10,11,12,13,14]].groupby("Gender",as_index=False).sum()

genderwiseIDE2019 = genderwiseIDE2019[(genderwiseIDE2019.Gender =="Male")| (genderwiseIDE2019.Gender =="Female")]

genmeltIDE19 = genderwiseIDE2019.melt(id_vars="Gender",var_name="IDE",value_name="Users")



sns.catplot(y="Gender", x="Users",data=genmeltIDE19,

            kind='bar', height=9, aspect=1.5,hue="IDE");

plt.title("Frequency of users of different IDE by their Gender")

plt.figtext(0.5, 1.1, "Figure 7 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
genderwiseIDE2019
resData = sts.chi2_contingency(genderwiseIDE2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
CountrywiseJupyter2019 = combineIDE19.iloc[:,[2,5,6,7,8,9,10,11,12,13,14]].groupby("Country",as_index=False).sum()

CountrywiseJupyter2019 = CountrywiseJupyter2019[CountrywiseJupyter2019.Country.isin(top10Country.Country.to_list())]

CountrymeltedIDE19 = CountrywiseJupyter2019.melt(id_vars="Country",var_name="IDE",value_name="Users")

CountrymeltedIDE19.head()

sns.catplot(y="Country", x="Users",data=CountrymeltedIDE19, 

            kind='bar', height=9, aspect=1.5,hue="IDE");

plt.title("Frequency of users of different IDE by their Country")

plt.figtext(0.5, 1.1, "Figure 8 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");

CountrywiseJupyter2019
resData = sts.chi2_contingency(CountrywiseJupyter2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
EducationwiseIDE2019 = combineIDE19.iloc[:,[3,5,6,7,8,9,10,11,12,13,14]].groupby("Education",as_index=False).sum()



EducationmeltedIDE19 = EducationwiseIDE2019.melt(id_vars="Education",var_name="IDE",value_name="Users")

EducationmeltedIDE19.head()

sns.catplot(y="Education", x="Users",data=EducationmeltedIDE19, 

            kind='bar', height=9, aspect=1.5,hue="IDE");

plt.title("Frequency of users of different IDE by their Education")

plt.figtext(0.5, 1.1, "Figure 9 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");

EducationwiseIDE2019
resData = sts.chi2_contingency(EducationwiseIDE2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
JobrolewiseIDE2019 = combineIDE19.iloc[:,[4,5,6,7,8,9,10,11,12,13,14]].groupby("Jobrole",as_index=False).sum()



JobrolemeltedIDE19 = JobrolewiseIDE2019.melt(id_vars="Jobrole",var_name="IDE",value_name="Users")

JobrolemeltedIDE19.head()

sns.catplot(y="Jobrole", x="Users",data=JobrolemeltedIDE19, 

            kind='bar', height=9, aspect=1.5,hue="IDE");

plt.title("Frequency of users of different IDE by their JobRole")

plt.figtext(0.5, 1.1, "Figure 10 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
JobrolewiseIDE2019
resData = sts.chi2_contingency(JobrolewiseIDE2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
# What programming languages do you use on a regular basis (Done)

python19 = survey2019.iloc[:,82].notnull().astype('int')

r19 = survey2019.iloc[:,83].notnull().astype('int')

sql19 = survey2019.iloc[:,84].notnull().astype('int')

c19 = survey2019.iloc[:,85].notnull().astype('int')

cplusplus19 = survey2019.iloc[:,86].notnull().astype('int')

java19 = survey2019.iloc[:,87].notnull().astype('int')

javascript19 = survey2019.iloc[:,88].notnull().astype('int')

typescript19 = survey2019.iloc[:,89].notnull().astype('int')

bash19 = survey2019.iloc[:,90].notnull().astype('int')

matlab19 = survey2019.iloc[:,91].notnull().astype('int')



languagedf19 = pd.DataFrame({

    "Python" : python19,

    "R": r19,

    "Sql": sql19,

    "C": c19,

    "C++": cplusplus19,

    "Java": java19,

    "Javascript" : javascript19,

    "Typescript": typescript19,

    "Bash": bash19,

    "Matlab" : matlab19

})

proused19 = languagedf19.sum(axis=0).sort_values(ascending=False)

proused19 = proused19.reset_index() 

proused19.columns = ["Programming Language","Number of Users"]

sns.catplot(y="Programming Language", x="Number of Users",data=proused19, 

            kind='bar', height=6, aspect=1.5);

plt.figtext(0.5, 1.1, "Figure 11 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
combineLanguage19 = pd.concat([df19, languagedf19], axis = 1)
combineLanguage19
agewiseLan2019 = combineLanguage19.iloc[:,[0,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Age",as_index=False).sum()

meltedLan19 = agewiseLan2019.melt(id_vars="Age",var_name="Programming Language",value_name="Users")

sns.catplot(y="Age", x="Users",data=meltedLan19, 

            kind='bar', height=9, aspect=1.5,hue="Programming Language");

plt.title("Frequency of different Programming Language users by their age group");

plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
agewiseLan2019
resData = sts.chi2_contingency(agewiseLan2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
genderLan2019 = combineLanguage19.iloc[:,[1,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Gender",as_index=False).sum()

genderLan2019 = genderLan2019[(genderLan2019.Gender =="Male")| (genderLan2019.Gender =="Female")]

meltedLan19 = genderLan2019.melt(id_vars="Gender",var_name="Programming Language",value_name="Users")

sns.catplot(y="Gender", x="Users",data=meltedLan19, 

            kind='bar', height=9, aspect=1.5,hue="Programming Language");

plt.title("Frequency of different Programming Language users by their Gender");

plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
genderLan2019
resData = sts.chi2_contingency(genderLan2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
countryLan2019 = combineLanguage19.iloc[:,[2,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Country",as_index=False).sum()

countryLan2019 = countryLan2019[countryLan2019.Country.isin(top10Country.Country.to_list())]

meltedLan19 = countryLan2019.melt(id_vars="Country",var_name="Programming Language",value_name="Users")

sns.catplot(y="Country", x="Users",data=meltedLan19, 

            kind='bar', height=9, aspect=1.5,hue="Programming Language");

plt.title("Frequency of different Programming Language users by their Country");

plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
countryLan2019
resData = sts.chi2_contingency(countryLan2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
eduLan2019 = combineLanguage19.iloc[:,[3,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Education",as_index=False).sum()

meltedLan19 = eduLan2019.melt(id_vars="Education",

                                 var_name="Programming Language",

                                 value_name="Users")

sns.catplot(y="Education", x="Users",data=meltedLan19, 

            kind='bar', height=9, aspect=1.5,hue="Programming Language");

plt.title("Frequency of different Programming Language users by their Education");

plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
eduLan2019
resData = sts.chi2_contingency(eduLan2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
jobRoleLan2019 = combineLanguage19.iloc[:,[4,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Jobrole",as_index=False).sum()

meltedLan19 = jobRoleLan2019.melt(id_vars="Jobrole",var_name="Programming Language",value_name="Users")

sns.catplot(y="Jobrole", x="Users",data=meltedLan19, 

            kind='bar', height=9, aspect=1.5,hue="Programming Language");

plt.title("Frequency of different Programming Language users by their Jobrole");

plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
jobRoleLan2019
resData = sts.chi2_contingency(jobRoleLan2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
# Which of the following cloud computing platforms do you use on a regular basis (Done)



googleCloudPlatform19 = survey2019.iloc[:,168].notnull().astype('int')

amazonWebServices19 = survey2019.iloc[:,169].notnull().astype('int')

microsoftAzure19 = survey2019.iloc[:,170].notnull().astype('int')

ibmCloud19 = survey2019.iloc[:,171].notnull().astype('int')

alibabaCloud19 = survey2019.iloc[:,172].notnull().astype('int')

salesforceCloud19 = survey2019.iloc[:,173].notnull().astype('int')

oracleCloud19 = survey2019.iloc[:,174].notnull().astype('int')

sapCloud19 = survey2019.iloc[:,175].notnull().astype('int')

vmwareCloud19 = survey2019.iloc[:,176].notnull().astype('int')

redHatCloud19 = survey2019.iloc[:,177].notnull().astype('int')



cloudPlatform19 = pd.DataFrame({

    "GoogleCloudPlatform" : googleCloudPlatform19,

    "AmazonWebServices": amazonWebServices19,

    "MicrosoftAzure": microsoftAzure19,

    "IBMCloud": ibmCloud19,

    "AlibabaCloud": alibabaCloud19,

    "SalesforceCloud": salesforceCloud19,

    "OracleCloud" : oracleCloud19,

    "SAPCloud": sapCloud19,

    "VMwareCloud": vmwareCloud19,

    "RedHatCloud" : redHatCloud19

})



combinecloud19 = pd.concat([df19, cloudPlatform19], axis = 1)

cloudused19 = cloudPlatform19.sum(axis=0).sort_values(ascending=False)

cloudused19 = cloudused19.reset_index() 

cloudused19.columns = ["Cloud Tools","Number of Users"]

sns.catplot(y="Cloud Tools", x="Number of Users",data=cloudused19, 

            kind='bar', height=6, aspect=1.5);

plt.figtext(0.5, 1.1, "Figure 11 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
ageCloud2019 = combinecloud19.iloc[:,[0,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Age",as_index=False).sum()

meltedCloud19 = ageCloud2019.melt(id_vars="Age",var_name="Cloud Tools",value_name="Users")

sns.catplot(y="Age", x="Users",data=meltedCloud19, 

            kind='bar', height=9, aspect=1.5,hue="Cloud Tools");

plt.title("Frequency of different Cloud Tools Platform users by their age group");

plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
ageCloud2019

resData = sts.chi2_contingency(ageCloud2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
genderCloud2019 = combinecloud19.iloc[:,[1,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Gender",as_index=False).sum()



genderCloud2019 = genderCloud2019[(genderCloud2019.Gender =="Male")| (genderCloud2019.Gender =="Female")]



melted19 = genderCloud2019.melt(id_vars="Gender",var_name="Cloud Tools",value_name="Users")

sns.catplot(y="Gender", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="Cloud Tools");

plt.title("Frequency of different Cloud Platform users by their Gender");

plt.figtext(0.5, 1.1, "Figure 13 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
genderCloud2019
resData = sts.chi2_contingency(genderCloud2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
countryCloud2019 = combinecloud19.iloc[:,[2,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Country",as_index=False).sum()

melted19 = countryCloud2019.melt(id_vars="Country",var_name="Cloud Tools",value_name="Users")

sns.catplot(y="Country", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="Cloud Tools");

plt.title("Frequency of different  Cloud platform users by their Country");

plt.figtext(0.5, 1.1, "Figure 14 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
countryCloud2019
resData = sts.chi2_contingency(countryCloud2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
eduCloud2019 = combinecloud19.iloc[:,[3,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Education",as_index=False).sum()

melted19 = eduCloud2019.melt(id_vars="Education",var_name="Cloud Tools",value_name="Users")

sns.catplot(y="Education", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="Cloud Tools");

plt.title("Frequency of different Cloud platform users by their Education");

plt.figtext(0.5, 1.1, "Figure 15 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
eduCloud2019
resData = sts.chi2_contingency(eduCloud2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
jobrole2019 = combinecloud19.iloc[:,[4,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Jobrole",as_index=False).sum()

melted19 = jobrole2019.melt(id_vars="Jobrole",var_name="Cloud Tools",value_name="Users")

sns.catplot(y="Jobrole", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="Cloud Tools");

plt.title("Frequency of different Cloud platform users by their Jobrole");

plt.figtext(0.5, 1.1, "Figure 16 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
jobrole2019
# Which specificff big data / analytics products do you use on a regular basis? (Done)



googleBigQuery19 = survey2019.iloc[:,194].notnull().astype('int')

awsRedshift19 = survey2019.iloc[:,195].notnull().astype('int')

databricks19 = survey2019.iloc[:,196].notnull().astype('int')

awsElasticMapReduce19 = survey2019.iloc[:,197].notnull().astype('int')

teradata19 = survey2019.iloc[:,198].notnull().astype('int')

microsoftAnalysisServices19 = survey2019.iloc[:,199].notnull().astype('int')

googleCloudDataflow19 = survey2019.iloc[:,200].notnull().astype('int')

awsAthena19 = survey2019.iloc[:,201].notnull().astype('int')

awsKinesis19 = survey2019.iloc[:,202].notnull().astype('int')

googleCloudPubSub19 = survey2019.iloc[:,203].notnull().astype('int')



dfn19 = pd.DataFrame({

    "googleBigQuery" : googleBigQuery19,

    "awsRedshift": awsRedshift19,

    "databricks": databricks19,

    "awsElasticMapReduce": awsElasticMapReduce19,

    "teradata": teradata19,

    "microsoftAnalysisServices": microsoftAnalysisServices19,

    "googleCloudDataflow" : googleCloudDataflow19,

    "awsAthena": awsAthena19,

    "awsKinesis": awsKinesis19,

    "googleCloudPubSub" : googleCloudPubSub19

})



combine19 = pd.concat([df19, dfn19], axis = 1)



dfdn = dfn19.sum(axis=0).sort_values(ascending=False)

dfdn = dfdn.reset_index() 

dfdn.columns = ["BigDataTools","Number of Users"]

sns.catplot(y="BigDataTools", x="Number of Users",data=dfdn, 

            kind='bar', height=6, aspect=1.5);

plt.figtext(0.5, 1.1, "Figure 17 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
age2019 = combine19.iloc[:,[0,5,6,

                            7,8,9,10,11,

                            12,13,14]].groupby("Age",as_index=False).sum()



melted19 = age2019.melt(id_vars="Age",var_name="BigDataTools",value_name="Users")

sns.catplot(y="Age", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="BigDataTools");

plt.title("Frequency of different BigDataTools Platform users by their age group");

plt.figtext(0.5, 1.1, "Figure 18 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
age2019
resData = sts.chi2_contingency(age2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
gender2019 = combine19.iloc[:,[1,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Gender",as_index=False).sum()



gender2019 = gender2019[(gender2019.Gender =="Male")| (genderCloud2019.Gender =="Female")]



melted19 = gender2019.melt(id_vars="Gender",var_name="BigDataTools",value_name="Users")

sns.catplot(y="Gender", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="BigDataTools");

plt.title("Frequency of different Cloud Platform users by their Gender");

plt.figtext(0.5, 1.1, "Figure 19 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
gender2019
###################################################################################################



resData = sts.chi2_contingency(gender2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
country2019 = combine19.iloc[:,[2,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Country",as_index=False).sum()

country2019 = country2019[country2019.Country.isin(top10Country.Country.to_list())]

melted19 = country2019.melt(id_vars="Country",var_name="BigDataTools",value_name="Users")

sns.catplot(y="Country", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="BigDataTools");

plt.title("Frequency of different  Cloud platform users by their Country");

plt.figtext(0.5, 1.1, "Figure 20 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
country2019
resData = sts.chi2_contingency(country2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
edu2019 = combine19.iloc[:,[3,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Education",as_index=False).sum()

melted19 = edu2019.melt(id_vars="Education",var_name="BigDataTools",value_name="Users")

sns.catplot(y="Education", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="BigDataTools");

plt.title("Frequency of different  Big Data Tools users by their Education");

plt.figtext(0.5, 1.1, "Figure 21 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
edu2019
resData = sts.chi2_contingency(edu2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
jobrole2019 = combine19.iloc[:,[4,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Jobrole",as_index=False).sum()

melted19 = jobrole2019.melt(id_vars="Jobrole",var_name="BigDataTools",value_name="Users")

sns.catplot(y="Jobrole", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="BigDataTools");

plt.title("Frequency of different Cloud platform users by their Jobrole");

plt.figtext(0.5, 1.1, "Figure 22 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
# Which Database (Done)



mySQL19 = survey2019.iloc[:,233].notnull().astype('int')

postgresSQL19 = survey2019.iloc[:,234].notnull().astype('int')

sqlite19 = survey2019.iloc[:,235].notnull().astype('int')

microsoftSQLServer19 = survey2019.iloc[:,236].notnull().astype('int')

oracleDatabase19 = survey2019.iloc[:,237].notnull().astype('int')

microsoftAccess19 = survey2019.iloc[:,238].notnull().astype('int')

awsRelationalDatabaseService19 = survey2019.iloc[:,239].notnull().astype('int')

awsDynamoDB19 = survey2019.iloc[:,240].notnull().astype('int')

azureSQLDatabase19 = survey2019.iloc[:,241].notnull().astype('int')

googleCloudSQL19 = survey2019.iloc[:,242].notnull().astype('int')



dfn19 = pd.DataFrame({

    "MySQL" : mySQL19,

    "PostgresSQL": postgresSQL19,

    "SQLite": sqlite19,

    "MicrosoftSQLServer": microsoftSQLServer19,

    "OracleDatabase": oracleDatabase19,

    "MicrosoftAccess": microsoftAccess19,

    "AWSRelationalDatabase" : awsRelationalDatabaseService19,

    "AWSDynamoDB": awsDynamoDB19,

    "AzureSQLDatabase": azureSQLDatabase19,

    "GoogleCloudSQL" : googleCloudSQL19

})



combine19 = pd.concat([df19, dfn19], axis = 1)



dfdn = dfn19.sum(axis=0).sort_values(ascending=False)

dfdn = dfdn.reset_index() 

dfdn.columns = ["DataBase","Number of Users"]

sns.catplot(y="DataBase", x="Number of Users",data=dfdn, 

            kind='bar', height=6, aspect=1.5);

plt.figtext(0.5, 1.1, "Figure 23 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
###################################################################################################

age2019 = combine19.iloc[:,[0,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Age",as_index=False).sum()

melted19 = age2019.melt(id_vars="Age",var_name="DataBase",value_name="Users")

sns.catplot(y="Age", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="DataBase");

plt.title("Frequency of different DataBase Platform users by their age group");

plt.figtext(0.5, 1.1, "Figure 24 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
age2019
resData = sts.chi2_contingency(age2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
gender2019 = combine19.iloc[:,[1,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Gender",as_index=False).sum()



gender2019 = gender2019[(gender2019.Gender =="Male")| (genderCloud2019.Gender =="Female")]



melted19 = gender2019.melt(id_vars="Gender",var_name="DataBase",value_name="Users")

sns.catplot(y="Gender", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="DataBase");

plt.title("Frequency of different Database users by their Gender");

plt.figtext(0.5, 1.1, "Figure 25 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
gender2019
resData = sts.chi2_contingency(gender2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
country2019 = combine19.iloc[:,[2,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Country",as_index=False).sum()

country2019 = country2019[country2019.Country.isin(top10Country.Country.to_list())]

melted19 = country2019.melt(id_vars="Country",var_name="DataBase",value_name="Users")

sns.catplot(y="Country", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="DataBase");

plt.title("Frequency of different  Database users by their Country");

plt.figtext(0.5, 1.1, "Figure 26 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
country2019
resData = sts.chi2_contingency(country2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
edu2019 = combine19.iloc[:,[3,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Education",as_index=False).sum()

melted19 = edu2019.melt(id_vars="Education",var_name="DataBase",value_name="Users")

sns.catplot(y="Education", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="DataBase");

plt.title("Frequency of different Database users by their Education");

plt.figtext(0.5, 1.1, "Figure 27 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
edu2019
resData = sts.chi2_contingency(edu2019.iloc[:,1:])

print("p-value of chi-square test is : ", resData[1])
jobrole2019 = combine19.iloc[:,[4,5,6,

                                          7,8,9,10,11,

                                          12,13,

                                          14]].groupby("Jobrole",as_index=False).sum()

melted19 = jobrole2019.melt(id_vars="Jobrole",var_name="DataBase",value_name="Users")

sns.catplot(y="Jobrole", x="Users",data=melted19, 

            kind='bar', height=9, aspect=1.5,hue="DataBase");

plt.title("Frequency of different Database users by their Jobrole");

plt.figtext(0.5, 1.1, "Figure 28 :", wrap=True, horizontalalignment='center',

            fontsize=15,color="blue",alpha=1,fontweight="black");
jobrole2019