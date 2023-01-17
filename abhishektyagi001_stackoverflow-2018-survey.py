import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from math import pi

import seaborn as sns

%matplotlib inline 



import warnings

warnings.filterwarnings("ignore")



# Print all rows and columns

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

IS_LOCAL = True



import os



if(IS_LOCAL):

    PATH="../input/stack-overflow-2018-developer-survey"

else:

    PATH="../input"

print(os.listdir(PATH))
data_df = pd.read_csv(PATH+"/survey_results_public.csv")

schema_df = pd.read_csv(PATH+"/survey_results_schema.csv")
print("Stack Overflow 2018 Developer Survey data -  rows:",data_df.shape[0]," columns:", data_df.shape[1])
print("Stack Overflow 2018 Developer Survey schema -  rows:",schema_df.shape[0]," columns:", schema_df.shape[1])
data_df.head()
total = data_df.isnull().sum().sort_values(ascending = False)

percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)

tmp = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



def plot_missing_data(start, end, text):

    tmp1 = tmp[start:end]

    plt.figure(figsize = (16,4))

    plt.title('Missing data - percents of missing data (part %s)' % text)

    s = sns.barplot(x=tmp1.index,y=tmp1['Percent'])

    s.set_xticklabels(s.get_xticklabels(),rotation=90)

    plt.show()    

    



plot_missing_data(1,65,"I")
plot_missing_data(66,129,"II")
def plot_stats(feature, text, size=2):

    temp = data_df[feature].dropna().value_counts().head(50)

    df1 = pd.DataFrame({feature: temp.index,'Number of respondents': temp.values})

    plt.figure(figsize = (8*size,4))

    plt.title(text,fontsize=14)

    s = sns.barplot(x=feature,y='Number of respondents',data=df1)

    s.set_xticklabels(s.get_xticklabels(),rotation=90)

    plt.show()   

    

plot_stats('Country','Countries')
plot_stats('FormalEducation','Formal Education',1)
plot_stats('UndergradMajor','Undergraduate Major',1)
plot_stats('Student','Respondent is currently student',1)
plot_stats('EducationParents','Education of respondent\'s parents',1)
plot_stats('Employment','Employment',1)
plot_stats('CompanySize','Company size',1)
plot_stats('JobSatisfaction','Job satisfaction',1)
plot_stats('CareerSatisfaction','Career satisfaction',1)
plot_stats('HopeFiveYears','Where the respondents see themselves in 5 years',1)
plot_stats('JobSearchStatus','Status with searching for a new job',1)
plot_stats('LastNewJob','How much they were with the last new job',1)
schema_df[schema_df['Column'].str.contains('AssessJob')]
def plot_heatmap(feature, text, color="Blues"):

    tmp = schema_df[schema_df['Column'].str.contains(feature)]

    features = list(tmp['Column'])

    dim = len(features)

    temp1 = pd.DataFrame(np.random.randint(low=0, high=10,size=(1+dim, dim)),columns=features)

    for feature in features:

        temp1[feature] = data_df[feature].dropna().value_counts()



    fig, (ax1) = plt.subplots(ncols=1, figsize=(16,4))

    sns.heatmap(temp1[1::], 

        xticklabels=temp1.columns,

        yticklabels=temp1.index[1::],annot=True,ax=ax1,linewidths=.1,cmap=color)

    plt.title(text, fontsize=14)

    plt.show()
plot_heatmap('AssessJob','Heatmap with Assess Job priorities 1-10 (respondants count)')
schema_df[schema_df['Column'].str.contains('AssessBenefits')]
plot_heatmap('AssessBenefits','Heatmap with Assess Job benefits priorities 1-11 (respondants count)',"Greens")
schema_df[schema_df['Column'].str.contains('JobContactPriorities')]
plot_heatmap('JobContactPriorities','Heatmap with Job contact priorities 1-5 (respondants count)',"Reds")
schema_df[schema_df['Column'].str.contains('JobEmailPriorities')]
plot_heatmap('JobEmailPriorities','Heatmap with Job email priorities 1-7 (respondants count)',"Purples")
plot_stats('Currency','Salary currency',1)
plot_stats('SalaryType','Salary type',0.75)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax = ax1, x="SalaryType", y="ConvertedSalary", hue="SalaryType",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="SalaryType", y="ConvertedSalary", hue="SalaryType",data=data_df, palette="PRGn",showfliers=False)

plt.show();
plot_stats('DevType','Development type',2)
plot_stats('YearsCoding','Number of years coding',1)
plot_stats('YearsCodingProf','Number of years coding professionaly',1)
plot_stats('LanguageWorkedWith','Languages worked with',2)
plot_stats('LanguageDesireNextYear','Language desired next year',2)
plot_stats('DatabaseWorkedWith','Databases worked with',2)
plot_stats('DatabaseDesireNextYear','Databases desired next year',2)
plot_stats('PlatformWorkedWith','Platform Worked With',2)
plot_stats('PlatformDesireNextYear','Platform Desired Next Year',2)
plot_stats('OperatingSystem','Operating System',0.5)
schema_df[schema_df['Column'].str.contains('AdBlocker')]
plot_stats('AdBlocker','Have ad-blocking software installed',0.75)
plot_stats('AdBlockerDisable','Disabled ad-blocking last month',0.75)
plot_stats('AdBlockerReasons','Why disabled ad-blocking last month',2)
schema_df[schema_df['Column'].str.contains('AdsPriorities')]
plot_heatmap('AdsPriorities','Heatmap with Ads Priorities questions (respondants count)',"Blues")
plot_stats('EthicsChoice','Ethics Choice',0.5)
plot_stats('EthicsReport','Ethics Report',0.5)
plot_stats('EthicsResponsible','Ethics Responsible',0.5)
plot_stats('EthicalImplications','Ethical Implications',0.5)
schema_df[schema_df['Column'].str.contains('AI')]['QuestionText'][96]
plot_stats('AIDangerous','Most dangerous aspects of AI',0.5)
plot_stats('AIInteresting','Most exciting aspects of AI',0.5)
plot_stats('AIResponsible','Primarly responsible to consider consequences of AI',0.75) 
plot_stats('AIFuture','Future of AI',0.75) 
plot_stats('Age','Age of responders',1)
plot_stats('Gender','Gender declared by respondents',1)
plot_stats('SexualOrientation','Sexual orientation',1)
plot_stats('RaceEthnicity','Race and Ethnicity',2)
plot_stats('Exercise','How much exercise',0.5)
plot_stats('HoursOutside','How many hours spent outside',0.5)
def plot_heatmap_mean(feature1, feature2, feature3, color, title):

    tmp = data_df.groupby([feature1, feature2])[feature3].mean()

    df1 = tmp.reset_index()

    matrix = df1.pivot(feature1, feature2, feature3)

    fig, (ax1) = plt.subplots(ncols=1, figsize=(16,6))

    sns.heatmap(matrix, 

        xticklabels=matrix.columns,

        yticklabels=matrix.index,ax=ax1,linewidths=.1,annot=True,cmap=color)

    plt.title(title, fontsize=14)

    plt.show()



plot_heatmap_mean('Employment', 'Gender','ConvertedSalary', "Greens", "Heatmap with mean(Converted Salary - USD) per Employment type and Gender")
def plot_heatmap_count(feature1, feature2, color, title):

    tmp = data_df.groupby([feature1, feature2])['Country'].count()

    df1 = tmp.reset_index()

    matrix = df1.pivot(feature1, feature2, 'Country')

    fig, (ax1) = plt.subplots(ncols=1, figsize=(16,6))

    sns.heatmap(matrix, 

        xticklabels=matrix.columns,

        yticklabels=matrix.index,ax=ax1,linewidths=.1,annot=True,cmap=color)

    plt.title(title, fontsize=14)

    plt.show()
plot_heatmap_count('Employment', 'Gender',"Blues", "Heatmap with number of respondents per Employment type and Gender")
plot_heatmap_mean('Employment', 'YearsCoding','ConvertedSalary', "Reds", "Heatmap with mean(Converted Salary - USD) per Employment type and Years Coding")
plot_heatmap_count('OperatingSystem', 'Gender',"Blues", "Heatmap with number of respondents per Operating system and Gender")
plot_heatmap_count('OperatingSystem', 'Exercise',"Greens", "Heatmap with number of respondents per Operating system and Exercise")
plot_heatmap_count('Age', 'AIDangerous',"Reds", "Heatmap with number of respondents per AI Dangers and Age")
plot_heatmap_count('AIDangerous', 'OperatingSystem', "Blues", "Heatmap with number of respondents per AI Dangers and Operating System")
plot_heatmap_count('Age', 'AIInteresting', "Reds", "Heatmap with number of respondents per AI Interesting and Age")
plot_heatmap_count('AIInteresting','OperatingSystem',  "Blues", "Heatmap with number of respondents per AI Interesting and Operating System")
plot_heatmap_count('Age', 'AIResponsible', "Reds", "Heatmap with number of respondents per AI Responsibility and Age")
plot_heatmap_count('AIResponsible','OperatingSystem',  "Blues", "Heatmap with number of respondents per AI Responsibility and Operating System")