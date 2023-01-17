from matplotlib import pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns
pd.options.display.max_columns = 999

dataset = pd.read_csv("../input/so-survey-2017/survey_results_public.csv")

dataset.shape
dataset.head()
countries = dataset['Country'].unique()

print("Total Country: {0}".format(len(countries)))
country_freq = {}

for cnt in dataset['Country']:

  if cnt in country_freq:

    country_freq[cnt] += 1

  else:

    country_freq[cnt] =1
country_series = pd.Series(country_freq)

plt.figure(figsize=(40,15))

country_series.plot.bar()

plt.xlabel("Country Name")

plt.ylabel("Country Frequency")

plt.title("Country Frequency Graph from Stackoverflow dataset")

plt.show()
university = dataset['University'].value_counts()

university
plt.figure(figsize=(10,7))

sns.set(style="whitegrid")

ax = sns.countplot(x = 'University', data=dataset)

plt.xlabel("University Education Type")

plt.ylabel("Education Type Frequency")

plt.title("Education Type Frequency Graph from Stackoverflow dataset")

plt.show()
no_university_cnt = dataset[dataset['University']=='No']['Country']

no_university_cnt_frq = no_university_cnt.value_counts()

no_university_cnt_frq

plt.figure(figsize=(80,40))

sns.set(style="whitegrid")

ax = sns.countplot(x = 'Country', data=dataset[dataset['University']=='No'].iloc[:50])

plt.xlabel("Country")

plt.ylabel("'No' type University Frequency")

plt.title("'No' type University wise country frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
no_university_cnt = dataset[dataset['University']=='Yes, full-time']['Country']

no_university_cnt_frq = no_university_cnt.value_counts()

no_university_cnt_frq

plt.figure(figsize=(40,10))

sns.set(style="whitegrid")

ax = sns.countplot(x = 'Country', data=dataset[dataset['University']=='Yes, full-time'].iloc[:50])

plt.xlabel("Country")

plt.ylabel("'Yes, full-time' Type University Frequency")

plt.title("'Yes, full-time' type University wise country frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
no_university_cnt = dataset[dataset['University']=='Yes, part-time']['Country']

no_university_cnt_frq = no_university_cnt.value_counts()

no_university_cnt_frq

plt.figure(figsize=(40,10))

sns.set(style="whitegrid")

ax = sns.countplot(x = 'Country', data=dataset[dataset['University']=='Yes, part-time'].iloc[:50])

plt.xlabel("Country")

plt.ylabel("'Yes, part-time' Type University Frequency")

plt.title("'Yes, part-time' type University wise country frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
no_university_cnt = dataset[dataset['University']=='I prefer not to say']['Country']

no_university_cnt_frq = no_university_cnt.value_counts()

no_university_cnt_frq

plt.figure(figsize=(40,10))

sns.set(style="whitegrid")

ax = sns.countplot(x = 'Country', data=dataset[dataset['University']=='I prefer not to say'].iloc[:50])

plt.xlabel("Country")

plt.ylabel("'I prefer not to say' Type University Frequency")

plt.title("'I prefer not to say' type University wise country frequency graph from Stackoverflow dataset")

plt.xticks
dataset['EmploymentStatus'].value_counts()
EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Employed full-time']['Country'].value_counts()

print("Maximum Full time employee country: {0}".format(EmploymentStatusCnt.index[0]))
EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Independent contractor, freelancer, or self-employed']['Country'].value_counts()

print("Maximum (Independent contractor, freelancer, or self-employed) employee country: {0}".format(EmploymentStatusCnt.index[0]))
EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Employed part-time']['Country'].value_counts()

print("Maximum (Employed part-time) employee country: {0}".format(EmploymentStatusCnt.index[0]))
EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Not employed, and not looking for work']['Country'].value_counts()

print("Maximum (Not employed, and not looking for work ) employee country: {0}".format(EmploymentStatusCnt.index[0]))
EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Not employed, but looking for work']['Country'].value_counts()

print("Maximum (Not employed, but looking for work ) employee country: {0}".format(EmploymentStatusCnt.index[0]))
EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'I prefer not to say']['Country'].value_counts()

print("Maximum (I prefer not to say) employee country: {0}".format(EmploymentStatusCnt.index[0]))
EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Retired']['Country'].value_counts()

print("Maximum (Retired) employee country: {0}".format(EmploymentStatusCnt.index[0]))
FormalEducationFreq = dataset['FormalEducation'].value_counts(normalize=True)*100

FormalEducationFreq
data_labels = ["Bachelor's degree","Master's degree","Some college/university study without earning a bachelor's degree","Secondary school","Doctoral degree","I prefer not to answer","Primary/elementary school","Professional degree","I never completed any formal education "]

plt.figure(figsize=(10,8))

plt.pie(FormalEducationFreq,labels=data_labels,autopct='%1.1f%%',)
FormalEducationBachelor = dataset[dataset['FormalEducation'] == "Bachelor's degree"]["Country"].value_counts().iloc[:50]

plt.figure(figsize=(30,10))

sns.set(style="whitegrid")

FormalEducationBachelor.plot.bar()

plt.xlabel("FormalEducation")

plt.ylabel("Bachelor's degree Type Country Frequency")

plt.title("Bachelor's degree wise country frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
FormalEducationBachelor = dataset[dataset['FormalEducation'] == "Master's degree"]["Country"].value_counts().iloc[:50]

plt.figure(figsize=(30,10))

sns.set(style="whitegrid")

FormalEducationBachelor.plot.bar()

plt.xlabel("FormalEducation")

plt.ylabel("Master's degree Type Country Frequency")

plt.title("Master's degree wise country frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
MajorUndergradFreq = dataset['MajorUndergrad'].value_counts(normalize=True)*100

MajorUndergradFreq
data_labels = ["Computer science or software engineering","Computer engineering or electrical/electronics engineering","Computer programming or Web development","Information technology, networking, or system administration","A natural science","A non-computer-focused engineering discipline","Mathematics or statistics","Something else","A humanities discipline","A business discipline","Management information systems","Fine arts or performing arts","A social science","I never declared a major","Psychology","A health science"]

plt.figure(figsize=(25,8))

plt.pie(MajorUndergradFreq,labels=data_labels,autopct='%1.1f%%',)
MajorUndergradCSECountryFreq = dataset[dataset['MajorUndergrad'] == "Computer science or software engineering"]["Country"].value_counts().iloc[:40]

plt.figure(figsize=(30,10))

MajorUndergradCSECountryFreq.plot.bar()

plt.xlabel("Country")

plt.ylabel("CSE undergrade frequency")

plt.title("CSE undergrade frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
MajorUndergradCSECountryFreq = dataset[dataset['MajorUndergrad'] == "Computer engineering or electrical/electronics engineering"]["Country"].value_counts().iloc[:40]

plt.figure(figsize=(30,10))

MajorUndergradCSECountryFreq.plot.bar()

plt.xlabel("Country")

plt.ylabel("CSE/EEE undergrade frequency")

plt.title("CSE/EEE undergrade  frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
YearsCodedJobFreq = dataset["YearsCodedJob"].value_counts()

YearsCodedJobFreq
plt.figure(figsize=(20,10))

sns.set(style="whitegrid")

ax = sns.countplot(x = 'YearsCodedJob', data=dataset)

plt.xlabel("YearsCodedJob")

plt.ylabel("YearsCodedJob Frequency")

plt.title("YearsCodedJob frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
dataset["DeveloperType"].isnull().sum()
dataset["DeveloperType"].fillna("No Type", inplace=True)

dataset["DeveloperType"].isnull().sum()
dataset["CareerSatisfaction"].isnull().sum()
dataset["CareerSatisfaction"].fillna(0, inplace=True)

dataset["CareerSatisfaction"].isnull().sum()
CareerSatisfactionFreq = dataset["CareerSatisfaction"].value_counts(normalize=True)*100

CareerSatisfactionFreq
data_labels = ["8.0","7.0","0.0","9.0","10.0","6.0","5.0","4.0","3.0","2.0","1.0"]

plt.figure(figsize=(25,8))

plt.pie(CareerSatisfactionFreq,labels=data_labels,autopct='%1.1f%%',)

plt.show()
MostCareerSatisfactionCountry = dataset[dataset["CareerSatisfaction"] == 8.0]["Country"].value_counts().iloc[:50]

MostCareerSatisfactionCountry
plt.figure(figsize=(30,10))

MostCareerSatisfactionCountry.plot.bar()

plt.xlabel("Country")

plt.ylabel("Most CareerSatisfaction Country frequency")

plt.title("Most CareerSatisfaction Country frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
MostCareerSatisfactionYearsCodedJob = dataset[dataset["CareerSatisfaction"] == 8.0]["YearsCodedJob"].value_counts().iloc[:50]

MostCareerSatisfactionYearsCodedJob
plt.figure(figsize=(22,10))

sns.set(style="whitegrid")

ax = sns.boxplot(x = 'YearsCodedJob',y="CareerSatisfaction", data=dataset.sort_values(by="YearsCodedJob"))

plt.xlabel("YearsCodedJob")

plt.ylabel("CareerSatisfaction")

plt.title("Boxplot graph for CareerSatisfaction and YearsCodedJob from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(20,10))

sns.set(style="whitegrid")

ax = sns.countplot(x = 'YearsCodedJob', data=dataset[dataset["CareerSatisfaction"] == 8.0].sort_values(by="YearsCodedJob"))

plt.xlabel("YearsCodedJob")

plt.ylabel("CareerSatisfaction (8.0 )YearsCodedJob Frequency")

plt.title("Most CareerSatisfaction (8.0) YearsCodedJob frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
mostSeniorCarrerSatisfyCountry = dataset[(dataset["CareerSatisfaction"] == 8.0) & (dataset["YearsCodedJob"] == '20 or more years')]['Country'].value_counts(normalize=True)*100

mostSeniorCarrerSatisfyCountry
plt.figure(figsize=(25,10))

sns.set(style="whitegrid")

ax = sns.countplot(x = 'Country', data=dataset[(dataset["CareerSatisfaction"] == 8.0) & (dataset["YearsCodedJob"] == '20 or more years')].sort_values(by="Country"))

plt.xlabel("Country")

plt.ylabel("Most senior (20 years or more)CareerSatisfaction (8.0 ) Frequency")

plt.title("Most senior (20 years or more)CareerSatisfaction (8.0 ) frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(20,10))

sns.set(style="whitegrid")

ax = sns.countplot(x = 'YearsCodedJob', data=dataset[dataset["CareerSatisfaction"] == 1.0].sort_values(by="YearsCodedJob"))

plt.xlabel("YearsCodedJob")

plt.ylabel("CareerSatisfaction (1.0 )YearsCodedJob Frequency")

plt.title("Less CareerSatisfaction (1.0) YearsCodedJob frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(25,10))

sns.set(style="whitegrid")

ax = sns.countplot(x = 'Country', data=dataset[(dataset["CareerSatisfaction"] == 1.0) & (dataset["YearsCodedJob"] == '1 to 2 years')].sort_values(by="Country"))

plt.xlabel("Country")

plt.ylabel("(1 to 2 years) CareerSatisfaction (1.0 ) Frequency")

plt.title("(1 to 2 years) CareerSatisfaction (1.0 ) frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(28,11))

sns.set(style="whitegrid")

ax = sns.countplot(x = 'Country', data=dataset[(dataset["CareerSatisfaction"] == 1.0) & (dataset["YearsCodedJob"] == 'Less than a year')].sort_values(by="Country"))

plt.xlabel("Country")

plt.ylabel("(Less than a year) CareerSatisfaction (1.0 ) Frequency")

plt.title("(Less than a year) CareerSatisfaction (1.0 ) frequency graph from Stackoverflow dataset")

plt.xticks(rotation=45)

plt.show()
JobSatisfactionFreq = dataset["JobSatisfaction"].value_counts(normalize=True)*100

JobSatisfactionFreq
data_labels = ["8.0","7.0","9.0","6.0","10.0","5.0","4.0","3.0","2.0","0.0","1.0"]

plt.figure(figsize=(25,8))

plt.pie(JobSatisfactionFreq,labels=data_labels,autopct='%1.1f%%',)

plt.show()
JobSatisfactionIndex = dataset["JobSatisfaction"].value_counts().index

JobSatisfactionIndex
JobSatisfactionDF = dataset[dataset['JobSatisfaction']== 10.0][['Country','FormalEducation','University','YearsCodedJob']]

JobSatisfactionDF.head()
JobSatisfactionDFCountryFreq = JobSatisfactionDF['Country'].value_counts(normalize=True)*100

JobSatisfactionDFCountryFreq.iloc[:20]
JobSatisfactionDFFormalEducationFreq = JobSatisfactionDF['FormalEducation'].value_counts(normalize=True)*100

JobSatisfactionDFFormalEducationFreq.iloc[:20]
JobSatisfactionDFFormalColFreq = JobSatisfactionDF['University'].value_counts(normalize=True)*100

JobSatisfactionDFFormalColFreq.iloc[:20]
JobSatisfactionDFFormalColFreq = JobSatisfactionDF['YearsCodedJob'].value_counts(normalize=True)*100

JobSatisfactionDFFormalColFreq.iloc[:20]
JobSatisfactionDFExplore = JobSatisfactionDF[JobSatisfactionDF['Country']=='United States'][['FormalEducation','University','YearsCodedJob']]

JobSatisfactionDFExplore.head()
JobSatisfactionDFExplore['FormalEducation'].value_counts(normalize=True)*100
JobSatisfactionDFExplore['University'].value_counts(normalize=True)*100
data_labels = ["No","Yes, full-time","Yes, part-time","I prefer not to say"]

plt.figure(figsize=(25,8))

plt.pie(JobSatisfactionDFExplore['University'].value_counts(normalize=True)*100,labels=data_labels,autopct='%1.1f%%',)

plt.show()
JobSatisfactionDFExplore['YearsCodedJob'].value_counts(normalize=True)*100
JobSatisfactionDF_8 = dataset[dataset['JobSatisfaction']== 8.0][['Country','FormalEducation','University','YearsCodedJob']]

JobSatisfactionDF_8.head()
JobSatisfactionDFCountryFreq = JobSatisfactionDF_8['Country'].value_counts(normalize=True)*100

JobSatisfactionDFCountryFreq.iloc[:20]
JobSatisfactionDFFormalEducationFreq = JobSatisfactionDF_8['FormalEducation'].value_counts(normalize=True)*100

JobSatisfactionDFFormalEducationFreq.iloc[:20]
JobSatisfactionDFExplore['University'].value_counts(normalize=True)*100
data_labels = ["No","Yes, full-time","Yes, part-time","I prefer not to say"]

plt.figure(figsize=(25,8))

plt.pie(JobSatisfactionDFExplore['University'].value_counts(normalize=True)*100,labels=data_labels,autopct='%1.1f%%',)

plt.show()
JobSatisfactionDFExplore = JobSatisfactionDF_8[JobSatisfactionDF_8['Country']=='United States'][['FormalEducation','University','YearsCodedJob']]

JobSatisfactionDFExplore.head()
JobSatisfactionDFExplore['YearsCodedJob'].value_counts(normalize=True)*100
data_labels = ["20 or more years","2 to 3 years","1 to 2 years","3 to 4 years","4 to 5 years","Less than a year","5 to 6 years","6 to 7 years","10 to 11 years","9 to 10 years","7 to 8 years","8 to 9 years","15 to 16 years","14 to 15 years","11 to 12 years","12 to 13 years","16 to 17 years","13 to 14 years","19 to 20 years","17 to 18 years","18 to 19 years"]

plt.figure(figsize=(25,8))

plt.pie(JobSatisfactionDFExplore['YearsCodedJob'].value_counts(normalize=True)*100,labels=data_labels,autopct='%1.1f%%',)

plt.show()