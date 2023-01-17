import numpy as np 
import pandas as pd
import warnings
warnings.simplefilter("ignore")
from scipy.special import boxcox
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.set_config_file(offline=True)
import plotly as py
FreeFormDataSet = pd.read_csv('../input/freeFormResponses.csv', low_memory=False, header=[0,1])
FreeFormDataSet.columns = ['_'.join(col) for col in FreeFormDataSet.columns]
MultipleChoiceDataSet = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False, header=[0,1])
MultipleChoiceDataSet.columns = ['_'.join(col) for col in MultipleChoiceDataSet.columns]
SurveySchemaDataSet = pd.read_csv('../input/SurveySchema.csv', low_memory=False, header=[0,1])
SurveySchemaDataSet.columns = ['_'.join(col) for col in SurveySchemaDataSet.columns]
pd.set_option('display.max_columns', None)
FreeFormDataSet.head(5)
MultipleChoiceDataSet.head(5)
SurveySchemaDataSet.head(5)
MultipleChoiceDataSet = MultipleChoiceDataSet.rename({
                'Time from Start to Finish (seconds)_Duration (in seconds)' : 'Time Duration', 
                 'Q1_What is your gender? - Selected Choice' : 'Gender', 
                 'Q1_OTHER_TEXT_What is your gender? - Prefer to self-describe - Text' : 'Other_Gender', 
                 'Q2_What is your age (# years)?' : 'Age', 
                 'Q3_In which country do you currently reside?' : 'Country', 
                 'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?' : 'Higher_Education', 
                 'Q5_Which best describes your undergraduate major? - Selected Choice' : 'Undergraduate_Major', 
                 'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice' : 'Current_Role', 
                 'Q6_OTHER_TEXT_Select the title most similar to your current role (or most recent title if retired): - Other - Text' : 'Other_Current_Role', 
                 'Q7_In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice' : 'Current_Industry', 
                'Q7_OTHER_TEXT_In what industry is your current employer/contract (or your most recent employer if retired)? - Other - Text' : 'Other_Current_Industry', 
                'Q8_How many years of experience do you have in your current role?' : 'Years_Experience', 
                'Q9_What is your current yearly compensation (approximate $USD)?' : 'Current_Compensation'}, axis='columns')
MultipleChoiceDataSet.head(5)
TotalMissingTimeDuration = MultipleChoiceDataSet['Time Duration'].isnull().sum()
TotalTimeDuration = MultipleChoiceDataSet['Time Duration'].count()
TotalMissingTimeDurationPercentage = (round((TotalMissingTimeDuration / TotalTimeDuration) * 100,3))

TotalMissingGender = MultipleChoiceDataSet['Gender'].isnull().sum()
TotalGender = MultipleChoiceDataSet['Gender'].count()
TotalMissingGenderPercentage = (round((TotalMissingGender / TotalGender) * 100,3))

TotalMissingAge = MultipleChoiceDataSet['Age'].isnull().sum()
TotalAge = MultipleChoiceDataSet['Age'].count()
TotalMissingAgePercentage = (round((TotalMissingAge / TotalAge) * 100,3))

TotalMissingCountry = MultipleChoiceDataSet['Country'].isnull().sum()
TotalCountry = MultipleChoiceDataSet['Country'].count()
TotalMissingCountryPercentage = (round((TotalMissingCountry / TotalCountry) * 100,3))

TotalMissingHigherEducation = MultipleChoiceDataSet['Higher_Education'].isnull().sum()
TotalHigherEducation = MultipleChoiceDataSet['Higher_Education'].count()
TotalMissingHigherEducationPercentage = (round((TotalMissingHigherEducation / TotalHigherEducation) * 100,3))

TotalMissingUndergraduateMajor = MultipleChoiceDataSet['Undergraduate_Major'].isnull().sum()
TotalUndergraduateMajor = MultipleChoiceDataSet['Undergraduate_Major'].count()
TotalMissingUndergraduateMajorPercentage = (round((TotalMissingUndergraduateMajor / TotalUndergraduateMajor) * 100,3))

TotalMissingCurrentRole = MultipleChoiceDataSet['Current_Role'].isnull().sum()
TotalCurrentRole = MultipleChoiceDataSet['Current_Role'].count()
TotalMissingCurrentRolePercentage = (round((TotalMissingCurrentRole / TotalCurrentRole) * 100,3))

TotalMissingCurrentIndustry = MultipleChoiceDataSet['Current_Industry'].isnull().sum()
TotalCurrentIndustry = MultipleChoiceDataSet['Current_Industry'].count()
TotalMissingCurrentIndustryPercentage = (round((TotalMissingCurrentIndustry / TotalCurrentIndustry) * 100,3))

TotalMissingYearsExperience = MultipleChoiceDataSet['Years_Experience'].isnull().sum()
TotalYearsExperience = MultipleChoiceDataSet['Years_Experience'].count()
TotalMissingYearsExperiencePercentage = (round((TotalMissingYearsExperience / TotalYearsExperience) * 100,3))

TotalMissingCurrentCompensation = MultipleChoiceDataSet['Current_Compensation'].isnull().sum()
TotalCurrentCompensation = MultipleChoiceDataSet['Current_Compensation'].count()
TotalMissingCurrentCompensationPercentage = (round((TotalMissingCurrentCompensation / TotalCurrentCompensation) * 100,3))

Field_1 = pd.Series({'Field Name': 'Time Duration',
                        'Total Missing Records': TotalMissingTimeDuration,
                        'Missing Percentage': TotalMissingTimeDurationPercentage,                     
                         'Total Records' : TotalTimeDuration,
                    })
Field_2 = pd.Series({'Field Name': 'Gender',
                        'Total Missing Records': TotalMissingGender,
                        'Missing Percentage': TotalMissingGenderPercentage,                     
                         'Total Records' : TotalGender,
                    })
Field_3 = pd.Series({'Field Name': 'Age',
                        'Total Missing Records': TotalMissingAge,
                        'Missing Percentage': TotalMissingAgePercentage,                     
                         'Total Records' : TotalAge,
                    })
Field_4 = pd.Series({'Field Name': 'Country',
                        'Total Missing Records': TotalMissingCountry,
                        'Missing Percentage': TotalMissingCountryPercentage,                     
                         'Total Records' : TotalCountry,
                    })
Field_5 = pd.Series({'Field Name': 'Higher Education',
                        'Total Missing Records': TotalMissingHigherEducation,
                        'Missing Percentage': TotalMissingHigherEducationPercentage,                     
                         'Total Records' : TotalHigherEducation,
                    })
Field_6 = pd.Series({'Field Name': 'Undergraduate Major',
                        'Total Missing Records': TotalMissingUndergraduateMajor,
                        'Missing Percentage': TotalMissingUndergraduateMajorPercentage,                     
                         'Total Records' : TotalUndergraduateMajor,
                    })
Field_7 = pd.Series({'Field Name': 'Current Role',
                        'Total Missing Records': TotalMissingCurrentRole,
                        'Missing Percentage': TotalMissingCurrentRolePercentage,                     
                         'Total Records' : TotalCurrentRole,
                    })
Field_8 = pd.Series({'Field Name': 'Current Industry',
                        'Total Missing Records': TotalMissingCurrentIndustry,
                        'Missing Percentage': TotalMissingCurrentIndustryPercentage,                     
                         'Total Records' : TotalCurrentIndustry,
                    })
Field_9 = pd.Series({'Field Name': 'Years Experience',
                        'Total Missing Records': TotalMissingYearsExperience,
                        'Missing Percentage': TotalMissingYearsExperiencePercentage,                     
                         'Total Records' : TotalYearsExperience,
                    })
Field_10 = pd.Series({'Field Name': 'Current Compensation',
                        'Total Missing Records': TotalMissingCurrentCompensation,
                        'Missing Percentage': TotalMissingCurrentCompensationPercentage,                     
                         'Total Records' : TotalCurrentCompensation,
                    })
FieldSummary = pd.DataFrame([Field_1,Field_2,Field_3,Field_4,Field_5,Field_6,Field_7,Field_8,Field_9,Field_10]
                             , index=['1','2','3','4','5','6','7','8','9','10'])
FieldSummary
MultipleChoiceDataSet['Current_Role'].fillna("Unknown", inplace=True)
MultipleChoiceDataSet['Current_Industry'].fillna("Unknown", inplace=True)
MultipleChoiceDataSet['Years_Experience'].fillna("0", inplace=True)
MultipleChoiceDataSet['Current_Compensation'].fillna("0", inplace=True)
MultipleChoiceDataSet['Higher_Education'].fillna("Unknown", inplace=True)
MultipleChoiceDataSet['Undergraduate_Major'].fillna("Unknown", inplace=True)
TotalMissingTimeDuration = MultipleChoiceDataSet['Time Duration'].isnull().sum()
TotalTimeDuration = MultipleChoiceDataSet['Time Duration'].count()
TotalMissingTimeDurationPercentage = (round((TotalMissingTimeDuration / TotalTimeDuration) * 100,3))

TotalMissingGender = MultipleChoiceDataSet['Gender'].isnull().sum()
TotalGender = MultipleChoiceDataSet['Gender'].count()
TotalMissingGenderPercentage = (round((TotalMissingGender / TotalGender) * 100,3))

TotalMissingAge = MultipleChoiceDataSet['Age'].isnull().sum()
TotalAge = MultipleChoiceDataSet['Age'].count()
TotalMissingAgePercentage = (round((TotalMissingAge / TotalAge) * 100,3))

TotalMissingCountry = MultipleChoiceDataSet['Country'].isnull().sum()
TotalCountry = MultipleChoiceDataSet['Country'].count()
TotalMissingCountryPercentage = (round((TotalMissingCountry / TotalCountry) * 100,3))

TotalMissingHigherEducation = MultipleChoiceDataSet['Higher_Education'].isnull().sum()
TotalHigherEducation = MultipleChoiceDataSet['Higher_Education'].count()
TotalMissingHigherEducationPercentage = (round((TotalMissingHigherEducation / TotalHigherEducation) * 100,3))

TotalMissingUndergraduateMajor = MultipleChoiceDataSet['Undergraduate_Major'].isnull().sum()
TotalUndergraduateMajor = MultipleChoiceDataSet['Undergraduate_Major'].count()
TotalMissingUndergraduateMajorPercentage = (round((TotalMissingUndergraduateMajor / TotalUndergraduateMajor) * 100,3))

TotalMissingCurrentRole = MultipleChoiceDataSet['Current_Role'].isnull().sum()
TotalCurrentRole = MultipleChoiceDataSet['Current_Role'].count()
TotalMissingCurrentRolePercentage = (round((TotalMissingCurrentRole / TotalCurrentRole) * 100,3))

TotalMissingCurrentIndustry = MultipleChoiceDataSet['Current_Industry'].isnull().sum()
TotalCurrentIndustry = MultipleChoiceDataSet['Current_Industry'].count()
TotalMissingCurrentIndustryPercentage = (round((TotalMissingCurrentIndustry / TotalCurrentIndustry) * 100,3))

TotalMissingYearsExperience = MultipleChoiceDataSet['Years_Experience'].isnull().sum()
TotalYearsExperience = MultipleChoiceDataSet['Years_Experience'].count()
TotalMissingYearsExperiencePercentage = (round((TotalMissingYearsExperience / TotalYearsExperience) * 100,3))

TotalMissingCurrentCompensation = MultipleChoiceDataSet['Current_Compensation'].isnull().sum()
TotalCurrentCompensation = MultipleChoiceDataSet['Current_Compensation'].count()
TotalMissingCurrentCompensationPercentage = (round((TotalMissingCurrentCompensation / TotalCurrentCompensation) * 100,3))

Field_1 = pd.Series({'Field Name': 'Time Duration',
                        'Total Missing Records': TotalMissingTimeDuration,
                        'Missing Percentage': TotalMissingTimeDurationPercentage,                     
                         'Total Records' : TotalTimeDuration,
                    })
Field_2 = pd.Series({'Field Name': 'Gender',
                        'Total Missing Records': TotalMissingGender,
                        'Missing Percentage': TotalMissingGenderPercentage,                     
                         'Total Records' : TotalGender,
                    })
Field_3 = pd.Series({'Field Name': 'Age',
                        'Total Missing Records': TotalMissingAge,
                        'Missing Percentage': TotalMissingAgePercentage,                     
                         'Total Records' : TotalAge,
                    })
Field_4 = pd.Series({'Field Name': 'Country',
                        'Total Missing Records': TotalMissingCountry,
                        'Missing Percentage': TotalMissingCountryPercentage,                     
                         'Total Records' : TotalCountry,
                    })
Field_5 = pd.Series({'Field Name': 'Higher Education',
                        'Total Missing Records': TotalMissingHigherEducation,
                        'Missing Percentage': TotalMissingHigherEducationPercentage,                     
                         'Total Records' : TotalHigherEducation,
                    })
Field_6 = pd.Series({'Field Name': 'Undergraduate Major',
                        'Total Missing Records': TotalMissingUndergraduateMajor,
                        'Missing Percentage': TotalMissingUndergraduateMajorPercentage,                     
                         'Total Records' : TotalUndergraduateMajor,
                    })
Field_7 = pd.Series({'Field Name': 'Current Role',
                        'Total Missing Records': TotalMissingCurrentRole,
                        'Missing Percentage': TotalMissingCurrentRolePercentage,                     
                         'Total Records' : TotalCurrentRole,
                    })
Field_8 = pd.Series({'Field Name': 'Current Industry',
                        'Total Missing Records': TotalMissingCurrentIndustry,
                        'Missing Percentage': TotalMissingCurrentIndustryPercentage,                     
                         'Total Records' : TotalCurrentIndustry,
                    })
Field_9 = pd.Series({'Field Name': 'Years Experience',
                        'Total Missing Records': TotalMissingYearsExperience,
                        'Missing Percentage': TotalMissingYearsExperiencePercentage,                     
                         'Total Records' : TotalYearsExperience,
                    })
Field_10 = pd.Series({'Field Name': 'Current Compensation',
                        'Total Missing Records': TotalMissingCurrentCompensation,
                        'Missing Percentage': TotalMissingCurrentCompensationPercentage,                     
                         'Total Records' : TotalCurrentCompensation,
                    })
FieldSummary = pd.DataFrame([Field_1,Field_2,Field_3,Field_4,Field_5,Field_6,Field_7,Field_8,Field_9,Field_10]
                             , index=['1','2','3','4','5','6','7','8','9','10'])
FieldSummary
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Country"] == "United States of America", "Country"] = 'USA'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Country"] == "United Kingdom of Great Britain and Northern Ireland", "Country"] = 'UK and N.Irland'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Country"] == "I do not wish to disclose my location", "Country"] = 'Confidential'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Country"] == "Iran, Islamic Republic of...", "Country"] = 'Iran, Islamic'

Field_1 = pd.Series({'Actual Value': 'United States of America',
                        'New Value': 'USA',                     
                    })
Field_2 = pd.Series({'Actual Value': 'United Kingdom of Great Britain and Northern Ireland',
                        'New Value': 'UK and N.Irland',                     
                    })
Field_3 = pd.Series({'Actual Value': 'I do not wish to disclose my location',
                        'New Value': 'Confidential',                     
                    })
Field_4 = pd.Series({'Actual Value': 'Iran, Islamic Republic of...',
                        'New Value': 'Iran, Islamic',                     
                    })
FieldUpdateSummary = pd.DataFrame([Field_1,Field_2,Field_3,Field_4],index=['1','2','3','4'])
FieldUpdateSummary
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Current_Industry"] == "Online Service/Internet-based Services", "Current_Industry"] = 'Online Service'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Current_Industry"] == "Online Business/Internet-based Sales", "Current_Industry"] = 'Online Business'

Field_1 = pd.Series({'Actual Value': 'Online Service/Internet-based Services',
                        'New Value': 'Online Service',                     
                    })
Field_2 = pd.Series({'Actual Value': 'Online Business/Internet-based Sales',
                        'New Value': 'Online Business',                     
                    })
FieldUpdateSummary = pd.DataFrame([Field_1,Field_2],index=['1','2'])
FieldUpdateSummary
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Higher_Education"] == "Some college/university study without earning a bachelor’s degree", "Higher_Education"] = 'Some College'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Higher_Education"] == "No formal education past high school", "Higher_Education"] = 'No formal Education'

Field_1 = pd.Series({'Actual Value': 'Some college/university study without earning a bachelor’s degree',
                        'New Value': 'Some College',                     
                    })
Field_2 = pd.Series({'Actual Value': 'No formal education past high school',
                        'New Value': 'No formal Education',                     
                    })
FieldUpdateSummary = pd.DataFrame([Field_1,Field_2],index=['1','2'])
FieldUpdateSummary
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Undergraduate_Major"] == "Computer science (software engineering, etc.)", "Undergraduate_Major"] = 'Computer Science'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Undergraduate_Major"] == "Engineering (non-computer focused)", "Undergraduate_Major"] = 'Engineering'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Undergraduate_Major"] == "A business discipline (accounting, economics, finance, etc.)", "Undergraduate_Major"] = 'A Business Discipline'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Undergraduate_Major"] == "Information technology, networking, or system administration", "Undergraduate_Major"] = 'Information Technology'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Undergraduate_Major"] == "Medical or life sciences (biology, chemistry, medicine, etc.)", "Undergraduate_Major"] = 'Medical'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Undergraduate_Major"] == "Social sciences (anthropology, psychology, sociology, etc.)", "Undergraduate_Major"] = 'Social Sciences'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Undergraduate_Major"] == "Humanities (history, literature, philosophy, etc.)", "Undergraduate_Major"] = 'Humanities'
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Undergraduate_Major"] == "Environmental science or geology", "Undergraduate_Major"] = 'Environmental Science'

Field_1 = pd.Series({'Actual Value': 'Computer science (software engineering, etc.)',
                        'New Value': 'Computer Science',                     
                    })
Field_2 = pd.Series({'Actual Value': 'Engineering (non-computer focused)',
                        'New Value': 'Engineering',                     
                    })
Field_3 = pd.Series({'Actual Value': 'A business discipline (accounting, economics, finance, etc.)',
                        'New Value': 'A Business Discipline',                     
                    })
Field_4 = pd.Series({'Actual Value': 'Information technology, networking, or system administration',
                        'New Value': 'Information Technology',                     
                    })
Field_5 = pd.Series({'Actual Value': 'Medical or life sciences (biology, chemistry, medicine, etc.)',
                        'New Value': 'Medical',                     
                    })
Field_6 = pd.Series({'Actual Value': 'Social sciences (anthropology, psychology, sociology, etc.)',
                        'New Value': 'Social Sciences',                     
                    })
Field_7 = pd.Series({'Actual Value': 'Humanities (history, literature, philosophy, etc.)',
                        'New Value': 'Humanities',                     
                    })
Field_8 = pd.Series({'Actual Value': 'Environmental science or geology',
                        'New Value': 'Environmental Science',                     
                    })
FieldUpdateSummary = pd.DataFrame([Field_1,Field_2,Field_3,Field_4,Field_5,Field_6,Field_7,Field_8],index=['1','2','3','4','5','6','7','8'])
FieldUpdateSummary
MultipleChoiceDataSet.loc[MultipleChoiceDataSet["Current_Compensation"] == "I do not wish to disclose my approximate yearly compensation", "Current_Compensation"] = 'Confidential'

Field_1 = pd.Series({'Actual Value': 'I do not wish to disclose my approximate yearly compensation',
                        'New Value': 'Confidential',                     
                    })
FieldUpdateSummary = pd.DataFrame([Field_1],index=['1'])
FieldUpdateSummary
TotalMale = MultipleChoiceDataSet[MultipleChoiceDataSet['Gender']=='Male']['Gender'].value_counts()
TotalMalePercentage = round(TotalMale / len(MultipleChoiceDataSet.Gender) * 100,2)
TotalFemale = MultipleChoiceDataSet[MultipleChoiceDataSet['Gender']=='Female']['Gender'].value_counts()
TotalFemalePercentage = round(TotalFemale / len(MultipleChoiceDataSet.Gender) * 100,2)
TotalPreferNot = MultipleChoiceDataSet[MultipleChoiceDataSet['Gender']=='Prefer not to say']['Gender'].value_counts()
TotalPreferNotPercentage = round(TotalPreferNot / len(MultipleChoiceDataSet.Gender) * 100,2)
TotalSelf = MultipleChoiceDataSet[MultipleChoiceDataSet['Gender']=='Prefer to self-describe']['Gender'].value_counts()
TotalSelfPercentage = round(TotalSelf / len(MultipleChoiceDataSet.Gender) * 100,2)
TotalGenderPercentage = round(len(MultipleChoiceDataSet.Gender) / len(MultipleChoiceDataSet.Gender) * 100,2)

Field_1 = pd.Series({'Description': 'Male',
                        'Total Records': int(TotalMale.values),
                         'Percentage' : float(TotalMalePercentage.values),
                    })
Field_2 = pd.Series({'Description': 'Female',
                        'Total Records': int(TotalFemale.values),
                         'Percentage' : float(TotalFemalePercentage.values),                     
                    })
Field_3 = pd.Series({'Description': 'Prefer not to say',
                        'Total Records': int(TotalPreferNot.values),
                         'Percentage' : float(TotalPreferNotPercentage.values),                     
                    })
Field_4 = pd.Series({'Description': 'Prefer to self-describe',
                        'Total Records': int(TotalSelf.values),
                         'Percentage' : float(TotalSelfPercentage.values),                     
                    })
Field_5 = pd.Series({'Description': 'Total Gender',
                        'Total Records': MultipleChoiceDataSet['Gender'].count(),
                         'Percentage' : TotalGenderPercentage})

GenderSummary = pd.DataFrame([Field_1,Field_2,Field_3,Field_4,Field_5], index=['1','2','3','4','5'])
GenderSummary
TotalGender = MultipleChoiceDataSet["Gender"].value_counts()
labels = (np.array(TotalGender.index))
sizes = (np.array((TotalGender / TotalGender.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title="Gender")
dat = [trace]
fig = go.Figure(data=dat, layout=layout)
py.offline.iplot( fig, validate=False, filename='Gender' )
TotalMaleWiseCountry = MultipleChoiceDataSet[MultipleChoiceDataSet["Gender"]=='Male']['Country'].value_counts()
print("Male wise Country\n")
print(TotalMaleWiseCountry)
MultipleChoiceDataSet[MultipleChoiceDataSet["Gender"]=='Male']['Country'].value_counts().plot.bar(figsize=(20,5),legend=True,fontsize='16', color=['#6495ED'])
plt.title('Male wise Country\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Country', fontsize='16')
TotalFemaleWiseCountry = MultipleChoiceDataSet[MultipleChoiceDataSet["Gender"]=='Female']['Country'].value_counts()
print("Female wise Country\n")
print(TotalFemaleWiseCountry)
MultipleChoiceDataSet[MultipleChoiceDataSet["Gender"]=='Female']['Country'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#FFC0CB'])
plt.title('Female wise Country\n' ,fontsize='16')
plt.ylabel('Records' ,fontsize='16')
plt.xlabel('Country', fontsize='16')
TotalMaleAge = MultipleChoiceDataSet[MultipleChoiceDataSet["Gender"]=='Male']['Age'].value_counts()
print("Total Male Age Group\n")
print(TotalMaleAge)
MultipleChoiceDataSet[MultipleChoiceDataSet["Gender"]=='Male']['Age'].value_counts().plot.barh(figsize=(20,7),legend=True, fontsize='16', color=['#6495ED'])
plt.title('Male wise Age Group\n', fontsize='16')
plt.ylabel('Age Group' ,fontsize='16')
plt.xlabel('Records', fontsize='16')
TotalFemaleAge = MultipleChoiceDataSet[MultipleChoiceDataSet["Gender"]=='Female']['Age'].value_counts()
print("Total Female Age Group\n")
print(TotalFemaleAge)
MultipleChoiceDataSet[MultipleChoiceDataSet["Gender"]=='Female']['Age'].value_counts().plot.barh(figsize=(20,7),legend=True, fontsize='16', color=['#FFC0CB'])
plt.title('Female wise Age Group\n', fontsize='16')
plt.ylabel('Age Group', fontsize='16')
plt.xlabel('Records', fontsize='16')
print("Total Age Group")
MultipleChoiceDataSet['Age'].value_counts() 
plt.figure(figsize=(20,7))
MultipleChoiceDataSet['Age'].value_counts().plot(kind='barh',  legend=False, fontsize='16', color=['#9B59B6'])
plt.title('Age Group\n', fontsize='16')
plt.xlabel('Records', fontsize='16')
plt.ylabel('Age Group', fontsize='16')
TotalUndergraduateMajorAgeGroup = MultipleChoiceDataSet[MultipleChoiceDataSet["Age"]=='25-29']['Undergraduate_Major'].value_counts()
print("Undergraduate Major 25-29 Age Group\n")
print(TotalUndergraduateMajorAgeGroup)
MultipleChoiceDataSet[MultipleChoiceDataSet["Age"]=='25-29']['Undergraduate_Major'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#9B59B6'])
plt.title('Undergraduate Major 25-29 Age Group\n,', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Undergraduate Major', fontsize='16')
TotalHigherEducationAgeGroup = MultipleChoiceDataSet[MultipleChoiceDataSet["Age"]=='25-29']['Higher_Education'].value_counts()
print("Higher Education 25-29 Age Group\n")
print(TotalHigherEducationAgeGroup)
MultipleChoiceDataSet[MultipleChoiceDataSet["Age"]=='25-29']['Higher_Education'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#9B59B6'])
plt.title('Higher Education 25-29 Age Group\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Higher Education', fontsize='16')
print("Total Country Name" )
MultipleChoiceDataSet['Country'].value_counts()
plt.figure(figsize=(20,17))
MultipleChoiceDataSet['Country'].value_counts().plot(kind='barh',  legend=True, fontsize='16', color=['#336B87', '#90AFC5', '#763623'])
plt.title('Country\n', fontsize='16')
plt.xlabel('Records', fontsize='16')
plt.ylabel('Country', fontsize='16')
TotalCurrentRole = MultipleChoiceDataSet[MultipleChoiceDataSet["Country"]=='USA']['Current_Role'].value_counts()
print("Current Role in USA\n")
print(TotalCurrentRole)
MultipleChoiceDataSet[MultipleChoiceDataSet["Country"]=='USA']['Current_Role'].value_counts().plot.bar(figsize=(20,5),legend=True,fontsize='16',  color=['#336B87', '#90AFC5', '#763623'])
plt.title('Current Role in USA\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Current Role', fontsize='16')
TotalCurrentRole = MultipleChoiceDataSet[MultipleChoiceDataSet["Country"]=='India']['Current_Role'].value_counts()
print("Current Role in India\n")
print(TotalCurrentRole)
MultipleChoiceDataSet[MultipleChoiceDataSet["Country"]=='India']['Current_Role'].value_counts().plot.bar(figsize=(20,5),legend=True,fontsize='16', color=['#336B87', '#90AFC5', '#763623'])
plt.title('Current Role in India\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Current Role' , fontsize='16')
TotalCurrentRole = MultipleChoiceDataSet[MultipleChoiceDataSet["Country"]=='China']['Current_Role'].value_counts()
print("Current Role in China\n")
print(TotalCurrentRole)
MultipleChoiceDataSet[MultipleChoiceDataSet["Country"]=='China']['Current_Role'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#336B87', '#90AFC5', '#763623'])
plt.title('Current Role in China\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Current Role', fontsize='16')
TopThreeCountrySum = MultipleChoiceDataSet['Country'].value_counts()
TopThreeCountryPer = ((TopThreeCountrySum.head(3).sum())/(TopThreeCountrySum.sum()) * 100)
print("Total USA, India and China Records : ", TopThreeCountrySum.head(3).sum())
print("Total Country Records : ", len(MultipleChoiceDataSet['Country']))
print("USA, India and China Percentage : ", round(TopThreeCountryPer,2),"%")
plt.figure(figsize=(20,5))
MultipleChoiceDataSet['Country'].value_counts().tail(54).plot(kind='bar',  legend=True, fontsize='16', color=['#336B87', '#90AFC5', '#763623'])
plt.title('Rest of The Country\n', fontsize='16')
plt.xlabel('Country', fontsize='16')
plt.ylabel('Records', fontsize='16')
CountryGraph = MultipleChoiceDataSet['Country'].value_counts()
data = [ dict( type = 'choropleth',
            locations = CountryGraph.index,
            z = CountryGraph.values,
            text = CountryGraph.index,
            locationmode = 'country names',
            colorscale='Viridis',
            autocolorscale = False,
            reversescale = True,
            marker = dict( line = dict (
                    color = 'rgb(128,0,128)',width = 0.3
                ) ),
            colorbar = dict( autotick = False,
                title = 'Response'),
      ) ]

layout = dict(
    title = 'Kaggle Survey - Response - 2018',
    geo = dict(showland = True,
               landcolor = "rgb(95,158,160)",
        showframe = False,
        showcoastlines = True,
        projection = dict( type = 'Mercator')
    ))

fig = dict( data=data, layout=layout )
py.offline.iplot( fig, validate=False, filename='world-map' )
print("Higher Education Name and Count" )
MultipleChoiceDataSet['Higher_Education'].value_counts()
plt.figure(figsize=(20,5))
MultipleChoiceDataSet['Higher_Education'].value_counts().plot(kind='bar', fontsize='16', legend=True, color=['#45B39D'])
plt.title('Higher Education\n', fontsize='16')
plt.xlabel('Higher Education', fontsize='16')
plt.ylabel('Records', fontsize='16')
TotalMasterDegree = MultipleChoiceDataSet[MultipleChoiceDataSet["Higher_Education"]=='Master’s degree']['Country'].value_counts()
print("Country wise Master Degree\n")
print(TotalMasterDegree)
MultipleChoiceDataSet[MultipleChoiceDataSet["Higher_Education"]=='Master’s degree']['Country'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#5F9EA0'])
plt.title('Country wise Master Degree\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Country', fontsize='16')
TotalMasterDegree = MultipleChoiceDataSet[MultipleChoiceDataSet["Higher_Education"]=='Master’s degree']['Current_Role'].value_counts()
print("Current Role wise Master Degree\n")
print(TotalMasterDegree)
MultipleChoiceDataSet[MultipleChoiceDataSet["Higher_Education"]=='Master’s degree']['Current_Role'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#5F9EA0'])
plt.title('Current Role wise Master Degree\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Country', fontsize='16')
TotalMasterDegree = MultipleChoiceDataSet[MultipleChoiceDataSet["Higher_Education"]=='Master’s degree']['Current_Compensation'].value_counts()
print("Current Compensation wise Master Degree\n")
print(TotalMasterDegree)
MultipleChoiceDataSet[MultipleChoiceDataSet["Higher_Education"]=='Master’s degree']['Current_Compensation'].value_counts().plot.bar(figsize=(20,5),legend=True,fontsize='16',  color=['#5F9EA0'])
plt.title('Current Compensation wise Master Degree\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('\nCurrent Compensation', fontsize='16')
print("Undergraduate Major Name and Count" )
MultipleChoiceDataSet['Undergraduate_Major'].value_counts()
plt.figure(figsize=(20,5))
MultipleChoiceDataSet['Undergraduate_Major'].value_counts().plot(kind='bar',  legend=True,fontsize='16',  color=['#45B39D'])
plt.title('Undergraduate Major\n', fontsize='16')
plt.xlabel('Undergraduate Major', fontsize='16')
plt.ylabel('Records', fontsize='16')
TotalComputerScience = MultipleChoiceDataSet[MultipleChoiceDataSet["Undergraduate_Major"]=='Computer Science']['Country'].value_counts()
print("Country wise Computer Science\n")
print(TotalComputerScience)
MultipleChoiceDataSet[MultipleChoiceDataSet["Undergraduate_Major"]=='Computer Science']['Country'].value_counts().plot.bar(figsize=(20,5),legend=True,fontsize='16', color=['#45B39D'])
plt.title('Country wise Computer Science\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Country', fontsize='16')
TotalComputerScienceAge = MultipleChoiceDataSet[MultipleChoiceDataSet["Undergraduate_Major"]=='Computer Science']['Age'].value_counts()
print("Age Group wise Computer Science\n")
print(TotalComputerScienceAge)
MultipleChoiceDataSet[MultipleChoiceDataSet["Undergraduate_Major"]=='Computer Science']['Age'].value_counts().plot.bar(figsize=(20,5),legend=True,fontsize='16', color=['#45B39D'])
plt.title('Age Group wise Computer Science\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('\nAge Group', fontsize='16')
print("Current Role Value Count" )
MultipleChoiceDataSet['Current_Role'].value_counts()
plt.figure(figsize=(20,8))
MultipleChoiceDataSet['Current_Role'].value_counts().plot(kind='barh',  legend=True,fontsize='16', color=['#707B7C'])
plt.title('Current Role\n', fontsize='16')
plt.xlabel('\nCurrent Role',fontsize='16')
plt.ylabel('Records', fontsize='16')
TotalStudent = MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Role"]=='Student']['Gender'].value_counts()
print("Gender wise Student\n")
print(TotalStudent)
MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Role"]=='Student']['Gender'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#707B7C'])
plt.title('Gender wise Student\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Gender', fontsize='16')
TotalDataScientist = MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Role"]=='Data Scientist']['Gender'].value_counts()
print("Data Scientist wise Gender\n")
print(TotalDataScientist)
MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Role"]=='Data Scientist']['Gender'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#707B7C'])
plt.title('Data Scientist wise Gender\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Gender', fontsize='16')
TotalStudent = MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Role"]=='Student']['Higher_Education'].value_counts()
print("Age Group wise Student\n")
print(TotalStudent)
MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Role"]=='Student']['Higher_Education'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#707B7C'])
plt.title('Age Group wise Student\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Age Group', fontsize='16')
TotalDataScientist = MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Role"]=='Data Scientist']['Age'].value_counts()
print("Age Group wise Data Scientist\n")
print(TotalDataScientist)
MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Role"]=='Data Scientist']['Age'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#707B7C'])
plt.title('Age Group wise Data Scientist\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('\nAge Group', fontsize='16')
TotalStdent = MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Role"]=='Data Scientist']['Years_Experience'].value_counts()
print("Years Experience wise Data Scientist\n")
print(TotalStdent)
MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Role"]=='Data Scientist']['Years_Experience'].value_counts().plot.bar(figsize=(20,5),legend=True,fontsize='16',  color=['#707B7C'])
plt.title('Years Experience wise Data Scientist\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('\nYears Experience', fontsize='16')
print("Current Industry Value Count" )
MultipleChoiceDataSet['Current_Industry'].value_counts()
plt.figure(figsize=(20,5))
MultipleChoiceDataSet['Current_Industry'].value_counts().plot(kind='bar',  legend=True, fontsize='16', color=['#800080'])
plt.title('Current Industry\n', fontsize='16')
plt.xlabel('Current Industry', fontsize='16')
plt.ylabel('Records', fontsize='16')
TotalComputerTechnology = MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Industry"]=='Computers/Technology']['Age'].value_counts()
print("Computers/Technology\n")
print(TotalComputerTechnology)
MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Industry"]=='Computers/Technology']['Age'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#800080'])
plt.title('Computers/Technology\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('\nAge Group', fontsize='16')
TotalComputerTechnology = MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Industry"]=='Computers/Technology']['Country'].value_counts()
print("Computers/Technology\n")
print(TotalComputerTechnology)
MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Industry"]=='Computers/Technology']['Country'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#800080'])
plt.title('Computers/Technology\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Country', fontsize='16')
print("Years Experience Value Count" )
MultipleChoiceDataSet['Years_Experience'].value_counts()
plt.figure(figsize=(20,5))
MultipleChoiceDataSet['Years_Experience'].value_counts().plot(kind='bar',  legend=True, fontsize='16', color=['#45B39D'])
plt.title('Total Years Experience Wise Records\n',fontsize='16')
plt.xlabel('\nYears Experience', fontsize='16')
plt.ylabel('Records', fontsize='16')
TotalCountry = MultipleChoiceDataSet[MultipleChoiceDataSet["Years_Experience"]=='0-1']['Current_Industry'].value_counts()
print("Years Experience\n")
print(TotalCountry)
MultipleChoiceDataSet[MultipleChoiceDataSet["Years_Experience"]=='0-1']['Current_Industry'].value_counts().plot.bar(figsize=(20,5),legend=True, fontsize='16', color=['#5F9EA0'])
plt.title('Years Experience\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Current Industry', fontsize='16')
TotalCountry = MultipleChoiceDataSet[MultipleChoiceDataSet["Years_Experience"]=='0-1']['Country'].value_counts()
print("Years Experience\n")
print(TotalCountry)
MultipleChoiceDataSet[MultipleChoiceDataSet["Years_Experience"]=='0-1']['Country'].value_counts().plot.bar(figsize=(20,5),legend=True,fontsize='16',  color=['#5F9EA0'])
plt.title('Years Experience\n', fontsize='16')
plt.ylabel('Records', fontsize='16')
plt.xlabel('Country', fontsize='16')
print("Current Compensation Value Count" )
MultipleChoiceDataSet['Current_Compensation'].value_counts()
TotalCurrentCompensation = MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Compensation"]!='Confidential']['Current_Compensation'].value_counts()
print("Current Compensation without Confidential\n")
print(TotalCurrentCompensation)
MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Compensation"]!='Confidential']['Current_Compensation'].value_counts().plot.bar(figsize=(20,5), fontsize='16')
plt.title('Current Compensation without Confidential\n',  fontsize='16')
plt.ylabel('Amount in USD($)', fontsize='16')
plt.xlabel('\nCurrent Compensation', fontsize='16')
TotalConfidencial = MultipleChoiceDataSet['Current_Compensation'] == 'Confidential'
TotalCurrentCompensation = len(MultipleChoiceDataSet['Current_Compensation'])
print("Total Current Compensation Records: ", TotalCurrentCompensation)
print("Total Confidencial: ", TotalConfidencial.sum())
TotalCurrentCompensationPercentage = (round((TotalConfidencial.sum() / TotalCurrentCompensation) *100,2))
print("Total Confidencial Percentage:", TotalCurrentCompensationPercentage, '%')
TotalCountry = MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Compensation"]!='Confidential']['Country'].value_counts()
print("Country wise Current Compensation without Confidential\n")
print(TotalCountry)
MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Compensation"]!='Confidential']['Country'].value_counts().plot.bar(figsize=(20,5), fontsize='16')
plt.title('Country wise Current Compensation without Confidential\n', fontsize='16')
plt.ylabel('Amount in USD($)', fontsize='16')
plt.xlabel('Country', fontsize='16')
TotalCurrentRole = MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Compensation"]!='Confidential']['Current_Role'].value_counts()
print("Current Role wise Current Compensation without Confidential\n")
print(TotalCurrentRole)
MultipleChoiceDataSet[MultipleChoiceDataSet["Current_Compensation"]!='Confidential']['Current_Role'].value_counts().plot.bar(figsize=(20,5),fontsize='16')
plt.title('Current Role Current Compensation without Confidential\n', fontsize='16')
plt.ylabel('Amount in USD($)', fontsize='16')
plt.xlabel('Current Role', fontsize='16')