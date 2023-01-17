import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import squarify
from itertools import chain
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:85% !important; }</style>"))

# return lista de palabras resultantes de partir de una cadena por un separador
def chainer(s):
    return list(chain.from_iterable(s.str.split(';')))

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:,})'.format(p=pct,v=val)
    return my_autopct

#Dataset con los datos de la encuesta
df_survey=pd.read_csv('../input/stack-overflow-2018-developer-survey/survey_results_public.csv',dtype={'Column': str,'Respondent': int,'Hobby': str,'OpenSource': str,'Country': str,'Student': str,'Employment': str,'FormalEducation': str,'UndergradMajor': str,'CompanySize': str,'DevType': str,'YearsCoding': str,'YearsCodingProf': str,'JobSatisfaction': str,'CareerSatisfaction': str,'HopeFiveYears': str,'JobSearchStatus': str,'LastNewJob': str,'AssessJob1': str,'AssessJob2': str,'AssessJob3': str,'AssessJob4': str,'AssessJob5': str,'AssessJob6': str,'AssessJob7': str,'AssessJob8': str,'AssessJob9': str,'AssessJob10': str,'AssessBenefits1': str,'AssessBenefits2': str,'AssessBenefits3': str,'AssessBenefits4': str,'AssessBenefits5': str,'AssessBenefits6': str,'AssessBenefits7': str,'AssessBenefits8': str,'AssessBenefits9': str,'AssessBenefits10': str,'AssessBenefits11': str,'JobContactPriorities1': str,'JobContactPriorities2': str,'JobContactPriorities3': str,'JobContactPriorities4': str,'JobContactPriorities5': str,'JobEmailPriorities1': str,'JobEmailPriorities2': str,'JobEmailPriorities3': str,'JobEmailPriorities4': str,'JobEmailPriorities5': str,'JobEmailPriorities6': str,'JobEmailPriorities7': str,'UpdateCV': str,'Currency': str,'Salary': str,'SalaryType': str,'ConvertedSalary': str,'CurrencySymbol': str,'CommunicationTools': str,'TimeFullyProductive': str,'EducationTypes': str,'SelfTaughtTypes': str,'TimeAfterBootcamp': str,'HackathonReasons': str,'AgreeDisagree1': str,'AgreeDisagree2': str,'AgreeDisagree3': str,'LanguageWorkedWith': str,'LanguageDesireNextYear': str,'DatabaseWorkedWith': str,'DatabaseDesireNextYear': str,'PlatformWorkedWith': str,'PlatformDesireNextYear': str,'FrameworkWorkedWith': str,'FrameworkDesireNextYear': str,'IDE': str,'OperatingSystem': str,'NumberMonitors': str,'Methodology': str,'VersionControl': str,'CheckInCode': str,'AdBlocker': str,'AdBlockerDisable': str,'AdBlockerReasons': str,'AdsAgreeDisagree1': str,'AdsAgreeDisagree2': str,'AdsAgreeDisagree3': str,'AdsActions': str,'AdsPriorities1': str,'AdsPriorities2': str,'AdsPriorities3': str,'AdsPriorities4': str,'AdsPriorities5': str,'AdsPriorities6': str,'AdsPriorities7': str,'AIDangerous': str,'AIInteresting': str,'AIResponsible': str,'AIFuture': str,'EthicsChoice': str,'EthicsReport': str,'EthicsResponsible': str,'EthicalImplications': str,'StackOverflowRecommend': str,'StackOverflowVisit': str,'StackOverflowHasAccount': str,'StackOverflowParticipate': str,'StackOverflowJobs': str,'StackOverflowDevStory': str,'StackOverflowJobsRecommend': str,'StackOverflowConsiderMember': str,'HypotheticalTools1': str,'HypotheticalTools2': str,'HypotheticalTools3': str,'HypotheticalTools4': str,'HypotheticalTools5': str,'WakeTime': str,'HoursComputer': str,'HoursOutside': str,'SkipMeals': str,'ErgonomicDevices': str,'Exercise': str,'Gender': str,'SexualOrientation': str,'EducationParents': str,'RaceEthnicity': str,'Age': str,'Dependents': str,'MilitaryUS': str,'SurveyTooLong': str,'SurveyEasy':str})
#Dataset con los datos de países y continentes para el análisis por regiones
df_countries_continents=pd.read_csv('../input/countries-and-continents/countries_and_continents.csv', keep_default_na=False)

df_survey['Age']=df_survey['Age'].apply(lambda x: 'No response' if pd.isna(x) else x)
df_ages=df_survey.groupby(['Age'])['Respondent'].agg(['count']).reset_index()
df_ages=df_ages.rename(columns={'count':'Total'})
df_ages['Age'].replace(['18 - 24 years old'], '18 - 24', inplace=True)
df_ages['Age'].replace(['25 - 34 years old'], '25 - 34', inplace=True)
df_ages['Age'].replace(['35 - 44 years old'], '35 - 44', inplace=True)
df_ages['Age'].replace(['45 - 54 years old'], '45 - 54', inplace=True)
df_ages['Age'].replace(['55 - 64 years old'], '55 - 64', inplace=True)
df_ages['Age'].replace(['65 years or older'], '> 65', inplace=True)
df_ages['Age'].replace(['Under 18 years old'], '< 18', inplace=True)
df_ages = df_ages.reindex([7,0,1,2,3,4,5,6])

fig, ax = plt.subplots(1,2,figsize=(24, 8))
ax[0].pie(df_ages['Total'], labels=df_ages['Age'], autopct=make_autopct(df_ages['Total']))
ax[0].set_title('Rango de edades de los participantes')

# Elimino del dataset los que no rellenaron el campo edad
df_ages_filtered=df_ages[df_ages['Age']!='No response']

x = df_ages_filtered['Age']
y = df_ages_filtered['Total']

ax[1].bar(x, y, color = 'green')
ax[1].set_title('Participantes por rango de edad (no se muestran los que no votaron)')
print()
df_survey_by_gender=df_survey[pd.isna(df_survey.Gender)==False]
lens = df_survey_by_gender['Gender'].str.split(';').map(len)
df_survey_by_gender = pd.DataFrame({'Respondent': np.repeat(df_survey_by_gender['Respondent'], lens), 'Gender': chainer(df_survey_by_gender['Gender'])})
df_survey_by_gender = df_survey_by_gender.groupby(['Gender'])['Respondent'].agg(['count']).reset_index()
df_survey_by_gender = df_survey_by_gender[(df_survey_by_gender.Gender!='Non-binary, genderqueer, or gender non-conforming') & (df_survey_by_gender.Gender!='Transgender')]

df_education=df_survey.groupby(['FormalEducation'])['Respondent'].agg(['count']).reset_index()

x = df_survey_by_gender['Gender']
y = df_survey_by_gender['count']

fig, ax = plt.subplots(1,2,figsize=(24, 12))
ax[0].bar(x, y, color = 'skyblue')
ax[0].set_title('Participantes por género')

ax[1].pie(df_education['count'], labels=df_education['FormalEducation'], autopct=make_autopct(df_education['count']))
ax[1].set_title('Formación académica')
print()
df_employments=df_survey.groupby(['Employment'])['Respondent'].agg(['count']).reset_index()
df_survey['DevType']=df_survey['DevType'].apply(lambda x: 'No response' if pd.isna(x) else x)
lens = df_survey['DevType'].str.split(';').map(len)

res = pd.DataFrame({'Respondent': np.repeat(df_survey['Respondent'], lens), 'DevType': chainer(df_survey['DevType'])})

res=res.groupby(['DevType'])['Respondent'].agg(['count'])
res=res.sort_values(by='count', ascending=False).reset_index()

fig, ax = plt.subplots(1,2,figsize=(24, 8))
ax[0].pie(df_employments['count'], labels=df_employments['Employment'], autopct=make_autopct(df_employments['count']))
ax[0].set_title('Situación profesional')

x = df_ages['Age']
y = df_ages['Total']

ax[1].pie(res['count'], labels=res['DevType'], autopct='%1.2f%%')
ax[1].set_title('Ocupaciones profesionales')
print()
df_assessjob1 = df_survey.groupby(['AssessJob1'])['Respondent'].agg(['count']).reset_index()
df_assessjob1 = df_assessjob1.reindex([0,2,3,4,5,6,7,8,9,1])
df_assessjob1 = df_assessjob1.set_index('AssessJob1')
df_assessjob1=df_assessjob1.rename(columns={'count':'TotalSum'})
total=df_assessjob1['TotalSum'].sum()
df_assessjob1['pct']=df_assessjob1.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessjob2 = df_survey.groupby(['AssessJob2'])['Respondent'].agg(['count']).reset_index()
df_assessjob2 = df_assessjob2.reindex([0,2,3,4,5,6,7,8,9,1])
df_assessjob2 = df_assessjob2.set_index('AssessJob2')
df_assessjob2=df_assessjob2.rename(columns={'count':'TotalSum'})
total=df_assessjob2['TotalSum'].sum()
df_assessjob2['pct']=df_assessjob2.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessjob3 = df_survey.groupby(['AssessJob3'])['Respondent'].agg(['count']).reset_index()
df_assessjob3 = df_assessjob3.reindex([0,2,3,4,5,6,7,8,9,1])
df_assessjob3 = df_assessjob3.set_index('AssessJob3')
df_assessjob3=df_assessjob3.rename(columns={'count':'TotalSum'})
total=df_assessjob3['TotalSum'].sum()
df_assessjob3['pct']=df_assessjob3.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessjob4 = df_survey.groupby(['AssessJob4'])['Respondent'].agg(['count']).reset_index()
df_assessjob4 = df_assessjob4.reindex([0,2,3,4,5,6,7,8,9,1])
df_assessjob4 = df_assessjob4.set_index('AssessJob4')
df_assessjob4=df_assessjob4.rename(columns={'count':'TotalSum'})
total=df_assessjob4['TotalSum'].sum()
df_assessjob4['pct']=df_assessjob4.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessjob5 = df_survey.groupby(['AssessJob5'])['Respondent'].agg(['count']).reset_index()
df_assessjob5 = df_assessjob5.reindex([0,2,3,4,5,6,7,8,9,1])
df_assessjob5 = df_assessjob5.set_index('AssessJob5')
df_assessjob5=df_assessjob5.rename(columns={'count':'TotalSum'})
total=df_assessjob5['TotalSum'].sum()
df_assessjob5['pct']=df_assessjob5.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessjob6 = df_survey.groupby(['AssessJob6'])['Respondent'].agg(['count']).reset_index()
df_assessjob6 = df_assessjob6.reindex([0,2,3,4,5,6,7,8,9,1])
df_assessjob6 = df_assessjob6.set_index('AssessJob6')
df_assessjob6=df_assessjob6.rename(columns={'count':'TotalSum'})
total=df_assessjob6['TotalSum'].sum()
df_assessjob6['pct']=df_assessjob6.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessjob7 = df_survey.groupby(['AssessJob7'])['Respondent'].agg(['count']).reset_index()
df_assessjob7 = df_assessjob7.reindex([0,2,3,4,5,6,7,8,9,1])
df_assessjob7 = df_assessjob7.set_index('AssessJob7')
df_assessjob7=df_assessjob7.rename(columns={'count':'TotalSum'})
total=df_assessjob7['TotalSum'].sum()
df_assessjob7['pct']=df_assessjob7.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessjob8 = df_survey.groupby(['AssessJob8'])['Respondent'].agg(['count']).reset_index()
df_assessjob8 = df_assessjob8.reindex([0,2,3,4,5,6,7,8,9,1])
df_assessjob8 = df_assessjob8.set_index('AssessJob8')
df_assessjob8=df_assessjob8.rename(columns={'count':'TotalSum'})
total=df_assessjob8['TotalSum'].sum()
df_assessjob8['pct']=df_assessjob8.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessjob9 = df_survey.groupby(['AssessJob9'])['Respondent'].agg(['count']).reset_index()
df_assessjob9 = df_assessjob9.reindex([0,2,3,4,5,6,7,8,9,1])
df_assessjob9 = df_assessjob9.set_index('AssessJob9')
df_assessjob9=df_assessjob9.rename(columns={'count':'TotalSum'})
total=df_assessjob9['TotalSum'].sum()
df_assessjob9['pct']=df_assessjob9.apply(lambda row: row.TotalSum*100/total, axis=1)
fig, ax =  plt.subplots(figsize=(15, 8))

# Pintamos cada una de las series
ax.plot(df_assessjob1['pct'], color = 'red', label='Industria en la que trabajar')
ax.plot(df_assessjob2['pct'], color = 'green', label='Comportamiento financiero y situación económica de la empresa')
ax.plot(df_assessjob3['pct'], color = 'blue', label='Departamento o equipo en el que voy a trabajar')
ax.plot(df_assessjob4['pct'], color = 'yellow', label='Lenguajes, frameworks, y tecnologías')
ax.plot(df_assessjob5['pct'], color = 'skyblue', label='Compensación y beneficios')
ax.plot(df_assessjob6['pct'], color = 'pink', label='Ambiente y cultura de la empresa')
ax.plot(df_assessjob7['pct'], color = 'orange', label='Posibilidad de teletrabajar')
ax.plot(df_assessjob8['pct'], color = 'grey', label='Oportunidades de desarrollo profesional')
ax.plot(df_assessjob9['pct'], color = 'black', label='Diversidad cultural de la empresa')
# Añadimos Formato al gráfico
ax.set_title('Distribución global de las preferencias a la hora de valorar un cambio de trabajo')
ax.set_xlabel('Importancia. 1 lo más importante, 10 lo menos importante')
ax.set_ylabel('Procentaje de votos')
ax.grid(True)
plt.legend()
print()
df_countries_continents=df_countries_continents.drop(['official_name_en','official_name_en','official_name_fr','ISO3166-1-Alpha-2','ISO3166-1-Alpha-3','M49','ITU','MARC','WMO','DS','Dial','FIFA','FIPS','GAUL','IOC','ISO4217-currency_alphabetic_code','ISO4217-currency_country_name','ISO4217-currency_minor_unit','ISO4217-currency_name','ISO4217-currency_numeric_code','is_independent','Capital','TLD','Languages','Geoname ID','EDGAR'],axis=1)
df_countries_continents=df_countries_continents[pd.isna(df_countries_continents.name)==False]
df_countries_continents=df_countries_continents[pd.isna(df_countries_continents.Continent)==False]
df_countries_continents=df_countries_continents.rename(columns={'name':'Country'})
df_survey['Country'].replace(['United States'], 'US', inplace=True)
#Ahora toca hacer el join para incluir la información por continentes
df_survey_joined = pd.merge(df_survey, df_countries_continents, how='inner', on=['Country'])
def generateForContinentAndAS(df_survey_continentAS_, continent, assessjob):
    df_survey_continent_ = df_survey_continentAS_[df_survey_continentAS_.Continent==continent]
    df_survey_continent_ = df_survey_continent_.sort_values(by=assessjob, ascending=True)
    df_survey_continent_[assessjob]=pd.to_numeric(df_survey_continent_[assessjob])
    df_survey_continent_=df_survey_continent_.sort_values(by=assessjob)
    df_survey_continent_=df_survey_continent_.set_index(assessjob)
    df_survey_continent_=df_survey_continent_.rename(columns={'count':'TotalSum'})
    total_=df_survey_continent_['TotalSum'].sum()
    df_survey_continent_['pct']=df_survey_continent_.apply(lambda row: row.TotalSum*100/total_, axis=1)
    return df_survey_continent_
df_survey_continentAS1 = df_survey_joined.groupby(['Continent','AssessJob1'])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AS1_US = generateForContinentAndAS(df_survey_continentAS1,"NA",'AssessJob1')
df_survey_continent_AS1_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob1')
df_survey_continent_AS1_AF = generateForContinentAndAS(df_survey_continentAS1,"AF",'AssessJob1')
df_survey_continent_AS1_OC = generateForContinentAndAS(df_survey_continentAS1,"OC",'AssessJob1')
df_survey_continent_AS1_AS = generateForContinentAndAS(df_survey_continentAS1,"AS",'AssessJob1')
df_survey_continent_AS1_SA = generateForContinentAndAS(df_survey_continentAS1,"SA",'AssessJob1')
df_survey_continentAS1 = df_survey_joined.groupby(['Continent','AssessJob2'])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AS2_US = generateForContinentAndAS(df_survey_continentAS1,"NA",'AssessJob2')
df_survey_continent_AS2_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob2')
df_survey_continent_AS2_AF = generateForContinentAndAS(df_survey_continentAS1,"AF",'AssessJob2')
df_survey_continent_AS2_OC = generateForContinentAndAS(df_survey_continentAS1,"OC",'AssessJob2')
df_survey_continent_AS2_AS = generateForContinentAndAS(df_survey_continentAS1,"AS",'AssessJob2')
df_survey_continent_AS2_SA = generateForContinentAndAS(df_survey_continentAS1,"SA",'AssessJob2')
df_survey_continentAS1 = df_survey_joined.groupby(['Continent','AssessJob3'])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AS3_US = generateForContinentAndAS(df_survey_continentAS1,"NA",'AssessJob3')
df_survey_continent_AS3_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob3')
df_survey_continent_AS3_AF = generateForContinentAndAS(df_survey_continentAS1,"AF",'AssessJob3')
df_survey_continent_AS3_OC = generateForContinentAndAS(df_survey_continentAS1,"OC",'AssessJob3')
df_survey_continent_AS3_AS = generateForContinentAndAS(df_survey_continentAS1,"AS",'AssessJob3')
df_survey_continent_AS3_SA = generateForContinentAndAS(df_survey_continentAS1,"SA",'AssessJob3')
#Mostramos las tres gráficas anteriores juntas

fig, ax = plt.subplots(1,3,figsize=(24, 8))
ax[0].plot(df_survey_continent_AS1_US['pct'], color = 'red', label='North America')
ax[0].plot(df_survey_continent_AS1_EU['pct'], color = 'green', label='Europe')
ax[0].plot(df_survey_continent_AS1_AF['pct'], color = 'blue', label='Africa')
ax[0].plot(df_survey_continent_AS1_AS['pct'], color = 'orange', label='Asia')
ax[0].plot(df_survey_continent_AS1_OC['pct'], color = 'black', label='Oceania')
ax[0].plot(df_survey_continent_AS1_SA['pct'], color = 'gray', label='South America')
ax[0].set_title('Industria en la que trabajar')
ax[0].set_xlabel('Importancia. 1 lo más importante, 10 lo menos importante')
ax[0].set_ylabel('Porcentaje de votos')
ax[0].grid(True)

ax[1].plot(df_survey_continent_AS2_US['pct'], color = 'red', label='North America')
ax[1].plot(df_survey_continent_AS2_EU['pct'], color = 'green', label='Europe')
ax[1].plot(df_survey_continent_AS2_AF['pct'], color = 'blue', label='Africa')
ax[1].plot(df_survey_continent_AS2_AS['pct'], color = 'orange', label='Asia')
ax[1].plot(df_survey_continent_AS2_OC['pct'], color = 'black', label='Oceania')
ax[1].plot(df_survey_continent_AS2_SA['pct'], color = 'gray', label='South America')
ax[1].set_title('Comportamiento financiero y situación económica de la empresa')
ax[1].set_xlabel('Importancia. 1 lo más importante, 10 lo menos importante')
ax[1].set_ylabel('Porcentaje de votos')
ax[1].grid(True)

ax[2].plot(df_survey_continent_AS3_US['pct'], color = 'red', label='North America')
ax[2].plot(df_survey_continent_AS3_EU['pct'], color = 'green', label='Europe')
ax[2].plot(df_survey_continent_AS3_AF['pct'], color = 'blue', label='Africa')
ax[2].plot(df_survey_continent_AS3_AS['pct'], color = 'orange', label='Asia')
ax[2].plot(df_survey_continent_AS3_OC['pct'], color = 'black', label='Oceania')
ax[2].plot(df_survey_continent_AS3_SA['pct'], color = 'gray', label='South America')
ax[2].set_title('El equipo o departamento para el que trabajaría')
ax[2].set_xlabel('Importancia. 1 lo más importante, 10 lo menos importante')
ax[2].set_ylabel('Porcentaje de votos')
ax[2].grid(True)
df_survey_continentAS1 = df_survey_joined.groupby(['Continent','AssessJob4'])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AS4_US = generateForContinentAndAS(df_survey_continentAS1,"NA",'AssessJob4')
df_survey_continent_AS4_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob4')
df_survey_continent_AS4_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob4')
df_survey_continent_AS4_AF = generateForContinentAndAS(df_survey_continentAS1,"AF",'AssessJob4')
df_survey_continent_AS4_OC = generateForContinentAndAS(df_survey_continentAS1,"OC",'AssessJob4')
df_survey_continent_AS4_AS = generateForContinentAndAS(df_survey_continentAS1,"AS",'AssessJob4')
df_survey_continent_AS4_SA = generateForContinentAndAS(df_survey_continentAS1,"SA",'AssessJob4')
df_survey_continentAS1 = df_survey_joined.groupby(['Continent','AssessJob5'])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AS5_US = generateForContinentAndAS(df_survey_continentAS1,"NA",'AssessJob5')
df_survey_continent_AS5_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob5')
df_survey_continent_AS5_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob5')
df_survey_continent_AS5_AF = generateForContinentAndAS(df_survey_continentAS1,"AF",'AssessJob5')
df_survey_continent_AS5_OC = generateForContinentAndAS(df_survey_continentAS1,"OC",'AssessJob5')
df_survey_continent_AS5_AS = generateForContinentAndAS(df_survey_continentAS1,"AS",'AssessJob5')
df_survey_continent_AS5_SA = generateForContinentAndAS(df_survey_continentAS1,"SA",'AssessJob5')
df_survey_continentAS1 = df_survey_joined.groupby(['Continent','AssessJob6'])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AS6_US = generateForContinentAndAS(df_survey_continentAS1,"NA",'AssessJob6')
df_survey_continent_AS6_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob6')
df_survey_continent_AS6_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob6')
df_survey_continent_AS6_AF = generateForContinentAndAS(df_survey_continentAS1,"AF",'AssessJob6')
df_survey_continent_AS6_OC = generateForContinentAndAS(df_survey_continentAS1,"OC",'AssessJob6')
df_survey_continent_AS6_AS = generateForContinentAndAS(df_survey_continentAS1,"AS",'AssessJob6')
df_survey_continent_AS6_SA = generateForContinentAndAS(df_survey_continentAS1,"SA",'AssessJob6')
#Pintamos las tres gráficas anteriores juntas

fig, ax = plt.subplots(1,3,figsize=(24, 8))
ax[0].plot(df_survey_continent_AS4_US['pct'], color = 'red', label='North America')
ax[0].plot(df_survey_continent_AS4_EU['pct'], color = 'green', label='Europe')
ax[0].plot(df_survey_continent_AS4_AF['pct'], color = 'blue', label='Africa')
ax[0].plot(df_survey_continent_AS4_AS['pct'], color = 'orange', label='Asia')
ax[0].plot(df_survey_continent_AS4_OC['pct'], color = 'black', label='Oceania')
ax[0].plot(df_survey_continent_AS4_SA['pct'], color = 'gray', label='South America')
ax[0].set_title('Lenguajes, frameworks y tecnologías')
ax[0].set_xlabel('Importancia. 1 lo más importante, 10 lo menos importante')
ax[0].set_ylabel('Porcentaje de votos')
ax[0].grid(True)

ax[1].plot(df_survey_continent_AS5_US['pct'], color = 'red', label='North America')
ax[1].plot(df_survey_continent_AS5_EU['pct'], color = 'green', label='Europe')
ax[1].plot(df_survey_continent_AS5_AF['pct'], color = 'blue', label='Africa')
ax[1].plot(df_survey_continent_AS5_AS['pct'], color = 'orange', label='Asia')
ax[1].plot(df_survey_continent_AS5_OC['pct'], color = 'black', label='Oceania')
ax[1].plot(df_survey_continent_AS5_SA['pct'], color = 'gray', label='South America')
ax[1].set_title('Compensación y beneficios')
ax[1].set_xlabel('Importancia. 1 lo más importante, 10 lo menos importante')
ax[1].set_ylabel('Porcentaje de votos')
ax[1].grid(True)

ax[2].plot(df_survey_continent_AS6_US['pct'], color = 'red', label='North America')
ax[2].plot(df_survey_continent_AS6_EU['pct'], color = 'green', label='Europe')
ax[2].plot(df_survey_continent_AS6_AF['pct'], color = 'blue', label='Africa')
ax[2].plot(df_survey_continent_AS6_AS['pct'], color = 'orange', label='Asia')
ax[2].plot(df_survey_continent_AS6_OC['pct'], color = 'black', label='Oceania')
ax[2].plot(df_survey_continent_AS6_SA['pct'], color = 'gray', label='South America')
ax[2].set_title('Ambiente y cultura empresarial')
ax[2].set_xlabel('Importancia. 1 lo más importante, 10 lo menos importante')
ax[2].set_ylabel('Porcentaje de votos')
ax[2].grid(True)
df_survey_continentAS1 = df_survey_joined.groupby(['Continent','AssessJob7'])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AS7_US = generateForContinentAndAS(df_survey_continentAS1,"NA",'AssessJob7')
df_survey_continent_AS7_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob7')
df_survey_continent_AS7_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob7')
df_survey_continent_AS7_AF = generateForContinentAndAS(df_survey_continentAS1,"AF",'AssessJob7')
df_survey_continent_AS7_OC = generateForContinentAndAS(df_survey_continentAS1,"OC",'AssessJob7')
df_survey_continent_AS7_AS = generateForContinentAndAS(df_survey_continentAS1,"AS",'AssessJob7')
df_survey_continent_AS7_SA = generateForContinentAndAS(df_survey_continentAS1,"SA",'AssessJob7')
df_survey_continentAS1 = df_survey_joined.groupby(['Continent','AssessJob8'])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AS8_US = generateForContinentAndAS(df_survey_continentAS1,"NA",'AssessJob8')
df_survey_continent_AS8_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob8')
df_survey_continent_AS8_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob8')
df_survey_continent_AS8_AF = generateForContinentAndAS(df_survey_continentAS1,"AF",'AssessJob8')
df_survey_continent_AS8_OC = generateForContinentAndAS(df_survey_continentAS1,"OC",'AssessJob8')
df_survey_continent_AS8_AS = generateForContinentAndAS(df_survey_continentAS1,"AS",'AssessJob8')
df_survey_continent_AS8_SA = generateForContinentAndAS(df_survey_continentAS1,"SA",'AssessJob8')
df_survey_continentAS1 = df_survey_joined.groupby(['Continent','AssessJob9'])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AS9_US = generateForContinentAndAS(df_survey_continentAS1,"NA",'AssessJob9')
df_survey_continent_AS9_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob9')
df_survey_continent_AS9_EU = generateForContinentAndAS(df_survey_continentAS1,"EU",'AssessJob9')
df_survey_continent_AS9_AF = generateForContinentAndAS(df_survey_continentAS1,"AF",'AssessJob9')
df_survey_continent_AS9_OC = generateForContinentAndAS(df_survey_continentAS1,"OC",'AssessJob9')
df_survey_continent_AS9_AS = generateForContinentAndAS(df_survey_continentAS1,"AS",'AssessJob9')
df_survey_continent_AS9_SA = generateForContinentAndAS(df_survey_continentAS1,"SA",'AssessJob9')
#Pintamos las tres gráficas anteriores juntas

fig, ax = plt.subplots(1,3,figsize=(24, 8))
ax[0].plot(df_survey_continent_AS7_US['pct'], color = 'red', label='North America')
ax[0].plot(df_survey_continent_AS7_EU['pct'], color = 'green', label='Europe')
ax[0].plot(df_survey_continent_AS7_AF['pct'], color = 'blue', label='Africa')
ax[0].plot(df_survey_continent_AS7_AS['pct'], color = 'orange', label='Asia')
ax[0].plot(df_survey_continent_AS7_OC['pct'], color = 'black', label='Oceania')
ax[0].plot(df_survey_continent_AS7_SA['pct'], color = 'gray', label='South America')
ax[0].set_title('Posibilidad de teletrabajar')
ax[0].set_xlabel('Importancia. 1 lo más importante, 10 lo menos importante')
ax[0].set_ylabel('Porcentaje de votos')
ax[0].grid(True)

ax[1].plot(df_survey_continent_AS8_US['pct'], color = 'red', label='North America')
ax[1].plot(df_survey_continent_AS8_EU['pct'], color = 'green', label='Europe')
ax[1].plot(df_survey_continent_AS8_AF['pct'], color = 'blue', label='Africa')
ax[1].plot(df_survey_continent_AS8_AS['pct'], color = 'orange', label='Asia')
ax[1].plot(df_survey_continent_AS8_OC['pct'], color = 'black', label='Oceania')
ax[1].plot(df_survey_continent_AS8_SA['pct'], color = 'gray', label='South America')
ax[1].set_title('Oportunidades de desarrollo profesional')
ax[1].set_xlabel('Importancia. 1 lo más importante, 10 lo menos importante')
ax[1].set_ylabel('Porcentaje de votos')
ax[1].grid(True)

ax[2].plot(df_survey_continent_AS9_US['pct'], color = 'red', label='North America')
ax[2].plot(df_survey_continent_AS9_EU['pct'], color = 'green', label='Europe')
ax[2].plot(df_survey_continent_AS9_AF['pct'], color = 'blue', label='Africa')
ax[2].plot(df_survey_continent_AS9_AS['pct'], color = 'orange', label='Asia')
ax[2].plot(df_survey_continent_AS9_OC['pct'], color = 'black', label='Oceania')
ax[2].plot(df_survey_continent_AS9_SA['pct'], color = 'gray', label='South America')
ax[2].set_title('Diversidad cultural de la organización')
ax[2].set_xlabel('Importancia. 1 lo más importante, 10 lo menos importante')
ax[2].set_ylabel('Porcentaje de votos')
ax[2].grid(True)
df_assessbenefits1 = df_survey.groupby(['AssessBenefits1'])['Respondent'].agg(['count']).reset_index()
df_assessbenefits1 = df_assessbenefits1.reindex([0,3,4,5,6,7,8,9,10,1,2])
df_assessbenefits1 = df_assessbenefits1.set_index('AssessBenefits1')
df_assessbenefits1=df_assessbenefits1.rename(columns={'count':'TotalSum'})
total=df_assessbenefits1['TotalSum'].sum()
df_assessbenefits1['pct']=df_assessbenefits1.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessbenefits2 = df_survey.groupby(['AssessBenefits2'])['Respondent'].agg(['count']).reset_index()
df_assessbenefits2 = df_assessbenefits2.reindex([0,3,4,5,6,7,8,9,10,1,2])
df_assessbenefits2 = df_assessbenefits2.set_index('AssessBenefits2')
df_assessbenefits2=df_assessbenefits2.rename(columns={'count':'TotalSum'})
total=df_assessbenefits2['TotalSum'].sum()
df_assessbenefits2['pct']=df_assessbenefits2.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessbenefits3 = df_survey.groupby(['AssessBenefits3'])['Respondent'].agg(['count']).reset_index()
df_assessbenefits3 = df_assessbenefits3.reindex([0,3,4,5,6,7,8,9,10,1,2])
df_assessbenefits3 = df_assessbenefits3.set_index('AssessBenefits3')
df_assessbenefits3=df_assessbenefits3.rename(columns={'count':'TotalSum'})
total=df_assessbenefits3['TotalSum'].sum()
df_assessbenefits3['pct']=df_assessbenefits3.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessbenefits4 = df_survey.groupby(['AssessBenefits4'])['Respondent'].agg(['count']).reset_index()
df_assessbenefits4 = df_assessbenefits4.reindex([0,3,4,5,6,7,8,9,10,1,2])
df_assessbenefits4 = df_assessbenefits4.set_index('AssessBenefits4')
df_assessbenefits4=df_assessbenefits4.rename(columns={'count':'TotalSum'})
total=df_assessbenefits4['TotalSum'].sum()
df_assessbenefits4['pct']=df_assessbenefits4.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessbenefits5 = df_survey.groupby(['AssessBenefits5'])['Respondent'].agg(['count']).reset_index()
df_assessbenefits5 = df_assessbenefits5.reindex([0,3,4,5,6,7,8,9,10,1,2])
df_assessbenefits5 = df_assessbenefits5.set_index('AssessBenefits5')
df_assessbenefits5=df_assessbenefits5.rename(columns={'count':'TotalSum'})
total=df_assessbenefits5['TotalSum'].sum()
df_assessbenefits5['pct']=df_assessbenefits5.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessbenefits6 = df_survey.groupby(['AssessBenefits6'])['Respondent'].agg(['count']).reset_index()
df_assessbenefits6 = df_assessbenefits6.reindex([0,3,4,5,6,7,8,9,10,1,2])
df_assessbenefits6 = df_assessbenefits6.set_index('AssessBenefits6')
df_assessbenefits6=df_assessbenefits6.rename(columns={'count':'TotalSum'})
total=df_assessbenefits6['TotalSum'].sum()
df_assessbenefits6['pct']=df_assessbenefits6.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessbenefits7 = df_survey.groupby(['AssessBenefits7'])['Respondent'].agg(['count']).reset_index()
df_assessbenefits7 = df_assessbenefits7.reindex([0,3,4,5,6,7,8,9,10,1,2])
df_assessbenefits7 = df_assessbenefits7.set_index('AssessBenefits7')
df_assessbenefits7=df_assessbenefits7.rename(columns={'count':'TotalSum'})
total=df_assessbenefits7['TotalSum'].sum()
df_assessbenefits7['pct']=df_assessbenefits7.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessbenefits9 = df_survey.groupby(['AssessBenefits9'])['Respondent'].agg(['count']).reset_index()
df_assessbenefits9 = df_assessbenefits9.reindex([0,3,4,5,6,7,8,9,10,1,2])
df_assessbenefits9 = df_assessbenefits9.set_index('AssessBenefits9')
df_assessbenefits9=df_assessbenefits9.rename(columns={'count':'TotalSum'})
total=df_assessbenefits9['TotalSum'].sum()
df_assessbenefits9['pct']=df_assessbenefits9.apply(lambda row: row.TotalSum*100/total, axis=1)

df_assessbenefits10 = df_survey.groupby(['AssessBenefits10'])['Respondent'].agg(['count']).reset_index()
df_assessbenefits10 = df_assessbenefits10.reindex([0,3,4,5,6,7,8,9,10,1,2])
df_assessbenefits10 = df_assessbenefits10.set_index('AssessBenefits10')
df_assessbenefits10=df_assessbenefits10.rename(columns={'count':'TotalSum'})
total=df_assessbenefits10['TotalSum'].sum()
df_assessbenefits10['pct']=df_assessbenefits10.apply(lambda row: row.TotalSum*100/total, axis=1)

fig, ax =  plt.subplots(figsize=(15, 8))

# Pintamos cada una de las series
ax.plot(df_assessbenefits1['pct'], color = 'red', label='Salario y bonos')
ax.plot(df_assessbenefits2['pct'], color = 'green', label='Stock options o acciones')
ax.plot(df_assessbenefits3['pct'], color = 'blue', label='Seguro médico')
ax.plot(df_assessbenefits4['pct'], color = 'yellow', label='Baja parental')
ax.plot(df_assessbenefits5['pct'], color = 'red', label='Beneficios o descuentos en fitness y bienestar')
ax.plot(df_assessbenefits6['pct'], color = 'pink', label='Plan de pensiones')
ax.plot(df_assessbenefits7['pct'], color = 'blue', label='Snack o comidas proporcionadas por la empresa')
ax.plot(df_assessbenefits9['pct'], color = 'black', label='Beneficios para el cuidado de los niños')
ax.plot(df_assessbenefits10['pct'], color = 'orange', label='Beneficios en transporte')
# Añadimos Formato al gráfico
ax.set_title('Comparativa global de la importancia que tienen la compensación y los beneficios')
ax.set_xlabel('Importancia. 1 lo más importante, 11 lo menos importante')
ax.set_ylabel('Total de votos')
ax.grid(True)
plt.legend()
print()
AssessBenefits='AssessBenefits1'
df_survey_continentAB = df_survey_joined.groupby(['Continent',AssessBenefits])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AB1_US = generateForContinentAndAS(df_survey_continentAB,"NA",AssessBenefits)
df_survey_continent_AB1_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB1_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB1_AF = generateForContinentAndAS(df_survey_continentAB,"AF",AssessBenefits)
df_survey_continent_AB1_OC = generateForContinentAndAS(df_survey_continentAB,"OC",AssessBenefits)
df_survey_continent_AB1_AS = generateForContinentAndAS(df_survey_continentAB,"AS",AssessBenefits)
df_survey_continent_AB1_SA = generateForContinentAndAS(df_survey_continentAB,"SA",AssessBenefits)
AssessBenefits='AssessBenefits2'
df_survey_continentAB = df_survey_joined.groupby(['Continent',AssessBenefits])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AB2_US = generateForContinentAndAS(df_survey_continentAB,"NA",AssessBenefits)
df_survey_continent_AB2_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB2_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB2_AF = generateForContinentAndAS(df_survey_continentAB,"AF",AssessBenefits)
df_survey_continent_AB2_OC = generateForContinentAndAS(df_survey_continentAB,"OC",AssessBenefits)
df_survey_continent_AB2_AS = generateForContinentAndAS(df_survey_continentAB,"AS",AssessBenefits)
df_survey_continent_AB2_SA = generateForContinentAndAS(df_survey_continentAB,"SA",AssessBenefits)
AssessBenefits='AssessBenefits3'
df_survey_continentAB = df_survey_joined.groupby(['Continent',AssessBenefits])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AB3_US = generateForContinentAndAS(df_survey_continentAB,"NA",AssessBenefits)
df_survey_continent_AB3_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB3_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB3_AF = generateForContinentAndAS(df_survey_continentAB,"AF",AssessBenefits)
df_survey_continent_AB3_OC = generateForContinentAndAS(df_survey_continentAB,"OC",AssessBenefits)
df_survey_continent_AB3_AS = generateForContinentAndAS(df_survey_continentAB,"AS",AssessBenefits)
df_survey_continent_AB3_SA = generateForContinentAndAS(df_survey_continentAB,"SA",AssessBenefits)
#Pintamos las tres gráficas anteriores juntas

fig, ax = plt.subplots(1,3,figsize=(24, 8))
ax[0].plot(df_survey_continent_AB1_US['pct'], color = 'red', label='North America')
ax[0].plot(df_survey_continent_AB1_EU['pct'], color = 'green', label='Europe')
ax[0].plot(df_survey_continent_AB1_AF['pct'], color = 'blue', label='Africa')
ax[0].plot(df_survey_continent_AB1_AS['pct'], color = 'orange', label='Asia')
ax[0].plot(df_survey_continent_AB1_OC['pct'], color = 'black', label='Oceania')
ax[0].plot(df_survey_continent_AB1_SA['pct'], color = 'gray', label='South America')
ax[0].set_title('Salario y bonos')
ax[0].set_xlabel('Importancia. 1 lo más importante, 11 lo menos importante')
ax[0].set_ylabel('Porcentaje de votos')
ax[0].grid(True)

ax[1].plot(df_survey_continent_AB2_US['pct'], color = 'red', label='North America')
ax[1].plot(df_survey_continent_AB2_EU['pct'], color = 'green', label='Europe')
ax[1].plot(df_survey_continent_AB2_AF['pct'], color = 'blue', label='Africa')
ax[1].plot(df_survey_continent_AB2_AS['pct'], color = 'orange', label='Asia')
ax[1].plot(df_survey_continent_AB2_OC['pct'], color = 'black', label='Oceania')
ax[1].plot(df_survey_continent_AB2_SA['pct'], color = 'gray', label='South America')
ax[1].set_title('Stock options o acciones')
ax[1].set_xlabel('Importancia. 1 lo más importante, 11 lo menos importante')
ax[1].set_ylabel('Porcentaje de votos')
ax[1].grid(True)

ax[2].plot(df_survey_continent_AB3_US['pct'], color = 'red', label='North America')
ax[2].plot(df_survey_continent_AB3_EU['pct'], color = 'green', label='Europe')
ax[2].plot(df_survey_continent_AB3_AF['pct'], color = 'blue', label='Africa')
ax[2].plot(df_survey_continent_AB3_AS['pct'], color = 'orange', label='Asia')
ax[2].plot(df_survey_continent_AB3_OC['pct'], color = 'black', label='Oceania')
ax[2].plot(df_survey_continent_AB3_SA['pct'], color = 'gray', label='South America')
ax[2].set_title('Seguro médico')
ax[2].set_xlabel('Importancia. 1 lo más importante, 11 lo menos importante')
ax[2].set_ylabel('Porcentaje de votos')
ax[2].grid(True)
AssessBenefits='AssessBenefits4'
df_survey_continentAB = df_survey_joined.groupby(['Continent',AssessBenefits])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AB4_US = generateForContinentAndAS(df_survey_continentAB,"NA",AssessBenefits)
df_survey_continent_AB4_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB4_AF = generateForContinentAndAS(df_survey_continentAB,"AF",AssessBenefits)
df_survey_continent_AB4_OC = generateForContinentAndAS(df_survey_continentAB,"OC",AssessBenefits)
df_survey_continent_AB4_AS = generateForContinentAndAS(df_survey_continentAB,"AS",AssessBenefits)
df_survey_continent_AB4_SA = generateForContinentAndAS(df_survey_continentAB,"SA",AssessBenefits)
AssessBenefits='AssessBenefits5'
df_survey_continentAB = df_survey_joined.groupby(['Continent',AssessBenefits])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AB5_US = generateForContinentAndAS(df_survey_continentAB,"NA",AssessBenefits)
df_survey_continent_AB5_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB5_AF = generateForContinentAndAS(df_survey_continentAB,"AF",AssessBenefits)
df_survey_continent_AB5_OC = generateForContinentAndAS(df_survey_continentAB,"OC",AssessBenefits)
df_survey_continent_AB5_AS = generateForContinentAndAS(df_survey_continentAB,"AS",AssessBenefits)
df_survey_continent_AB5_SA = generateForContinentAndAS(df_survey_continentAB,"SA",AssessBenefits)
AssessBenefits='AssessBenefits6'
df_survey_continentAB = df_survey_joined.groupby(['Continent',AssessBenefits])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AB6_US = generateForContinentAndAS(df_survey_continentAB,"NA",AssessBenefits)
df_survey_continent_AB6_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB6_AF = generateForContinentAndAS(df_survey_continentAB,"AF",AssessBenefits)
df_survey_continent_AB6_OC = generateForContinentAndAS(df_survey_continentAB,"OC",AssessBenefits)
df_survey_continent_AB6_AS = generateForContinentAndAS(df_survey_continentAB,"AS",AssessBenefits)
df_survey_continent_AB6_SA = generateForContinentAndAS(df_survey_continentAB,"SA",AssessBenefits)
#Pintamos las tres gráficas anteriores juntas

fig, ax = plt.subplots(1,3,figsize=(24, 8))
ax[0].plot(df_survey_continent_AB4_US['pct'], color = 'red', label='North America')
ax[0].plot(df_survey_continent_AB4_EU['pct'], color = 'green', label='Europe')
ax[0].plot(df_survey_continent_AB4_AF['pct'], color = 'blue', label='Africa')
ax[0].plot(df_survey_continent_AB4_AS['pct'], color = 'orange', label='Asia')
ax[0].plot(df_survey_continent_AB4_OC['pct'], color = 'black', label='Oceania')
ax[0].plot(df_survey_continent_AB4_SA['pct'], color = 'gray', label='South America')
ax[0].set_title('Baja parental')
ax[0].set_xlabel('Importancia. 1 lo más importante, 11 lo menos importante')
ax[0].set_ylabel('Porcentaje de votos')
ax[0].grid(True)

ax[1].plot(df_survey_continent_AB5_US['pct'], color = 'red', label='North America')
ax[1].plot(df_survey_continent_AB5_EU['pct'], color = 'green', label='Europe')
ax[1].plot(df_survey_continent_AB5_AF['pct'], color = 'blue', label='Africa')
ax[1].plot(df_survey_continent_AB5_AS['pct'], color = 'orange', label='Asia')
ax[1].plot(df_survey_continent_AB5_OC['pct'], color = 'black', label='Oceania')
ax[1].plot(df_survey_continent_AB5_SA['pct'], color = 'gray', label='South America')
ax[1].set_title('Beneficios o descuentos en fitness y bienestar')
ax[1].set_xlabel('Importancia. 1 lo más importante, 11 lo menos importante')
ax[1].set_ylabel('Porcentaje de votos')
ax[1].grid(True)

ax[2].plot(df_survey_continent_AB6_US['pct'], color = 'red', label='North America')
ax[2].plot(df_survey_continent_AB6_EU['pct'], color = 'green', label='Europe')
ax[2].plot(df_survey_continent_AB6_AF['pct'], color = 'blue', label='Africa')
ax[2].plot(df_survey_continent_AB6_AS['pct'], color = 'orange', label='Asia')
ax[2].plot(df_survey_continent_AB6_OC['pct'], color = 'black', label='Oceania')
ax[2].plot(df_survey_continent_AB6_SA['pct'], color = 'gray', label='South America')
ax[2].set_title('Plan de pensiones')
ax[2].set_xlabel('Importancia. 1 lo más importante, 11 lo menos importante')
ax[2].set_ylabel('Porcentaje de votos')
ax[2].grid(True)
AssessBenefits='AssessBenefits7'
df_survey_continentAB = df_survey_joined.groupby(['Continent',AssessBenefits])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AB7_US = generateForContinentAndAS(df_survey_continentAB,"NA",AssessBenefits)
df_survey_continent_AB7_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB7_AF = generateForContinentAndAS(df_survey_continentAB,"AF",AssessBenefits)
df_survey_continent_AB7_OC = generateForContinentAndAS(df_survey_continentAB,"OC",AssessBenefits)
df_survey_continent_AB7_AS = generateForContinentAndAS(df_survey_continentAB,"AS",AssessBenefits)
df_survey_continent_AB7_SA = generateForContinentAndAS(df_survey_continentAB,"SA",AssessBenefits)
AssessBenefits='AssessBenefits9'
df_survey_continentAB = df_survey_joined.groupby(['Continent',AssessBenefits])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AB9_US = generateForContinentAndAS(df_survey_continentAB,"NA",AssessBenefits)
df_survey_continent_AB9_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB9_AF = generateForContinentAndAS(df_survey_continentAB,"AF",AssessBenefits)
df_survey_continent_AB9_OC = generateForContinentAndAS(df_survey_continentAB,"OC",AssessBenefits)
df_survey_continent_AB9_AS = generateForContinentAndAS(df_survey_continentAB,"AS",AssessBenefits)
df_survey_continent_AB9_SA = generateForContinentAndAS(df_survey_continentAB,"SA",AssessBenefits)
AssessBenefits='AssessBenefits10'
df_survey_continentAB = df_survey_joined.groupby(['Continent',AssessBenefits])['Respondent'].agg(['count']).reset_index()
df_survey_continent_AB10_US = generateForContinentAndAS(df_survey_continentAB,"NA",AssessBenefits)
df_survey_continent_AB10_EU = generateForContinentAndAS(df_survey_continentAB,"EU",AssessBenefits)
df_survey_continent_AB10_AF = generateForContinentAndAS(df_survey_continentAB,"AF",AssessBenefits)
df_survey_continent_AB10_OC = generateForContinentAndAS(df_survey_continentAB,"OC",AssessBenefits)
df_survey_continent_AB10_AS = generateForContinentAndAS(df_survey_continentAB,"AS",AssessBenefits)
df_survey_continent_AB10_SA = generateForContinentAndAS(df_survey_continentAB,"SA",AssessBenefits)
#Pintamos las tres gráficas anteriores juntas

fig, ax = plt.subplots(1,3,figsize=(24, 8))
ax[0].plot(df_survey_continent_AB7_US['pct'], color = 'red', label='North America')
ax[0].plot(df_survey_continent_AB7_EU['pct'], color = 'green', label='Europe')
ax[0].plot(df_survey_continent_AB7_AF['pct'], color = 'blue', label='Africa')
ax[0].plot(df_survey_continent_AB7_AS['pct'], color = 'orange', label='Asia')
ax[0].plot(df_survey_continent_AB7_OC['pct'], color = 'black', label='Oceania')
ax[0].plot(df_survey_continent_AB7_SA['pct'], color = 'gray', label='South America')
ax[0].set_title('Snacks o comida proporcionada por la empresa')
ax[0].set_xlabel('Importancia. 1 lo más importante, 11 lo menos importante')
ax[0].set_ylabel('Porcentaje de votos')
ax[0].grid(True)

ax[1].plot(df_survey_continent_AB9_US['pct'], color = 'red', label='North America')
ax[1].plot(df_survey_continent_AB9_EU['pct'], color = 'green', label='Europe')
ax[1].plot(df_survey_continent_AB9_AF['pct'], color = 'blue', label='Africa')
ax[1].plot(df_survey_continent_AB9_AS['pct'], color = 'orange', label='Asia')
ax[1].plot(df_survey_continent_AB9_OC['pct'], color = 'black', label='Oceania')
ax[1].plot(df_survey_continent_AB9_SA['pct'], color = 'gray', label='South America')
ax[1].set_title('Beneficios para el cuidado de los niños')
ax[1].set_xlabel('Importancia. 1 lo más importante, 11 lo menos importante')
ax[1].set_ylabel('Porcentaje de votos')
ax[1].grid(True)

ax[2].plot(df_survey_continent_AB10_US['pct'], color = 'red', label='North America')
ax[2].plot(df_survey_continent_AB10_EU['pct'], color = 'green', label='Europe')
ax[2].plot(df_survey_continent_AB10_AF['pct'], color = 'blue', label='Africa')
ax[2].plot(df_survey_continent_AB10_AS['pct'], color = 'orange', label='Asia')
ax[2].plot(df_survey_continent_AB10_OC['pct'], color = 'black', label='Oceania')
ax[2].plot(df_survey_continent_AB10_SA['pct'], color = 'gray', label='South America')
ax[2].set_title('Beneficios para el transporte')
ax[2].set_xlabel('Importancia. 1 lo más importante, 11 lo menos importante')
ax[2].set_ylabel('Porcentaje de votos')
ax[2].grid(True)
## Clasificamos por lenguages con qué querrían trabajar los profesionales de big data
df_survey=df_survey[(df_survey.Employment=='Employed full-time') | (df_survey.Employment=='Employed part-time') | (df_survey.Employment=='Independent contractor, freelancer, or self-employed')]


lens = df_survey['DevType'].str.split(';').map(len)
df_survey_bd = pd.DataFrame({'Respondent': np.repeat(df_survey['Respondent'], lens),
                    'DevType': chainer(df_survey['DevType']),'LanguageDesireNextYear': df_survey['LanguageDesireNextYear'], 'DatabaseWorkedWith':df_survey['DatabaseWorkedWith'], 'PlatformWorkedWith':df_survey['PlatformWorkedWith'],'FrameworkWorkedWith':df_survey['FrameworkWorkedWith'],'IDE':df_survey['IDE'],'OperatingSystem':df_survey['OperatingSystem']})
df_survey_bd=df_survey_bd[(df_survey_bd.DevType=='Data or business analyst') | (df_survey_bd.DevType=='Data scientist or machine learning specialist')]

df_survey_lenguage=df_survey_bd[pd.isna(df_survey_bd.LanguageDesireNextYear)==False]
lensLenguage = df_survey_lenguage['LanguageDesireNextYear'].str.split(';').map(len)
df_survey_lenguage = pd.DataFrame({'Respondent': np.repeat(df_survey_lenguage['Respondent'], lensLenguage),
                    'LanguageDesireNextYear': chainer(df_survey_lenguage['LanguageDesireNextYear'])})

df_survey_lenguage=df_survey_lenguage.groupby(['LanguageDesireNextYear'])['Respondent'].agg(['count']).reset_index()
df_survey_lenguage=df_survey_lenguage.sort_values(by='count', ascending=False)
df_survey_lenguage.head(15)

import squarify
fig, ax = plt.subplots(figsize=(20, 10))
squarify.plot(ax=ax, sizes=df_survey_lenguage['count'], label=df_survey_lenguage['LanguageDesireNextYear'], alpha=0.5)
ax.set_title('Lenguajes de programación con que querrían trabajar los profesionales de Big Data')
plt.show()
## Clasificamos bases de datos con qué querrían trabajar los profesionales de big data

lens = df_survey['DevType'].str.split(';').map(len)
df_survey_bd = pd.DataFrame({'Respondent': np.repeat(df_survey['Respondent'], lens),
                    'DevType': chainer(df_survey['DevType']),'DatabaseDesireNextYear': df_survey['DatabaseDesireNextYear'], 'DatabaseWorkedWith':df_survey['DatabaseWorkedWith'], 'PlatformDesireNextYear':df_survey['PlatformDesireNextYear'],'FrameworkWorkedWith':df_survey['FrameworkWorkedWith'],'IDE':df_survey['IDE'],'OperatingSystem':df_survey['OperatingSystem']})
df_survey_bd=df_survey_bd[(df_survey_bd.DevType=='Data or business analyst') | (df_survey_bd.DevType=='Data scientist or machine learning specialist')]

df_survey_lenguage=df_survey_bd[pd.isna(df_survey_bd.DatabaseDesireNextYear)==False]
lensLenguage = df_survey_lenguage['DatabaseDesireNextYear'].str.split(';').map(len)
df_survey_lenguage = pd.DataFrame({'Respondent': np.repeat(df_survey_lenguage['Respondent'], lensLenguage),
                    'DatabaseDesireNextYear': chainer(df_survey_lenguage['DatabaseDesireNextYear'])})

df_survey_lenguage=df_survey_lenguage.groupby(['DatabaseDesireNextYear'])['Respondent'].agg(['count']).reset_index()
df_survey_lenguage=df_survey_lenguage.sort_values(by='count', ascending=False)

fig, ax = plt.subplots(figsize=(20, 10))
squarify.plot(ax=ax, sizes=df_survey_lenguage['count'], label=df_survey_lenguage['DatabaseDesireNextYear'], alpha=0.5)
ax.set_title('Bases de datos con las que querrían trabajar los profesionales de Big Data')
plt.show()
## Clasificamos por plataformas con qué querrían trabajar los profesionales de big data

lens = df_survey['DevType'].str.split(';').map(len)
df_survey_bd = pd.DataFrame({'Respondent': np.repeat(df_survey['Respondent'], lens),
                    'DevType': chainer(df_survey['DevType']),'PlatformDesireNextYear': df_survey['PlatformDesireNextYear'], 'DatabaseWorkedWith':df_survey['DatabaseWorkedWith'], 'PlatformWorkedWith':df_survey['PlatformWorkedWith'],'FrameworkWorkedWith':df_survey['FrameworkWorkedWith'],'IDE':df_survey['IDE'],'OperatingSystem':df_survey['OperatingSystem']})
df_survey_bd=df_survey_bd[(df_survey_bd.DevType=='Data or business analyst') | (df_survey_bd.DevType=='Data scientist or machine learning specialist')]

df_survey_lenguage=df_survey_bd[pd.isna(df_survey_bd.PlatformDesireNextYear)==False]
lensLenguage = df_survey_lenguage['PlatformDesireNextYear'].str.split(';').map(len)
df_survey_lenguage = pd.DataFrame({'Respondent': np.repeat(df_survey_lenguage['Respondent'], lensLenguage),
                    'PlatformDesireNextYear': chainer(df_survey_lenguage['PlatformDesireNextYear'])})

df_survey_lenguage=df_survey_lenguage.groupby(['PlatformDesireNextYear'])['Respondent'].agg(['count']).reset_index()
df_survey_lenguage=df_survey_lenguage.sort_values(by='count', ascending=False)
df_survey_lenguage.head(15)

import squarify
fig, ax = plt.subplots(figsize=(20, 10))
squarify.plot(ax=ax, sizes=df_survey_lenguage['count'], label=df_survey_lenguage['PlatformDesireNextYear'], alpha=0.5)
ax.set_title('Plataformas con las que querrían trabajar los profesionales de Big Data')
plt.show()
## Clasificamos por frameworks con qué querrían trabajar los profesionales de big data

lens = df_survey['DevType'].str.split(';').map(len)
df_survey_bd = pd.DataFrame({'Respondent': np.repeat(df_survey['Respondent'], lens),
                    'DevType': chainer(df_survey['DevType']),'FrameworkDesireNextYear': df_survey['FrameworkDesireNextYear'], 'DatabaseWorkedWith':df_survey['DatabaseWorkedWith'], 'PlatformWorkedWith':df_survey['PlatformWorkedWith'],'FrameworkWorkedWith':df_survey['FrameworkWorkedWith'],'IDE':df_survey['IDE'],'OperatingSystem':df_survey['OperatingSystem']})
df_survey_bd=df_survey_bd[(df_survey_bd.DevType=='Data or business analyst') | (df_survey_bd.DevType=='Data scientist or machine learning specialist')]

df_survey_lenguage=df_survey_bd[pd.isna(df_survey_bd.FrameworkDesireNextYear)==False]
lensLenguage = df_survey_lenguage['FrameworkDesireNextYear'].str.split(';').map(len)
df_survey_lenguage = pd.DataFrame({'Respondent': np.repeat(df_survey_lenguage['Respondent'], lensLenguage),
                    'FrameworkDesireNextYear': chainer(df_survey_lenguage['FrameworkDesireNextYear'])})

df_survey_lenguage=df_survey_lenguage.groupby(['FrameworkDesireNextYear'])['Respondent'].agg(['count']).reset_index()
df_survey_lenguage=df_survey_lenguage.sort_values(by='count', ascending=False)
df_survey_lenguage.head(15)

import squarify
fig, ax = plt.subplots(figsize=(20, 10))
squarify.plot(ax=ax, sizes=df_survey_lenguage['count'], label=df_survey_lenguage['FrameworkDesireNextYear'], alpha=0.5)
ax.set_title('Frameworks con los que querrrían trabajar los profesionales de Big Data')
plt.show()
#Sacamos los salarios por continente y puesto
res = pd.DataFrame({'Respondent': np.repeat(df_survey['Respondent'], lens),
                    'DevType': chainer(df_survey['DevType']),'Country': df_survey['Country'], 'ConvertedSalary':df_survey['ConvertedSalary'], 'Salary':df_survey['Salary']})

df_salaries=pd.merge(res, df_countries_continents, how='inner', on=['Country'])
df_salaries=df_salaries.groupby(['Continent','DevType'])['Respondent'].agg(['mean']).reset_index()
df_salaries=df_salaries.sort_values(by=['Continent','mean'], ascending=False)
df_salaries=df_salaries[(df_salaries.DevType=='Data or business analyst') | (df_salaries.DevType=='Data scientist or machine learning specialist')]
df_salaries=df_salaries.rename(columns={'mean':'Salario anual medio'})

df_salaries['Continent'].replace(['SA'], 'South America', inplace=True)
df_salaries['Continent'].replace(['OC'], 'Oceania', inplace=True)
df_salaries['Continent'].replace(['NA'], 'North America', inplace=True)
df_salaries['Continent'].replace(['EU'], 'Europe', inplace=True)
df_salaries['Continent'].replace(['AS'], 'Asia', inplace=True)
df_salaries['Continent'].replace(['AF'], 'Africa', inplace=True)
ind = np.arange(len(df_salaries['Salario anual medio']))  # the x locations for the groups
width = 0.35  # the width of the bars
mean_salaries=map(int, df_salaries['Salario anual medio'])
mean_salaries_list = list(mean_salaries)

fig, ax = plt.subplots(figsize=(20, 10))
rects1 = ax.bar(ind - width/2, mean_salaries_list, width, color='SkyBlue', label='Data or business analyst')
rects2 = ax.bar(ind + width/2, mean_salaries_list, width, color='IndianRed', label='Data scientist or machine learning specialist')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Annual salary in dolars')
ax.set_title('Annual salaries by position')
ax.set_xticks(ind)
ax.set_xticklabels(df_salaries['Continent'])
ax.legend()

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height, '{}'.format(height), ha=ha[xpos], va='bottom')

autolabel(rects1, "left")
autolabel(rects2, "right")

plt.show()
lens = df_survey['DevType'].str.split(';').map(len)
res = pd.DataFrame({'Respondent': np.repeat(df_survey['Respondent'], lens),
                    'DevType': chainer(df_survey['DevType']),'Country': df_survey['Country'], 'ConvertedSalary':df_survey['ConvertedSalary'], 'Salary':df_survey['Salary']})

df_salaries=pd.merge(res, df_countries_continents, how='inner', on=['Country'])

df_salaries=df_salaries[df_salaries.Continent=='EU']
df_salaries=df_salaries[pd.isna(df_salaries.ConvertedSalary)==False]
df_salaries=df_salaries[(df_salaries.DevType=='Data or business analyst') | (df_salaries.DevType=='Data scientist or machine learning specialist')]
df_salaries=df_salaries.groupby(['Country','DevType'])['Respondent'].agg(['mean']).reset_index()
df_salaries=df_salaries.sort_values(by='mean', ascending=False)
df_salaries=df_salaries.rename(columns={'mean':'Salario anual medio'})
df_salaries=df_salaries.head(30)
df_salaries
ind = np.arange(len(df_salaries['Salario anual medio']))  # the x locations for the groups
width = 0.35  # the width of the bars
mean_salaries=map(int, df_salaries['Salario anual medio'])
mean_salaries_list = list(mean_salaries)

fig, ax = plt.subplots(figsize=(20, 10))
rects1 = ax.bar(ind - width/2, mean_salaries_list, width, color='SkyBlue', label='Data or business analyst')
rects2 = ax.bar(ind + width/2, mean_salaries_list, width, color='IndianRed', label='Data scientist or machine learning specialist')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Salario anual en dólares')
ax.set_title('Salarios anuales por país y posición (Data or business analyst y Data scientist or machine learning specialist)')
ax.set_xticks(ind)
ax.set_xticklabels(df_salaries['Country'])
ax.legend()

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.3, 'right': 0.3, 'left': 0.6}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height, '{}'.format(height), ha=ha[xpos], va='bottom')

autolabel(rects1, "left")
autolabel(rects2, "right")

plt.show()
