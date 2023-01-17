import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/CAERS_ASCII_2004_2017Q2.csv')
df.shape
df.info()
df.head()
plt.figure(figsize=(12,9))
sns.heatmap(df.isnull(),
            cmap='plasma',
            yticklabels=False,
            cbar=False)
plt.title('Missing Data\n',fontsize=20)
plt.xticks(fontsize=15)
plt.show()
print('Duplicate Data?')
df.duplicated('RA_Report #').value_counts()
df.drop_duplicates(['RA_Report #'],keep='last',inplace=True)
len(df)
print('Duplicate Data?')
df.duplicated('RA_Report #').value_counts()
df.drop(['AEC_Event Start Date','CI_Age at Adverse Event'],axis=1,inplace=True)
plt.figure(figsize=(12,9))
sns.heatmap(df.isnull(),
            cmap='plasma',
            yticklabels=False,
            cbar=False)
plt.title('Missing Data\n',fontsize=20)
plt.xticks(fontsize=15)
plt.show()
print('Rows in the Dataframe?')
len(df)
df.head()
df.columns
plt.figure(figsize=(12,9))
df['PRI_Product Role'].value_counts().plot.bar()
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title('Suspect or Concomitant?\n',fontsize=20)
plt.show()
print('Suspect or Concomitant?\n')
print(df['PRI_Product Role'].value_counts())
plt.figure(figsize=(10,10))
df['CI_Gender'].value_counts()[:3].plot(kind='pie')
plt.title('Adverse Events Reported by Gender\n',fontsize=20)
plt.show()
print('Reported Events by Gender\n')
print(df['CI_Gender'].value_counts())
type(df['RA_CAERS Created Date'][1])
df['RA_CAERS Created Date'] = pd.to_datetime(df['RA_CAERS Created Date'])
type(df['RA_CAERS Created Date'][1])
df['Created Year'] = df['RA_CAERS Created Date'].apply(lambda x: x.year)
df['Created Month'] = df['RA_CAERS Created Date'].apply(lambda x: x.month)
plt.figure(figsize=(12,9))
df.groupby('Created Month').count()['RA_CAERS Created Date'].plot(kind='bar')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title('Reports by Month\n',fontsize=20)
plt.show()
print(df.groupby('Created Month').count()['RA_CAERS Created Date'])
plt.figure(figsize=(12,9))
df.groupby('Created Year').count()['RA_CAERS Created Date'].plot(kind='bar')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title('Reports by Year\n',fontsize=20)
plt.show()
print(df.groupby('Created Year').count()['RA_CAERS Created Date'])
plt.figure(figsize=(12,12))
df['AEC_One Row Outcomes'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('20 Most Common Adverse Event Outcome\n',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()
print('\nNumber of different Outcomes: ',len(df['AEC_One Row Outcomes'].value_counts()))
visit_count = 0
death_count = 0
non_serious = 0
serious = 0
dis_count = 0

for i in df['AEC_One Row Outcomes']:
    if 'HOSPITALIZATION' in i or 'VISITED A HEALTH CARE PROVIDER' in i or 'VISITED AN ER' in i:
        visit_count += 1
    if 'DEATH' in i:
        death_count += 1
    if 'NON-SERIOUS INJURIES/ ILLNESS' in i:
        non_serious += 1
    if  i == 'SERIOUS INJURIES/ ILLNESS' or ' SERIOUS INJURIES/ ILLNESS' in i or i[:25] == 'SERIOUS INJURIES/ ILLNESS':    
        serious += 1
    if 'DISABILITY' in i:
        dis_count += 1
        
print('PERCENTAGES OF OUTCOMES\n')        
print('VISITED AN ER, VISITED A HEALTH CARE PROVIDER, HOSPITALIZATION percentage: {}%'.format(round(visit_count/len(df),3)*100))
print('NON-SERIOUS INJURIES/ ILLNESS percentage: {}%'.format(round(non_serious/len(df),3)*100))
print('SERIOUS INJURIES/ ILLNESS percentage: {}%'.format(round(serious/len(df),2)*100))
print('DEATH percentage: {}%'.format(round(death_count/len(df),3)*100))
print('DISABILITY percentage: {}%'.format(round(dis_count/len(df),3)*100))
d = {'VISITED AN ER, VISITED A HEALTH CARE PROVIDER, HOSPITALIZATION':visit_count,
     'NON-SERIOUS INJURIES/ ILLNESS':non_serious,
     'SERIOUS INJURIES/ ILLNESS':serious,
     'DEATH':death_count,
     'DISABILITY':dis_count}

outcomesDF = pd.Series(data=d)
plt.figure(figsize=(10,8))
outcomesDF.sort_values().plot(kind='barh')
plt.title('Outcome Counts',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()
print(outcomesDF.sort_values(ascending=False))
df[df['AEC_One Row Outcomes']=="DISABILITY, LIFE THREATENING, HOSPITALIZATION, DEATH"]
plt.figure(figsize=(12,15))
df['PRI_FDA Industry Name'].value_counts()[:40].sort_values(ascending=True).plot(kind='barh')
plt.title('Reports by Industry\n',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()
print(df['PRI_FDA Industry Name'].value_counts()[:40])
plt.figure(figsize=(12,12))
df['PRI_Reported Brand/Product Name'].value_counts()[1:21].sort_values(ascending=True).plot(kind='barh')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title('Most Reported Brands\n',fontsize=20)
plt.show()
print(df['PRI_Reported Brand/Product Name'].value_counts()[:21])
plt.figure(figsize=(12,12))
df[df['PRI_Reported Brand/Product Name']=='REDACTED']['PRI_FDA Industry Name'].value_counts()[:21].sort_values(ascending=True).plot(kind='barh')
plt.title('"REDACTED" by Industry\n',fontsize=20)
plt.show()
print(df[df['PRI_Reported Brand/Product Name']=='REDACTED']['PRI_FDA Industry Name'].value_counts()[:21])
print('{}%'.format(round(5455 / len(df[df['PRI_Reported Brand/Product Name']=='REDACTED'])*100),3) + ' of the "REDACTED" instances belong to the cosmetics industry.')
redacted_df = df[df['PRI_Reported Brand/Product Name']=='REDACTED']

visit_count = 0
death_count = 0
non_serious = 0
serious = 0
dis_count = 0

for i in redacted_df['AEC_One Row Outcomes']:
    if 'HOSPITALIZATION' in i or 'VISITED A HEALTH CARE PROVIDER' in i or 'VISITED AN ER' in i:
        visit_count += 1
    if 'DEATH' in i:
        death_count += 1
    if 'NON-SERIOUS INJURIES/ ILLNESS' in i:
        non_serious += 1
    if  i == 'SERIOUS INJURIES/ ILLNESS' or ' SERIOUS INJURIES/ ILLNESS' in i or i[:25] == 'SERIOUS INJURIES/ ILLNESS':    
        serious += 1
    if 'DISABILITY' in i:
        dis_count += 1
redacted_dict = {'VISITED AN ER, VISITED A HEALTH CARE PROVIDER, HOSPITALIZATION':visit_count,
     'NON-SERIOUS INJURIES/ ILLNESS':non_serious,
     'SERIOUS INJURIES/ ILLNESS':serious,
     'DEATH':death_count,
     'DISABILITY':dis_count}

redacted_data = pd.Series(data=d)
plt.figure(figsize=(12,9))
redacted_data.sort_values().plot(kind='barh')
plt.title('Count of "REDACTED" Outcomes',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
print(redacted_data.sort_values(ascending=False))
redacted_df[redacted_df['AEC_One Row Outcomes']=='DEATH']['SYM_One Row Coded Symptoms'].value_counts()[:10]
df['SYM_One Row Coded Symptoms'].value_counts()[:25]
# Drop missing values from the column. Can not loop through column otherwise.

df['SYM_One Row Coded Symptoms'].dropna(axis=0,how='any',inplace=True)
cancer_count = 0
choking_count = 0
diarrhoea_count = 0
vomit_count = 0
nausea_count = 0
dysgeusia_count = 0
malaise_count = 0
alopecia_count = 0
abpain_count = 0
rash_count = 0
headache_count = 0
laceration_count = 0
convulsion_count = 0
hyper_count = 0

#If you know how to make this loop more pythonic, please let me know. 

for i in df['SYM_One Row Coded Symptoms']:
    if 'CHOKING' in i:
        choking_count += 1
    if 'DIARRHOEA' in i:
        diarrhoea_count += 1
    if 'CANCER' in i:
        cancer_count += 1
    if 'VOMIT' in i:
        vomit_count += 1
    if 'NAUSEA' in i:
        nausea_count += 1
    if 'DYSGEUSIA' in i:
        dysgeusia_count += 1
    if 'MALAISE' in i:
        malaise_count += 1
    if 'ALOPECIA' in i:
        alopecia_count += 1
    if 'ABDOMINAL PAIN' in i:
        abpain_count += 1
    if 'RASH' in i:
        rash_count += 1
    if 'HEADACHE' in i:
        headache_count += 1
    if 'LACERATION' in i:
        laceration_count += 1
    if 'CONVULSION' in i:
        convulsion_count += 1
    if 'HYPERSENSITIVITY' in i:
        hyper_count += 1
symptoms_dict = {
 'ABDOMINAL PAIN': abpain_count,
 'ALOPECIA': alopecia_count,
 'CHOKING': choking_count,
 'CONVULSION': convulsion_count,
 'DIARRHOEA': diarrhoea_count,
 'DYSGEUSIA': dysgeusia_count,
 'HEADACHE': headache_count,
 'HYPERSENSITIVITY': hyper_count,
 'LACERATION': laceration_count,
 'MALAISE': malaise_count,
 'NAUSEA': nausea_count,
 'CANCER': cancer_count,
 'RASH': rash_count,
 'VOMITING': vomit_count
}

for k,v in symptoms_dict.items():
    print(k + ': ',v)
symptom_df = pd.Series(symptoms_dict)
plt.figure(figsize=(12,9))
symptom_df.sort_values(ascending=True).plot(kind='barh')
plt.title('SYMPTOMS COUNT',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()
print(symptom_df.sort_values(ascending=False))
plt.figure(figsize=(12,9))
df[df['AEC_One Row Outcomes']=='NON-SERIOUS INJURIES/ ILLNESS']['PRI_FDA Industry Name'].value_counts()[:25].sort_values(ascending=True).plot(kind='barh')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title('Non-Serious Injuries or Illness by Industry\n',fontsize=20)
plt.show()
print(df[df['AEC_One Row Outcomes']=='NON-SERIOUS INJURIES/ ILLNESS']['PRI_FDA Industry Name'].value_counts()[:25])
plt.figure(figsize=(12,9))
df[df['AEC_One Row Outcomes']=='OTHER SERIOUS (IMPORTANT MEDICAL EVENTS)']['PRI_FDA Industry Name'].value_counts()[:25].sort_values(ascending=True).plot(kind='barh')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title('OTHER SERIOUS (IMPORTANT MEDICAL EVENTS by Industry\n',fontsize=20)
plt.show()
print(df[df['AEC_One Row Outcomes']=='OTHER SERIOUS (IMPORTANT MEDICAL EVENTS)']['PRI_FDA Industry Name'].value_counts()[:25])
plt.figure(figsize=(12,9))
df[df['AEC_One Row Outcomes']=='DEATH']['PRI_FDA Industry Name'].value_counts()[:25].sort_values(ascending=True).plot(kind='barh')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title('DEATH by Industry\n',fontsize=20)
plt.show()
print(df[df['AEC_One Row Outcomes']=='DEATH']['PRI_FDA Industry Name'].value_counts()[:25])
injury = df[df['AEC_One Row Outcomes']=='NON-SERIOUS INJURIES/ ILLNESS']

plt.figure(figsize=(20,6))
sns.countplot(injury['PRI_FDA Industry Name'],hue=injury['CI_Gender'])
plt.title('NON-SERIOUS INJURIES/ ILLNESS by Gender\n',fontsize=20)
plt.xticks(fontsize=15,rotation=90)
plt.yticks(fontsize=15)
plt.xlabel('Industry',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.legend(loc=7)
plt.show()
print('Reported Male Non Serious Injuries: ' + str(len(df[(df['CI_Gender']=='Male')&(df['AEC_One Row Outcomes']=='NON-SERIOUS INJURIES/ ILLNESS')]['PRI_FDA Industry Name'])))
print('Reported Female Non Serious Injuries: ' + str(len(df[(df['CI_Gender']=='Female')&(df['AEC_One Row Outcomes']=='NON-SERIOUS INJURIES/ ILLNESS')]['PRI_FDA Industry Name'])))
serious = df[df['AEC_One Row Outcomes']=='OTHER SERIOUS (IMPORTANT MEDICAL EVENTS)']

plt.figure(figsize=(20,6))
sns.countplot(serious['PRI_FDA Industry Name'],hue=serious['CI_Gender'])
plt.title('OTHER SERIOUS (IMPORTANT MEDICAL EVENTS) by Gender\n',fontsize=20)
plt.xticks(fontsize=15,rotation=90)
plt.yticks(fontsize=15)
plt.xlabel('Industry',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.legend(loc=7)
plt.show()
print('Reported Male Serious Injuries: ' + str(len(df[(df['CI_Gender']=='Male')&(df['AEC_One Row Outcomes']=='OTHER SERIOUS (IMPORTANT MEDICAL EVENTS)')]['PRI_FDA Industry Name'])))
print('Reported Female Serious Injuries: ' + str(len(df[(df['CI_Gender']=='Female')&(df['AEC_One Row Outcomes']=='OTHER SERIOUS (IMPORTANT MEDICAL EVENTS)')]['PRI_FDA Industry Name'])))
death = df[df['AEC_One Row Outcomes']=='DEATH']

plt.figure(figsize=(15,6))
sns.countplot(death['PRI_FDA Industry Name'],hue=death['CI_Gender'])
plt.title('DEATH by Gender\n',fontsize=20)
plt.xticks(fontsize=15,rotation=90)
plt.yticks(fontsize=15)
plt.xlabel('Industry',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.legend(loc=7)
plt.show()
print('Reported Male Deaths: ' + str(len(df[(df['CI_Gender']=='Male')&(df['AEC_One Row Outcomes']=='DEATH')]['PRI_FDA Industry Name'])))
print('Reported Female Deaths: ' + str(len(df[(df['CI_Gender']=='Female')&(df['AEC_One Row Outcomes']=='DEATH')]['PRI_FDA Industry Name'])))
df['SYM_One Row Coded Symptoms'].value_counts()[:20]
death = df[df['AEC_One Row Outcomes']=='DEATH']
death['SYM_One Row Coded Symptoms'].value_counts()[:20]
ovarian = 0
for i in death['SYM_One Row Coded Symptoms']:
    if 'OVARIAN CANCER' in i:
        ovarian += 1
print('{}%'.format(round(ovarian/len(death)*100),3) + ' of the symptoms in the "DEATH" dataframe had the term "OVERIAN CANCER" in it.')
plt.figure(figsize=(12,9))
death[death['SYM_One Row Coded Symptoms']=='OVARIAN CANCER']['Created Year'].value_counts().plot(kind='bar')
plt.title('Ovarian Cancer by Year Reported',fontsize=20)
plt.show()
print(death[death['SYM_One Row Coded Symptoms']=='OVARIAN CANCER']['Created Year'].value_counts())
ovarian_death = death[death['SYM_One Row Coded Symptoms']=='OVARIAN CANCER']

plt.figure(figsize=(12,9))
ovarian_death.groupby('Created Month').count()['RA_Report #'].plot(kind='bar')
plt.title('Ovarian Cancer by Month Reported',fontsize=20)
plt.show()
print(death.groupby('Created Month')['Created Month'].value_counts())
ovarian_death['PRI_FDA Industry Name'].value_counts()
ovarian_death['PRI_Reported Brand/Product Name'].value_counts()
ovarian_death[ovarian_death['RA_CAERS Created Date']=='2015-01-28']
ovarian_death[ovarian_death['RA_CAERS Created Date']=='2017-02-27']