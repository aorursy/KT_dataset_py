import pandas as pd
import numpy as np
import requests
import datetime
import json
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('max_colwidth', 800)
CT_GOV_URL = 'https://clinicaltrials.gov/api/query/study_fields?expr=COVID-19&min_rnk=1&max_rnk=1000&fmt=json'
rct_fields = [
    'NCTId',
    'DesignAllocation',
    'DesignMasking',
    'LeadSponsorClass',
    'DesignPrimaryPurpose',
    'EnrollmentCount',
    'InterventionName',
    'InterventionType',
    'LastKnownStatus',
    'LeadSponsorName',
    'OutcomeMeasureTimeFrame',
    'LocationCountry',
    'StudyFirstPostDate',
    'StudyFirstPostDateType',
    'StudyFirstSubmitDate',
    'StudyFirstSubmitQCDate',
    'OverallStatus',
    'StudyType',
    'WhyStopped'
]
query_url = f'{CT_GOV_URL}&fields={",".join(rct_fields)}'
print(query_url)
r = requests.get(query_url)
r.status_code
j = json.loads(r.content)
df1 = pd.DataFrame(j['StudyFieldsResponse']['StudyFields'])
df1.shape
# Let's get the next 1000
CT_GOV_URL = 'https://clinicaltrials.gov/api/query/study_fields?expr=COVID-19&min_rnk=1001&max_rnk=2000&fmt=json'
query_url = f'{CT_GOV_URL}&fields={",".join(rct_fields)}'
r = requests.get(query_url)
r.status_code
j = json.loads(r.content)
df2 = pd.DataFrame(j['StudyFieldsResponse']['StudyFields'])
df2.shape
df_base = pd.concat([df1, df2])
df_base.shape
df = df_base
df.head()
temp = df['NCTId'].str[0]
temp.head()
for col in df.columns[1:]:
    print(col)
    df[col] = df[col].str[0]
df.head()
df["StudyFirstPostDate"] = df["StudyFirstPostDate"].astype("datetime64")
df.StudyFirstPostDate.max(), df.StudyFirstPostDate.min()
df.StudyFirstPostDate
date_time_str = '01/01/20'
date_obj = datetime.strptime(date_time_str, '%d/%m/%y')
date_obj
# Select trials registered only after Jan 1st 2020
df = df[df.StudyFirstPostDate>date_obj]
df.shape
df.StudyFirstPostDate.max(), df.StudyFirstPostDate.min()
df.groupby([df["StudyFirstPostDate"].dt.month]).NCTId.count()
# Let's plot the results on log scale since per month trials vary from 3 to 858
df.groupby([df["StudyFirstPostDate"].dt.month]).NCTId.count().plot(kind="bar", log=True)
# All covid-19 trials (post 01/01/2020)
df.shape[0]
# Select Interventional trials 
df = df[df.StudyType=='Interventional']
df.shape[0]
tt = df.groupby("InterventionType", as_index=False).NCTId.count().sort_values(by=['NCTId'], ascending=False)
tt
plt.figure(figsize=(14,8))
graph = sns.barplot(y="InterventionType", x="NCTId", data=tt, palette="Blues_d", log=True)
for p in graph.patches:
        graph.annotate('{:.0f}'.format(p.get_width()), (p.get_width()+(p.get_width()/10), p.get_y()+0.6),
                    ha='center', va='bottom', color= 'black')
tt = df.groupby("DesignPrimaryPurpose", as_index=False).NCTId.count().sort_values(by=['NCTId'], ascending=False)
tt
plt.figure(figsize=(14,6))
graph = sns.barplot(y="DesignPrimaryPurpose", x="NCTId", data=tt, palette="Blues_d")
for p in graph.patches:
        graph.annotate('{:.0f}'.format(p.get_width()), (p.get_width()+6, p.get_y()+0.6),
                    ha='center', va='bottom', color= 'black')
# Adding log scale to make the viz better
plt.figure(figsize=(14,6))
graph = sns.barplot(y="DesignPrimaryPurpose", x="NCTId", data=tt, palette="Blues_d", log=True)
for p in graph.patches:
        graph.annotate('{:.0f}'.format(p.get_width()), (p.get_width()+(p.get_width()/10), p.get_y()+0.6),
                    ha='center', va='bottom', color= 'black')
tt = df.loc[df.DesignPrimaryPurpose=='Treatment',]
tt.shape[0]
tt1 = tt.groupby("InterventionType", as_index=False).NCTId.count().sort_values(by=['NCTId'], ascending=False)
tt1
plt.figure(figsize=(14,8))
graph = sns.barplot(y="InterventionType", x="NCTId", data=tt1, palette="Blues_d", log=True)
for p in graph.patches:
        graph.annotate('{:.0f}'.format(p.get_width()), (p.get_width()+(p.get_width()/10), p.get_y()+0.6),
                    ha='center', va='bottom',
                    color= 'black')
graph.set_title('Intervention Type distribution where Purpose = Treatment')
tt = df.loc[(df.DesignPrimaryPurpose=='Treatment') & (df.InterventionType=='Drug'),]
covid_drugs = tt.groupby('InterventionName', as_index=False).NCTId.count().sort_values('NCTId', ascending=False)
covid_drugs.reset_index(drop=True)
covid_drugs.iloc[:15,]
df.InterventionName.replace(to_replace='Hydroxychloroquine Sulfate',value='Hydroxychloroquine',inplace=True)
df.InterventionName.replace(to_replace='Hydroxychloroquine (HCQ)',value='Hydroxychloroquine',inplace=True)
df.InterventionName.replace(to_replace='hydroxychloroquine',value='Hydroxychloroquine',inplace=True)

tt = df.loc[(df.DesignPrimaryPurpose=='Treatment') & (df.InterventionType=='Drug'),]
tt1 = tt.groupby('InterventionName', as_index=False).NCTId.count()
covid_drugs = tt1.sort_values('NCTId', ascending=False)
covid_drugs.iloc[:15,]
tt = df.loc[(df.DesignPrimaryPurpose=='Treatment') & (df.InterventionType=='Drug'),]
covid_drugs = tt.groupby('InterventionName', as_index=False).NCTId.count().sort_values('NCTId', ascending=False)
covid_drugs[covid_drugs.InterventionName.str.lower().str.contains('hydroxychloroquine')]
# Let's look at the top 15 records
tt = df.loc[(df.DesignPrimaryPurpose=='Treatment') & (df.InterventionType=='Biological'),]
covid_biological = tt.groupby('InterventionName', as_index=False).NCTId.count().sort_values('NCTId', ascending=False)
covid_biological.iloc[:15,]
tt = df.loc[(df.DesignPrimaryPurpose=='Treatment') & (df.InterventionType=='Dietary Supplement'),]
covid_supplement = tt.groupby('InterventionName', as_index=False).NCTId.count().sort_values('NCTId', ascending=False)
covid_supplement
len(df.LocationCountry.unique()), df.LocationCountry.unique()
df['EnrollmentCount'] = pd.to_numeric(df['EnrollmentCount'])
# Let's explore the treatment trial distribution across geographies.
tt = df[(df.DesignPrimaryPurpose=='Treatment') & (df.StudyType=='Interventional')]
tt.groupby('LocationCountry', as_index=False).agg({'EnrollmentCount': ['count','sum', 'mean']}).sort_values([('EnrollmentCount', 'count')], ascending=False)[:20]
df.EnrollmentCount.sum()
df.boxplot(column='EnrollmentCount')
df.EnrollmentCount.max()
df[df.EnrollmentCount==df.EnrollmentCount.max()].NCTId
tt = df[df.DesignPrimaryPurpose=='Treatment']
tt.EnrollmentCount.sum()
tt.boxplot(column='EnrollmentCount')
tt[tt.EnrollmentCount==tt.EnrollmentCount.max()].NCTId
tt.EnrollmentCount.describe()
tt.EnrollmentCount.plot(kind="bar", log=True)
len(df.LeadSponsorName.unique())
df.shape
tt1 = tt.groupby('LeadSponsorName', as_index=False).EnrollmentCount.sum()
covid_drugs = tt1.sort_values('EnrollmentCount', ascending=False)
covid_drugs.iloc[:15,]
tt1 = tt.groupby('LeadSponsorClass', as_index=False).EnrollmentCount.sum()
sponsor_class = tt1.sort_values('EnrollmentCount', ascending=False)
sponsor_class
ss = tt[tt.LeadSponsorClass=='INDUSTRY']
tt1 = ss.groupby('LeadSponsorName', as_index=False).agg({'EnrollmentCount': ['sum', 'count', 'mean']})
tt1.sort_values([('EnrollmentCount', 'sum')], ascending=False)[:20]
tt1.sort_values([('EnrollmentCount', 'count')], ascending=False)[:10]
# Sorted by total enrollment per sponsor
ss = tt[tt.LeadSponsorClass=='OTHER']
tt1 = ss.groupby('LeadSponsorName', as_index=False).agg({'EnrollmentCount': ['sum', 'count', 'mean']})
tt1.sort_values([('EnrollmentCount', 'sum')], ascending=False)[:10]
# Sorted by total number of trials per sponsor
ss = tt[tt.LeadSponsorClass=='OTHER']
tt1 = ss.groupby('LeadSponsorName', as_index=False).agg({'EnrollmentCount': ['count', 'sum', 'mean']})
tt1.sort_values([('EnrollmentCount', 'count')], ascending=False)[:10]
# Sorted by total enrollment per sponsor
ss = tt[tt.LeadSponsorClass=='OTHER_GOV']
tt1 = ss.groupby('LeadSponsorName', as_index=False).agg({'EnrollmentCount': ['sum', 'count', 'mean']})
tt1.sort_values([('EnrollmentCount', 'sum')], ascending=False)[:10]
# Sorted by total number of trials per sponsor
ss = tt[tt.LeadSponsorClass=='OTHER_GOV']
tt1 = ss.groupby('LeadSponsorName', as_index=False).agg({'EnrollmentCount': ['sum', 'count', 'mean']})
tt1.sort_values([('EnrollmentCount', 'count')], ascending=False)[:5]
df.groupby('OverallStatus', as_index=False).NCTId.count().sort_values([('NCTId')], ascending=False)
df.groupby('WhyStopped', as_index=False).NCTId.count().sort_values([('NCTId')], ascending=False)
# In all interventional trials
df.groupby('DesignMasking', as_index=False).NCTId.count().sort_values([('NCTId')], ascending=False)
# In treatment-only interventional trials
tt.groupby('DesignMasking', as_index=False).NCTId.count().sort_values([('NCTId')], ascending=False)
# Masking in Hydroxychloroquine (as mono or combo therapy) trials
tt[tt.InterventionName.str.lower().str.contains('hydroxychloroquine')].groupby('DesignMasking', as_index=False).NCTId.count().sort_values([('NCTId')], ascending=False)
tt.groupby('DesignAllocation', as_index=False).NCTId.count().sort_values([('NCTId')], ascending=False)
tt[tt.DesignAllocation=='Randomized'].groupby(['DesignAllocation','DesignMasking'], as_index=False).NCTId.count().sort_values([('NCTId')], ascending=False)
# Hydroxychloroquine monotherapy
tt[(tt.InterventionName=='Hydroxychloroquine') & (tt.DesignAllocation=='Randomized')].groupby(['DesignAllocation','DesignMasking'], as_index=False).NCTId.count().sort_values([('NCTId')], ascending=False)
# Hydroxychloroquine mono or combo therapy
tt[(tt.InterventionName.str.lower().str.contains('hydroxychloroquine')) & (tt.DesignAllocation=='Randomized')].groupby(['DesignAllocation','DesignMasking'], as_index=False).NCTId.count().sort_values([('NCTId')], ascending=False)
# Remdesivir mono or combo therapy
tt[(tt.InterventionName.str.lower().str.contains('remdesivir')) & (tt.DesignAllocation=='Randomized')].groupby(['DesignAllocation','DesignMasking'], as_index=False).NCTId.count().sort_values([('NCTId')], ascending=False)