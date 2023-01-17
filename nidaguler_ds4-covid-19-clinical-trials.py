import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os



import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/covid19-clinical-trials-dataset/COVID clinical trials.csv")
df.head()
df.tail()
for i,col in enumerate(df.columns):

    print(i+1,". column is", col)
df.columns=["rank","nct_number","title","acronym","status","study_results","conditions","interventions",

           "outcome_measures","sponsor_collaborators","gender","age","phases","enrollment","funded_bys",

           "study_type","study_designs","other_ids","start_date","primary_completion_date","completion_date","first_posted",

           "results_first_posted","last_update_posted","locations","study_documents","url"]
df.head()
df.drop(["nct_number","other_ids","start_date","primary_completion_date","completion_date","first_posted",

        "results_first_posted","last_update_posted","study_documents","url"],axis=1,inplace=True)
df.head()
df.drop(["phases","enrollment"],axis=1,inplace=True)
df.head()
df.isna().sum()
df=df.dropna()
df.title.unique()
df.drop(["title"],axis=1,inplace=True)
df.acronym.unique()
df.drop(["acronym"],axis=1,inplace=True)
df.status.unique()
df.study_results.unique()
df.conditions.unique()
df.drop(["conditions"],axis=1,inplace=True)
df.interventions.unique()
df.drop(["interventions"],axis=1,inplace=True)
df.outcome_measures.unique()
df.drop(["outcome_measures"],axis=1,inplace=True)
df.sponsor_collaborators.unique()
df.drop(["sponsor_collaborators"],axis=1,inplace=True)
df.gender.unique()
df.age.unique()
df.funded_bys.unique()
df.study_type.unique()
df.study_designs.unique()
df.drop(["study_designs"],axis=1,inplace=True)
df.locations.unique()
df.drop(["locations"],axis=1,inplace=True)
df.head()
df.isna().sum()
df.info()
df.status.unique()
plt.figure(figsize=(20,7))

sns.barplot(x=df['status'].value_counts().index,

              y=df['status'].value_counts().values)

plt.xlabel('status')

plt.ylabel('Frequency')

plt.title('Show of status Bar Plot')

plt.show()
sns.barplot(x=df['study_results'].value_counts().index,

              y=df['study_results'].value_counts().values)

plt.xlabel('study_results')

plt.ylabel('Frequency')

plt.title('Show of study_results Bar Plot')

plt.show()
sns.barplot(x=df['study_results'].value_counts().index,

              y=df['study_results'].value_counts().values)

plt.xlabel('study_results')

plt.ylabel('Frequency')

plt.title('Show of study_results Bar Plot')

plt.show()
sns.barplot(x=df['gender'].value_counts().index,

              y=df['gender'].value_counts().values)

plt.xlabel('gender')

plt.ylabel('Frequency')

plt.title('Show of gender Bar Plot')

plt.show()
plt.figure(figsize=(20,3))

sns.barplot(x=df['funded_bys'].value_counts().index,

              y=df['funded_bys'].value_counts().values)

plt.xlabel('funded_bys')

plt.ylabel('Frequency')

plt.title('Show of funded_bys Bar Plot')

plt.show()
df.columns


sns.barplot(x=df['study_type'].value_counts().index,

              y=df['study_type'].value_counts().values)

plt.xlabel('study_type')

plt.ylabel('Frequency')

plt.title('Show of study_type Bar Plot')

plt.show()
df.age.unique()
df["age"]=df["age"].apply(lambda x:str(x).replace('Years','')if 'Years' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('\xa0','')if '\xa0' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('to','')if 'to' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('up','')if 'up' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('18','')if '18' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('Year','')if 'Year' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('48','')if '48' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('99','')if '99' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('60','')if '60' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('and','')if 'and' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('70','')if '70' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('1','')if '1' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('30','')if '30' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('90','')if '90' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('00','')if '00' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('6','')if '6' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('40','')if '40' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('80','')if '80' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('75','')if '75' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('2','')if '2' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('8','')if '8' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('5','')if '5' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('9','')if '9' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('0','')if '0' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('3','')if '3' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('4','')if '4' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('7','')if '7' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace('(','')if '(' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace(')','')if ')' in str(x) else str(x))
df["age"]=df["age"].apply(lambda x:str(x).replace(' ','')if ' ' in str(x) else str(x))
df.age.unique()
plt.figure(figsize=(20,3))

sns.barplot(x=df['age'].value_counts().index,

              y=df['age'].value_counts().values)

plt.xlabel('age')

plt.ylabel('Frequency')

plt.title('Show of age Bar Plot')

plt.xticks(rotation=90)

plt.show()
df.drop(["age"],axis=1,inplace=True)
df.head()