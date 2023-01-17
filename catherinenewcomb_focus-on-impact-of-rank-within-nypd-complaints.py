# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/civilian-complaints-against-nyc-police-officers/allegations_202007271729.csv')
df.head()
df.info()
rank_abbrevs = pd.read_csv('/kaggle/input/nypd-ranks/rank_abbreviations.csv')

rank_abbrevs.drop('Rank', axis=1, inplace=True)

rank_abbrevs.info()
rank_dict = rank_abbrevs['Abbreviation'].to_dict()

rank_dict = dict((v,k) for k,v in rank_dict.items())



rank_dict
def enumerate_rank(officer):

    if officer['rank_abbrev_incident'] in rank_dict.keys():

        number = rank_dict[officer['rank_abbrev_incident']]

        return number

   

    

df['rank_incident_number'] = df.apply(enumerate_rank, axis=1)
df.rank_incident_number.value_counts()
def enumerate_rank(officer):

    if officer['rank_abbrev_now'] in rank_dict.keys():

        number = rank_dict[officer['rank_abbrev_now']]

        return number

   

    

df['rank_now_number'] = df.apply(enumerate_rank, axis=1)
df.loc[df.rank_incident_number >= 22, "rank_incident_number"] = 22

df.loc[df.rank_now_number >= 22, "rank_now_number"] = 22



df.rank_now_number.value_counts()
df["rank_change"] = df["rank_now_number"] - df["rank_incident_number"]
sum(df.unique_mos_id.unique())
sum(df.unique_mos_id.duplicated())
df_officers_first = df.copy()
#make complaint info a datetime

df_officers_first['complaint_dt'] = pd.to_datetime(dict(year=df_officers_first.year_received, month=df_officers_first.month_received, day = 1))
df_officers_first.complaint_dt.sample(5)

df_officers_first.unique_mos_id.astype(int)

df_officers_first.info()
df_officers_first = df_officers_first.sort_values('complaint_dt').drop_duplicates('unique_mos_id',keep='first')

sum(df_officers_first.unique_mos_id.duplicated())
officer_complaints = pd.DataFrame(df.unique_mos_id.value_counts())
officer_complaints
officer_complaints.rename(columns = {'unique_mos_id' : 'number_complaints'}, inplace=True)

officer_complaints.index.name = 'unique_mos_id'

officer_complaints
df_officers_first = df_officers_first.join(officer_complaints, on='unique_mos_id')

df_officers_first.info()
import matplotlib.pyplot as plt



plt.scatter(df_officers_first['rank_change'], df_officers_first['number_complaints']);

plt.ylabel('Number of Complaints')

plt.xlabel('Change in Rank Since First Complaint')

import statsmodels.api as sm



df_officers_first.dropna(subset=['rank_change'], inplace=True)



df_officers_first["intercept"] = 1

lm = sm.OLS(df_officers_first["number_complaints"], df_officers_first[["intercept", "rank_change"]])

results = lm.fit()

results.summary()



import plotly.express as px

fig = px.scatter(df_officers_first, x="rank_change", y = "number_complaints", trendline='ols')

fig.show();

fig1 = plt.hist(df["rank_incident_number"], alpha=.7);

fig2 = plt.hist(df["rank_now_number"], alpha = .5);
df.board_disposition.unique()
df_substantiated = df[df['board_disposition'].str.contains("Substantiated*")]
df_substantiated.board_disposition.unique()

df_substantiated.rank_now_number.dropna()
fig1 = plt.hist(df_substantiated["rank_incident_number"], alpha=.7);

fig2 = plt.hist(df_substantiated["rank_now_number"], alpha = .5);
df_substantiated.info()
officer_complaints_subst = pd.DataFrame(df_substantiated.unique_mos_id.value_counts())

officer_complaints_subst.rename(columns = {'unique_mos_id' : 'number_complaints'}, inplace=True)

officer_complaints_subst.index.name = 'unique_mos_id'





df_substantiated.complaint_dt = df_substantiated['complaint_dt'] = pd.to_datetime(dict(year=df_substantiated.year_received, month=df_substantiated.month_received, day = 1))

df_substantiated_first = df_substantiated.sort_values('complaint_dt').drop_duplicates('unique_mos_id',keep='first')



df_substantiated_first = df_substantiated_first.join(officer_complaints_subst, on='unique_mos_id')



fig = px.scatter(df_substantiated_first, x="rank_change", y = "number_complaints", trendline='ols')

fig.show();
df_substantiated_first.dropna(subset=['rank_change'], inplace=True)



df_substantiated_first["intercept"] = 1

lm = sm.OLS(df_substantiated_first["number_complaints"], df_substantiated_first[["intercept", "rank_change"]])

results = lm.fit()

results.summary()
substantiated_rate = df_substantiated.shape[0] / df.shape[0]

print(substantiated_rate)
df.complainant_ethnicity.value_counts().plot(kind='bar');
df_substantiated.complainant_ethnicity.value_counts().plot(kind='bar');
df.info()
df.groupby('fado_type')["complainant_ethnicity"].value_counts().plot(kind='bar', figsize=(15, 10))
%matplotlib inline



fado_pie = df.groupby('complainant_ethnicity')["fado_type"].value_counts().unstack()



#time_pie = df.groupby("no_show").time_diff_bins.value_counts().unstack()

labels = "Abuse of Authority", "Discourtesy", "Force", "Offensive Language"

titles= "Black", "Hispanic", "White", "Unknown", "Other Race", "Asian", "Refused", "American Indian"

plt.style.use("bmh")



fado_pie.plot(kind="pie", subplots=True, sharey=False, sharex = False, layout = (8, 1), 

             legend=False, figsize=(80,60), autopct='%1.1f%%', fontsize= 11);
fado_by_ethnicity = pd.DataFrame(df.groupby('fado_type')["complainant_ethnicity"].value_counts().unstack())
