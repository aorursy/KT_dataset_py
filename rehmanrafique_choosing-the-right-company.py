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
import numpy as np 

import pandas as pd 

import seaborn as sns

#sns.set(color_codes=True)

#sns.set_style({"axes.facecolor": "1.0", 'grid.linestyle': '--', 'grid.color': '.8'})

sns.set_style("whitegrid")

#colors = ["#F28E2B", "#4E79A7","#79706E"]



colors = {'Data Science': "#F28E2B", 'Data/SW Engineering': "#4E79A7", 'Business/Management': "#79706E"}

colors_entr = {'Large Enterprise': "#17BECF", 'SME': "#BCBD22", 'SMB': "#C7C7C7", 'NA': "#FF7F0E"}



import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from IPython.display import display, HTML

init_notebook_mode(connected=True)

display(HTML("""

<style>

.output {

    display: flex;

    align-items: left;

    text-align: center;

}

</style>

"""))



data_19 = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv", skiprows = range(1,2))

other_responses = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')



conditions = [

    (data_19['Q5'] == 'Data Scientist') | (data_19['Q5'] == 'Statistician') | (data_19['Q5'] == 'Data Analyst') | (data_19['Q5'] == 'Research Scientist'), 

    (data_19['Q5'] == 'Software Engineer') | (data_19['Q5'] == 'Data Engineer') | (data_19['Q5'] == 'DBA/Database Engineer'),

    (data_19['Q5'] == 'Business Analyst') | (data_19['Q5'] == 'Product/Project Manager'),

    (data_19['Q5'] == 'Student') | (data_19['Q5'] == 'Not employed')]

choices = ['Data Science', 'Data/SW Engineering', 'Business/Management','Student']

data_19['JobDomain'] = np.select(conditions, choices, default='Others')





conditions = [

    (data_19['Q6'] == '1000-9,999 employees') | (data_19['Q6'] == '> 10,000 employees') , 

    (data_19['Q6'] == '50-249 employees') | (data_19['Q6'] == '250-999 employees') ,

    (data_19['Q6'] == '0-49 employees') ]

choices = ['Large Enterprise','SME', 'SMB', ]

data_19['Vertical'] = np.select(conditions, choices, default='NA')



conditions = [

    (data_19['Q3'] == 'United States of America') , 

    (data_19['Q3'] == 'India') ,

    (data_19['Q3'] == 'Russia'),

    (data_19['Q3'] == 'Japan'),

    (data_19['Q3'] == 'Brazil'),]

choices = ['USA','India', 'Russia','Japan','Brazil' ]

data_19['CountryGroup'] = np.select(conditions, choices, default='NA')



compensation_replace_dict = {

    '$0-999': '< 10,000','1,000-1,999': '< 10,000','2,000-2,999': '< 10,000','3,000-3,999': '< 10,000',

    '4,000-4,999': '< 10,000','5,000-7,499': '< 10,000','7,500-9,999': '< 10,000','10,000-14,999': '10,000 - 50,000',

    '15,000-19,999': '10,000 - 50,000','20,000-24,999': '10,000 - 50,000','25,000-29,999': '10,000 - 50,000',

    '30,000-39,999': '10,000 - 50,000','40,000-49,999': '10,000 - 50,000','50,000-59,999': '50,000 - 99,000',

    '60,000-69,999': '50,000 - 99,000','70,000-79,999': '50,000 - 99,000','80,000-89,999': '50,000 - 99,000',

    '90,000-99,999': '50,000 - 99,000','100,000-124,999': '> 100,000','125,000-149,999': '> 100,000',

    '150,000-199,999': '> 100,000','200,000-249,999': '> 100,000','250,000-299,999': '> 100,000',

    '300,000-500,000': '> 100,000','> $500,000': '> 100,000'}



data_19['Q10'] = data_19['Q10'].replace(compensation_replace_dict)



df = data_19.query(" JobDomain != 'Student' & JobDomain != 'Others'")
ax = sns.countplot(data=df, x="JobDomain",palette=colors)#sns.color_palette(colors))



ax.set_title('Number of Respondents by Title/Job Category\n')

ax.set_ylabel('')

ax.set_xlabel('')





plt.show()
fig, axs = plt.subplots(figsize=(10, 6),sharey=True)

country = (df.groupby(['JobDomain'])['CountryGroup']

                     .value_counts(normalize=True)

                     .rename('Percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('CountryGroup'))



r = sns.barplot(x="CountryGroup", y="Percentage", hue="JobDomain", data=country[country['CountryGroup']!= 'NA'], palette=colors)

r.set_title('Job Groups Compared by Countries\n')

r.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),ncol=2)

_ = plt.setp(r.get_xticklabels(), rotation=90)
edu = (df.groupby(['JobDomain'])['Q4']

                     .value_counts(normalize=True)

                     .rename('Percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q4'))



coding = (df.groupby(['JobDomain'])['Q15']

                     .value_counts(normalize=True)

                     .rename('Percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q15'))







fig, axs = plt.subplots(ncols=2,figsize=(20, 6),sharey=True)

plt.subplots_adjust(wspace=0.4)

p = sns.barplot(x="Q4", y="Percentage", hue="JobDomain", data=edu, ax=axs[0],palette=colors)

q = sns.barplot(x="Q15", y="Percentage", hue="JobDomain", data=coding, ax=axs[1],palette=colors)



p.set_title('Comparison by Education \n')

q.set_title('Years of coding experience for data analysis\n')

_ = plt.setp(p.get_xticklabels(), rotation=90)

_ = plt.setp(q.get_xticklabels(), rotation=90)
fig, axs = plt.subplots(figsize=(10, 6),sharey=True)

pay = (df.groupby(['JobDomain'])['Q10']

                     .value_counts(normalize=True)

                     .rename('Percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q10'))



r = sns.barplot(x="Q10", y="Percentage", hue="JobDomain", data=pay[:-1], palette=colors)

r.set_title('Annual Compensation\n')

r.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),ncol=2)

_ = plt.setp(r.get_xticklabels(), rotation=90)

#will only work with data science community created in previous sections

df_ds = data_19.query(" JobDomain == 'Data Science' ")



ax = sns.countplot(data=df_ds, x="Vertical",palette=colors_entr)



ax.set_title('Number people from DS Cohort working in different businesses\n')

ax.set_ylabel('')

ax.set_xlabel('')





plt.show()
df_ds_Excl = df_ds.query(" Vertical != 'NA' ")



fig, axs = plt.subplots(figsize=(10, 6),sharey=True)

country1 = (df_ds_Excl.groupby(['Vertical'])['CountryGroup']

                     .value_counts(normalize=True)

                     .rename('Percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('CountryGroup'))



r = sns.barplot(x="CountryGroup", y="Percentage", hue="Vertical", data=country1[country1['CountryGroup']!= 'NA'], palette=colors_entr)

r.set_title('Job Cohorts Compared by Top 5 Countries\n')

r.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),ncol=4)

_ = plt.setp(r.get_xticklabels(), rotation=90)
educ = (df_ds_Excl.groupby(['Vertical'])['Q4']

                     .value_counts(normalize=True)

                     .rename('Percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q4'))



sal = (df_ds_Excl.groupby(['Vertical'])['Q10']

                     .value_counts(normalize=True)

                     .rename('Percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q10'))







fig, axs = plt.subplots(ncols=2,figsize=(18, 6),sharey=True)

plt.subplots_adjust(wspace=0.2)

p = sns.barplot(x="Q4", y="Percentage", hue="Vertical", data=educ, ax=axs[0],palette=colors_entr)

q = sns.barplot(x="Q10", y="Percentage", hue="Vertical", data=sal, ax=axs[1],palette=colors_entr)



p.set_title('Education difference in DS community by Org type\n')

q.set_title('Salary difference in DS community by Org type\n')

_ = plt.setp(p.get_xticklabels(), rotation=90)

_ = plt.setp(q.get_xticklabels(), rotation=90)
df_ds_Excl = df_ds.query(" Vertical != 'NA' ")





ML = (df_ds_Excl.groupby(['Vertical'])['Q8']

                     .value_counts(normalize=True)

                     .rename('Percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q8'))



tool = (df_ds_Excl.groupby(['Vertical'])['Q14']

                     .value_counts(normalize=True)

                     .rename('Percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q14'))







fig, axs = plt.subplots(ncols=2,figsize=(18, 6),sharey=True)

plt.subplots_adjust(wspace=0.2)

p = sns.barplot(x="Q8", y="Percentage", hue="Vertical", data=ML, ax=axs[0],palette=colors_entr)

q = sns.barplot(x="Q14", y="Percentage", hue="Vertical", data=tool, ax=axs[1],palette=colors_entr)



p.set_title('ML methods adoptation by Org type\n')

q.set_title('Tools used to analyze data by Org type\n')

_ = plt.setp(p.get_xticklabels(), rotation=90)

_ = plt.setp(q.get_xticklabels(), rotation=90)
mysql_ds = 100 * df_ds_Excl.groupby(['Q34_Part_1']).size()/len(df_ds_Excl)

sql_ds = 100 * df_ds_Excl.groupby(['Q34_Part_4']).size()/len(df_ds_Excl)

orc_ds = 100 * df_ds_Excl.groupby(['Q34_Part_5']).size()/len(df_ds_Excl)

post_ds = 100 * df_ds_Excl.groupby(['Q34_Part_2']).size()/len(df_ds_Excl)

lite_ds = 100 * df_ds_Excl.groupby(['Q34_Part_3']).size()/len(df_ds_Excl)

db_perc_all = pd.concat([mysql_ds, sql_ds,orc_ds,post_ds,lite_ds], axis=0)



db_perc_all_srt = db_perc_all.sort_values(ascending=False)



q = sns.barplot(db_perc_all_srt.index, db_perc_all_srt.values)

q.set_title('Top DB Engines usage by DS Community\n')

q.set(ylabel='Percentage')

_ = plt.setp(q.get_xticklabels(), rotation=90)
ora_ds = 100 * df_ds_Excl.groupby(['Q34_Part_5']).size()/len(df_ds_Excl)

sql_ds = 100 * df_ds_Excl.groupby(['Q34_Part_4']).size()/len(df_ds_Excl)

db_perc = pd.concat([ora_ds, sql_ds], axis=0)



df_ds_large = df_ds_Excl.query(" Vertical == 'Large Enterprise' ")



ora_ds_large = 100 * df_ds_large.groupby(['Q34_Part_5']).size()/len(df_ds_large)

sql_ds_large = 100 * df_ds_large.groupby(['Q34_Part_4']).size()/len(df_ds_large)

db_perc_large = pd.concat([ora_ds_large, sql_ds_large], axis=0)





fig, axs = plt.subplots(ncols=2,figsize=(18, 6),sharey=True)

plt.subplots_adjust(wspace=0.2)

p = sns.barplot(db_perc.index, db_perc.values, ax=axs[0])

q = sns.barplot(db_perc_large.index, db_perc_large.values, ax=axs[1])



p.set_title('SQL Server/Oracle DB usage by DS Community\n')

q.set_title('SQL Server/Oracle DB usage by DS Community in Large Organisations\n')



p.set(ylabel='Percentage')



_ = plt.setp(p.get_xticklabels(), rotation=0)

_ = plt.setp(q.get_xticklabels(), rotation=0)
post_ds = 100 * df_ds_Excl.groupby(['Q34_Part_2']).size()/len(df_ds_Excl)





df_ds_med = df_ds_Excl.query(" Vertical == 'SME' ")

post_ds_med = 100 * df_ds_med.groupby(['Q34_Part_2']).size()/len(df_ds_med)



fig, axs = plt.subplots(ncols=2,figsize=(18, 6),sharey=True)

plt.subplots_adjust(wspace=0.2)



p = sns.barplot(post_ds.index, post_ds.values, ax=axs[0])

q = sns.barplot(post_ds_med.index, post_ds_med.values, ax=axs[1])







p.set_title('Enterprise DB usage by DS Community\n')

q.set_title('Enterprise DB usage by DS Community in SME Organisations\n')



p.set(ylabel='Percentage')

p.set(xlabel='')

q.set(xlabel='')



_ = plt.setp(p.get_xticklabels(), rotation=0)

_ = plt.setp(q.get_xticklabels(), rotation=0)
mysql_ds = 100 * df_ds_Excl.groupby(['Q34_Part_1']).size()/len(df_ds_Excl)

lite_ds = 100 * df_ds_Excl.groupby(['Q34_Part_3']).size()/len(df_ds_Excl)

db_perc = pd.concat([mysql_ds, lite_ds], axis=0)



df_ds_small = df_ds_Excl.query(" Vertical == 'SMB' ")



mysql_small = 100 * df_ds_small.groupby(['Q34_Part_1']).size()/len(df_ds_small)

lite_ds_small = 100 * df_ds_small.groupby(['Q34_Part_3']).size()/len(df_ds_small)

db_perc_small = pd.concat([mysql_small, lite_ds_small], axis=0)





fig, axs = plt.subplots(ncols=2,figsize=(18, 6),sharey=True)

#plt.subplots_adjust(wspace=0.2)

p = sns.barplot(db_perc.index, db_perc.values, ax=axs[0])

q = sns.barplot(db_perc_small.index, db_perc_small.values, ax=axs[1])



p.set_title('Enterprise DB usage by DS Community\n')

q.set_title('Enterprise DB usage by DS Community in SMB (Small Size) Organisations\n')



p.set(ylabel='Percentage')



_ = plt.setp(p.get_xticklabels(), rotation=0)

_ = plt.setp(q.get_xticklabels(), rotation=0)
gcp_ds = 100 * df_ds_Excl.groupby(['Q29_Part_1']).size()/len(df_ds_Excl)

aws_ds = 100 * df_ds_Excl.groupby(['Q29_Part_2']).size()/len(df_ds_Excl)

azure_ds = 100 * df_ds_Excl.groupby(['Q29_Part_3']).size()/len(df_ds_Excl)

db_perc_all = pd.concat([gcp_ds, aws_ds,azure_ds], axis=0)



db_perc_all_srt = db_perc_all.sort_values(ascending=False)



q = sns.barplot(db_perc_all_srt.index, db_perc_all_srt.values)

q.set_title('Top 3 Cloud Computing Platforms usage by DS Community\n')

q.set(ylabel='Percentage')

_ = plt.setp(q.get_xticklabels(), rotation=90)
gcp_ds_lg = 100 * df_ds_large.groupby(['Q29_Part_1']).size()/len(df_ds_large)

aws_ds_lg = 100 * df_ds_large.groupby(['Q29_Part_2']).size()/len(df_ds_large)

azure_ds_lg = 100 * df_ds_large.groupby(['Q29_Part_3']).size()/len(df_ds_large)

db_perc_lg = pd.concat([gcp_ds_lg, aws_ds_lg,azure_ds_lg], axis=0)



db_perc_lg = db_perc_lg.sort_values(ascending=False)



gcp_ds_med = 100 * df_ds_med.groupby(['Q29_Part_1']).size()/len(df_ds_med)

aws_ds_med = 100 * df_ds_med.groupby(['Q29_Part_2']).size()/len(df_ds_med)

azure_ds_med = 100 * df_ds_med.groupby(['Q29_Part_3']).size()/len(df_ds_med)

db_perc_med = pd.concat([gcp_ds_med, aws_ds_med,azure_ds_med], axis=0)



db_perc_med = db_perc_med.sort_values(ascending=False)



gcp_ds_sm = 100 * df_ds_small.groupby(['Q29_Part_1']).size()/len(df_ds_small)

aws_ds_sm = 100 * df_ds_small.groupby(['Q29_Part_2']).size()/len(df_ds_small)

azure_ds_sm = 100 * df_ds_small.groupby(['Q29_Part_3']).size()/len(df_ds_small)

db_perc_sm = pd.concat([gcp_ds_sm, aws_ds_sm,azure_ds_sm], axis=0)



db_perc_sm = db_perc_sm.sort_values(ascending=False)



fig, axs = plt.subplots(ncols=3,figsize=(18, 6),sharey=True)

#plt.subplots_adjust(wspace=0.2)

l = sns.barplot(db_perc_lg.index, db_perc_lg.values, ax=axs[0])

m = sns.barplot(db_perc_med.index, db_perc_med.values, ax=axs[1])

s = sns.barplot(db_perc_sm.index, db_perc_sm.values, ax=axs[2])





l.set_title('CCP usage by DS Community in Large Enterprise\n')

m.set_title('CCP usage by DS Community in SME\n')

s.set_title('CCP usage by DS Community in SMB\n')



l.set(ylabel='Percentage')



_ = plt.setp(l.get_xticklabels(), rotation=90)

_ = plt.setp(m.get_xticklabels(), rotation=90)

_ = plt.setp(s.get_xticklabels(), rotation=90)
ec2_ds = 100 * df_ds_Excl.groupby(['Q30_Part_1']).size()/len(df_ds_Excl)

gce_ds = 100 * df_ds_Excl.groupby(['Q30_Part_2']).size()/len(df_ds_Excl)

#lamb_ds = 100 * df_ds_Excl.groupby(['Q30_Part_3']).size()/len(df_ds_Excl)

azure_vm_ds = 100 * df_ds_Excl.groupby(['Q30_Part_4']).size()/len(df_ds_Excl)

#g_ae_ds = 100 * df_ds_Excl.groupby(['Q30_Part_5']).size()/len(df_ds_Excl)

#g_cf_ds = 100 * df_ds_Excl.groupby(['Q30_Part_6']).size()/len(df_ds_Excl)

#aws_eb_ds = 100 * df_ds_Excl.groupby(['Q30_Part_7']).size()/len(df_ds_Excl)

#gk_ds = 100 * df_ds_Excl.groupby(['Q30_Part_8']).size()/len(df_ds_Excl)

#aws_b_ds = 100 * df_ds_Excl.groupby(['Q30_Part_9']).size()/len(df_ds_Excl)

#azure_c_ds = 100 * df_ds_Excl.groupby(['Q30_Part_10']).size()/len(df_ds_Excl)



cc_prod_all = pd.concat([ec2_ds, gce_ds,azure_vm_ds], axis=0)



cc_prod_all_srt = cc_prod_all.sort_values(ascending=False)



q = sns.barplot(cc_prod_all_srt.index, cc_prod_all_srt.values)

q.set_title('Top 3 Cloud Computing Products usage by DS Community\n')

q.set(ylabel='Percentage')

_ = plt.setp(q.get_xticklabels(), rotation=90)
ec2_ds_lg = 100 * df_ds_large.groupby(['Q30_Part_1']).size()/len(df_ds_large)

gce_ds_lg = 100 * df_ds_large.groupby(['Q30_Part_2']).size()/len(df_ds_large)

azure_vm_ds_lg = 100 * df_ds_large.groupby(['Q30_Part_4']).size()/len(df_ds_large)

db_perc_lg = pd.concat([ec2_ds_lg, gce_ds_lg,azure_vm_ds_lg], axis=0)



db_perc_lg = db_perc_lg.sort_values(ascending=False)



ec2_ds_med = 100 * df_ds_med.groupby(['Q30_Part_1']).size()/len(df_ds_med)

gce_ds_med = 100 * df_ds_med.groupby(['Q30_Part_2']).size()/len(df_ds_med)

azure_vm_ds_med = 100 * df_ds_med.groupby(['Q30_Part_4']).size()/len(df_ds_med)

db_perc_med = pd.concat([ec2_ds_med, gce_ds_med,azure_vm_ds_med], axis=0)



db_perc_med = db_perc_med.sort_values(ascending=False)



ec2_ds_sm = 100 * df_ds_small.groupby(['Q30_Part_1']).size()/len(df_ds_small)

gce_ds_sm = 100 * df_ds_small.groupby(['Q30_Part_2']).size()/len(df_ds_small)

azure_vm_ds_sm = 100 * df_ds_small.groupby(['Q30_Part_4']).size()/len(df_ds_small)

db_perc_sm = pd.concat([ec2_ds_sm, gce_ds_sm,azure_vm_ds_sm], axis=0)



db_perc_sm = db_perc_sm.sort_values(ascending=False)



fig, axs = plt.subplots(ncols=3,figsize=(18, 6),sharey=True)

#plt.subplots_adjust(wspace=0.2)

l = sns.barplot(db_perc_lg.index, db_perc_lg.values, ax=axs[0])

m = sns.barplot(db_perc_med.index, db_perc_med.values, ax=axs[1])

s = sns.barplot(db_perc_sm.index, db_perc_sm.values, ax=axs[2])





l.set_title('Cloud products used by DS Community in Large Enterprise\n')

m.set_title('Cloud products used by DS Community in SME\n')

s.set_title('Cloud products used by DS Community in SMB\n')



l.set(ylabel='Percentage')



_ = plt.setp(l.get_xticklabels(), rotation=90)

_ = plt.setp(m.get_xticklabels(), rotation=90)

_ = plt.setp(s.get_xticklabels(), rotation=90)
q1_ds = 100 * df_ds_Excl.groupby(['Q24_Part_1']).size()/len(df_ds_Excl)

q2_ds = 100 * df_ds_Excl.groupby(['Q24_Part_2']).size()/len(df_ds_Excl)

q3_ds = 100 * df_ds_Excl.groupby(['Q24_Part_3']).size()/len(df_ds_Excl)

q4_ds = 100 * df_ds_Excl.groupby(['Q24_Part_4']).size()/len(df_ds_Excl)

q5_ds = 100 * df_ds_Excl.groupby(['Q24_Part_5']).size()/len(df_ds_Excl)

q6_ds = 100 * df_ds_Excl.groupby(['Q24_Part_6']).size()/len(df_ds_Excl)

q7_ds = 100 * df_ds_Excl.groupby(['Q24_Part_7']).size()/len(df_ds_Excl)

q8_ds = 100 * df_ds_Excl.groupby(['Q24_Part_8']).size()/len(df_ds_Excl)

q9_ds = 100 * df_ds_Excl.groupby(['Q24_Part_9']).size()/len(df_ds_Excl)

q10_ds = 100 * df_ds_Excl.groupby(['Q24_Part_10']).size()/len(df_ds_Excl)



algo_prod_all = pd.concat([q1_ds, q2_ds,q3_ds,q4_ds,q5_ds,q6_ds,q7_ds,q8_ds,q9_ds,q10_ds], axis=0)



algo_prod_all_srt = algo_prod_all.sort_values(ascending=False)



q = sns.barplot(algo_prod_all_srt.index, algo_prod_all_srt.values)

q.set_title('Usage of ML Algos by DS Community\n')

q.set(ylabel='Percentage')

_ = plt.setp(q.get_xticklabels(), rotation=90)
q1_ds = 100 * df_ds_large.groupby(['Q24_Part_1']).size()/len(df_ds_large)

q2_ds = 100 * df_ds_large.groupby(['Q24_Part_2']).size()/len(df_ds_large)

q3_ds = 100 * df_ds_large.groupby(['Q24_Part_3']).size()/len(df_ds_large)

q4_ds = 100 * df_ds_large.groupby(['Q24_Part_4']).size()/len(df_ds_large)

q5_ds = 100 * df_ds_large.groupby(['Q24_Part_5']).size()/len(df_ds_large)

q6_ds = 100 * df_ds_large.groupby(['Q24_Part_6']).size()/len(df_ds_large)

q7_ds = 100 * df_ds_large.groupby(['Q24_Part_7']).size()/len(df_ds_large)

q8_ds = 100 * df_ds_large.groupby(['Q24_Part_8']).size()/len(df_ds_large)

q9_ds = 100 * df_ds_large.groupby(['Q24_Part_9']).size()/len(df_ds_large)

q10_ds = 100 * df_ds_large.groupby(['Q24_Part_10']).size()/len(df_ds_large)



algo_prod_lg = pd.concat([q1_ds, q2_ds,q3_ds,q4_ds,q5_ds,q6_ds,q7_ds,q8_ds,q9_ds,q10_ds], axis=0)





q1_ds = 100 * df_ds_med.groupby(['Q24_Part_1']).size()/len(df_ds_med)

q2_ds = 100 * df_ds_med.groupby(['Q24_Part_2']).size()/len(df_ds_med)

q3_ds = 100 * df_ds_med.groupby(['Q24_Part_3']).size()/len(df_ds_med)

q4_ds = 100 * df_ds_med.groupby(['Q24_Part_4']).size()/len(df_ds_med)

q5_ds = 100 * df_ds_med.groupby(['Q24_Part_5']).size()/len(df_ds_med)

q6_ds = 100 * df_ds_med.groupby(['Q24_Part_6']).size()/len(df_ds_med)

q7_ds = 100 * df_ds_med.groupby(['Q24_Part_7']).size()/len(df_ds_med)

q8_ds = 100 * df_ds_med.groupby(['Q24_Part_8']).size()/len(df_ds_med)

q9_ds = 100 * df_ds_med.groupby(['Q24_Part_9']).size()/len(df_ds_med)

q10_ds = 100 * df_ds_med.groupby(['Q24_Part_10']).size()/len(df_ds_med)



algo_prod_med = pd.concat([q1_ds, q2_ds,q3_ds,q4_ds,q5_ds,q6_ds,q7_ds,q8_ds,q9_ds,q10_ds], axis=0)





q1_ds = 100 * df_ds_small.groupby(['Q24_Part_1']).size()/len(df_ds_small)

q2_ds = 100 * df_ds_small.groupby(['Q24_Part_2']).size()/len(df_ds_small)

q3_ds = 100 * df_ds_small.groupby(['Q24_Part_3']).size()/len(df_ds_small)

q4_ds = 100 * df_ds_small.groupby(['Q24_Part_4']).size()/len(df_ds_small)

q5_ds = 100 * df_ds_small.groupby(['Q24_Part_5']).size()/len(df_ds_small)

q6_ds = 100 * df_ds_small.groupby(['Q24_Part_6']).size()/len(df_ds_small)

q7_ds = 100 * df_ds_small.groupby(['Q24_Part_7']).size()/len(df_ds_small)

q8_ds = 100 * df_ds_small.groupby(['Q24_Part_8']).size()/len(df_ds_small)

q9_ds = 100 * df_ds_small.groupby(['Q24_Part_9']).size()/len(df_ds_small)

q10_ds = 100 * df_ds_small.groupby(['Q24_Part_10']).size()/len(df_ds_small)



algo_prod_sm = pd.concat([q1_ds, q2_ds,q3_ds,q4_ds,q5_ds,q6_ds,q7_ds,q8_ds,q9_ds,q10_ds], axis=0)





al_usg = pd.concat([algo_prod_all,algo_prod_lg,algo_prod_med,algo_prod_sm], axis=1)

al_usg.columns = ['All', 'Large', 'SME','SMB']



from matplotlib.colors import ListedColormap

fig, axs = plt.subplots(ncols=2,figsize=(14, 8),sharey=True)



with sns.axes_style('white'):

      p = sns.heatmap(al_usg,

                cbar=False,

                square=False,

                annot=True,

                fmt='g',

                cmap=ListedColormap(['white']),

                linewidths=0.2,ax=axs[0])



tab_n = al_usg.div(al_usg.max(axis=1), axis=0)

q = sns.heatmap(tab_n,annot=False,cmap="YlGnBu", cbar=False, linewidths=0.5,ax=axs[1])

bottom, top = q.get_ylim()

q.set_ylim(bottom + 0.5, top - 0.5)



p.set(ylabel='Percentage')

p.set(title='% of Usage of Algos by Enterprise Type')



_ = plt.setp(p.get_xticklabels(), rotation=90)

_ = plt.setp(q.get_xticklabels(), rotation=90)
q1_ds = 100 * df_ds_Excl.groupby(['Q28_Part_1']).size()/len(df_ds_Excl)

q2_ds = 100 * df_ds_Excl.groupby(['Q28_Part_2']).size()/len(df_ds_Excl)

q3_ds = 100 * df_ds_Excl.groupby(['Q28_Part_3']).size()/len(df_ds_Excl)

q4_ds = 100 * df_ds_Excl.groupby(['Q28_Part_4']).size()/len(df_ds_Excl)

q5_ds = 100 * df_ds_Excl.groupby(['Q28_Part_5']).size()/len(df_ds_Excl)

q6_ds = 100 * df_ds_Excl.groupby(['Q28_Part_6']).size()/len(df_ds_Excl)

q7_ds = 100 * df_ds_Excl.groupby(['Q28_Part_7']).size()/len(df_ds_Excl)

q8_ds = 100 * df_ds_Excl.groupby(['Q28_Part_8']).size()/len(df_ds_Excl)

q9_ds = 100 * df_ds_Excl.groupby(['Q28_Part_9']).size()/len(df_ds_Excl)

q10_ds = 100 * df_ds_Excl.groupby(['Q28_Part_10']).size()/len(df_ds_Excl)



ml_fw_all = pd.concat([q1_ds, q2_ds,q3_ds,q4_ds,q5_ds,q6_ds,q7_ds,q8_ds,q9_ds,q10_ds], axis=0)



ml_fw_all_srt = ml_fw_all.sort_values(ascending=False)



q = sns.barplot(ml_fw_all_srt.index, ml_fw_all_srt.values)

q.set_title('Usage of ML Frameworks by DS Community\n')

q.set(ylabel='Percentage')

_ = plt.setp(q.get_xticklabels(), rotation=90)
q1_ds = 100 * df_ds_large.groupby(['Q28_Part_1']).size()/len(df_ds_large)

q2_ds = 100 * df_ds_large.groupby(['Q28_Part_2']).size()/len(df_ds_large)

q3_ds = 100 * df_ds_large.groupby(['Q28_Part_3']).size()/len(df_ds_large)

q4_ds = 100 * df_ds_large.groupby(['Q28_Part_4']).size()/len(df_ds_large)

q5_ds = 100 * df_ds_large.groupby(['Q28_Part_5']).size()/len(df_ds_large)

q6_ds = 100 * df_ds_large.groupby(['Q28_Part_6']).size()/len(df_ds_large)

q7_ds = 100 * df_ds_large.groupby(['Q28_Part_7']).size()/len(df_ds_large)

q8_ds = 100 * df_ds_large.groupby(['Q28_Part_8']).size()/len(df_ds_large)

q9_ds = 100 * df_ds_large.groupby(['Q28_Part_9']).size()/len(df_ds_large)

q10_ds = 100 * df_ds_large.groupby(['Q28_Part_10']).size()/len(df_ds_large)



ml_fw_lg = pd.concat([q1_ds, q2_ds,q3_ds,q4_ds,q5_ds,q6_ds,q7_ds,q8_ds,q9_ds,q10_ds], axis=0)





q1_ds = 100 * df_ds_med.groupby(['Q28_Part_1']).size()/len(df_ds_med)

q2_ds = 100 * df_ds_med.groupby(['Q28_Part_2']).size()/len(df_ds_med)

q3_ds = 100 * df_ds_med.groupby(['Q28_Part_3']).size()/len(df_ds_med)

q4_ds = 100 * df_ds_med.groupby(['Q28_Part_4']).size()/len(df_ds_med)

q5_ds = 100 * df_ds_med.groupby(['Q28_Part_5']).size()/len(df_ds_med)

q6_ds = 100 * df_ds_med.groupby(['Q28_Part_6']).size()/len(df_ds_med)

q7_ds = 100 * df_ds_med.groupby(['Q28_Part_7']).size()/len(df_ds_med)

q8_ds = 100 * df_ds_med.groupby(['Q28_Part_8']).size()/len(df_ds_med)

q9_ds = 100 * df_ds_med.groupby(['Q28_Part_9']).size()/len(df_ds_med)

q10_ds = 100 * df_ds_med.groupby(['Q28_Part_10']).size()/len(df_ds_med)



ml_fw_med = pd.concat([q1_ds, q2_ds,q3_ds,q4_ds,q5_ds,q6_ds,q7_ds,q8_ds,q9_ds,q10_ds], axis=0)





q1_ds = 100 * df_ds_small.groupby(['Q28_Part_1']).size()/len(df_ds_small)

q2_ds = 100 * df_ds_small.groupby(['Q28_Part_2']).size()/len(df_ds_small)

q3_ds = 100 * df_ds_small.groupby(['Q28_Part_3']).size()/len(df_ds_small)

q4_ds = 100 * df_ds_small.groupby(['Q28_Part_4']).size()/len(df_ds_small)

q5_ds = 100 * df_ds_small.groupby(['Q28_Part_5']).size()/len(df_ds_small)

q6_ds = 100 * df_ds_small.groupby(['Q28_Part_6']).size()/len(df_ds_small)

q7_ds = 100 * df_ds_small.groupby(['Q28_Part_7']).size()/len(df_ds_small)

q8_ds = 100 * df_ds_small.groupby(['Q28_Part_8']).size()/len(df_ds_small)

q9_ds = 100 * df_ds_small.groupby(['Q28_Part_9']).size()/len(df_ds_small)

q10_ds = 100 * df_ds_small.groupby(['Q28_Part_10']).size()/len(df_ds_small)



ml_fw_sm = pd.concat([q1_ds, q2_ds,q3_ds,q4_ds,q5_ds,q6_ds,q7_ds,q8_ds,q9_ds,q10_ds], axis=0)





al_usg = pd.concat([ml_fw_all,ml_fw_lg,ml_fw_med,ml_fw_sm], axis=1)

al_usg.columns = ['All', 'Large', 'SME','SMB']



from matplotlib.colors import ListedColormap

fig, axs = plt.subplots(ncols=2,figsize=(14, 8),sharey=True)



with sns.axes_style('white'):

      p = sns.heatmap(al_usg,

                cbar=False,

                square=False,

                annot=True,

                fmt='g',

                cmap=ListedColormap(['white']),

                linewidths=0.2,ax=axs[0])



tab_n = al_usg.div(al_usg.max(axis=1), axis=0)

q = sns.heatmap(tab_n,annot=False,cmap="YlGnBu", cbar=False, linewidths=0.5,ax=axs[1])

bottom, top = q.get_ylim()

q.set_ylim(bottom + 0.5, top - 0.5)



p.set(ylabel='Percentage')

p.set(title='% of Usage of ML Framwork by Enterprise Type')



_ = plt.setp(p.get_xticklabels(), rotation=90)

_ = plt.setp(q.get_xticklabels(), rotation=90)