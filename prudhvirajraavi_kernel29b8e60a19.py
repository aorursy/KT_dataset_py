import numpy as np

import pandas as pd



from subprocess import check_output



# Checking the files present in current working directory

print(check_output(["ls", "-la", "../input"]).decode("utf-8"))

source_df = pd.read_csv("../input/h-1b-visa/h1b_kaggle.csv")
source_df.shape
source_df.head(2)
source_df.drop(['Unnamed: 0'],inplace = True,axis = 1)
source_df.head(4)
source_df.info()
source_df.isnull().any()
source_df.isnull().sum().sort_values(ascending = False)
source_df['EMPLOYER_NAME'].value_counts()[:10]
import matplotlib.pyplot as plt

import seaborn as sb
employer = source_df['EMPLOYER_NAME'].value_counts()[:20]

sb.barplot(x= employer.values, y= employer.index)

#OMG Desi companies outplayed tech gaints!!!
soc_name = source_df['SOC_NAME'].value_counts()[:20]

sb.barplot(x= soc_name.values, y= soc_name.index)
job_title = source_df['JOB_TITLE'].value_counts()[:20]

sb.barplot(x= job_title.values, y= job_title.index)

#Ofcourse, programming is the most favorite job.
year = source_df["YEAR"].value_counts()[:10]

sb.barplot(x= year.index, y= year.values, saturation = 1)

#By every year no. of applications are increasing!
source_df["PREVAILING_WAGE"].sum()/source_df.shape[0]
work_place = source_df["WORKSITE"].value_counts()[:20]

sb.barplot(x= work_place.values, y= work_place.index)

# More people are working in New York city.
pd.options.display.float_format = '{:,.2f}'.format

source_df.groupby('WORKSITE').agg({'PREVAILING_WAGE':'mean'}).sort_values(by = ['PREVAILING_WAGE'], ascending = False)[:10]

source_df[source_df['WORKSITE'] == 'WASHINGTON, NA']
#status = source_df['CASE_STATUS'].value_counts()

#sb.barplot(x = status.values, y = status.index)

source_df['CASE_STATUS'].value_counts()
sb.countplot(source_df['FULL_TIME_POSITION'])
sample = source_df[(source_df['EMPLOYER_NAME'] == 'INFOSYS LIMITED') | (source_df['EMPLOYER_NAME'] == 'TATA CONSULTANCY SERVICES LIMITED') | (source_df['EMPLOYER_NAME'] == 'WIPRO LIMITED')]



sample.groupby(["EMPLOYER_NAME", "YEAR"]).count()['CASE_STATUS']
fig, ax = plt.subplots(figsize = (10,6))



sb.countplot(palette = 'gist_ncar',x= 'YEAR',hue = 'EMPLOYER_NAME',data = sample)
fig, ax = plt.subplots(figsize = (10,6))

sb.barplot(x= 'EMPLOYER_NAME',hue = 'SOC_NAME',y = 'CASE_STATUS', data = sample.groupby(['SOC_NAME','EMPLOYER_NAME']).count()['CASE_STATUS'].sort_values(ascending = False)[:20].reset_index())