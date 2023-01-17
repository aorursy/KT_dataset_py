# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# For notebook plotting
%matplotlib inline

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Dir = '../input/'
freeform_df = pd.read_csv(Dir + 'freeFormResponses.csv', low_memory=False, header=[0,1])
multi_df = pd.read_csv(Dir + 'multipleChoiceResponses.csv', low_memory=False, header=[0,1])
SurveySchema = pd.read_csv(Dir + 'SurveySchema.csv', low_memory=False, header=[0,1])
freeform_df.columns = freeform_df.columns.map('_'.join)
multi_df.columns = multi_df.columns.map('_'.join)
SurveySchema.columns = SurveySchema.columns.map('_'.join)
data_scientist = multi_df['Q26_Do you consider yourself to be a data scientist?'] == "Definitely yes"
#datasci_df = multi_df[data_scientist]
top_earning_sal = {"100-125,000","125-150,000","150-200,000","200-250,000","250-300,000","300-400,000","400-500,000","500,000+"}
top_earn = multi_df['Q9_What is your current yearly compensation (approximate $USD)?'].isin(top_earning_sal)
top_dsci_df  = multi_df[data_scientist & top_earn]
age_df = multi_df['Q2_What is your age (# years)?'].value_counts()
age_df.index.name = 'Age'
age_df.sort_index(inplace=True)
age_df.plot(kind='bar',rot=20, title='Age distribution of Kagglers',figsize=(14,5));
age_teds_df = top_dsci_df['Q2_What is your age (# years)?'].value_counts()
age_teds_df.index.name = 'Age'
age_teds_df.sort_index(inplace=True)
age_teds_df.plot(kind='bar',rot=20, title='Age distribution of Top Earning Data Scientist Kagglers',figsize=(14,5));
country_df = multi_df['Q3_In which country do you currently reside?'].value_counts()
country_df.index.name = 'Country'
#country_df.sort_index(inplace=True)
country_df.plot(kind='barh',rot=0, title='Countries of Kagglers',figsize=(15,15), colormap = 'Paired');
country_teds_df = top_dsci_df['Q3_In which country do you currently reside?'].value_counts()
country_teds_df.index.name = 'Country'
#country_teds_df.sort_index(inplace=True)
country_teds_df.plot(kind='barh',rot=0, title='Countries of Top Earning Data Science Kagglers',figsize=(15,15), colormap = 'Paired');
edu_df = multi_df['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts()
edu_df.index.name = 'Education'
#edu_df.sort_index(inplace=True)
edu_df.plot(kind='barh',rot=0, title='Education of Kagglers',figsize=(15,5), colormap = 'Paired');
edu_teds_df = top_dsci_df['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts()
edu_teds_df.index.name = 'Education'
#edu_teds_df.sort_index(inplace=True)
edu_teds_df.plot(kind='barh',rot=0, title='Education of Top Earning Data Scientists Kagglers',figsize=(15,5), colormap = 'Paired');
reclang_teds_df = top_dsci_df['Q18_What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts()
reclang_teds_df.index.name = 'Recommendation'
#reclang_teds_df.sort_index(inplace=True)
reclang_teds_df.plot(kind='bar',rot=80, title='Recommended Programming Language of Top Earning Data Science Kagglers',figsize=(15,5));
reclang_teds_df = top_dsci_df['Q39_Part_1_How do you perceive the quality of online learning platforms and in-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions? - Online learning platforms and MOOCs:'].value_counts()
reclang_teds_df.index.name = 'MOOCS Better or Worse'
#reclang_teds_df.sort_index(inplace=True)
temp_new_df = reclang_teds_df.reindex(["Much better", "Slightly better", "Neither better nor worse", "Slightly worse", "Much worse", "No opinion; I do not know"])
temp_new_df.plot(kind='barh',rot=40, title='What Top Earning Data Science Kagglers Think of MOOCS and Online Learning Platforms',figsize=(15,5));
onlinelearn_teds_df = top_dsci_df['Q37_On which online platform have you spent the most amount of time? - Selected Choice'].value_counts()
onlinelearn_teds_df.index.name = 'MOOCS and Online Learning Platforms'
#onlinelearn_teds_df.sort_index(inplace=True)
onlinelearn_teds_df.plot(kind='barh',rot=40, title='MOOCS and Online Learning Platforms the Top Earning Data Scientists Use',figsize=(15,5), colormap = 'Spectral');
edu_online_df = multi_df['Q37_On which online platform have you spent the most amount of time? - Selected Choice'].value_counts()
edu_online_df.index.name = 'MOOCS and Online Learning Platforms'
#edu_online_df.sort_index(inplace=True)
edu_online_df.plot(kind='barh',rot=40, title='MOOCS and Online Learning Platforms of Kagglers',figsize=(15,5), colormap = 'Spectral');
top_earning_india = {"20-30,000","30-40,000","40-50,000","50-60,000","60-70,000","70-80,000","80-90,000","90-100,000","100-125,000","125-150,000","150-200,000","200-250,000","250-300,000","300-400,000","400-500,000","500,000+"}
top_earn_india = multi_df['Q9_What is your current yearly compensation (approximate $USD)?'].isin(top_earning_india)
india_res = multi_df['Q3_In which country do you currently reside?'] == "India"
indiatop_dsci_df  = multi_df[data_scientist & top_earn_india & india_res]
age_indiatop_df = indiatop_dsci_df['Q2_What is your age (# years)?'].value_counts()
age_indiatop_df.index.name = 'Age'
age_indiatop_df.sort_index(inplace=True)
age_indiatop_df.plot(kind='bar',rot=20, title='Age distribution of Top Earning Data Scientist Kagglers in India',figsize=(14,5));
edu_indiatop_df = indiatop_dsci_df['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts()
edu_indiatop_df.index.name = 'Education'
#edu_teds_df.sort_index(inplace=True)
edu_indiatop_df.plot(kind='barh',rot=0, title='Education of Top Earning Data Scientists Kagglers in India',figsize=(15,5), colormap = 'Paired');
reclang_indiateds_df = indiatop_dsci_df['Q18_What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts()
reclang_indiateds_df.index.name = 'Recommendation'
#reclang_teds_df.sort_index(inplace=True)
reclang_indiateds_df.plot(kind='bar',rot=80, title='Recommended Programming Language of Top Earning Data Science Kagglers in India',figsize=(15,5));
reclang_indiateds_df = indiatop_dsci_df['Q39_Part_1_How do you perceive the quality of online learning platforms and in-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions? - Online learning platforms and MOOCs:'].value_counts()
reclang_indiateds_df.index.name = 'MOOCS Better or Worse'
#reclang_teds_df.sort_index(inplace=True)
temp_indianew_df = reclang_indiateds_df.reindex(["Much better", "Slightly better", "Neither better nor worse", "Slightly worse", "Much worse", "No opinion; I do not know"])
temp_indianew_df.plot(kind='barh',rot=40, title='What Top Earning Data Science Kagglers in India Think of MOOCS and Online Learning Platforms',figsize=(15,5));
indiaonlinelearn_teds_df = indiatop_dsci_df['Q37_On which online platform have you spent the most amount of time? - Selected Choice'].value_counts()
indiaonlinelearn_teds_df.index.name = 'MOOCS and Online Learning Platforms'
#onlinelearn_teds_df.sort_index(inplace=True)
indiaonlinelearn_teds_df.plot(kind='barh',rot=40, title='MOOCS and Online Learning Platforms the Top Earning Data Scientists in India Use',figsize=(15,5), colormap = 'Spectral');