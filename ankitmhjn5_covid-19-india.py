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
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
age_grp_details = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv") 
individual_details = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")
age_grp_details.head()
plt.figure(figsize=(10,10))
ax = sns.barplot(x=age_grp_details['AgeGroup'], y=age_grp_details['TotalCases'])
plt.show()
individual_details.head()
# remove columns not used for analysis
individual_details = individual_details.drop(['id', 'government_id', 'status_change_date'], axis=1)
# finding the number of missing values
individual_details.isnull().sum()
# lets change the missing age with 0
individual_details['age'] = individual_details['age'].fillna(0)
# lets change the missing gender to Missing
individual_details['gender'] = individual_details['gender'].fillna('Missing')
# lets change the missing city, district to Missing
individual_details['detected_city'] = individual_details['detected_city'].fillna('Missing')
individual_details['detected_district'] = individual_details['detected_district'].fillna('Missing')
# lets change the missing nationaliuty
individual_details['nationality'] = individual_details['nationality'].fillna('Missing')
individual_details['nationality'] = individual_details['nationality'].apply(lambda row: 'India' if row.lower() == 'indian' else row)
individual_details.isnull().sum()
data = individual_details['gender'].value_counts().reset_index()
# plotting the number of males and females who suffered from the disease
data = individual_details['gender'].value_counts().reset_index()
cols = data['index']
values = data['gender']

fig1, ax1 = plt.subplots()
ax1.pie(values, labels=cols, wedgeprops=dict(width=0.6), autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
plt.show()
# getting the different district people are suffering from the disease
distinct_city = ','.join(individual_details[individual_details['detected_city'] != 'Missing']['detected_city'].unique())

wordcloud = WordCloud(max_font_size=40).generate(distinct_city)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# getting the count for people with different nationality 
fig, ax1 = plt.subplots(figsize=(10,10))
ax1.set_yscale('log')
chart = sns.countplot(x=individual_details['nationality'])
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
# getting the count for people with different nationality 
fig, ax1 = plt.subplots(figsize=(20,10))
ax1.set_yscale('log')
chart = sns.countplot(x=individual_details['detected_state'])
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
# state wise count based on gender lets drop the gender with missing values
data = individual_details[individual_details['gender'] != 'Missing']
fig, ax1 = plt.subplots(figsize=(20,10))
ax1.set_yscale('log')
chart = sns.countplot(x=data['detected_state'], hue=data['gender'])
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
individual_details['age'] = individual_details['age'].apply(lambda row: 33 if row == "28-35" else row)
individual_details['age'] = individual_details['age'].astype(float)
# lets perform the binning for age column

def getAgeBins(row):
    if row == 0:
        return "0"
    elif row > 0 and row <= 9:
        return "0-9"
    elif row > 9 and row <= 19:
        return "10-19"
    elif row > 19 and row <= 29:
        return "20-29"
    elif row > 29 and row <= 39:
        return "30-39"
    elif row > 39 and row <= 49:
        return "40-49"
    elif row > 49 and row <= 59:
        return "50-59"
    elif row > 59 and row <= 69:
        return "60-69"
    elif row > 69 and row <= 79:
        return "70-79"
    elif row > 79 and row <= 89:
        return "80-89"
    else:
        return ">90"
    
individual_details['age_bins'] = individual_details['age'].apply(lambda row: getAgeBins(row))
individual_details['age_bins'].head()
fig, ax1 = plt.subplots(figsize=(10,10))
ax1.set_yscale('log')
sns.countplot(x=individual_details['age_bins'], order=individual_details['age_bins'].value_counts().sort_values().index)
plt.show()