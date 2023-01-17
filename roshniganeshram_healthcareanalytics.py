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
h1 = pd.read_csv('/kaggle/input/healthcare-dataset/Rreview_transaction_coo.csv')

h2 = pd.read_csv('/kaggle/input/healthcare-dataset/Inpatient_Pat.csv')

h3 = pd.read_csv('/kaggle/input/healthcare-dataset/Outpatient_Pat.csv')

h4 = pd.read_csv('/kaggle/input/healthcare-dataset/Inpatient_provdr.csv')

h5 = pd.read_csv('/kaggle/input/healthcare-dataset/Review_patient_history_samp.csv')

h6 = pd.read_csv('/kaggle/input/healthcare-dataset/Transaction_coo.csv')

h7 = pd.read_csv('/kaggle/input/healthcare-dataset/Patient_history_samp.csv')

h8 = pd.read_csv('/kaggle/input/healthcare-dataset/Outpatient_provdr.csv')

#Review Transaction related data

h1.head(4)

h1.nunique()
h1.sum().isnull()
#In - Patient Data

h2.head(4)
h2.nunique()
h2.sum().isnull()
#Correlation between variables in Inpatient Data

import matplotlib

import matplotlib.pyplot as plt

plt.scatter('Total Discharges' ,  'Average Total Payments' , data = h2)

plt.show()
h2.corr()

# From the below output we can see that Total Discharges has weak negative correlation with Average Covered Charges,

# Average Total Payments and Average Medicare Payments
#Outpatient data

h3.nunique()
h3.sum().isnull()
plt.scatter('Average Estimated Submitted Charges' ,  'Average Total Payments' , data = h3)

plt.show()

# From the below graph we can see that Estimated Charges and Total Payments have strong positive correlation

# This shows that the estimates are in line with actual payments
h4.head(5)
plt.figure(figsize=(15,10))

plot3 = sns.barplot(x= 'Provider State', y ='Average Total Payments', data = h4)
#InPatient Provider City

city_count  = h4['Provider City'].value_counts()

city_count = city_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(city_count.index, city_count.values, alpha=0.8)

plt.title('InPatient Provider in top 10 cities')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('city', fontsize=12)

plt.show()

# Inpatient by State

state_count  = h4['Provider State'].value_counts()

state_count = state_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(state_count.index, state_count.values, alpha=0.8)

plt.title('InPatient  in top 10 states ')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('State', fontsize=12)

plt.show()
# Inpatient by Zip Code

zipcode_count  = h4['Provider Zip Code'].value_counts()

zipcode_count = zipcode_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(zipcode_count.index,zipcode_count.values, alpha=0.8)

plt.title('InPatient  in top 10 Zip Code ')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Zipcode', fontsize=12)

plt.show()
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

encoder.fit(h4['Provider Name'].drop_duplicates())

h4['Provider Name']=encoder.transform(h4['Provider Name'])

encoder.fit(h4['Provider Street Address'].drop_duplicates())

h4['Provider Street Address']=encoder.transform(h4['Provider Street Address'])

encoder.fit(h4['Provider City'].drop_duplicates())

h4['Provider City']=encoder.transform(h4['Provider City'])

encoder.fit(h4['Provider State'].drop_duplicates())

h4['Provider State']=encoder.transform(h4['Provider State'])





corr = h4.corr()

plt.figure(figsize=(15,10))

sns.color_palette("Blues")

sns.heatmap(corr , annot=True , cbar=False)

plt.show()
#Review_Patient_Sample_Code

h5 = pd.read_csv('/kaggle/input/healthcare-dataset/Review_patient_history_samp.csv')

h5.head(5)
# Review_Patient_Sample_Code

income_ct  = h5['income'].value_counts()

income_ct = income_ct[:5,]

plt.figure(figsize=(10,5))

sns.barplot(income_ct.index,income_ct.values, alpha=0.8)

plt.title('Review_Patient_Top5_IncomeLevels ')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('income', fontsize=12)

plt.show()
# Review_Patient_Sample_Code

age_ct  = h5['age'].value_counts()

age_ct = age_ct[:5,]

plt.figure(figsize=(10,5))

sns.barplot(age_ct.index,age_ct.values, alpha=0.8)

plt.title('Review_Patient_Top5_AgeGroups ')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Agegroup', fontsize=12)

plt.show()
#Transaction_COO

h6.head(5)
# Patient_History_Sample

h7.head(5)
# Patient_History_Sample

income_ct  = h7['income'].value_counts()

income_ct = income_ct[:5,]

plt.figure(figsize=(10,5))

sns.barplot(income_ct.index,income_ct.values, alpha=0.8)

plt.title('Patient_History_Top5_IncomeLevels ')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('income', fontsize=12)

plt.show()
# Patient_History_Sample

age_ct  = h7['age'].value_counts()

age_ct = age_ct[:5,]

plt.figure(figsize=(10,5))

sns.barplot(age_ct.index,age_ct.values, alpha=0.8)

plt.title('Patient_History_Top5_AgeGroups ')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Agegroup', fontsize=12)

plt.show()
h8.head(5)
#Outpatient_Provider

city_count  = h8['Provider City'].value_counts()

city_count = city_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(city_count.index, city_count.values, alpha=0.8)

plt.title('OutPatient Provider in top 10 cities')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('city', fontsize=12)

plt.show()
# Outpatient by State

state_count  = h8['Provider State'].value_counts()

state_count = state_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(state_count.index, state_count.values, alpha=0.8)

plt.title('OutPatient  in top 10 states ')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('State', fontsize=12)

plt.show()
# Outpatient by Zip Code

zipcode_count  = h8['Provider Zip Code'].value_counts()

zipcode_count = zipcode_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(zipcode_count.index,zipcode_count.values, alpha=0.8)

plt.title('OutPatient  in top 10 Zip Code ')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Zipcode', fontsize=12)

plt.show()
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

encoder.fit(h8['Provider Name'].drop_duplicates())

h8['Provider Name']=encoder.transform(h8['Provider Name'])

encoder.fit(h8['Provider Street Address'].drop_duplicates())

h8['Provider Street Address']=encoder.transform(h8['Provider Street Address'])

encoder.fit(h8['Provider City'].drop_duplicates())

h8['Provider City']=encoder.transform(h8['Provider City'])

encoder.fit(h8['Provider State'].drop_duplicates())

h8['Provider State']=encoder.transform(h8['Provider State'])





corr = h8.corr()

plt.figure(figsize=(15,10))

sns.color_palette("Blues")

sns.heatmap(corr , annot=True , cbar=False)

plt.show()