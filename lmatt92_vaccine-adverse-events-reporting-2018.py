import numpy as np

import pandas as pd

pd.options.display.max_seq_items = 4000

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# PATIENT REPORTED TABLE

data_file_path = '/kaggle/input/2018vaers_data.csv'



df_patient = pd.read_csv(data_file_path)

df_patient.drop(['CAGE_YR', 'CAGE_MO', 'SYMPTOM_TEXT', 'DATEDIED', 'L_THREAT', 'HOSPITAL', 'HOSPDAYS', 

                 'X_STAY', 'DISABLE', 'VAX_DATE', 'ONSET_DATE', 'NUMDAYS', 'LAB_DATA', 'V_ADMINBY', 'V_FUNDBY', 

                 'OTHER_MEDS', 'CUR_ILL', 'SPLTTYPE', 'FORM_VERS', 'TODAYS_DATE', 'BIRTH_DEFECT', 

                 'ER_ED_VISIT', 'ALLERGIES', 'RPT_DATE', 'HISTORY', 'PRIOR_VAX'], inplace=True, axis=1)

df_patient.head()

vax_file_path = '/kaggle/input/2018vaers_vax.csv'



# VACCINE TYPE TABLE

df_vax = pd.read_csv(vax_file_path)

df_vax.drop(['VAX_MANU', 'VAX_LOT', 'VAX_DOSE_SERIES', 'VAX_ROUTE', 'VAX_SITE'], inplace=True, axis=1)

df_vax.head()

# MERGE TABLES ON VAERS_ID

df_merged = pd.merge(df_patient, df_vax, how='inner', on='VAERS_ID')

df_merged.head()
# REMOVE DUPLICATES TO GET ACCURATE ACCOUNTS OF DEATH, ER VISITS, AND AGE (WHEN NO NEED FOR VACCINE TYPE)

df_no_duplicates = df_merged.copy()

df_no_duplicates.drop_duplicates(subset='VAERS_ID', keep='first', inplace=True)

df_no_duplicates.head()
num_vaccines = df_merged.groupby('VAX_TYPE').VAERS_ID.nunique()

#median_age_vax_type = df_merged.groupby('VAX_TYPE').AGE_YRS.median()



#ages of dataset

ages = df_merged.AGE_YRS.dropna().apply(np.ceil)

ages_under12 = df_merged[df_merged.AGE_YRS < 13]

ages_13to18 = df_merged[(df_merged.AGE_YRS > 12) & (df_merged.AGE_YRS < 19)]

ages_19to30 = df_merged[(df_merged.AGE_YRS > 18) & (df_merged.AGE_YRS < 31)]

ages_31to50 = df_merged[(df_merged.AGE_YRS > 30) & (df_merged.AGE_YRS < 51)]

ages_51to65 = df_merged[(df_merged.AGE_YRS > 50) & (df_merged.AGE_YRS < 66)]

ages_over65 = df_merged[(df_merged.AGE_YRS > 65)]



#Type of Vaccine

vax_type = df_merged.VAX_TYPE.unique()



#cases where deaths were recorded - USE JUST VAERSDATA FILE NOT VAXTYPE TO AVOID DUPS

#unique_str = df_merged.SYMPTOM_TEXT.unique()

#deaths = df_merged.DIED.notnull()

#deaths_unique = deaths[deaths.SYMPTOM_TEXT.unique()]





sns.distplot(a=ages, kde=False)

plt.xlabel("Age in Years")

plt.ylabel("Number of ADRs")

plt.title("Number of ADRs by Age")

print("Ages under 12: {}".format(len(ages_under12)))

print("Ages between 13 and 18: {}".format(len(ages_13to18)))

print("Ages between 19 and 30: {}".format(len(ages_19to30)))

print("Ages between 31 and 50: {}".format(len(ages_31to50)))

print("Ages between 51 and 65: {}".format(len(ages_51to65)))

print("Ages under 65: {}".format(len(ages_over65)))

print("Total: {}".format(len(ages)))
#Most Reported by Vax Type

order_by_vax_type = df_merged.groupby(['VAX_TYPE']).count().sort_values(by=['VAERS_ID'], ascending=False)



#TEN MOST REPORTED

ten_most_reported = order_by_vax_type.drop(['RECVDATE', 'STATE', 'AGE_YRS', 'SEX', 'DIED', 'ER_VISIT', 'RECOVD', 

                                            'OFC_VISIT', 'VAX_NAME'], axis=1).head(10)

ten_most_reported



#ten_most_reported.plot(y='VAERS_ID', kind='bar')

    

ax = ten_most_reported.plot.bar()

#ax = sns.lineplot(data=num_reported_per_vax_inorder.head(10))

ax.set_xlabel("Vaccine")

ax.set_ylabel("Number of ADRs")

ax.set_title("10 Most Reported Vaccine ADRs for 2018")

ax.get_legend().remove()

ax.tick_params(axis='x', rotation=45)
#REPORTED DEATHS

deaths = df_no_duplicates[df_no_duplicates.DIED == 'Y']

num_deaths = deaths.DIED.count()

                    

deaths_by_vax = deaths.groupby(['VAX_TYPE']).count().sort_values(by=['DIED'], ascending=False)

ten_most_reported_deaths = deaths_by_vax[['DIED']].head(10)

ten_most_reported_deaths



# print("Number of Recorded Deaths: {}\n".format(num_deaths)) #157



#Make bar chart of top 5 vaxs and num of deaths for each

ax = ten_most_reported_deaths.plot.bar()

ax.set_xlabel("Vaccine")

ax.set_ylabel("Number of Deaths Reported")

ax.set_title("Ten Most Reported Vaccines Where Death Occurred")

ax.get_legend().remove()

ax.tick_params(axis='x', rotation=45)