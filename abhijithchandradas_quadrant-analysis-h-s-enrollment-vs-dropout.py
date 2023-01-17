import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



plt.style.use('seaborn-darkgrid')

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_enroll=pd.read_csv('../input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv')

df_drop=pd.read_csv('../input/indian-school-education-statistics/dropout-ratio-2012-2015.csv')

df_enroll.head(10)
#Dropout data 

df_drop.head(5)
df_enroll.columns
df_enroll=df_enroll[['State_UT', 'Year','Higher_Secondary_Boys', 'Higher_Secondary_Girls',

       'Higher_Secondary_Total']]

df_enroll=df_enroll.rename(columns={'Higher_Secondary_Boys':'hs_boys',

                          'Higher_Secondary_Girls':'hs_girls',

                          'Higher_Secondary_Total':'hs_total'})

df_enroll.head()
df_enroll.Year.unique()
df_enroll_13_14=df_enroll[df_enroll.Year=='2013-14']

df_enroll_15_16=df_enroll[df_enroll.Year=='2015-16']

df_enroll_13_14.drop('Year', axis=1, inplace=True)

df_enroll_15_16.drop('Year', axis=1, inplace=True)
df_drop.columns
df_drop=df_drop[['State_UT', 'year','HrSecondary_Boys', 'HrSecondary_Girls', 'HrSecondary_Total']]

df_drop=df_drop.rename(columns={'HrSecondary_Boys':'hs_boys',

                          'HrSecondary_Girls':'hs_girls',

                          'HrSecondary_Total':'hs_total'})

df_drop.head()
df_drop.year.unique()
df_drop_12_13=df_drop[df_drop.year=='2012-13']

df_drop_14_15=df_drop[df_drop.year=='2014-15']

df_drop_12_13.drop('year', axis=1, inplace=True)

df_drop_14_15.drop('year', axis=1, inplace=True)
df_drop_12_13.head()
cols=df_drop_12_13.columns[1:]
#Function to check NR value 

def check_nr():

    df_enroll_13_14.name='df_enroll_13_14'

    df_enroll_15_16.name='df_enroll_15_16'

    df_drop_12_13.name='df_drop_12_13'

    df_drop_14_15.name='df_drop_14_15'

    df_list=[df_enroll_13_14,df_enroll_15_16,df_drop_12_13,df_drop_14_15]



    for df in df_list:

        print(df.name)

        for col in cols:

            try:

                x=df[col].value_counts()['NR']

            except KeyError:

                x=0

            print(str(col),x)
check_nr()
# selecting data with non NR values

df_enroll_13_14=df_enroll_13_14[df_enroll_13_14['hs_total']!='NR']

df_drop_12_13=df_drop_12_13[df_drop_12_13['hs_girls']!='NR']

df_drop_14_15=df_drop_14_15[df_drop_14_15['hs_girls']!='NR']
check_nr()
df_drop_14_15=df_drop_14_15[df_drop_14_15['hs_boys']!='NR']
check_nr()
# merging dropout datasets

df_dr=pd.merge(df_drop_12_13,df_drop_14_15,on='State_UT',suffixes=('_past','_new'))

df_dr
#Merging enrollment datasets

df_en=pd.merge(df_enroll_13_14,df_enroll_15_16,on='State_UT',suffixes=('_past','_new'))

df_en.head()
df_en.hs_total_new=df_en.hs_total_new.astype('float')

df_en.hs_total_past=df_en.hs_total_past.astype('float')

df_dr.hs_total_new=df_dr.hs_total_new.astype('float')

df_dr.hs_total_past=df_dr.hs_total_past.astype('float')
# Calculating change in enrollment and dropout ratios and percentage change

df_en=df_en[['State_UT','hs_total_past','hs_total_new']]

df_en['ch']=df_en.hs_total_new-df_en.hs_total_past

df_en['ch_perc']=round(df_en.ch*100/df_en.hs_total_past,2)



df_dr=df_dr[['State_UT','hs_total_past','hs_total_new']]

df_dr['ch']=df_dr.hs_total_new-df_dr.hs_total_past

df_dr['ch_perc']=round(-df_dr.ch*100/df_dr.hs_total_past,2)

df_dr.head()
#columns in enrollment data but not in dropout data

set(df_en.State_UT)-set(df_dr.State_UT)
#Columns in Dropout dataset only

set(df_dr.State_UT)-set(df_en.State_UT)
#Syncing the statenames in both datasets wherever possible

df_dr.State_UT[df_dr.State_UT=='A & N Islands']='Andaman & Nicobar Islands'

df_dr.State_UT[df_dr.State_UT=='Jammu & Kashmir']='Jammu And Kashmir'
#Using only percentage reduction dropout rate and percentage increase in enrollment rate

df_dr=df_dr[['State_UT','ch_perc']]

df_en=df_en[['State_UT','ch_perc']]
# Final dataset with only the required data

df_final=pd.merge(df_dr,df_en,on='State_UT',suffixes=('_drop','_enroll'))

df_final.head()
#Box plot to visually examine distribution of data and identify outliers

plt.subplot(2,1,1)

sns.boxplot(df_final.ch_perc_enroll)

plt.subplot(2,1,2)

sns.boxplot(df_final.ch_perc_drop)

plt.tight_layout()

plt.show()
#Plotting the data

sns.scatterplot('ch_perc_enroll','ch_perc_drop', data=df_final)

plt.show()
#identifying outliers

df_final.sort_values(by='ch_perc_drop', ascending=True).head()
# Examining the reason for outliers

df_drop[df_drop.State_UT.isin(['Gujarat','Arunachal Pradesh'])]
#Removing outliers

df_final=df_final[df_final.State_UT.isin(['Gujarat','Arunachal Pradesh'])==False].reset_index()

df_final.head()
#Plotting quadrant analysis



plt.figure(figsize=(8,6))

plot=sns.scatterplot('ch_perc_enroll','ch_perc_drop', data=df_final)

plt.axhline(y=0,color='k', alpha=0.5)

plt.axvline(x=0,color='k', alpha=0.5)

for i in range(0,df_final.shape[0]):

    plot.text(df_final.ch_perc_enroll[i],df_final.ch_perc_drop[i],df_final.State_UT[i])

plt.xlabel("Decrease in Dropout Rate")

plt.ylabel("Increase in Enrollment Rate")

plt.show()