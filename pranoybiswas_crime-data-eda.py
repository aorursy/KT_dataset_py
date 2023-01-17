# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
vic_rape_df = pd.read_csv("../input/20_Victims_of_rape.csv")

#Exploring first few rows of 'Victims of Rape' dataset

vic_rape_df.head()
#Selecting data for Maharashtra, for the latest year available - 2010



mah_vic=vic_rape_df.loc[vic_rape_df['Area_Name']=='Maharashtra']

mah_vic_2010_total = mah_vic [(mah_vic['Year']==2010) & (mah_vic['Subgroup']=='Total Rape Victims')]



#Plotting age breakup of victims

ax = mah_vic_2010_total[['Victims_Upto_10_Yrs','Victims_Between_10-14_Yrs','Victims_Between_14-18_Yrs','Victims_Between_18-30_Yrs','Victims_Between_30-50_Yrs']].plot(kind='bar',legend=True, title = 'Age Breakup of rape victims (Maharashtra)')

ax.set_ylabel("No of Victims", fontsize=12)

ax.set_xticklabels([])





#Selecting data on incest rape in Maharashtra

mah_vic_2010_inc = mah_vic [(mah_vic['Year']==2010) & (mah_vic['Subgroup']=='Victims of Incest Rape')]



#Plotting age breakup of victims of incest rape

ax = mah_vic_2010_inc[['Victims_Upto_10_Yrs','Victims_Between_10-14_Yrs','Victims_Between_14-18_Yrs','Victims_Between_18-30_Yrs','Victims_Between_30-50_Yrs']].plot(kind='bar',title = 'Age breakup of victims of incest rape (Maharashtra)',legend=True)

ax.set_ylabel("No of Victims", fontsize=12)

ax.set_xticklabels([''])

mah_vic_total = mah_vic.loc[mah_vic['Subgroup']=='Total Rape Victims']

ax = mah_vic_total['Rape_Cases_Reported'].plot(kind='bar', title = 'Total rape cases reported (Maharashtra 2001-10)')

ax.set_xlabel("Year", fontsize=15)

ax.set_ylabel("No of Victims", fontsize=15)

ax.set_xticklabels(mah_vic_total['Year'],rotation='horizontal')

vic_rape_2010_total = vic_rape_df[(vic_rape_df['Year']==2010) & (vic_rape_df['Subgroup']== 'Total Rape Victims')]

ax1 = vic_rape_2010_total['Victims_of_Rape_Total'].plot(kind='barh',figsize=(15, 10))

ax1.set_xlabel("Number of rape victims (2010)", fontsize=15)

ax1.set_yticklabels(vic_rape_2010_total['Area_Name'])

vic_rape_2001_total = vic_rape_df[(vic_rape_df['Year']==2001) & (vic_rape_df['Subgroup']== 'Total Rape Victims')]

df1 = vic_rape_2001_total[['Area_Name','Victims_of_Rape_Total']]

df2 = vic_rape_2010_total[['Area_Name','Victims_of_Rape_Total']]



#Renaming column name in order to differentiate by year

df1 ['Total no of rape victims (2001)'] = df1 ['Victims_of_Rape_Total']

df2 ['Total no of rape victims (2010)'] = df2 ['Victims_of_Rape_Total']

df1.drop(['Victims_of_Rape_Total'], axis = 1, inplace = True)

df2.drop(['Victims_of_Rape_Total'], axis = 1, inplace = True)

fig = plt.figure()

ax = fig.add_subplot(111) # Create matplotlib axes



width = 0.4



df1.plot(kind='barh', color='red', ax=ax, width=width, position=0,figsize=(15,15))

df2.plot(kind='barh', color='blue', ax=ax, width=width, position=1,figsize=(15,15))

ax.set_xlabel("Number of Victims", fontsize=15)

ax.set_yticklabels(df1['Area_Name'])



plt.show()
murder_df = pd.read_csv("../input/32_Murder_victim_age_sex.csv")

mah_murder_df=murder_df.loc[murder_df['Area_Name']=='Maharashtra']

mah_murder_2010_total = mah_murder_df [(mah_murder_df['Year']==2010) & (mah_murder_df['Sub_Group_Name']=='1. Male Victims')]

ax = mah_murder_2010_total [['Victims_Upto_10_Yrs','Victims_Upto_10_15_Yrs','Victims_Upto_15_18_Yrs','Victims_Upto_18_30_Yrs','Victims_Upto_30_50_Yrs','Victims_Above_50_Yrs']].plot(kind='bar',title='Age Breakup of Male Murder Victims (Maharashtra 2010)',legend=True)

ax.set_xlabel("Age Breakup", fontsize=12)

ax.set_ylabel("No of Victims", fontsize=12)

#murder_df.head()
mah_murder_2010_total = mah_murder_df [(mah_murder_df['Year']==2010) & (mah_murder_df['Sub_Group_Name']=='2. Female Victims')]

ax = mah_murder_2010_total [['Victims_Upto_10_Yrs','Victims_Upto_10_15_Yrs','Victims_Upto_15_18_Yrs','Victims_Upto_18_30_Yrs','Victims_Upto_30_50_Yrs','Victims_Above_50_Yrs']].plot(kind='bar',legend=True)

ax.set_xlabel("Age Breakup", fontsize=12)

ax.set_ylabel("No of Victims", fontsize=12)

vic_murd_2001_total = murder_df[(murder_df['Year']==2001) & (murder_df['Sub_Group_Name']== '3. Total')]

vic_murd_2010_total = murder_df[(murder_df['Year']==2010) & (murder_df['Sub_Group_Name']== '3. Total')]

df1 = vic_murd_2001_total[['Area_Name','Victims_Total']]

df2 = vic_murd_2010_total[['Area_Name','Victims_Total']]

df1 ['Total no of murder victims (2001)'] = df1 ['Victims_Total']

df2 ['Total no of murder victims (2010)'] = df2 ['Victims_Total']

df1.drop(['Victims_Total'], axis = 1, inplace = True)

df2.drop(['Victims_Total'], axis = 1, inplace = True)

fig = plt.figure()

ax = fig.add_subplot(111) # Create matplotlib axes



width = 0.4



df1.plot(kind='barh', color='red', ax=ax, width=width, position=0,figsize=(15,15))

df2.plot(kind='barh', color='blue', ax=ax, width=width, position=1,figsize=(15,15))

ax.set_xlabel("Number of Victims", fontsize=15)

ax.set_yticklabels(df1['Area_Name'])



plt.show()
cases_women_df = pd.read_csv("../input/42_Cases_under_crime_against_women.csv",error_bad_lines=False)

cases_women_df.head()
df_cas_wom = cases_women_df.loc[cases_women_df['Year'] == 2010]

#df_cas_wom.head()

mah_cas_wom = df_cas_wom[(df_cas_wom['Area_Name'] == 'Maharashtra') & (df_cas_wom['Year']==2010)]

#mah_cas_wom.columns.values

df = mah_cas_wom[mah_cas_wom.Group_Name != 'Total Crime Against Women']

ax = df[['Group_Name','Total_Cases_for_Trial']].plot(kind='bar', legend = True, title = "Category of cases for trial (Women)",figsize = (15,10))

ax.set_xticklabels(mah_cas_wom['Group_Name'])

ax.set_xlabel('Category of Case', fontsize = 20)

ax.set_ylabel('Number of cases for trial', fontsize = 20)

plt.show()