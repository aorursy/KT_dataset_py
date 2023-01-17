

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
vic_murder_df = pd.read_csv("../input/32_Murder_victim_age_sex.csv")

#Exploring first few rows of 'Victims of Murder' dataset

vic_murder_df.head()
#Selecting data for Delhi, for the year- 2010



mah_murder_df=vic_murder_df.loc[vic_murder_df['Area_Name']=='Delhi']

mah_murder_2010_total = mah_murder_df [(mah_murder_df['Year']==2010) & (mah_murder_df['Sub_Group_Name']=='1. Male Victims')]

ax = mah_murder_2010_total [['Victims_Upto_10_Yrs','Victims_Upto_10_15_Yrs','Victims_Upto_15_18_Yrs','Victims_Upto_18_30_Yrs','Victims_Upto_30_50_Yrs','Victims_Above_50_Yrs']].plot(kind='bar',title='Age of Male Murder Victims (Delhi)',legend=True)

ax.set_xlabel("Age ", fontsize=12)

ax.set_ylabel("No of Victims", fontsize=12)







mah_murder_df=vic_murder_df.loc[vic_murder_df['Area_Name']=='Delhi']



mah_murder_2010_total = mah_murder_df [(mah_murder_df['Year']==2010) & (mah_murder_df['Sub_Group_Name']=='2. Female Victims')]

ax = mah_murder_2010_total [['Victims_Upto_10_Yrs','Victims_Upto_10_15_Yrs','Victims_Upto_15_18_Yrs','Victims_Upto_18_30_Yrs','Victims_Upto_30_50_Yrs','Victims_Above_50_Yrs']].plot(kind='bar',title='Age of Male Murder Victims (Delhi)',legend=True)

ax.set_xlabel("Age ", fontsize=12)

ax.set_ylabel("No of Victims", fontsize=12)

vic_murd_2002_total = vic_murder_df[(vic_murder_df['Year']==2002) & (vic_murder_df['Sub_Group_Name']== '3. Total')]

vic_murd_2009_total = vic_murder_df[(vic_murder_df['Year']==2009) & (vic_murder_df['Sub_Group_Name']== '3. Total')]

df1 = vic_murd_2002_total[['Area_Name','Victims_Total']]

df2 = vic_murd_2009_total[['Area_Name','Victims_Total']]

df1 ['Total no of murder victims (2002)'] = df1 ['Victims_Total']

df2 ['Total no of murder victims (2009)'] = df2 ['Victims_Total']

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
murF_vic=vic_murder_df.loc[vic_murder_df['Area_Name']=='Uttar Pradesh']

murF_vic_2002_total = murF_vic [(murF_vic['Year']==2002) & (murF_vic['Sub_Group_Name']=='2. Female Victims')]



ax = murF_vic_2002_total[['Victims_Upto_10_Yrs','Victims_Upto_10_15_Yrs','Victims_Upto_15_18_Yrs','Victims_Upto_18_30_Yrs','Victims_Upto_30_50_Yrs','Victims_Above_50_Yrs']].plot(kind='bar',legend=True, title = 'Age murder female victims (Uttar Pradesh)')

ax.set_ylabel("No of Victims", fontsize=12)

ax.set_xlabel("Age of Female Victims", fontsize=12)
murF_vic=vic_murder_df.loc[vic_murder_df['Area_Name']=='Uttar Pradesh']

murF_vic_2002_total = murF_vic [(murF_vic['Year']==2002) & (murF_vic['Sub_Group_Name']=='1. Male Victims')]



ax = murF_vic_2002_total[['Victims_Upto_10_Yrs','Victims_Upto_10_15_Yrs','Victims_Upto_15_18_Yrs','Victims_Upto_18_30_Yrs','Victims_Upto_30_50_Yrs','Victims_Above_50_Yrs']].plot(kind='bar',legend=True, title = 'Age murder Male victims (Uttar pradesh)')

ax.set_ylabel("No of Victims", fontsize=12)

ax.set_xlabel("Age of Male Victims", fontsize=12)