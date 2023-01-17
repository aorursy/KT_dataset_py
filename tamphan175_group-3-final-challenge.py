import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_palette('husl')

from datetime import date

import numpy as np



kivaLoans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv') 

kivaMPIregion = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv') 

loan_theme_ids = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv') 

loan_themes_by_region = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv') 



# kivaLoans.columns

# kivaMPIregion.head()

# loan_theme_ids

# kivaLoans.columns

# kiva_loans['lender_count'].describe()
# Figure 1:

plt.figure(figsize = (10,5))

topCountry = kivaMPIregion['world_region'].value_counts()

graph1 = sns.barplot(x = topCountry.values, y = topCountry.index)

for i, v in enumerate(topCountry.values):

    plt.text(2, i, v, color = 'white', fontsize = 12)

graph1.set_title('Frequency receiving loans on Kiva', fontsize = 12)

graph1.set_ylabel('Country', fontsize = 12)

graph1.set_xlabel('Frequency of loans', fontsize = 12)
# Figure 2:

countryList = kivaMPIregion.loc[:,['country','world_region']]

data2 = kivaLoans.merge(kivaMPIregion, on = 'country')

data3 = data2.groupby(['world_region'])['loan_amount'].sum().sort_values(ascending = False)

data3 = data3.head(10)



plt.figure(figsize = (12,8))

cmap = plt.get_cmap('Set3')

colors = [cmap(i) for i in np.linspace(0, 1, 8)]



plt.pie(data3.values, labels = data3.index, autopct='%1.1f%%', colors = colors, shadow = True, startangle=90)

plt.title('Total loan amount', fontsize = 14)

plt.axis('equal')

plt.show()
#Figure 3: MPI distribution of each world regions

fig, ax = plt.subplots(figsize = (15,5))

countryList = kivaLoans.loc[:,['country','term_in_months']]

data4 = countryList.merge(kivaMPIregion, on = 'country')

sns.boxplot(x = 'world_region', y = 'MPI', data = data4, ax = ax)

plt.show()
# Figure 4:

plt.figure(figsize = (5,5))

topGender = kivaLoans['borrower_genders'].value_counts()

topGender = topGender.head(2)

plt.pie(topGender.values, labels = topGender.index, autopct='%1.1f%%', 

        colors = ['gold', 'yellowgreen'], shadow = True, startangle=90)

plt.title('Loan distribution by genders', fontsize = 14)

plt.show()
# Figure 5: Top sectors receiving mean loan amount

data2 = kivaLoans.groupby(['sector'])['loan_amount'].mean().sort_values(ascending = False)

data2 = data2.head(10)

plt.figure(figsize = (10,5))

graph3 = sns.barplot(x = data2.values, y = data2.index)

graph3.set_xlabel('Mean loan amount')
# Figure 6: Top sectors receiving loan

data2 = kivaLoans['sector'].value_counts().sort_values(ascending = False)

data2 = data2.head(10)

plt.figure(figsize = (10,5))

graph3 = sns.barplot(x = data2.values, y = data2.index)

graph3.set_xlabel('Loan Frequency')

graph3.set_ylabel('Sector')
# Figure 7: Top activities receiving loan

data2 = kivaLoans['activity'].value_counts().sort_values(ascending = False)

data2 = data2.head(10)

plt.figure(figsize = (10,5))

graph3 = sns.barplot(x = data2.values, y = data2.index)

graph3.set_xlabel('Loan Frequency')

graph3.set_ylabel('Activity')
# Figure 8: Top uses receiving loan

data2 = kivaLoans['use'].value_counts().sort_values(ascending = False)

data2 = data2.head(10)

plt.figure(figsize = (10,5))

graph3 = sns.barplot(x = data2.values, y = data2.index)

graph3.set_xlabel('Loan Frequency')

graph3.set_ylabel('Use')
#Figure 9: Distribution of loan amount

plt.figure(figsize = (9,8))

upperlimit = np.percentile(kivaLoans.loan_amount.values, 99)

data3 = kivaLoans.loan_amount.loc[kivaLoans.loan_amount < upperlimit]

# data3.plot.hist()

graph4 = sns.distplot(data3, bins = 50, kde = False)
# Figure 10: Relationship between Sector and repayment intervals

cm = sns.light_palette('red', as_cmap = True)

intervalSector = pd.crosstab(kivaLoans['sector'], kivaLoans['repayment_interval'])

intervalSector.plot(kind = 'bar', stacked = True, figsize = (10, 6))
df= pd.DataFrame(kivaLoans)



df['repayment_group'] = df['repayment_interval'].apply(lambda x: x if x != 'bullet' and x!='irregular' else 'bullet or irregular')



df.head()

# groupby country, region, and repayment group



sub_group_data=df.groupby(["country", "region","repayment_group"])["loan_amount"].sum().reset_index(name="sub_group_count")

sub_group_data.head()



# groupby country, region

group_data=df.groupby(["country","region"])["loan_amount"].sum().reset_index(name="main_group_count")

group_data.head()



# join table

join_table_1=pd.merge(sub_group_data,group_data,on=['country','region'],how='inner')

join_table_1["%"]=join_table_1["sub_group_count"]/join_table_1["main_group_count"]

join_table_1.head()



# join with mpi table



mpi=kivaMPIregion[["country", "region", "MPI"]]

mpi.drop_duplicates()

mpi.head()



join_table_2=pd.merge(join_table_1,mpi,on=['country','region'],how='inner')

join_table_2.head()

#scatter plot visualization



join_table_2.repayment_group.unique()

plt.scatter(join_table_2[join_table_2["repayment_group"]=="bullet or irregular"]

                                      ["%"],join_table_2[join_table_2["repayment_group"]=="bullet or irregular"]

                                      ["MPI"])

plt.scatter(join_table_2[join_table_2["repayment_group"]=="monthly"]

                                      ["%"],join_table_2[join_table_2["repayment_group"]=="monthly"]

                                      ["MPI"])

plt.show()