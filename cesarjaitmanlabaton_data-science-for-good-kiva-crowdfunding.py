# Importing dependencies:

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



print('Installed all dependencies!')
# Exploring the data:

# I'm creating this notebook locally so the path of the files differ from Kaggle.



# Kiva dataset - Kaggle format:

kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')

kiva_mpi_region_locations = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding//kiva_mpi_region_locations.csv')

loan_theme_ids = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv')

loan_themes_by_region = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv')



# Multidimensional Poverty Index - Kaggle format:

MPI_national = pd.read_csv('../input/mpi/MPI_national.csv')

MPI_subnational = pd.read_csv('../input/mpi/MPI_subnational.csv')



# # Kiva dataset:

# kiva_loans = pd.read_csv('../kiva_loans.csv')

# kiva_mpi_region_locations = pd.read_csv('../kiva_mpi_region_locations.csv')

# loan_theme_ids = pd.read_csv('../loan_theme_ids.csv')

# loan_themes_by_region = pd.read_csv('../loan_themes_by_region.csv')



# # Multidimensional Poverty Index:

# MPI_national = pd.read_csv('../MPI_national.csv')

# MPI_subnational = pd.read_csv('../MPI_subnational.csv')



MPI_subnational.head()
# Converting dataset to lowercase for easier future use:



kiva_loans.columns = [x.lower() for x in kiva_loans.columns]

kiva_mpi_region_locations.columns = [x.lower() for x in kiva_mpi_region_locations.columns]

loan_theme_ids.columns = [x.lower() for x in loan_theme_ids.columns]

loan_themes_by_region.columns = [x.lower() for x in loan_themes_by_region.columns]

MPI_national.columns = [x.lower() for x in MPI_national.columns]

MPI_subnational.columns = [x.lower() for x in MPI_subnational.columns]
# Analizing the dataset:



print('kiva_loans Shape: ', kiva_loans.shape)

print('-' * 40)

print(pd.DataFrame(kiva_loans.info()))
# Overview of the dataset:



kiva_loans.describe(include=['O'])
# Changing the date column to to_datetime:



kiva_loans['date'] = pd.to_datetime(kiva_loans['date'])

kiva_loans['year'] = pd.DataFrame(kiva_loans['date'].dt.year)

kiva_loans['month'] = pd.DataFrame(kiva_loans['date'].dt.month)

kiva_loans['day'] = pd.DataFrame(kiva_loans['date'].dt.day)

kiva_loans.head()
null_values = kiva_loans.isnull().sum()

null_values.columns = ['total_null']

total_cells = np.product(kiva_loans.shape)

missing_values = null_values.sum()



print('Only ', (missing_values/total_cells) * 100, 'of the dataset is missing.')
kiva_loan_regions = pd.DataFrame(kiva_mpi_region_locations['world_region'].value_counts())

kiva_loan_regions.reset_index(inplace=True)

kiva_loan_regions.columns = ['world_region', 'total_amount']



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=kiva_loan_regions['total_amount'], y=kiva_loan_regions['world_region'])

barplot.set(xlabel='', ylabel='')

plt.title('Regions that got most of the loans:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
kiva_loans_countries = pd.DataFrame(kiva_loans['country'].value_counts(sort=['loan_amount']))

kiva_loans_countries.reset_index(inplace=True)

kiva_loans_countries.columns = ['country', 'total_loaned']



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=kiva_loans_countries['total_loaned'][:20], y=kiva_loans_countries['country'][:20])

barplot.set(xlabel='', ylabel='')

plt.title('Top 20 countries that got the most loans:', fontsize=20)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.show()
plt.figure(figsize=(20, 20))



pointplot = sns.pointplot(x=kiva_mpi_region_locations['mpi'], y=kiva_mpi_region_locations['country'], hue=kiva_mpi_region_locations['world_region'])

pointplot.set(xlabel='', ylabel='')

plt.yticks(fontsize=17)

plt.yticks(fontsize=12)

plt.show()
kiva_currency = pd.DataFrame(kiva_loans['currency'].value_counts(sort='country'))



plt.figure(figsize=(20, 7))

sns.set_style("whitegrid")



barplot = sns.barplot(x=kiva_currency.index[:15], y=kiva_currency['currency'][:15])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the 15 most popular currency used:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
kiva_loans['loan_amount_log'] = np.log(kiva_loans['loan_amount'])



plt.figure(figsize=(20, 7))



sns.set_style("whitegrid")

boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=kiva_loans)

boxplot.set(xlabel='', ylabel='')

plt.title('Displaying all sectors that got loans:', fontsize=20)

plt.xticks(rotation=60, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
kiva_loans_sectors = pd.DataFrame(kiva_loans.groupby(['sector'])['loan_amount'].mean())

kiva_loans_sectors.reset_index(inplace=True)

kiva_loans_sectors.columns = ['sector', 'average_frequent_sectors']



plt.figure(figsize=(20, 7))



sns.set_style("whitegrid")

boxplot = sns.barplot(x='sector', y='average_frequent_sectors', data=kiva_loans_sectors)

boxplot.set(xlabel='', ylabel='')

plt.title('Displaying the most frequent sectors that get loans:', fontsize=20)

plt.xticks(rotation=60, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
kiva_loans_counting_entertainment = pd.DataFrame(kiva_loans[kiva_loans['sector'] == 'Entertainment']['loan_amount'].value_counts())

kiva_loans_counting_entertainment.reset_index(inplace=True)

kiva_loans_counting_entertainment.columns = ['total_amount', 'times_invested']

kiva_loans_counting_wholesale = pd.DataFrame(kiva_loans[kiva_loans['sector'] == 'Wholesale']['loan_amount'].value_counts())

kiva_loans_counting_wholesale.reset_index(inplace=True)

kiva_loans_counting_wholesale.columns = ['total_amount', 'times_invested']

kiva_loans_counting_agriculture = pd.DataFrame(kiva_loans[kiva_loans['sector'] == 'Agriculture']['loan_amount'].value_counts())

kiva_loans_counting_agriculture.reset_index(inplace=True)

kiva_loans_counting_agriculture.columns = ['total_amount', 'times_invested']



fig = plt.figure(figsize=(20, 10))

mpl.rcParams['xtick.labelsize'] = 12

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(311)

ax1 = sns.pointplot(x=kiva_loans_counting_entertainment['times_invested'], y=kiva_loans_counting_entertainment['total_amount'], color='green')

ax1.set(xlabel='Times Invested', ylabel='Amount')

ax1.set_title('Displaying the frequency and values of loans in entertainment:', fontsize=20)



plt.subplot(312)

ax1 = sns.pointplot(x=kiva_loans_counting_wholesale['times_invested'], y=kiva_loans_counting_wholesale['total_amount'], color='purple')

ax1.set(xlabel='Times Invested', ylabel='Amount')

ax1.set_title('Displaying the frequency and values of loans in wholesale:', fontsize=20)



plt.subplot(313)

ax2 = sns.pointplot(x=kiva_loans_counting_agriculture['times_invested'], y=kiva_loans_counting_agriculture['total_amount'], color='pink')

ax2.set(xlabel='Times Invested', ylabel='Amount')

ax2.set_title('Displaying the frequency and values of loans in agriculture:', fontsize=20)



plt.tight_layout()

plt.show()
light_palette = sns.light_palette("green", as_cmap=True)

pd.crosstab(kiva_loans['year'], kiva_loans['sector']).style.background_gradient(cmap=light_palette)
kiva_use = pd.DataFrame(kiva_loans['use'].value_counts(sort='loan_amount'))

kiva_use.reset_index(inplace=True)

kiva_use.columns = ['use', 'total_amount']



plt.figure(figsize=(15, 10))



barplot = sns.barplot(x=kiva_use['total_amount'][:20], y=kiva_use['use'][:20])

barplot.set(xlabel='', ylabel='')

plt.title('Top 20 usages of loans:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=15)

plt.show()
plt.figure(figsize=(20, 10))

mpl.rcParams['xtick.labelsize'] = 12

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(221)

ax1 = plt.scatter(range(kiva_loans['sector'].shape[0]), np.sort(kiva_loans['loan_amount'].values))

# ax1.title('Displaying funding of usage of investments:', fontsize=20)



plt.subplot(222)

ax2 = plt.scatter(range(kiva_loans['sector'].shape[0]), np.sort(kiva_loans['funded_amount'].values))

# ax2.title('Displaying funding of usage of investments:', fontsize=20)



plt.tight_layout()

plt.show()
light_palette = sns.light_palette("green", as_cmap=True)

pd.crosstab(kiva_loans['year'], kiva_loans['activity']).style.background_gradient(cmap=light_palette)
kiva_loans['borrower_genders'] = kiva_loans['borrower_genders'].astype(str)

gender_list = pd.DataFrame(kiva_loans['borrower_genders'].str.split(',').tolist())

kiva_loans['clean_borrower_genders'] = gender_list[0]

kiva_loans.loc[kiva_loans['clean_borrower_genders'] == 'nan', 'clean_borrower_genders'] = np.nan



kiva_gender = kiva_loans['clean_borrower_genders'].value_counts()

labels = kiva_gender.index



plt.figure(figsize=(15, 5))



patches = plt.pie(kiva_gender, autopct='%1.1f%%')

plt.legend(labels, fontsize=20)

plt.axis('equal')

plt.tight_layout()

plt.show()
sex_mean = kiva_loans.groupby('clean_borrower_genders').count()



fig = plt.figure(figsize=(20, 10))

mpl.rcParams['xtick.labelsize'] = 17

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(211)

ax1 = sns.violinplot(kiva_loans['loan_amount'], kiva_loans['clean_borrower_genders'])

ax1.set(xlabel='', ylabel='')

ax1.set_title('Displaying the total amount of money loaned by gender:', fontsize=20)



plt.subplot(212)

ax2 = sns.violinplot(kiva_loans['loan_amount'], kiva_loans['clean_borrower_genders'])

ax2.set(xlabel='', ylabel='')

ax2.set_title('Displaying a closer look of the initial part of the violinplot for better visualization of distribution:', fontsize=20)

ax2.set_xlim(0, 2500)



plt.tight_layout()

plt.show()
kiva_loan = pd.DataFrame(kiva_loans['lender_count'].value_counts())

kiva_loan.reset_index(inplace=True)

kiva_loan.columns = ['lenders', 'total_amount']

kiva_loan



plt.figure(figsize=(20, 7))



pointplot = sns.pointplot(x=kiva_loan['lenders'], y=kiva_loan['total_amount'], color='g')

pointplot.set(xlabel='', ylabel='')

plt.title('Displaying the 25 most common amounts of lenders that invest in one loan:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.xlim(0, 25)

plt.show()
plt.figure(figsize=(20, 7))



boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=kiva_loans, hue='clean_borrower_genders')

boxplot.set(xlabel='', ylabel='')

plt.title('Displaying how each gender invest in each sector:', fontsize=17)

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
plt.figure(figsize=(25, 7))



boxplot = sns.pointplot(x='activity', y='loan_amount_log', data=kiva_loans, hue='clean_borrower_genders')

boxplot.set(xlabel='', ylabel='')

plt.title('Displaying how each gender invest in each activity:', fontsize=17)

plt.xticks(rotation=80, fontsize=8)

plt.yticks(fontsize=17)

plt.show()
facetgrid = sns.FacetGrid(kiva_loans, hue='repayment_interval', size=5, aspect=3)

facetgrid = (facetgrid.map(sns.kdeplot, 'loan_amount_log', shade=True).set_axis_labels('Months', 'Total Amount (log)').add_legend(fontsize=17))
facetgrid = sns.FacetGrid(kiva_loans, hue='year', size=5, aspect=3)

facetgrid = (facetgrid.map(sns.kdeplot, 'loan_amount_log', shade=True).set_axis_labels('Months', 'Total Amount (log)').add_legend(fontsize=17))
kiva_terms = pd.DataFrame(kiva_loans['term_in_months'].value_counts(sort='country'))

kiva_terms.reset_index(inplace=True)

kiva_terms.columns = ['term_in_months', 'total_amount']



plt.figure(figsize=(20, 7))



pointplot = sns.pointplot(x=kiva_terms['term_in_months'], y=kiva_terms['total_amount'], color='g')

pointplot.set(xlabel='', ylabel='')

plt.title('Displaying how long in average the monthly terms are:', fontsize=20)

plt.yticks(fontsize=15)

plt.xticks(fontsize=15)

plt.xlim(0, 30)

plt.show()
# Analyzing loan themes by region:

loan_themes_by_region.head()
loan_partner = pd.DataFrame(loan_themes_by_region['field partner name'].value_counts(sort=['amount']))

loan_partner.reset_index(inplace=True)

loan_partner.columns = ['partner_name', 'total_amount']

loan_partner.head()



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=loan_partner['total_amount'][:20], y=loan_partner['partner_name'][:20])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the partners that invest the most times:', fontsize=20)

plt.xticks(rotation=90, fontsize=15)

plt.yticks(fontsize=17)

plt.show()
loan_amount = loan_themes_by_region.groupby('field partner name').sum().sort_values(by='amount', ascending=False)

loan_amount.reset_index(inplace=True)

loan_amount.head()



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=loan_amount['amount'][:20], y=loan_amount['field partner name'][:20])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the lenders that invested the most money:', fontsize=20)

plt.xticks(rotation=80, fontsize=15)

plt.yticks(fontsize=17)

plt.show()
loan_region = pd.DataFrame(loan_themes_by_region['region'].value_counts())

loan_region.reset_index(inplace=True)

loan_region.columns = ['region', 'total_amount']



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=loan_region['total_amount'][:20], y=loan_region['region'][:20])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the most common regions that partners invest in:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
loan_country = pd.DataFrame(loan_themes_by_region['country'].value_counts())

loan_country.reset_index(inplace=True)

loan_country.columns = ['region', 'total_amount']



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=loan_country['total_amount'][:20], y=loan_country['region'][:20])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the most common countries that partners invest in:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
loan_theme = pd.DataFrame(loan_themes_by_region['loan theme type'].value_counts()).reset_index()

loan_theme.columns = ['theme', 'total_amount']



plt.figure(figsize=(20, 7))



barplot = sns.pointplot(x=loan_theme['theme'][:15], y=loan_theme['total_amount'][:15], color='g')

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the most common themes that partners invest in:', fontsize=20)

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
loan_general = pd.DataFrame(loan_themes_by_region[loan_themes_by_region['loan theme type'] == 'General'])

loan_general = pd.DataFrame(loan_general['sector'].value_counts().reset_index())

loan_general.columns = ['sector', 'total_amount']



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=loan_general['sector'][:10], y=loan_general['total_amount'][:10])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the most common sector with general theme that partners invest in:', fontsize=20)

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
# loans_kiva = pd.DataFrame(loan_themes_by_region['forkiva'].value_counts().reset_index())



plt.figure(figsize=(20, 7))



pointplot = sns.pointplot(x='sector', y='amount', hue='forkiva', data=loan_themes_by_region)

pointplot.set(xlabel='', ylabel='')

plt.title('Displaying loans that were for Kiva based in sector:', fontsize=20)

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
philippines = pd.DataFrame(kiva_loans[kiva_loans['country'] == 'Philippines'])

philippines_partners = pd.DataFrame(loan_themes_by_region[loan_themes_by_region['country'] == 'Philippines'])

philippines.head()
kiva_php = 52.0640

kiva_usd = 1.00

# source: http://www.xe.com/currencyconverter/convert/?Amount=1&From=USD&To=PHP



philippines_php = philippines['loan_amount'].sum()

philippines_transform = philippines_php / kiva_php

print('Total amount invested in PHP: ₱', philippines_php)

print('Amount invested in USD: #',  philippines_transform)
kiva_mpi_region_locations_philippines = pd.DataFrame(kiva_mpi_region_locations[kiva_mpi_region_locations['country'] == 'Philippines'])

kiva_mpi_region_locations_philippines_mpi = kiva_mpi_region_locations_philippines['mpi'].mean()

print('Philippines has a MPI of: ', kiva_mpi_region_locations_philippines_mpi)
plt.figure(figsize=(20, 7))



sns.set_style('whitegrid')

boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=philippines)

boxplot.set(xlabel='', ylabel='')

plt.title("Total loaned in Philippines' sectors :", fontsize=20)

plt.xticks(rotation=60, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
philippines_sector_average = pd.DataFrame(philippines.groupby(['sector'])['loan_amount'].mean().reset_index())



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=philippines_sector_average['sector'], y=philippines_sector_average['loan_amount'])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the mean of loans in each sector:', fontsize=20)

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
philippines_counting_agriculture = pd.DataFrame(philippines[philippines['sector'] == 'Agriculture']['loan_amount'].value_counts().reset_index())

philippines_counting_wholesale = pd.DataFrame(philippines[philippines['sector'] == 'Wholesale']['loan_amount'].value_counts().reset_index())



plt.figure(figsize=(20, 7))

mpl.rcParams['xtick.labelsize'] = 12

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(211)

ax1 = sns.pointplot(x=philippines_counting_agriculture['loan_amount'], y=philippines_counting_agriculture['index'], color='purple')

ax1.set(xlabel='Times Invested', ylabel='Amount')

ax1.set_title('Displaying the frequency and values of loans in agriculture:', fontsize=20)



plt.subplot(212)

ax2 = sns.pointplot(x=philippines_counting_wholesale['loan_amount'], y=philippines_counting_wholesale['index'], color='pink')

ax2.set(xlabel='Times Invested', ylabel='Amount')

ax2.set_title('Displaying the frequency and values of loans in wholesale:', fontsize=20)



plt.tight_layout()

plt.show()
plt.figure(figsize=(20, 7))



boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=philippines, hue='clean_borrower_genders')

boxplot.set(xlabel='', ylabel='')

plt.title('Displaying how each sector got funded based on gender:', fontsize=20)

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
philippines_sector_partner = pd.DataFrame(philippines_partners['sector'].value_counts().reset_index())

philippines_sector_partner.head()



plt.figure(figsize=[20, 7])



barplot = sns.barplot(x='sector', y='index', data=philippines_sector_partner)

barplot.set(xlabel='', ylabel='')

plt.title('Displaying how each sector got funded by partners:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
light_palette = sns.light_palette("green", as_cmap=True)

pd.crosstab(philippines['year'], philippines['sector']).style.background_gradient(cmap=light_palette)
philippines_activity = pd.DataFrame(philippines['activity'].value_counts().reset_index())

philippines_activity.columns = ['activity', 'total_amount']



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=philippines_activity['total_amount'][:20], y=philippines_activity['activity'][:20])

barplot.set(xlabel='', ylabel='')

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
philippines_generalstore = pd.DataFrame(philippines[philippines['activity'] == 'General Store'])

philippines_generalstore = pd.DataFrame(philippines_generalstore['use'].value_counts())

philippines_generalstore.reset_index(inplace=True)

philippines_generalstore.columns = ['use', 'total_amount']



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=philippines_generalstore['total_amount'][:20], y=philippines_generalstore['use'][:20])

barplot.set(xlabel='', ylabel='')

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
plt.figure(figsize=(25, 7))



pointplot = sns.pointplot(x='activity', y='loan_amount_log', data=philippines, hue='clean_borrower_genders')

pointplot.set(xlabel='', ylabel='')

plt.title('Displaying how each gender invest in each activity:', fontsize=17)

plt.xticks(rotation=80, fontsize=8)

plt.yticks(fontsize=17)

plt.show()
light_palette = sns.light_palette("green", as_cmap=True)

pd.crosstab(philippines['year'], philippines['activity']).style.background_gradient(cmap=light_palette)
philippines_use = pd.DataFrame(philippines['use'].value_counts())

philippines_use.reset_index(inplace=True)

philippines_use.columns = ['use', 'total']



# Displaying the results in a pie chart:



plt.figure(figsize=(20, 8))



barplot = sns.barplot(x=philippines_use['total'][:20], y=philippines_use['use'][:20])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the top 20 most common usage of loans:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
philippines_amount_loan = philippines.groupby('year').count().sort_values(by='loan_amount', ascending=False)

philippines_amount_loan.reset_index(inplace=True)



plt.figure(figsize=(20, 7))



barplot = sns.pointplot(x=philippines_amount_loan['year'], y=philippines_amount_loan['loan_amount'], color='g')

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the yearly loan amounts:', fontsize=20)

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
plt.figure(figsize=(20, 10))

mpl.rcParams['xtick.labelsize'] = 12

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(221)

ax1 = plt.scatter(range(philippines['sector'].shape[0]), np.sort(philippines['loan_amount'].values))

# ax1.title('Displaying funding of usage of investments:', fontsize=20)



plt.subplot(222)

ax2 = plt.scatter(range(philippines['sector'].shape[0]), np.sort(philippines['funded_amount'].values))

# ax2.title('Displaying funding of usage of investments:', fontsize=20)



plt.tight_layout()

plt.show()
facetgrid = sns.FacetGrid(philippines, hue='repayment_interval', size=5, aspect=3)

facetgrid = (facetgrid.map(sns.kdeplot, 'loan_amount_log', shade=True).set_axis_labels('Months', 'Total Amount (log)').add_legend(fontsize=17))
philippines_terms = pd.DataFrame(philippines['term_in_months'].value_counts(sort='country'))

philippines_terms.reset_index(inplace=True)

philippines_terms.columns = ['term_in_months', 'total_amount']



plt.figure(figsize=(20, 7))



pointplot = sns.pointplot(x=philippines_terms['term_in_months'], y=philippines_terms['total_amount'], color='g')

pointplot.set(xlabel='', ylabel='')

plt.title('Displaying how long in average the monthly terms are:', fontsize=20)

plt.yticks(fontsize=15)

plt.xticks(fontsize=15)

plt.xlim(0, 30)

plt.show()
sex_mean = philippines.groupby('clean_borrower_genders').count()



fig = plt.figure(figsize=(20, 10))

mpl.rcParams['xtick.labelsize'] = 17

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(211)

ax1 = sns.violinplot(philippines['loan_amount'], philippines['clean_borrower_genders'])

ax1.set(xlabel='', ylabel='')

ax1.set_title('Displaying the total amount of money loaned by gender:', fontsize=20)



plt.subplot(212)

ax2 = sns.violinplot(philippines['loan_amount'], philippines['clean_borrower_genders'])

ax2.set(xlabel='', ylabel='')

ax2.set_title('Displaying a closer look of the initial part of the violinplot for better visualization of distribution:', fontsize=20)

ax2.set_xlim(0, 2500)



plt.tight_layout()

plt.show()
philippines_partners_count = pd.DataFrame(philippines_partners['field partner name'].value_counts().reset_index())

plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=philippines_partners_count['field partner name'], y=philippines_partners_count['index'])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying partners that invest in the Philippines:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
kenya = pd.DataFrame(kiva_loans[kiva_loans['country'] == 'Kenya'])

kenya_partners = pd.DataFrame(loan_themes_by_region[loan_themes_by_region['country'] == 'Kenya'])

kenya.head()
kenya_mpi = pd.DataFrame(MPI_subnational[MPI_subnational['country'] == 'Kenya'])

kenya_mpi = kenya_mpi['mpi national'].mean()

print("Kenya's MPI is:", kenya_mpi)
kiva_kes = 100.801



kenya_loan_amount = kenya['loan_amount'].sum()

# sounrce: http://www.xe.com/currencyconverter/convert/?Amount=1&From=USD&To=KES



kiva_transform = kenya_loan_amount / kiva_kes

print('Total amount invested in Kenya: KSh', kenya_loan_amount)

print('Amount invested in USD: $', kiva_transform)
# Creating and saving the different sectors:

plt.figure(figsize=(20, 7))



sns.set_style('whitegrid')

boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=kenya)

boxplot.set(xlabel='', ylabel='')

plt.title('Displaying all the sectors and their repective loan amounts:', fontsize=20)

plt.xticks(rotation=60, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
kenya_sector_average = kenya.groupby(['sector'])['loan_amount'].mean().reset_index()

kenya_sector_average.head()



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=kenya_sector_average['sector'], y=kenya_sector_average['loan_amount'])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the average loan amount in each sector:', fontsize=20)

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
kenya_average_health = pd.DataFrame(kenya[kenya['sector'] == 'Health']['loan_amount'].value_counts().reset_index())

kenya_average_agriculture = pd.DataFrame(kenya[kenya['sector'] == 'Agriculture']['loan_amount'].value_counts().reset_index())



plt.figure(figsize=(20, 7))

mpl.rcParams['xtick.labelsize'] = 12

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(211)

ax1 = sns.pointplot(x=kenya_average_health['loan_amount'], y=kenya_average_health['index'], color='green')

ax1.set(xlabel='Times Invested', ylabel='Amount')

ax1.set_title('Displaying the frequency and values of loans in Health:', fontsize=20)



plt.subplot(212)

ax2 = sns.pointplot(x=kenya_average_agriculture['loan_amount'], y=kenya_average_agriculture['index'], color='pink')

ax2.set(xlabel='Times Invested', ylabel='Amount')

ax2.set_title('Displaying the frequency and values of loans in Agriculture:', fontsize=20)



plt.tight_layout()

plt.show()
plt.figure(figsize=(20, 7))



boxplot = sns.boxplot(x='sector', y='loan_amount_log', hue='clean_borrower_genders', data=kenya)

boxplot.set(xlabel='', ylabel='')

plt.title('Displaying genders investment in each sector:', fontsize=20)

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
kenya_sector_partners = pd.DataFrame(kenya_partners['sector'].value_counts().reset_index())



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=kenya_sector_partners['sector'], y=kenya_sector_partners['index'])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying how each partner invest in each sector:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
light_palette = sns.light_palette('green', as_cmap=True)

pd.crosstab(kenya['year'], kenya['sector']).style.background_gradient(cmap=light_palette)
kenya_activity = pd.DataFrame(kenya['activity'].value_counts(sort=['loan_amount']))

kenya_activity.reset_index(inplace=True)

kenya_activity.columns = ['activity', 'total_amount']



# Displaying each activity in a pie chart:



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=kenya_activity['total_amount'][:20], y=kenya_activity['activity'][:20])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the activities in the agriculture sector:', fontsize=20)

plt.show()
kenya_farming = pd.DataFrame(kenya[kenya['activity'] == 'Farming']['use'].value_counts().reset_index())



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=kenya_farming['use'][:20], y=kenya_farming['index'][:20])

plt.show()
plt.figure(figsize=(25, 7))



pointplot = sns.pointplot(x='activity', y='loan_amount_log', data=kenya, hue='clean_borrower_genders')

pointplot.set(xlabel='', ylabel='')

plt.title('Displaying how each gender invest in each activity:', fontsize=17)

plt.xticks(rotation=80, fontsize=8)

plt.yticks(fontsize=17)

plt.show()
light_palette = sns.light_palette("green", as_cmap=True)

pd.crosstab(kenya['year'], kenya['activity']).style.background_gradient(cmap=light_palette)
kenya_use = pd.DataFrame(kenya['use'].value_counts().reset_index())



plt.figure(figsize=(20, 8))



barplot = sns.barplot(x=kenya_use['use'][:20], y=kenya_use['index'][:20])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the top 20 most common usage of loans:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
kenya_amount_year = kenya.groupby('year').count().sort_values(by='loan_amount', ascending=False)

kenya_amount_year.reset_index(inplace=True)



plt.figure(figsize=(20, 7))



pointplot = sns.pointplot(x=kenya_amount_year['year'], y=kenya_amount_year['loan_amount'], color='g')

pointplot.set(xlabel='', ylabel='')

plt.title('Displaying the yearly loan amounts:', fontsize=20)

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
plt.figure(figsize=(20, 10))

mpl.rcParams['xtick.labelsize'] = 12

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(221)

ax1 = plt.scatter(range(kenya['sector'].shape[0]), np.sort(kenya['loan_amount'].values))

# ax1.title('Displaying funding of usage of investments:', fontsize=20)



plt.subplot(222)

ax2 = plt.scatter(range(kenya['sector'].shape[0]), np.sort(kenya['funded_amount'].values))

# ax2.title('Displaying funding of usage of investments:', fontsize=20)



plt.tight_layout()

plt.show()
facetgrid = sns.FacetGrid(kenya, hue='repayment_interval', size=5, aspect=3)

facetgrid = (facetgrid.map(sns.kdeplot, 'loan_amount_log', shade=True).set_axis_labels('Months', 'Total Amount (log)').add_legend(fontsize=17))
kenya_terms = pd.DataFrame(kenya['term_in_months'].value_counts(sort='country'))

kenya_terms.reset_index(inplace=True)

kenya_terms.columns = ['term_in_months', 'total_amount']



plt.figure(figsize=(20, 7))



pointplot = sns.pointplot(x=kenya_terms['term_in_months'], y=kenya_terms['total_amount'], color='g')

pointplot.set(xlabel='', ylabel='')

plt.title('Displaying how long in average the monthly terms are:', fontsize=20)

plt.yticks(fontsize=15)

plt.xticks(fontsize=15)

plt.xlim(0, 30)

plt.show()
fig = plt.figure(figsize=(20, 10))

mpl.rcParams['xtick.labelsize'] = 17

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(211)

ax1 = sns.violinplot(kenya['loan_amount'], kenya['clean_borrower_genders'])

ax1.set(xlabel='', ylabel='')

ax1.set_title('Displaying the total amount of money loaned by gender:', fontsize=20)



plt.subplot(212)

ax2 = sns.violinplot(kenya['loan_amount'], kenya['clean_borrower_genders'])

ax2.set(xlabel='', ylabel='')

ax2.set_title('Displaying a closer look of the initial part of the violinplot for better visualization of distribution:', fontsize=20)

ax2.set_xlim(0, 2000)



plt.tight_layout()

plt.show()
kenya_partners_count = pd.DataFrame(loan_themes_by_region['field partner name'].value_counts().reset_index())



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=kenya_partners_count['field partner name'][:20], y=kenya_partners_count['index'][:20])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying partners that invest in the Kenya:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
salvador = kiva_loans[kiva_loans['country'] == 'El Salvador']

salvador_partners = pd.DataFrame(loan_themes_by_region[loan_themes_by_region['country'] == 'El Salvador'])

salvador.head()
salvador_mpi = pd.DataFrame(MPI_subnational[MPI_subnational['country'] == 'El Salvador'])

salvador_mpi = salvador_mpi['mpi national'].mean()

print("El Salvador's MPI is:", salvador_mpi)
plt.figure(figsize=(20, 7))



sns.set_style('whitegrid')

boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=salvador)

boxplot.set(xlabel='', ylabel='')

plt.title('Displaying each sector with their respective loans invested:', fontsize=20)

plt.xticks(rotation=60, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
salvador_sector_average = salvador.groupby('sector')['loan_amount'].mean().reset_index()



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=salvador_sector_average['sector'], y=salvador_sector_average['loan_amount'])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the average investment in sectors:', fontsize=(20))

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
salvador_sector_wholesale = pd.DataFrame(salvador[salvador['sector'] == 'Wholesale']['loan_amount'].value_counts().reset_index())

salvador_sector_transportation = pd.DataFrame(salvador[salvador['sector'] == 'Transportation']['loan_amount'].value_counts().reset_index())

salvador_sector_agriculture = pd.DataFrame(salvador[salvador['sector'] == 'Agriculture']['loan_amount'].value_counts().reset_index())



plt.figure(figsize=(20, 7))

mpl.rcParams['xtick.labelsize'] = 12

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(311)

ax1 = sns.pointplot(x='loan_amount', y='index', data=salvador_sector_wholesale, color='red')

ax1.set(xlabel='Times Invested', ylabel='Amount')

ax1.set_title('Displaying the frequency and values of loans in Wholesale:', fontsize=20)



plt.subplot(312)

ax2 = sns.pointplot(x='loan_amount', y='index', data=salvador_sector_transportation, color='purple')

ax2.set(xlabel='Times Invested', ylabel='Amount')

ax2.set_title('Displaying the frequency and values of loans in Transportation:', fontsize=20)



plt.subplot(313)

ax3 = sns.pointplot(x='loan_amount', y='index', data=salvador_sector_agriculture, color='pink')

ax3.set(xlabel='Times Invested', ylabel='Amount')

ax3.set_title('Displaying the frequency and values of loans in Agriculture:', fontsize=20)



plt.tight_layout()

plt.show()
plt.figure(figsize=(20, 7))



boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=salvador, hue='clean_borrower_genders')

boxplot.set(xlabel='', ylabel='')

plt.title('Displaying genders investment in each sector:', fontsize=20)

plt.xticks(rotation=80, fontsize=17)

plt.xticks(fontsize=17)

plt.show()
salvador_sector_partners = pd.DataFrame(salvador_partners['sector'].value_counts().reset_index())



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x='sector', y='index', data=salvador_sector_partners)

barplot.set(xlabel='', ylabel='')

plt.title('Displaying partners investments:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
salvador_partner_theme_type = pd.DataFrame(salvador_partners['loan theme type'].value_counts().reset_index())



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x='loan theme type', y='index', data=salvador_partner_theme_type)

barplot.set(xlabel='', ylabel='')

plt.title('Displaying themes from partners:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
light_palette = sns.light_palette("green", as_cmap=True)

pd.crosstab(salvador['year'], salvador['sector']).style.background_gradient(cmap=light_palette)
salvador_activity = pd.DataFrame(salvador['activity'].value_counts().reset_index())



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=salvador_activity['activity'][:20], y=salvador_activity['index'][:20])

barplot.set(xlabel='', ylabel='')

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
salvador_housing = pd.DataFrame(salvador[salvador['activity'] == 'Personal Housing Expenses']['use'].value_counts().reset_index())



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=salvador_housing['use'][:20], y=salvador_housing['index'][:20])

barplot.set(xlabel='', ylabel='')

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
plt.figure(figsize=(25, 7))



pointplot = sns.pointplot(x='activity', y='loan_amount_log', data=salvador, hue='clean_borrower_genders')

pointplot.set(xlabel='', ylabel='')

plt.title('Displaying how each gender invest in each activity:', fontsize=17)

plt.xticks(rotation=80, fontsize=8)

plt.yticks(fontsize=17)

plt.show()
light_palette = sns.light_palette("green", as_cmap=True)

pd.crosstab(salvador['year'], salvador['activity']).style.background_gradient(cmap=light_palette)
salvador_use = pd.DataFrame(salvador['use'].value_counts().reset_index())



plt.figure(figsize=(20, 8))



barplot = sns.barplot(x=salvador_use['use'][:20], y=salvador_use['index'][:20])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the top 20 most common usage of loans:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
salvador_amount_loan = salvador.groupby('year').count().sort_values(by='loan_amount', ascending=False).reset_index()



plt.figure(figsize=(20, 7))



barplot = sns.pointplot(x=salvador_amount_loan['year'], y=salvador_amount_loan['loan_amount'], color='g')

barplot.set(xlabel='', ylabel='')

plt.title('Displaying the yearly loan amounts:', fontsize=20)

plt.xticks(rotation=80, fontsize=17)

plt.yticks(fontsize=17)

plt.show()
plt.figure(figsize=(20, 10))

mpl.rcParams['xtick.labelsize'] = 12

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(221)

ax1 = plt.scatter(range(salvador['sector'].shape[0]), np.sort(salvador['loan_amount'].values))



plt.subplot(222)

ax2 = plt.scatter(range(salvador['sector'].shape[0]), np.sort(salvador['funded_amount'].values))



plt.tight_layout()

plt.show()
facetgrid = sns.FacetGrid(salvador, hue='repayment_interval', size=5, aspect=3)

facetgrid = (facetgrid.map(sns.kdeplot, 'loan_amount_log', shade=True).set_axis_labels('Months', 'Total Amount (log)').add_legend(fontsize=17))
salvador_terms = pd.DataFrame(salvador['term_in_months'].value_counts(sort='country').reset_index())



plt.figure(figsize=(20, 7))



pointplot = sns.pointplot(x=salvador_terms['index'], y=salvador_terms['term_in_months'], color='g')

pointplot.set(xlabel='', ylabel='')

plt.title('Displaying how long in average the monthly terms are:', fontsize=20)

plt.yticks(fontsize=15)

plt.xticks(fontsize=15)

plt.xlim(0, 30)

plt.show()
fig = plt.figure(figsize=(20, 10))

mpl.rcParams['xtick.labelsize'] = 17

mpl.rcParams['ytick.labelsize'] = 17



plt.subplot(211)

ax1 = sns.violinplot(salvador['loan_amount'], salvador['clean_borrower_genders'])

ax1.set(xlabel='', ylabel='')

ax1.set_title('Displaying the total amount of money loaned by gender:', fontsize=20)



plt.subplot(212)

ax2 = sns.violinplot(salvador['loan_amount'], salvador['clean_borrower_genders'])

ax2.set(xlabel='', ylabel='')

ax2.set_title('Displaying a closer look of the initial part of the violinplot for better visualization of distribution:', fontsize=20)

ax2.set_xlim(0, 1750)



plt.tight_layout()

plt.show()
salvador_partners_count = pd.DataFrame(salvador_partners['field partner name'].value_counts().reset_index())



plt.figure(figsize=(20, 7))



barplot = sns.barplot(x=salvador_partners_count['field partner name'], y=salvador_partners_count['index'])

barplot.set(xlabel='', ylabel='')

plt.title('Displaying partners that invest in the Philippines:', fontsize=20)

plt.xticks(fontsize=17)

plt.yticks(fontsize=17)

plt.show()
MPI_national.head()
MPI_subnational.head()
ordered_MPI_national = MPI_national.sort_values(by='intensity of deprivation rural', ascending=False)

ordered_MPI_subnational = MPI_subnational.sort_values(by='intensity of deprivation regional', ascending=False)



fig = plt.figure(figsize=(20, 10))

mpl.rcParams['xtick.labelsize'] = 17

mpl.rcParams['ytick.labelsize'] = 13



plt.subplot(121)

ax1 = sns.barplot(x=ordered_MPI_national['intensity of deprivation rural'][:20], y=ordered_MPI_national['country'][:20])

ax1.set(xlabel='', ylabel='')

ax1.set_title('Rural deprivation intensity:', fontsize=20)



plt.subplot(122)

ax2 = sns.barplot(x=ordered_MPI_subnational['intensity of deprivation regional'][:10], y=ordered_MPI_subnational['sub-national region'][:10])

ax2.set(xlabel='', ylabel='')

ax2.set_title('Sub-regional deprivation intensity:', fontsize=20)



plt.show()
philippines.head()