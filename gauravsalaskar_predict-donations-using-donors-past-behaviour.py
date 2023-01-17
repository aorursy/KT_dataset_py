# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime # convert string to date

import gc # for garbage collection to reduce RAM load

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load data
Donations = pd.read_csv('../input/Donations.csv', parse_dates=['Donation Received Date'])
Donors = pd.read_csv('../input/Donors.csv', dtype={'Donor Zip': 'str'})
Projects = pd.read_csv('../input/Projects.csv', parse_dates=['Project Posted Date', 'Project Expiration Date', 'Project Fully Funded Date'])
Resources = pd.read_csv('../input/Resources.csv')
Schools = pd.read_csv('../input/Schools.csv')
Teachers = pd.read_csv('../input/Teachers.csv', parse_dates=['Teacher First Project Posted Date'])
print('Data loaded')
# Check data loaded properly - column type
print('Donations')
print(Donations.dtypes)

print(' ')
print('Donors')
print(Donors.dtypes)

print(' ')
print('Projects') # WARNING: Column Project Essay contains line breaks
print(Projects.dtypes)

print(' ')
print('Resources')
print(Resources.dtypes)

print(' ')
print('Schools')
print(Schools.dtypes)

print(' ')
print('Teachers')
print(Teachers.dtypes)
# Select parameters for creating the subset later
Date = '2018-01-01'
SubsetNumDonors = 10000
SubsetNumProj = 100
RndSeed = 404
MinDonations = 5
print('Date: ' + Date)
print('Subset number of donors: ' + str(SubsetNumDonors))
print('Subset number of projects live at chosen date: ' + str(SubsetNumProj))
print('Random seed value for sampling subset: ' + str(RndSeed))
print('Minimum number of donations for inclusion in training: ' + str(MinDonations))
# Process subject category
ProjCat = Projects[['Project ID', 'Project Subject Category Tree']]
ProjCat['Project Subject Category Tree'] = ProjCat['Project Subject Category Tree'].str.replace(', ', '--')
ProjCat['Project Subject Category Tree'] = ProjCat['Project Subject Category Tree'].str.replace('h--', 'h, ')
ProjCat.loc[:, 'Cat_Pri'],ProjCat.loc[:, 'Cat_Sec'] = ProjCat['Project Subject Category Tree'].str.split('--').str

#idx_contains_warmth = ProjCat['Project Subject Category Tree'].str.contains('Warmth', na=False)
#ProjCat[idx_contains_warmth].head()
ProjCat.head()
#Process project subcategory
ProjSubcat = Projects[['Project ID', 'Project Subject Subcategory Tree']]
ProjSubcat['Project Subject Subcategory Tree'] = ProjSubcat['Project Subject Subcategory Tree'].str.replace(', ', '--')
ProjSubcat['Project Subject Subcategory Tree'] = ProjSubcat['Project Subject Subcategory Tree'].str.replace('h--', 'h, ')
ProjSubcat.loc[:, 'Subcat_Pri'],ProjSubcat.loc[:, 'Subcat_Sec'] = ProjSubcat['Project Subject Subcategory Tree'].str.split('--').str

ProjSubcat.head()
#Replace the category and subcategory columns with the new split columns
Projects_adj = (Projects.merge(ProjCat.drop(columns=['Project Subject Category Tree']), on='Project ID', how='inner')
                .merge(ProjSubcat.drop(columns=['Project Subject Subcategory Tree']), on='Project ID', how='inner')
               .drop(columns=['Project Subject Category Tree'])
               .drop(columns=['Project Subject Subcategory Tree']))

Projects_adj.head()
#Filter projects by date
print('Date: ' + Date)

# Seed value for random subset selection = 404
Projects_t = Projects_adj[(Projects_adj['Project Fully Funded Date'] < Date) | (Projects_adj['Project Expiration Date'] < Date)]
Projects_t.head()
#Filter donations to those associated with the historic set Project ID
#TrainingProjectID = pd.DataFrame(Projects_t['Project ID'])
Donations_t = Donations[(Donations['Donation Received Date'] < Date)]
Donations_t.head()
print('Subset number of donors: ' + str(SubsetNumDonors))
print('Minimum number of donations for inclusion in training: ' + str(MinDonations))
print('Random seed value for sampling subset: ' + str(RndSeed))

#Count the number of donations per donor in the training set
#DonationsPerDonor = Donations_t[['Donor ID', 'Donation ID']].groupby(['Donor ID']).agg('count')
#DonationsPerDonor.head()

#Filter out donors with not enough donations in the data
ValidDonors = Donations_t.groupby(['Donor ID']).filter(lambda x: len(x) >= MinDonations)
#ValidDonors.head()

#Choose a random subset of donors who have made enough donations
Donors_t = (Donors.merge(pd.DataFrame(ValidDonors['Donor ID']), on='Donor ID', how='inner')
           .sample(n=SubsetNumDonors, random_state=RndSeed))
Donors_t.head()
print('Subset number of projects live at chosen date: ' + str(SubsetNumProj))
print('Random seed value for sampling subset: ' + str(RndSeed))

#Find live projects at reference date
LiveProjects = (Projects_adj[(Projects_adj['Project Posted Date'] <= Date) & 
                            ((Projects_adj['Project Fully Funded Date'] >= Date) | (Projects_adj['Project Expiration Date'] >= Date))]
               .sample(n=SubsetNumProj, random_state=RndSeed))
#LiveProjects[LiveProjects['Project Fully Funded Date'].isnull()].head()
LiveProjects.head()
#Cross join donors with live projects
Donors_t_tmp = Donors_t
Donors_t_tmp['Cross_join_key'] = 1

LiveProjects_tmp = LiveProjects
LiveProjects_tmp['Cross_join_key'] = 1

DonorVsLiveProj = (Donors_t_tmp.merge(LiveProjects_tmp, on='Cross_join_key', how='outer')
                   .drop(columns=['Cross_join_key'])
                  )

del Donors_t_tmp
del LiveProjects_tmp
gc.collect()
print(DonorVsLiveProj.shape)
#Create variables
#Join tables to get donor and school columns together
DonorVsDonation = Donors_t.merge(Donations_t, on='Donor ID', how='left')
DonorVsDonationVsProject = DonorVsDonation.merge(Projects_t, on='Project ID', how='left')
DonorVsDonationVsProjectVsSchool = DonorVsDonationVsProject.merge(Schools, on='School ID', how='left')

print('Donor vs donation vs project vs school table dimensions')
print(DonorVsDonationVsProjectVsSchool.shape)
#Historical donations count in the data
NumHistDonations = (DonorVsDonation[['Donor ID', 'Donation ID']]
                    .groupby(['Donor ID'])
                    .agg('count')
                    .reset_index()
                    .rename(columns={'Donation ID': 'Num_hist_donations'})
                   )
NumHistDonations.head()
#Most recent donation date in the data
LastDonations = (DonorVsDonation[['Donor ID', 'Donation Received Date']]
                     .groupby(['Donor ID'])
                     .agg('max')
                     .reset_index()
                     .rename(columns={'Donation Received Date': 'Last_donation_date'})
                    )

#Days since most recent donation
LastDonations['Days_since_last_donation'] = (datetime.strptime(Date, '%Y-%m-%d') - LastDonations['Last_donation_date']) / np.timedelta64(1, 'D')
LastDonations.head()
#Earliest donation date in the data
EarliestDonations = (DonorVsDonation[['Donor ID', 'Donation Received Date']]
                     .groupby(['Donor ID'])
                     .agg('min')
                     .reset_index()
                     .rename(columns={'Donation Received Date': 'Earliest_donation_date'})
                    )

#Average number of days between donations
DonationFreq = (NumHistDonations
                .merge(LastDonations, on='Donor ID', how='inner')
                .merge(EarliestDonations, on='Donor ID', how='inner')
               )

DonationFreq['Avg_days_between_donations'] = (((DonationFreq['Last_donation_date'] - DonationFreq['Earliest_donation_date']) / DonationFreq['Num_hist_donations'])
                                              / np.timedelta64(1, 'D')
                                             )
del EarliestDonations
gc.collect()
DonationFreq.head()
#Average number of donations per project
NumDonationsPerProj = (DonorVsDonation[['Donor ID', 'Project ID', 'Donation ID']]
                       .groupby(['Donor ID', 'Project ID'])
                       .agg('count')
                       .reset_index()
                       .rename(columns={'Donation ID': 'Num_donations_per_proj'})
                      )

AvgDonationsPerProj = (NumDonationsPerProj[['Donor ID', 'Num_donations_per_proj']]
                       .groupby(['Donor ID'])
                       .agg('mean')
                       .reset_index()
                       .rename(columns={'Num_donations_per_proj': 'Avg_donations_per_proj'})
                      )

del NumDonationsPerProj
gc.collect()
AvgDonationsPerProj.head()
# Donor interest in optional donations
NumOptional = (DonorVsDonation[['Donor ID', 'Donation Included Optional Donation']]
                    .replace({'Yes': 1, 'No': 0})
                    .groupby(['Donor ID'])
                    .agg('sum')
                    .reset_index()
                    .rename(columns={'Donation Included Optional Donation': 'Num_optional'})
                   )

InterestInOptional = NumHistDonations.merge(NumOptional, on='Donor ID', how='left')
InterestInOptional['Interest_in_optional'] = InterestInOptional['Num_optional'] / InterestInOptional['Num_hist_donations']

del NumOptional
gc.collect()
InterestInOptional.head()
#Interest in subject category
InterestInCat_tmp = pd.concat(
                    [DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Cat_Pri']].rename(columns={'Cat_Pri': 'Cat'}),
                     DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Cat_Sec']].rename(columns={'Cat_Sec': 'Cat'})]
                    , axis=0
                    , join='outer'
                    )

NumCat = (InterestInCat_tmp
          .dropna()
          .groupby(['Donor ID', 'Cat'])
          .agg('count')
          .reset_index()
          .rename(columns={'Donation ID': 'Num_cat'})
         )

InterestInCat = NumHistDonations.merge(NumCat, on='Donor ID', how='left')
InterestInCat['Interest_in_cat'] = InterestInCat['Num_cat'] / InterestInCat['Num_hist_donations']

del InterestInCat_tmp
del NumCat
gc.collect()
InterestInCat.head()
#Interest in subject subcategory
InterestInSubcat_tmp = pd.concat(
                    [DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Subcat_Pri']].rename(columns={'Subcat_Pri': 'Subcat'}),
                     DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Subcat_Sec']].rename(columns={'Subcat_Sec': 'Subcat'})]
                    , axis=0
                    , join='outer'
                    ).dropna()

NumSubcat = (InterestInSubcat_tmp
          .groupby(['Donor ID', 'Subcat'])
          .agg('count')
          .reset_index()
          .rename(columns={'Donation ID': 'Num_subcat'})
         )

InterestInSubcat = NumHistDonations.merge(NumSubcat, on='Donor ID', how='left')
InterestInSubcat['Interest_in_subcat'] = InterestInSubcat['Num_subcat'] / InterestInSubcat['Num_hist_donations']

del InterestInSubcat_tmp
del NumSubcat
gc.collect()
InterestInSubcat.head()
#Interest in resource category
InterestInRes_tmp = (DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Project Resource Category']]
                     .groupby(['Donor ID', 'Project Resource Category'])
                     .agg('count')
                     .reset_index()
                     .rename(columns={'Donation ID': 'Num_res'})
                    )

InterestInRes = NumHistDonations.merge(InterestInRes_tmp, on='Donor ID', how='left')
InterestInRes['Interest_in_res'] = InterestInRes['Num_res'] / InterestInRes['Num_hist_donations']

del InterestInRes_tmp
gc.collect()
InterestInRes.head()
#Interest in grade level category
InterestInGrade_tmp = (DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Project Grade Level Category']]
                     .groupby(['Donor ID', 'Project Grade Level Category'])
                     .agg('count')
                     .reset_index()
                     .rename(columns={'Donation ID': 'Num_grade'})
                    )

InterestInGrade = NumHistDonations.merge(InterestInGrade_tmp, on='Donor ID', how='left')
InterestInGrade['Interest_in_grade'] = InterestInGrade['Num_grade'] / InterestInGrade['Num_hist_donations']

del InterestInGrade_tmp
gc.collect()
InterestInGrade.head()
#Interest in project type
InterestInProjType_tmp = (DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Project Type']]
                          .groupby(['Donor ID', 'Project Type'])
                          .agg('count')
                          .reset_index()
                          .rename(columns={'Donation ID': 'Num_proj_type'})
                         )

InterestInProjType = NumHistDonations.merge(InterestInProjType_tmp, on='Donor ID', how='left')
InterestInProjType['Interest_in_proj_type'] = InterestInProjType['Num_proj_type'] / InterestInProjType['Num_hist_donations']

del InterestInProjType_tmp
gc.collect()
InterestInProjType.head()
#Interest in school metro type
InterestInMetro_tmp = (DonorVsDonationVsProjectVsSchool[['Donor ID', 'Donation ID', 'School Metro Type']]
                          .groupby(['Donor ID', 'School Metro Type'])
                          .agg('count')
                          .reset_index()
                          .rename(columns={'Donation ID': 'Num_metro'})
                         )

InterestInMetro = NumHistDonations.merge(InterestInMetro_tmp, on='Donor ID', how='left')
InterestInMetro['Interest_in_metro'] = InterestInMetro['Num_metro'] / InterestInMetro['Num_hist_donations']

del InterestInMetro_tmp
gc.collect()
InterestInMetro.head()
#Interest in month
InterestInMonth_tmp = DonorVsDonation[['Donor ID', 'Donation ID', 'Donation Received Date']]
InterestInMonth_tmp['Donation_month'] = InterestInMonth_tmp['Donation Received Date'].dt.month

InterestInMonth_tmp_2 = (InterestInMonth_tmp.drop(columns=['Donation Received Date'])
                         .groupby(['Donor ID', 'Donation_month'])
                         .agg('count')
                         .reset_index()
                         .rename(columns={'Donation ID': 'Num_donations_in_month'})
                        )

InterestInMonth = NumHistDonations.merge(InterestInMonth_tmp_2, on='Donor ID', how='left')
InterestInMonth['Interest_in_month'] = InterestInMonth['Num_donations_in_month'] / InterestInMonth['Num_hist_donations']

del InterestInMonth_tmp
del InterestInMonth_tmp_2
gc.collect()
InterestInMonth.head()
#Prediction model