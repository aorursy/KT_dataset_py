#Table of results

#          | Baseline (no variables  | Best performance   | Improvement over
#          |   for past behaviour)   |                    |     baseline
#----------+-------------------------+--------------------+------------------
#Accuracy      0.6083333333333333      0.9541666666666667    57%

#Precision     0.6444444444444445      0.9753086419753086    51%

#Recall        0.5731225296442688      0.9367588932806324    63%

#Confusion    196  31                  221   6                13%  -81%
# matrix      55  198                  16  237               -71%   20%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # create graphs
from pandas.plotting import parallel_coordinates # for parallel coordinate plot
from datetime import datetime # convert string to date
import math # for extra math functions

from sklearn.model_selection import train_test_split # For splitting data into train and test sets for the prediction model
from sklearn.metrics import confusion_matrix # Confusion matrix for assessing model at the end
#from sklearn.ensemble import RandomForestClassifier # For creating a random forest classifier model
from xgboost import XGBClassifier # For XGBoost classifier model

import gc # for garbage collection to reduce RAM load

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Columns for donors
DonorCols = ['Donor ID', 'Donor City', 'Donor State', 'Donor Is Teacher']

# Columns for projects
ProjectCols = ['Project ID', 'School ID', 'Teacher ID', 'Teacher Project Posted Sequence', 'Project Type',
               'Project Subject Category Tree', 'Project Subject Subcategory Tree', 'Project Grade Level Category', 'Project Resource Category', 'Project Cost', 
               'Project Posted Date', 'Project Expiration Date', 'Project Fully Funded Date']

# Columns for schools
SchoolCols = ['School ID', 'School Metro Type', 'School Percentage Free Lunch', 'School State', 'School City', 'School Zip', 'School District', 'School County']

# Columns for resources
ResourceCols = ['Project ID', 'Resource Quantity', 'Resource Unit Price', 'Resource Vendor Name']

# Load data
Donations = pd.read_csv('../input/Donations.csv', parse_dates=['Donation Received Date'])
Donors = pd.read_csv('../input/Donors.csv', usecols=DonorCols, dtype={'Donor Zip': 'str'})
Projects = pd.read_csv('../input/Projects.csv', usecols=ProjectCols, parse_dates=['Project Posted Date', 'Project Expiration Date', 'Project Fully Funded Date'])
Resources = pd.read_csv('../input/Resources.csv', usecols=ResourceCols)
Schools = pd.read_csv('../input/Schools.csv', usecols=SchoolCols)
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
Date = '2018-01-01' #Default 2018-01-01
SubsetNumDonors = 1000 #Maximum subset size, default 1000
SubsetNumProj = 1000 #Maximum subset size, default 1000
RndSeed = 404 #Default 404
MinDonations = 5 #Default 5
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

del Projects
gc.collect()
Projects_adj.head()
#Filter projects by date
print('Date: ' + Date)

# Seed value for random subset selection = 404
Projects_t = Projects_adj[(Projects_adj['Project Fully Funded Date'] < Date) | (Projects_adj['Project Expiration Date'] < Date)]
Projects_t.head()
print('Subset number of projects live at chosen date: ' + str(SubsetNumProj))
print('Random seed value for sampling subset: ' + str(RndSeed))

#Find live projects at reference date
LiveProjects_tmp = (Projects_adj[(Projects_adj['Project Posted Date'] <= Date) & 
                                ((Projects_adj['Project Fully Funded Date'] >= Date) | (Projects_adj['Project Expiration Date'] >= Date))]
                   )

print("Total number of live projects: {}".format( LiveProjects_tmp.shape[0] ))

if LiveProjects_tmp.shape[0] > SubsetNumProj:
    LiveProjects = LiveProjects_tmp.sample(n=SubsetNumProj, random_state=RndSeed)
else:
    print("Not enough distinct projects to meet subset size, so using all projects")
    LiveProjects = LiveProjects_tmp

del LiveProjects_tmp
#del Projects_adj
gc.collect()
#LiveProjects[LiveProjects['Project Fully Funded Date'].isnull()].head()Projects_adj
print("Number of live projects selected: {}".format( LiveProjects.shape[0] ))
print("Number of distinct live projects selected: {}".format( LiveProjects.drop_duplicates().shape[0] ))
LiveProjects.head()
print('Subset number of donors: ' + str(SubsetNumDonors))
print('Minimum number of donations for inclusion in training: ' + str(MinDonations))
print('Random seed value for sampling subset: ' + str(RndSeed))

#Filter to keep donors who have donated to the live projects
DonorsWhoDonated_tmp = Donations.merge(LiveProjects[['Project ID']], on='Project ID', how='inner')
DonorsWhoDonated = DonorsWhoDonated_tmp[['Donor ID']].drop_duplicates()

print("Number of potential valid donors (who donatd to the selected live projects): {}".format( DonorsWhoDonated.shape[0] ))

#Filter donations to those associated with the historic set Project ID
#TrainingProjectID = pd.DataFrame(Projects_t['Project ID'])
Donations_t = Donations[(Donations['Donation Received Date'] < Date)]

#Count the number of donations per donor in the training set
#DonationsPerDonor = Donations_t[['Donor ID', 'Donation ID']].groupby(['Donor ID']).agg('count')
#DonationsPerDonor.head()

DonationsPerDonor = (Donations_t[['Donor ID', 'Donation ID']]
                     .groupby(['Donor ID'])
                     .agg('count')
                     .reset_index()
                     .rename(columns={'Donation ID': 'Num_donations'})
                    )

#Filter to keep donors with enough donations in the data and who have donated to the live projects
#This is to ensure the training set has donor-project pairs with donations
ValidDonors = (DonationsPerDonor[DonationsPerDonor['Num_donations'] > MinDonations]
               .merge(DonorsWhoDonated, on='Donor ID', how='inner')
              )
#ValidDonors.head()
print("Number of valid donors, after accounting for number of donations: {}".format( ValidDonors.shape[0] ))

#Choose a random subset of donors who have made enough donations
if ValidDonors.shape[0] > SubsetNumDonors:
    Donors_t = (Donors.merge(pd.DataFrame(ValidDonors['Donor ID']), on='Donor ID', how='inner')
                .drop_duplicates()
                .sample(n=SubsetNumDonors, random_state=RndSeed)
               )
else:
    print("Not enough distinct donors to meet subset size, so using all donors")
    Donors_t = (Donors.merge(pd.DataFrame(ValidDonors['Donor ID']), on='Donor ID', how='inner')
                .drop_duplicates()
               )
    
del Donors
del DonationsPerDonor
del DonorsWhoDonated_tmp
gc.collect()
print("Number of donors selected: {}".format( Donors_t.shape[0] ))
print("Number of distinct donors selected: {}".format( Donors_t.drop_duplicates().shape[0] ))
Donors_t.head()
#Get list of donors and all projects they donated towards
DonorsDonatedProjects_tmp = Donations.merge(Donors_t, on=['Donor ID'], how='inner')

DonorsDonatedProjects = DonorsDonatedProjects_tmp[['Donor ID', 'Project ID']].drop_duplicates()
DonorsDonatedProjects['Made_donation'] = 'Y'

del DonorsDonatedProjects_tmp
gc.collect()
DonorsDonatedProjects.head()
#Cross join donors with live projects
Donors_t_tmp = Donors_t
Donors_t_tmp['Cross_join_key'] = 1

LiveProjects_tmp = LiveProjects
LiveProjects_tmp['Cross_join_key'] = 1

DonorVsLiveProj_tmp = (Donors_t_tmp.merge(LiveProjects_tmp, on='Cross_join_key', how='outer')
                       .drop(columns=['Cross_join_key'])
                      )

DonorVsLiveProj = (DonorVsLiveProj_tmp
                   .merge(DonorsDonatedProjects, left_on=['Donor ID', 'Project ID'], 
                          right_on=['Donor ID', 'Project ID'], how='left')
                  )

DonorVsLiveProj['Donated_y_n'] = DonorVsLiveProj['Made_donation'].fillna('N')
DonorVsLiveProj.drop(columns=['Made_donation'], inplace=True)

del Donors_t_tmp
del LiveProjects_tmp
del DonorsDonatedProjects
del DonorVsLiveProj_tmp
gc.collect()
print("Size after cross-joining selected donors and selected projects: {}".format( DonorVsLiveProj.shape ))
print("Size after cross-joining and dropping duplicate records: {}".format( DonorVsLiveProj.drop_duplicates().shape ))
DonorVsLiveProj[DonorVsLiveProj['Donated_y_n'] == 'N'].head()
#Create variables
#Join tables to get donor and school columns together
DonorVsDonation = Donors_t.merge(Donations_t, on='Donor ID', how='left')
DonorVsDonationVsProject = DonorVsDonation.merge(Projects_t, on='Project ID', how='left')
DonorVsDonationVsProjectVsSchool = (DonorVsDonationVsProject.merge(Schools, on='School ID', how='left'))

del Donors_t
del Projects_t
print('Donor vs donation vs project vs school table dimensions')
print(DonorVsDonationVsProjectVsSchool.shape)
#Historical donations count in the data
NumHistDonations = (DonorVsDonation[['Donor ID', 'Donation ID']]
                    .groupby(['Donor ID'])
                    .agg('count')
                    .reset_index()
                    .rename(columns={'Donation ID': 'Num_hist_donations'})
                   )
print(NumHistDonations.shape)
NumHistDonations.head()
#Most recent donation date in the data
LastDonations_tmp = (DonorVsDonation[['Donor ID', 'Donation Received Date']]
                     .groupby(['Donor ID'])
                     .agg('max')
                     .reset_index()
                     .rename(columns={'Donation Received Date': 'Last_donation_date'})
                    )

#Days since most recent donation
LastDonations_tmp['Days_since_last_donation'] = ((datetime.strptime(Date, '%Y-%m-%d') - LastDonations_tmp['Last_donation_date']) / np.timedelta64(1, 'D')
                                                )

LastDonations = LastDonations_tmp[['Donor ID', 'Days_since_last_donation']]

print(LastDonations.shape)
LastDonations.head()
#Earliest donation date in the data
EarliestDonations = (DonorVsDonation[['Donor ID', 'Donation Received Date']]
                     .groupby(['Donor ID'])
                     .agg('min')
                     .reset_index()
                     .rename(columns={'Donation Received Date': 'Earliest_donation_date'})
                    )

#Average number of days between donations
# Use LastDonations_tmp because need Last_donation_date
DonationFreq_tmp = (NumHistDonations
                    .merge(LastDonations_tmp, on='Donor ID', how='inner')
                    .merge(EarliestDonations, on='Donor ID', how='inner')
                   )

DonationFreq_tmp['Avg_days_between_donations'] = (((DonationFreq_tmp['Last_donation_date'] - DonationFreq_tmp['Earliest_donation_date']) / DonationFreq_tmp['Num_hist_donations'])
                                                  / np.timedelta64(1, 'D')
                                                 )

DonationFreq = DonationFreq_tmp[['Donor ID', 'Avg_days_between_donations']]

del LastDonations_tmp
del EarliestDonations
del DonationFreq_tmp
gc.collect()
print(DonationFreq.shape)
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
print(AvgDonationsPerProj.shape)
AvgDonationsPerProj.head()
# Donor interest in optional donations
NumOptional = (DonorVsDonation[['Donor ID', 'Donation Included Optional Donation']]
                    .replace({'Yes': 1, 'No': 0})
                    .groupby(['Donor ID'])
                    .agg('sum')
                    .reset_index()
                    .rename(columns={'Donation Included Optional Donation': 'Num_optional'})
                   )

InterestInOptional_tmp = NumHistDonations.merge(NumOptional, on='Donor ID', how='left')
InterestInOptional_tmp['Interest_in_optional'] = InterestInOptional_tmp['Num_optional'] / InterestInOptional_tmp['Num_hist_donations']

InterestInOptional = InterestInOptional_tmp[['Donor ID', 'Interest_in_optional']]

del NumOptional
del InterestInOptional_tmp
gc.collect()
print(InterestInOptional.shape)
InterestInOptional.head()
#Interest in subject category
InterestInCat_tmp = pd.concat(
                    [DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Cat_Pri']].rename(columns={'Cat_Pri': 'Cat'}),
                     DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Cat_Sec']].rename(columns={'Cat_Sec': 'Cat'})]
                    , axis=0
                    , join='outer'
                    ).dropna()

NumCat = (InterestInCat_tmp
          .groupby(['Donor ID', 'Cat'])
          .agg('count')
          .reset_index()
          .rename(columns={'Donation ID': 'Num_cat'})
         )

InterestInCat_tmp2 = NumHistDonations.merge(NumCat, on='Donor ID', how='left')
InterestInCat_tmp2['Interest_in_cat'] = InterestInCat_tmp2['Num_cat'] / InterestInCat_tmp2['Num_hist_donations']

InterestInCat = InterestInCat_tmp2[['Donor ID', 'Cat', 'Interest_in_cat']]

del InterestInCat_tmp
del InterestInCat_tmp2
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

InterestInSubcat_tmp2 = NumHistDonations.merge(NumSubcat, on='Donor ID', how='left')
InterestInSubcat_tmp2['Interest_in_subcat'] = InterestInSubcat_tmp2['Num_subcat'] / InterestInSubcat_tmp2['Num_hist_donations']

InterestInSubcat = InterestInSubcat_tmp2[['Donor ID', 'Subcat', 'Interest_in_subcat']]

del InterestInSubcat_tmp
del InterestInSubcat_tmp2
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

InterestInRes_tmp2 = NumHistDonations.merge(InterestInRes_tmp, on='Donor ID', how='left')
InterestInRes_tmp2['Interest_in_res'] = InterestInRes_tmp2['Num_res'] / InterestInRes_tmp2['Num_hist_donations']

InterestInRes = InterestInRes_tmp2[['Donor ID', 'Project Resource Category', 'Interest_in_res']]

del InterestInRes_tmp
del InterestInRes_tmp2
gc.collect()
InterestInRes.head()
#Interest in grade level category
InterestInGrade_tmp = (DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Project Grade Level Category']]
                     .groupby(['Donor ID', 'Project Grade Level Category'])
                     .agg('count')
                     .reset_index()
                     .rename(columns={'Donation ID': 'Num_grade'})
                    )

InterestInGrade_tmp2 = NumHistDonations.merge(InterestInGrade_tmp, on='Donor ID', how='left')
InterestInGrade_tmp2['Interest_in_grade'] = InterestInGrade_tmp2['Num_grade'] / InterestInGrade_tmp2['Num_hist_donations']

InterestInGrade = InterestInGrade_tmp2[['Donor ID', 'Project Grade Level Category', 'Interest_in_grade']]

del InterestInGrade_tmp
del InterestInGrade_tmp2
gc.collect()
InterestInGrade.head()
#Interest in project type
InterestInProjType_tmp = (DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Project Type']]
                          .groupby(['Donor ID', 'Project Type'])
                          .agg('count')
                          .reset_index()
                          .rename(columns={'Donation ID': 'Num_proj_type'})
                         )

InterestInProjType_tmp2 = NumHistDonations.merge(InterestInProjType_tmp, on='Donor ID', how='left')
InterestInProjType_tmp2['Interest_in_proj_type'] = InterestInProjType_tmp2['Num_proj_type'] / InterestInProjType_tmp2['Num_hist_donations']

InterestInProjType = InterestInProjType_tmp2[['Donor ID', 'Project Type', 'Interest_in_proj_type']]

del InterestInProjType_tmp
del InterestInProjType_tmp2
gc.collect()
InterestInProjType.head()
#Interest in school metro type
InterestInMetro_tmp = (DonorVsDonationVsProjectVsSchool[['Donor ID', 'Donation ID', 'School Metro Type']]
                          .groupby(['Donor ID', 'School Metro Type'])
                          .agg('count')
                          .reset_index()
                          .rename(columns={'Donation ID': 'Num_metro'})
                         )

InterestInMetro_tmp2 = NumHistDonations.merge(InterestInMetro_tmp, on='Donor ID', how='left')
InterestInMetro_tmp2['Interest_in_metro'] = InterestInMetro_tmp2['Num_metro'] / InterestInMetro_tmp2['Num_hist_donations']

InterestInMetro = InterestInMetro_tmp2[['Donor ID', 'School Metro Type', 'Interest_in_metro']]

del InterestInMetro_tmp
del InterestInMetro_tmp2
gc.collect()
InterestInMetro.head()
#Interest in school county
InterestInCounty_tmp = (DonorVsDonationVsProjectVsSchool[['Donor ID', 'Donation ID', 'School County']]
                        .groupby(['Donor ID', 'School County'])
                        .agg('count')
                        .reset_index()
                        .rename(columns={'Donation ID': 'Num_county'})
                       )

InterestInCounty_tmp2 = NumHistDonations.merge(InterestInCounty_tmp, on='Donor ID', how='left')
InterestInCounty_tmp2['Interest_in_county'] = InterestInCounty_tmp2['Num_county'] / InterestInCounty_tmp2['Num_hist_donations']

InterestInCounty = InterestInCounty_tmp2[['Donor ID', 'School County', 'Interest_in_county']]

del InterestInCounty_tmp
del InterestInCounty_tmp2
gc.collect()
InterestInCounty.head()
#Interest in school district
InterestInDistrict_tmp = (DonorVsDonationVsProjectVsSchool[['Donor ID', 'Donation ID', 'School District']]
                          .groupby(['Donor ID', 'School District'])
                          .agg('count')
                          .reset_index()
                          .rename(columns={'Donation ID': 'Num_district'})
                         )

InterestInDistrict_tmp2 = NumHistDonations.merge(InterestInDistrict_tmp, on='Donor ID', how='left')
InterestInDistrict_tmp2['Interest_in_district'] = InterestInDistrict_tmp2['Num_district'] / InterestInDistrict_tmp2['Num_hist_donations']

InterestInDistrict = InterestInDistrict_tmp2[['Donor ID', 'School District', 'Interest_in_district']]

del InterestInDistrict_tmp
del InterestInDistrict_tmp2
gc.collect()
InterestInDistrict.head()
#Interest in school zip
InterestInZip_tmp = (DonorVsDonationVsProjectVsSchool[['Donor ID', 'Donation ID', 'School Zip']]
                     .groupby(['Donor ID', 'School Zip'])
                     .agg('count')
                     .reset_index()
                     .rename(columns={'Donation ID': 'Num_zip'})
                    )

InterestInZip_tmp2 = NumHistDonations.merge(InterestInZip_tmp, on='Donor ID', how='left')
InterestInZip_tmp2['Interest_in_zip'] = InterestInZip_tmp2['Num_zip'] / InterestInZip_tmp2['Num_hist_donations']

InterestInZip = InterestInZip_tmp2[['Donor ID', 'School Zip', 'Interest_in_zip']]

del InterestInZip_tmp
del InterestInZip_tmp2
gc.collect()
InterestInZip.head()
#Interest in month
InterestInMonth_tmp = DonorVsDonation[['Donor ID', 'Donation ID', 'Donation Received Date']]
InterestInMonth_tmp['Donation_month'] = InterestInMonth_tmp['Donation Received Date'].dt.month

InterestInMonth_tmp2 = (InterestInMonth_tmp.drop(columns=['Donation Received Date'])
                         .groupby(['Donor ID', 'Donation_month'])
                         .agg('count')
                         .reset_index()
                         .rename(columns={'Donation ID': 'Num_donations_in_month'})
                        )

InterestInMonth_tmp3 = NumHistDonations.merge(InterestInMonth_tmp2, on='Donor ID', how='left')
InterestInMonth_tmp3['Interest_in_month'] = InterestInMonth_tmp3['Num_donations_in_month'] / InterestInMonth_tmp3['Num_hist_donations']

InterestInMonth = InterestInMonth_tmp3[['Donor ID', 'Donation_month', 'Interest_in_month']]

del InterestInMonth_tmp
del InterestInMonth_tmp2
del InterestInMonth_tmp3
gc.collect()
InterestInMonth.head()
#To do to DonorVsLiveProj
#1. Attach columns from other tables
#2. Calculate project specific variables
#3. Attach pre-calculated variable values according to matching attributes

# Some clean up to release RAM
del DonorVsDonation
del DonorVsDonationVsProject
del DonorVsDonationVsProjectVsSchool
gc.collect()

# Calculate total resource cost of each project
ProjResCost_tmp = Resources[['Project ID', 'Resource Quantity', 'Resource Unit Price']]
ProjResCost_tmp['Resource_total_cost'] = ProjResCost_tmp['Resource Quantity'] * ProjResCost_tmp['Resource Unit Price']

ProjResCost = (ProjResCost_tmp[['Project ID', 'Resource_total_cost']]
               .groupby(['Project ID'])
               .agg('sum')
               .reset_index()
               .rename(columns={'Resource_total_cost': 'Project_res_total_cost'})
              )

del ProjResCost_tmp
del Resources
gc.collect()
print(ProjResCost.shape)
ProjResCost.head()
#Whole training set - initial
#Remove columns that will not be used for joining or for the prediction model
WholeTrainingSet_tmp = ((DonorVsLiveProj
                         .merge(Schools, on='School ID', how='left')
                         .merge(ProjResCost, on='Project ID', how='left')
                         .merge(Teachers[['Teacher ID', 'Teacher Prefix']].drop_duplicates(), on='Teacher ID', how='left')
                        )
                        .drop(columns=['Project Expiration Date', 'Project Fully Funded Date'])
                       )

del Schools
del ProjResCost
del Teachers
gc.collect()
print(WholeTrainingSet_tmp.shape)
WholeTrainingSet_tmp.head()
# Calculate other project specific values
WholeTrainingSet_tmp['Donor_vs_school_state'] = np.where(WholeTrainingSet_tmp['Donor State'] == WholeTrainingSet_tmp['School State'], 1, 0)
WholeTrainingSet_tmp['Donor_vs_school_city'] = np.where(
    np.logical_and(WholeTrainingSet_tmp['Donor_vs_school_state'] == 1, 
                   WholeTrainingSet_tmp['Donor City'] == WholeTrainingSet_tmp['School City'])
    , 1, 0)
WholeTrainingSet_tmp['Project_posted_month'] = pd.Series(WholeTrainingSet_tmp['Project Posted Date'].dt.month, index=WholeTrainingSet_tmp.index)
WholeTrainingSet_tmp['Project_posted_day_of_month'] = pd.Series(WholeTrainingSet_tmp['Project Posted Date'].dt.day, index=WholeTrainingSet_tmp.index)

print(WholeTrainingSet_tmp.shape)
WholeTrainingSet_tmp[WholeTrainingSet_tmp['Donor_vs_school_city'] == 1].head()
#Apply subject category interest to primary and secondary categories
TotalInterestInCat_tmp = (DonorVsLiveProj[['Donor ID', 'Project ID', 'Cat_Pri', 'Cat_Sec']]
                          .merge(InterestInCat, left_on=['Donor ID', 'Cat_Pri'], right_on=['Donor ID', 'Cat'], how='left')
                          .merge(InterestInCat, left_on=['Donor ID', 'Cat_Sec'], right_on=['Donor ID', 'Cat'], how='left')
                         )
TotalInterestInCat_tmp['Total_interest_in_cat'] = TotalInterestInCat_tmp['Interest_in_cat_x'].fillna(0) + TotalInterestInCat_tmp['Interest_in_cat_y'].fillna(0)
TotalInterestInCat = TotalInterestInCat_tmp[['Donor ID', 'Project ID', 'Total_interest_in_cat']]

TotalInterestInSubcat_tmp = (DonorVsLiveProj[['Donor ID', 'Project ID', 'Subcat_Pri', 'Subcat_Sec']]
                             .merge(InterestInSubcat, left_on=['Donor ID', 'Subcat_Pri'], right_on=['Donor ID', 'Subcat'], how='left')
                             .merge(InterestInSubcat, left_on=['Donor ID', 'Subcat_Sec'], right_on=['Donor ID', 'Subcat'], how='left')
                            )
TotalInterestInSubcat_tmp['Total_interest_in_subcat'] = TotalInterestInSubcat_tmp['Interest_in_subcat_x'].fillna(0) + TotalInterestInSubcat_tmp['Interest_in_subcat_y'].fillna(0)
TotalInterestInSubcat = TotalInterestInSubcat_tmp[['Donor ID', 'Project ID', 'Total_interest_in_subcat']]

print('TotalInterestInCat: {}'.format(TotalInterestInCat.shape))
print('TotalInterestInSubcat: {}'.format(TotalInterestInSubcat.shape))
TotalInterestInCat.head()
TotalInterestInSubcat.head()
# Merge values that are the same for all donor-project pairs
#Remove columns that are no longer needed to release RAM
WholeTrainingSet_tmp2 = ((WholeTrainingSet_tmp
                          .drop(columns=['School State', 'School City', 'Donor State', 'Donor City'])
                         )
                         .merge(NumHistDonations, on='Donor ID', how='left')
                         .merge(LastDonations, on='Donor ID', how='left')
                         .merge(DonationFreq, on='Donor ID', how='left')
                         .merge(AvgDonationsPerProj, on='Donor ID', how='left')
                         .merge(InterestInOptional, on='Donor ID', how='left')
                        )

del NumHistDonations
del LastDonations
del DonationFreq
del AvgDonationsPerProj
del InterestInOptional
del WholeTrainingSet_tmp
gc.collect()
print(WholeTrainingSet_tmp2.shape)
WholeTrainingSet_tmp2.head()
df_1 = WholeTrainingSet_tmp2.merge(TotalInterestInCat, left_on=['Donor ID', 'Project ID'], right_on=['Donor ID', 'Project ID'], how='left')

del WholeTrainingSet_tmp2
del TotalInterestInCat
gc.collect()

df_2 = df_1.merge(TotalInterestInSubcat, left_on=['Donor ID', 'Project ID'], right_on=['Donor ID', 'Project ID'], how='left')

del df_1
del TotalInterestInSubcat
gc.collect()

df_3 = df_2.merge(InterestInRes, left_on=['Donor ID', 'Project Resource Category'], right_on=['Donor ID', 'Project Resource Category'], how='left')

del df_2
del InterestInRes
gc.collect()

df_4 = df_3.merge(InterestInGrade, left_on=['Donor ID', 'Project Grade Level Category'], right_on=['Donor ID', 'Project Grade Level Category'], how='left')

del df_3
del InterestInGrade
gc.collect()

df_5 = df_4.merge(InterestInProjType, left_on=['Donor ID', 'Project Type'], right_on=['Donor ID', 'Project Type'], how='left')

del df_4
del InterestInProjType
gc.collect()

df_6 = df_5.merge(InterestInMetro, left_on=['Donor ID', 'School Metro Type'], right_on=['Donor ID', 'School Metro Type'], how='left')

del df_5
del InterestInMetro
gc.collect()

df_7 = df_6.merge(InterestInCounty, left_on=['Donor ID', 'School County'], right_on=['Donor ID', 'School County'], how='left')

del df_6
del InterestInCounty
gc.collect()

df_8 = df_7.merge(InterestInDistrict, left_on=['Donor ID', 'School District'], right_on=['Donor ID', 'School District'], how='left')

del df_7
del InterestInDistrict
gc.collect()

df_9 = df_8.merge(InterestInZip, left_on=['Donor ID', 'School Zip'], right_on=['Donor ID', 'School Zip'], how='left')

del df_8
del InterestInZip
gc.collect()

WholeTrainingSet = ((df_9.merge(InterestInMonth, left_on=['Donor ID', 'Project_posted_month'], right_on=['Donor ID', 'Donation_month'], how='left'))
                    .fillna({'Interest_in_optional': 0, 'Interest_in_res': 0, 'Interest_in_grade': 0, 'Interest_in_proj_type': 0, 'Interest_in_metro': 0,
                            'Interest_in_county': 0, 'Interest_in_district': 0, 'Interest_in_zip': 0})
                   )
del df_9
del InterestInMonth
gc.collect()
WholeTrainingSet.head()
#The final variable column
WholeTrainingSet['Total_project_interest'] = (WholeTrainingSet['Total_interest_in_cat'] + WholeTrainingSet['Total_interest_in_subcat'] 
                                              + WholeTrainingSet['Interest_in_res'] + WholeTrainingSet['Interest_in_grade']
                                              + WholeTrainingSet['Interest_in_proj_type'] + WholeTrainingSet['Interest_in_metro']
                                              + WholeTrainingSet['Interest_in_county'] + WholeTrainingSet['Interest_in_district']
                                              + WholeTrainingSet['Interest_in_zip']
                                             )

#Final column clean up
#Convert text type into number type
WholeTrainingSet['Donor Is Teacher'] = WholeTrainingSet['Donor Is Teacher'].map({'Yes': 1, 'No': 0})
WholeTrainingSet['Project Grade Level Category'] = (WholeTrainingSet['Project Grade Level Category']
                                                    .map({'Grades PreK-2': 1, 'Grades 3-5': 2, 'Grades 6-8': 3, 'Grades 9-12': 4})
                                                    .fillna(0)
                                                   )
WholeTrainingSet['Project Type'] = (WholeTrainingSet['Project Type']
                                    .map({'Student-Led': 1, 'Teacher-Led': 2, 'Professional Development': 3})
                                   )
WholeTrainingSet['School Metro Type'] = (WholeTrainingSet['School Metro Type']
                                         .map({'rural': 1, 'sururban': 2, 'urban': 3, 'town': 4})
                                         .fillna(0)
                                        )

#Drop columns that will not be used - basic
WholeTrainingSet.drop(columns=['Donor ID', 'Project ID', 'School ID', 'Teacher ID', 'Project Posted Date', 
                               'Project Resource Category', 'Cat_Pri', 'Cat_Sec', 'Subcat_Pri', 'Subcat_Sec',
                               'Teacher Prefix', 'School County', 'School District', 'School Zip'], inplace=True)

#Drop columns that will not be used - extra
# Option 1: Drop columns that are not dependent on both the donor and the project, or on the donor and ther past donations
# Option 2: Drop columns that are dependent on both the donor and the project, or on the donor and ther past donations; i.e. a baseline
DropExtra = False
DropBehaviour = False #Only proceeds if DropExtra is False
if DropExtra:
    WholeTrainingSet.drop(columns=['Teacher Project Posted Sequence', 'Project Type', 'Project Grade Level Category', 'Project Cost', 'School Metro Type', 
                                   'School Percentage Free Lunch', 'Project_res_total_cost', 'Project_posted_month', 'Project_posted_day_of_month', 'Donation_month',
                                   'Donor Is Teacher'],
                          inplace=True)
elif DropBehaviour:
    WholeTrainingSet.drop(columns=['Donor_vs_school_state', 'Donor_vs_school_city', 'Num_hist_donations', 'Days_since_last_donation', 'Avg_days_between_donations', 
                                   'Avg_donations_per_proj', 'Interest_in_optional', 'Total_interest_in_cat', 'Total_interest_in_subcat', 'Interest_in_res',
                                   'Interest_in_grade', 'Interest_in_proj_type', 'Interest_in_metro', 'Interest_in_month', 'Total_project_interest',
                                   'Interest_in_county', 'Interest_in_district', 'Interest_in_zip'],
                          inplace=True)
    
print(WholeTrainingSet.shape)
WholeTrainingSet.head()
print(WholeTrainingSet.dtypes)
#Control to drop 

#Count number of donor-project pairs with donations
SampleSize = WholeTrainingSet[WholeTrainingSet['Donated_y_n'] == 'Y'].shape[0]
print("Number of donor-project pairs with donations: {}".format( SampleSize ))

#Separate X by donor-project pairs with and without donations
PairsWithDonation = WholeTrainingSet[WholeTrainingSet['Donated_y_n'] == 'Y']
PairsWithoutDonation = WholeTrainingSet[WholeTrainingSet['Donated_y_n'] == 'N']

#Sample the donor-project pairs without donations
PairsWithoutDonation_sample = PairsWithoutDonation.sample(n=SampleSize, random_state=RndSeed)

#Union to create the final input data
FinalInput = pd.concat([PairsWithDonation, PairsWithoutDonation_sample], axis=0, join='outer')

del PairsWithDonation
del PairsWithoutDonation
del PairsWithoutDonation_sample
gc.collect()
print("Final input size: {}".format( FinalInput.shape ))
#Split data
TestSize = 0.2
y = pd.DataFrame(FinalInput['Donated_y_n'].map({'Y': 1, 'N': 0}))
X = FinalInput.drop(columns=['Donated_y_n'])

del FinalInput
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TestSize, random_state=RndSeed)

print("Fraction of training set that is donor-project pairs with donations: {}".format( y_train[y_train['Donated_y_n'] == 1].shape[0] / y_train.shape[0] ))
print("Fraction of   test   set that is donor-project pairs with donations: {}".format( y_test[y_test['Donated_y_n'] == 1].shape[0] / y_train.shape[0] ))
#Create general XGBoost model
model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)
#Default: max_depth=3, learning_rate=0.1, n_estimators=100
#from http://xgboost.readthedocs.io/en/latest/python/python_api.html

#Train model with data
model.fit(X_train, y_train)

#Make predictions with the trained model
y_pred = model.predict(X_test)

#Confusion matrix
ConfMat = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(ConfMat)

#Calculate more information
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

#Accuracy
print("Accuracy : {}".format( (tp + tn) / (tp + tn + fp + fn) ))
print("Precision: {}".format( tp / (tp + fp) ))
print("Recall   : {}".format( tp / (tp + fn) ))
#Baseline result, default inputs
#Reminder: baseline means running with these settings:
    #DropExtra = False
    #DropBehaviour = True
    # This means that donor past behaviour is not considered
#Confusion matrix: 
    #[[147 80] 
    # [108 145]] 
#Accuracy : 0.6083333333333333 
#Precision: 0.6444444444444445 
#Recall   : 0.5731225296442688    

#Past behaviour only, default inputs, before Dweikat's suggestion
    #DropExtra = True
    #DropBehaviour = False
#Confusion matrix: 
#    [[196 31] 
#     [ 55 198]] 
#Accuracy : 0.8208333333333333 -> 135% vs. baseline
#Precision: 0.8646288209606987 -> 134% vs. baseline
#Recall   : 0.782608695652174  -> 137% vs. baseline

#Past behaviour and baseline variables, default inputs, before Dweikat's suggestion
    #DropExtra = False
    #DropBehaviour = False
#Confusion matrix: 
    # [[196 31] 
    #  [ 50 203]] 
#Accuracy : 0.83125            -> 137% vs. baseline
#Precision: 0.8675213675213675 -> 135% vs. baseline
#Recall   : 0.8023715415019763 -> 140% vs. baseline

#Past behaviour and baseline variables, default inputs, after Dweikat's suggestion
    #DropExtra = False
    #DropBehaviour = False
#Confusion matrix: 
# [[221 6] 
# [ 16 237]] 
#Accuracy : 0.9541666666666667 -> 157%
#Precision: 0.9753086419753086 -> 151%
#Recall   : 0.9367588932806324 -> 163%
#Histograms and cumulative distribution graphs
NumBins = 10 #default 10

#Filter dataframe to get the desired columns
InterestVars = WholeTrainingSet[['Total_interest_in_cat', 'Total_interest_in_subcat', 'Interest_in_res', 'Interest_in_grade', 'Interest_in_proj_type',
                                'Interest_in_metro', 'Total_project_interest', 'Interest_in_county', 'Interest_in_district', 'Interest_in_zip',
                                'Donated_y_n']]

#Function to plot histogram and cumulative distribution
def HistAndCumuDist(df, ColName, NumBins):
    #Separate records with and without donations
    WithDonation = df[df['Donated_y_n'] == 'Y']
    NoDonation = df[df['Donated_y_n'] == 'N']
    
    #Format data for histogram
    values_WithD, base_WithD = np.histogram(WithDonation[ColName], bins=NumBins)
    values_NoD, base_NoD = np.histogram(NoDonation[ColName], bins=NumBins)
    
    #Format data for cumulative distribution, normalised
    CumuDist_WithD = np.cumsum(values_WithD) / WithDonation.shape[0]
    CumuDist_NoD = np.cumsum(values_NoD) / NoDonation.shape[0]
    
    #Plot the histogram
    fig, ax = plt.subplots()
    plt.hist(WithDonation[ColName], bins=base_WithD, alpha=0.3, label='Made donation', density=True)
    plt.hist(NoDonation[ColName], bins=base_NoD, alpha=0.3, label='No donation', density=True)
    plt.title('Donor-project pairs histogram with respect to ' + ColName)
    plt.ylabel('Density of pairs')
    plt.legend(loc='upper left')
    
    #Plot the cumulative function
    fig, ax = plt.subplots()
    plt.plot(base_WithD[:-1], CumuDist_WithD, label='Made donation')
    plt.plot(base_NoD[:-1], CumuDist_NoD, label='No donation', linestyle='dashed')
    plt.title('Cumulative distribution of donor-project pairs with respect to ' + ColName)
    plt.yticks((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    plt.ylabel('Fraction of pairs')
    plt.legend(loc='upper left')
    ax.grid()
    
HistAndCumuDist(InterestVars, 'Total_interest_in_cat', NumBins)
HistAndCumuDist(InterestVars, 'Total_interest_in_subcat', NumBins)
HistAndCumuDist(InterestVars, 'Interest_in_res', NumBins)
HistAndCumuDist(InterestVars, 'Interest_in_grade', NumBins)
HistAndCumuDist(InterestVars, 'Interest_in_proj_type', NumBins)
HistAndCumuDist(InterestVars, 'Interest_in_metro', NumBins)
HistAndCumuDist(InterestVars, 'Total_project_interest', NumBins)
HistAndCumuDist(InterestVars, 'Interest_in_county', NumBins)
HistAndCumuDist(InterestVars, 'Interest_in_district', NumBins)
HistAndCumuDist(InterestVars, 'Interest_in_zip', NumBins)
#Parallel coordinate graph
#Takes too long to run
#fig, ax = plt.subplots()
#parallel_coordinates(InterestVars, 'Donated_y_n')