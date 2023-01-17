import numpy as np 
import pandas as pd 

projects = pd.read_csv("../input/Projects.csv")
donations = pd.read_csv("../input/Donations.csv")
donors = pd.read_csv("../input/Donors.csv")
schools = pd.read_csv("../input/Schools.csv")
#Counting the number of donations for each donor
DonorsCount1 = pd.merge(donors,donations,how='inner',on='Donor ID').groupby('Donor ID')
DonorsCount2 = DonorsCount1.count()['Donor City'].reset_index()
DonorsCount2['NoDonations'] = DonorsCount2['Donor City']
#Merging donors table with DonorsCount2
#And adding a new column with the number of donation for each row
FinalDonors = pd.merge(donors,DonorsCount2,how = 'inner', on = 'Donor ID')
#To create a dataframe containing the corresponding project for each donation
DonationsProjects = pd.merge(donations,projects[['Project ID','School ID']],how ="left",on="Project ID")
#To create a dataframe containing the corresponding donor for each donation 
DonationsProjectsDonors = pd.merge(DonationsProjects,donors, how = "left", on = "Donor ID")
#To create a dataframe containing the corresponding school for each donation
DonationsProjectsDonorsSchools = pd.merge(DonationsProjectsDonors,schools, how = "left", on = "School ID")
#New column indicating if the donor city is the same as the school city
DonationsProjectsDonorsSchools["IsLocal"]=DonationsProjectsDonorsSchools.apply(lambda row: row["School City"] == row["Donor City"],axis=1)
#To add a new column with the number of donations for each donor
DonationsProjectsDonorsSchoolsNoDonations = pd.merge(DonationsProjectsDonorsSchools,FinalDonors[["Donor ID","NoDonations"]], how = "left", on = "Donor ID")
df = DonationsProjectsDonorsSchoolsNoDonations
#Donors with exact 2 donations
#This dataframe contains two records for each donor, first and second donation
df2 = pd.DataFrame(df[df['NoDonations']==2])[["Donor ID","Donation Received Date",'IsLocal']]
FirstDonation = pd.DataFrame(df2.groupby('Donor ID').min()).reset_index()
SecondDonation = pd.DataFrame(df2.groupby('Donor ID').max()).reset_index()
#First and second donation in the same row
FirstAndSecondDonation = pd.merge(FirstDonation,SecondDonation,how="inner",on='Donor ID')
FirstAndSecondDonation.head(10)
len(FirstDonation[FirstDonation['IsLocal']==True]) / len(FirstDonation)
len(SecondDonation[SecondDonation['IsLocal']==True]) / len(SecondDonation)
len(FirstAndSecondDonation[(FirstAndSecondDonation['IsLocal_x']==True) & (FirstAndSecondDonation['IsLocal_y']==False)])/len(FirstAndSecondDonation)
len(FirstAndSecondDonation[(FirstAndSecondDonation['IsLocal_x']==False) & (FirstAndSecondDonation['IsLocal_y']==True)])/len(FirstAndSecondDonation)