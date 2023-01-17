import pandas as pd
# List files
!ls ../input
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
donations = pd.read_csv('../input/Donations.csv')
# Merge donation data with donor data 
df = donations.merge(donors, on='Donor ID', how='inner')
df.head()
donation_count = pd.DataFrame()
donation_count['counts'] = df.groupby('Donor ID')['Donation ID'].count()
donation_count.describe()
