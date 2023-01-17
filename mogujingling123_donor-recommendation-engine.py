import os
import gc
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import math
import nltk
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
#from tensorflow.python.data import Dataset
from IPython import display
from sklearn import metrics

print(os.listdir("../input"))
in_path = os.path.join(os.path.pardir, 'input')
in_files = glob.glob(os.path.join(in_path, '*.csv'))
print('Input files found:\n\t{}'.format('\n\t'.join(in_files)))

def describe_df(df, df_name):
    id_cols = df.columns[df.columns.str.contains('ID')]
    n_rows, n_cols = df.shape[0], df.shape[1]
    print('{:10} dataframe has {:8} rows, {:2} columns an ID keys: {}'\
          .format(df_name, str(n_rows), str(n_cols), ', '.join(id_cols)))

donations_df = pd.read_csv('../input/Donations.csv')
describe_df(donations_df, 'Donations')

donors_df = pd.read_csv('../input/Donors.csv')
describe_df(donors_df, 'Donors')

projects_df = pd.read_csv('../input/Projects.csv')
describe_df(projects_df, 'Projects')

resources_df = pd.read_csv('../input/Resources.csv')
describe_df(resources_df, 'Resources')


schools_df = pd.read_csv('../input/Schools.csv')
describe_df(schools_df, 'Schools')

teachers_df = pd.read_csv('../input/Teachers.csv')
describe_df(teachers_df, 'Teachers')

School_donors_df=donors_df.merge(donations_df,left_on="Donor ID",right_on='Donor ID')\
                 .merge(projects_df[['Project ID','School ID']],left_on='Project ID',right_on='Project ID')\
                 .merge(schools_df,left_on='School ID',right_on='School ID')
School_donors_df=donors_df.merge(donations_df,left_on="Donor ID",right_on='Donor ID')\
                 .merge(projects_df[['Project ID','School ID']],left_on='Project ID',right_on='Project ID')\
                 .merge(schools_df,left_on='School ID',right_on='School ID')
print(School_donors_df.shape)
print(School_donors_df['Donation ID'].nunique())
donation_rep=donations_df.groupby('Donation ID')['Donor ID'].count().reset_index()
repeated=donation_rep.loc[donation_rep['Donor ID'] >1,]
print(repeated.shape)
School_donors_df=School_donors_df.drop_duplicates(subset=['Donation ID','Donor ID'])
print(School_donors_df.shape)                                                 
donor_frequency = School_donors_df.groupby('Donor ID')['Project ID'].nunique()
frequent_donors = donor_frequency[donor_frequency > 1]
frequent_donors = frequent_donors.index.tolist()
School_donors_df['Target'] = School_donors_df['Donor ID'].isin(frequent_donors)
print(School_donors_df.groupby('Target')['Donor ID'].nunique())
print('The percentage of donors with repeated donation is {}% '.format (round(len(frequent_donors)/School_donors_df['Donor ID'].nunique(),2) *100))
print('The number of repeated donor is {}, the total number of donors is {}'.format (len(frequent_donors),School_donors_df['Donor ID'].nunique()))
Target_donors = donor_frequency[donor_frequency==1]
Once_donor=Target_donors.index.tolist()
repeated_donors_df=School_donors_df[School_donors_df['Donor ID'].isin(frequent_donors)]
Target_donors_df = School_donors_df[School_donors_df['Donor ID'].isin(Once_donor)]
print(round(np.mean(repeated_donors_df['Donor State']==repeated_donors_df['School State']),2))
print(repeated_donors_df['Donor ID'].nunique())
repeated_donors_df['seq']=repeated_donors_df.groupby('Donor ID')['Donor Cart Sequence'].rank()
part1=repeated_donors_df[repeated_donors_df['seq']==1]
part2=repeated_donors_df[repeated_donors_df['seq']==2]
repeated_donors_t=part1.merge(part2,on='Donor ID')
repeated_donors_t['Same School']=repeated_donors_t['School ID_x']==repeated_donors_t['School ID_y']
repeated_donors_t['Same City']=repeated_donors_t['School City_x']==repeated_donors_t['School City_y']
repeated_donors_t['Same County']=repeated_donors_t['School County_x']==repeated_donors_t['School County_y']
repeated_donors_t['Same District']=repeated_donors_t['School District_x']==repeated_donors_t['School District_y']
repeated_donors_t['Same State']=repeated_donors_t['School State_x']==repeated_donors_t['School State_y']
repeated_donors_t['Same Zip']=repeated_donors_t['School Zip_x']==repeated_donors_t['School Zip_y']
print('{}% of repeated donors donating the second donation as the same school with the first donation.'. format(round(repeated_donors_t['Same School'].mean(),3)*100))
print('{}% of repeated donors donating the second donation as the same city with the first donation.'. format(round(repeated_donors_t['Same City'].mean(),3)*100) ) 
print('{}% of repeated donors donating the second donation as the same county with the first donation.'. format(round(repeated_donors_t['Same County'].mean(),3)*100) ) 
print('{}% of repeated donors donating the second donation as the same District with the first donation.'. format(round(repeated_donors_t['Same District'].mean(),3)*100) ) 
print('{}% of repeated donors donating the second donation as the same state with the first donation.'. format(round(repeated_donors_t['Same State'].mean(),2)*100))
print('{}% of repeated donors donating the second donation as the same Zip code with the first donation.'. format(round(repeated_donors_t['Same Zip'].mean(),2)*100))
repeated_donors_t['Time Elapse']=(pd.to_datetime(repeated_donors_t['Donation Received Date_y'])-pd.to_datetime(repeated_donors_t['Donation Received Date_x'])).apply(lambda x: x.days)
print(round(repeated_donors_t['Time Elapse'].describe(),2))
print(repeated_donors_t.groupby('Same School')['Time Elapse'].describe())
print(repeated_donors_t.groupby('Same State')['Time Elapse'].describe())
print(repeated_donors_t.groupby('Same City')['Time Elapse'].describe())
print(repeated_donors_t.groupby('Same County')['Time Elapse'].describe())
round(np.mean(repeated_donors_t['Donation Included Optional Donation_x']=='Yes'),2)*100

print(round(np.mean(repeated_donors_t['Donation Amount_x']< repeated_donors_t['Donation Amount_y']),2)*100)
print(round(repeated_donors_t['Donation Amount_x'].mean(),2))
print(round(repeated_donors_t['Donation Amount_y'].mean()),2)
print(abs((repeated_donors_t['Donation Amount_x']- repeated_donors_t['Donation Amount_y'])/repeated_donors_t['Donation Amount_x']).describe())