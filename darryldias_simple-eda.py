# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import datetime
import matplotlib.pyplot as plt

intTimeCheck = 0
datStart = datetime.datetime.now()
strStage = ""

datProgramStart = datetime.datetime.now()
dfTimeCheck = pd.DataFrame(columns=['Stage','Start','End', 'Seconds', 'Minutes'])

def start_time_check(str_stage_i):
    # created by darryldias 21may2018
    global intTimeCheck
    global strStage 
    global datStart 
    intTimeCheck += 1
    strStage = str_stage_i
    datStart = datetime.datetime.now()
    
def end_time_check():
    # created by darryldias 21may2018
    global intTimeCheck
    global strStage
    global datStart
    global dfTimeCheck
    datEnd = datetime.datetime.now()
    diffSeconds = (datEnd-datStart).total_seconds()
    diffMinutes = diffSeconds / 60.0
    dfTimeCheck.loc[intTimeCheck] = [strStage, datStart, datEnd, diffSeconds, diffMinutes]

def create_topline(df_input, str_item_column, str_count_column):
    # created by darryldias 21may2018; updated by darryldias 23may2018
    df_temp = df_input.groupby(str_item_column).size().reset_index(name=str_count_column)
    df_output = pd.DataFrame(columns=[str_item_column, str_count_column, 'Percent'])
    int_rows = df_temp.shape[0]
    int_columns = df_temp.shape[1]
    int_total = df_temp[str_count_column].sum()
    flt_total = float(int_total)
    for i in range(int_rows):
        str_item = df_temp.iloc[i][0]
        int_count = df_temp.iloc[i][1]
        flt_percent = round(int_count / flt_total * 100, 1)
        df_output.loc[i] = [str_item, int_count, flt_percent]
    
    df_output.loc[int_rows] = ['Total', int_total, 100.0]
    return df_output        

def get_dataframe_info(df_input):
    # created by darryldias 24may2018
    int_rows = df_input.shape[0]
    int_cols = df_input.shape[1]
    flt_rows = float(int_rows)
    
    df_output = pd.DataFrame(columns=["Column", "Type", "Not Null", 'Null', '% Not Null', '% Null'])
    df_output.loc[0] = ['Table Row Count', '', int_rows, '', '', '']
    df_output.loc[1] = ['Table Column Count', '', int_cols, '', '', '']
    int_table_row = 1
    for i in range(int_cols):
        str_column_name = df_input.columns.values[i]
        str_column_type = df_input.dtypes.values[i]
        int_not_null = df_input[str_column_name].count()
        int_null = sum( pd.isnull(df_input[str_column_name]) )
        flt_percent_not_null = round(int_not_null / flt_rows * 100, 1)
        flt_percent_null = 100 - flt_percent_not_null
        int_table_row += 1
        df_output.loc[int_table_row] = [str_column_name, str_column_type, int_not_null, int_null, flt_percent_not_null, flt_percent_null]
    
    return df_output

# Used some code from the following people's kernels (with thanks)
# https://www.kaggle.com/frkngnr 
# data4["Project Posted Date"] = pd.to_datetime(data4["Project Posted Date"])
start_time_check("Schools")
schools = pd.read_csv('../input/Schools.csv')
schools.head(10)

get_dataframe_info(schools)
create_topline(schools, 'School Metro Type', 'Count')
create_topline(schools, 'School State', 'Count')
end_time_check()
start_time_check("Teachers")
teachers = pd.read_csv('../input/Teachers.csv')
teachers.head(10)

get_dataframe_info(teachers)
create_topline(teachers, 'Teacher Prefix', 'Count')
end_time_check()
start_time_check("Donors")
donors = pd.read_csv('../input/Donors.csv')
donors.head(10)

get_dataframe_info(donors)
create_topline(donors, 'Donor State', 'Count')
create_topline(donors, 'Donor Is Teacher', 'Count')
end_time_check()
start_time_check("Donations")
donations = pd.read_csv('../input/Donations.csv')
donations.head(10)

get_dataframe_info(donations)
donations.describe()
create_topline(donations, 'Donation Included Optional Donation', 'Count')
def donation_amount_summary1 (row):
   if row['Donation Amount'] <= 10 :
      return '000.00 to 010.00'
   if row['Donation Amount'] <= 20 :
      return '010.01 to 020.00'
   if row['Donation Amount'] <= 30 :
      return '020.01 to 030.00'
   if row['Donation Amount'] <= 50 :
      return '030.01 to 050.00'
   if row['Donation Amount'] <= 100 :
      return '050.01 to 100.00'
   if row['Donation Amount'] <= 200 :
      return '100.01 to 200.00'
   if row['Donation Amount'] <= 500 :
      return '200.01 to 500.00'
   if row['Donation Amount'] > 500 :
      return '500.01+'
   return 'Other'

donations['Donation Amount Summary 1'] = donations.apply(donation_amount_summary1, axis=1)
create_topline(donations, 'Donation Amount Summary 1', 'Count')
donations["Donation Received Date"] = pd.to_datetime(donations["Donation Received Date"])
donations["Donation Received Date"].dt.year.describe()
donations["Donation Received Year"] = donations["Donation Received Date"].dt.year
create_topline(donations, 'Donation Received Year', 'Count')
end_time_check()
start_time_check("Donors Donation Merge")

grouped = donations.groupby('Donor ID')
dfGrouped = grouped['Donation Amount'].count().reset_index(name='Donation Amount Count')  
donors_merged = pd.merge(donors, dfGrouped, how='left', on=['Donor ID'])
dfGrouped = grouped['Donation Amount'].sum().reset_index(name='Donation Amount Sum')   
donors_merged = pd.merge(donors_merged, dfGrouped, how='left', on=['Donor ID'])
dfGrouped = grouped['Donation Amount'].mean().reset_index(name='Donation Amount Mean')   
donors_merged = pd.merge(donors_merged, dfGrouped, how='left', on=['Donor ID'])

dfGrouped = grouped['Donation Received Year'].min().reset_index(name='Donation Received Year Start')   
donors_merged = pd.merge(donors_merged, dfGrouped, how='left', on=['Donor ID'])
dfGrouped = grouped['Donation Received Year'].max().reset_index(name='Donation Received Year End')   
donors_merged = pd.merge(donors_merged, dfGrouped, how='left', on=['Donor ID'])

dfGrouped = grouped['Donation Received Date'].min().reset_index(name='Donation Received Date Start')   
donors_merged = pd.merge(donors_merged, dfGrouped, how='left', on=['Donor ID'])
dfGrouped = grouped['Donation Received Date'].max().reset_index(name='Donation Received Date End')   
donors_merged = pd.merge(donors_merged, dfGrouped, how='left', on=['Donor ID'])

datDonationNow = donors_merged['Donation Received Date End'].max() + pd.Timedelta('1 days')
donors_merged['Years Since Last Donation']=(datDonationNow-donors_merged['Donation Received Date End']) / pd.Timedelta(days=365)

def donation_count_summary1 (row):
   if row['Donation Amount Count'] == 1 :
      return 'Single Donor'
   if row['Donation Amount Count'] > 1 :
      return 'Multiple Donor'
   return 'Unknown'

donors_merged['Single Multiple Donor'] = donors_merged.apply(donation_count_summary1, axis=1)
create_topline(donors_merged, 'Single Multiple Donor', 'Count')
donors_merged = donors_merged[donors_merged['Single Multiple Donor'] != 'Unknown']
create_topline(donors_merged, 'Single Multiple Donor', 'Count')
def current_lapsed_donor (row):
   if row['Years Since Last Donation'] <= 1 :
      return 'Current Donor'
   if row['Years Since Last Donation'] > 1 :
      return 'Lapsed Donor'
   return 'Unknown'

donors_merged['Current Lapsed Donor'] = donors_merged.apply(current_lapsed_donor, axis=1)
create_topline(donors_merged, 'Current Lapsed Donor', 'Count')
def donation_total_summary (row):
   if row['Donation Amount Sum'] <= 50 :
      return 'Donated $50 or less'
   if row['Donation Amount Sum'] > 50 :
      return 'Donated over $50'
   return 'Unknown'

donors_merged['Donation Total Summary'] = donors_merged.apply(donation_total_summary, axis=1)
create_topline(donors_merged, 'Donation Total Summary', 'Count')
def donor_segment_1 (row):
   if row['Single Multiple Donor']=='Single Donor' and row['Current Lapsed Donor']=='Current Donor' and row['Donation Total Summary']=='Donated $50 or less':
      return 'Single Current $50 Less'
   if row['Single Multiple Donor']=='Single Donor' and row['Current Lapsed Donor']=='Current Donor' and row['Donation Total Summary']=='Donated over $50':
      return 'Single Current Over $50'
   if row['Single Multiple Donor']=='Single Donor' and row['Current Lapsed Donor']=='Lapsed Donor' and row['Donation Total Summary']=='Donated $50 or less':
      return 'Single Lapsed $50 Less'
   if row['Single Multiple Donor']=='Single Donor' and row['Current Lapsed Donor']=='Lapsed Donor' and row['Donation Total Summary']=='Donated over $50':
      return 'Single Lapsed Over $50'
   if row['Single Multiple Donor']=='Multiple Donor' and row['Current Lapsed Donor']=='Current Donor' and row['Donation Total Summary']=='Donated $50 or less':
      return 'Multiple Current $50 Less'
   if row['Single Multiple Donor']=='Multiple Donor' and row['Current Lapsed Donor']=='Current Donor' and row['Donation Total Summary']=='Donated over $50':
      return 'Multiple Current Over $50'
   if row['Single Multiple Donor']=='Multiple Donor' and row['Current Lapsed Donor']=='Lapsed Donor' and row['Donation Total Summary']=='Donated $50 or less':
      return 'Multiple Lapsed $50 Less'
   if row['Single Multiple Donor']=='Multiple Donor' and row['Current Lapsed Donor']=='Lapsed Donor' and row['Donation Total Summary']=='Donated over $50':
      return 'Multiple Lapsed Over $50'
   return 'Unknown'

donors_merged['Donor Segment 1'] = donors_merged.apply(donor_segment_1, axis=1)
create_topline(donors_merged, 'Donor Segment 1', 'Count')
dfGrouped = grouped['Project ID'].nunique().reset_index(name='Unique Projects')  
donors_merged = pd.merge(donors_merged, dfGrouped, how='left', on=['Donor ID'])

def unique_projects_summary (row):
   if row['Unique Projects'] == 1 :
      return 'Single Project'
   if row['Unique Projects'] > 1 :
      return 'Multiple Projects'
   return 'Unknown'

donors_merged['Unique Projects Summary'] = donors_merged.apply(unique_projects_summary, axis=1)
create_topline(donors_merged, 'Unique Projects Summary', 'Count')
donors_merged.info()

donors_merged.isnull().sum()
donors_merged.describe()
donors_merged.head(10)
donors_merged['Single Multiple Donor'].value_counts(sort=False).plot.pie(autopct='%1.1f%%')
plt.show()
donors_merged['Current Lapsed Donor'].value_counts(sort=False).plot.pie(autopct='%1.1f%%')
plt.show()
donors_merged['Donation Total Summary'].value_counts(sort=False).plot.pie(autopct='%1.1f%%')
plt.show()
donors_merged['Unique Projects Summary'].value_counts(sort=False).plot.pie(autopct='%1.1f%%')
plt.show()
dfChart = donors_merged['Donor Segment 1'].value_counts(sort=True, ascending=True)
ax = dfChart.plot.barh(title='Donor Segment 1', figsize=(10,10), colormap='summer')
plt.show()

end_time_check()
start_time_check("Projects")
projects = pd.read_csv('../input/Projects.csv')
projects.head(10)

get_dataframe_info(projects)
projects.describe()
create_topline(projects, 'Project Type', 'Count')
create_topline(projects, 'Project Subject Category Tree', 'Count')
create_topline(projects, 'Project Subject Subcategory Tree', 'Count')
create_topline(projects, 'Project Grade Level Category', 'Count')
create_topline(projects, 'Project Resource Category', 'Count')
create_topline(projects, 'Project Current Status', 'Count')
def project_cost_summary1 (row):
   if row['Project Cost'] <= 250 :
      return '0000.00 to 0250.00'
   if row['Project Cost'] <= 500 :
      return '0250.01 to 0500.00'
   if row['Project Cost'] <= 750 :
      return '0500.01 to 0750.00'
   if row['Project Cost'] <= 1000 :
      return '0750.01 to 1000.00'
   if row['Project Cost'] > 1000 :
      return '1000.01+'
   return 'Other'

projects['Project Cost Summary 1'] = projects.apply(project_cost_summary1, axis=1)
create_topline(projects, 'Project Cost Summary 1', 'Count')
end_time_check()
start_time_check("Donors Projects Merge")
grouped = donations.groupby(['Donor ID', 'Project ID'])
dfGrouped = grouped.size().reset_index(name='Size')  
df_temp = donors_merged.loc[:,['Donor ID', 'Donation Amount Count']]
dfGrouped = pd.merge(dfGrouped, df_temp, how='left', on=['Donor ID'])

#dfGrouped[ dfGrouped['Donation Amount Count'].isnull() ]
dfGrouped = dfGrouped[ dfGrouped['Donation Amount Count'] >= 1.0 ]
dfGrouped['Donation Amount Count'] = dfGrouped['Donation Amount Count'].astype(int)

df_temp = projects.loc[:,['Project ID', 'Project Subject Category Tree', 'Project Grade Level Category', 'Project Resource Category',  \
                          'Project Cost', 'Project Cost Summary 1', 'Project Current Status']] 
dfGrouped = pd.merge(dfGrouped, df_temp, how='left', on=['Project ID'])

dfGrouped = dfGrouped[ dfGrouped['Size'] == dfGrouped['Donation Amount Count'] ]
dfGrouped.drop(['Size', 'Donation Amount Count'], axis=1, inplace=True)
donors_merged = pd.merge(donors_merged, dfGrouped, how='left', on=['Donor ID'])
donors_merged.head(15)
create_topline(donors_merged, 'Project Subject Category Tree', 'Count')
create_topline(donors_merged, 'Project Grade Level Category', 'Count')
create_topline(donors_merged, 'Project Resource Category', 'Count')
create_topline(donors_merged, 'Project Cost Summary 1', 'Count')
create_topline(donors_merged, 'Project Current Status', 'Count')
end_time_check()
start_time_check("Resources")
resources = pd.read_csv('../input/Resources.csv')
resources.head(10)

get_dataframe_info(resources)
create_topline(resources, 'Resource Vendor Name', 'Count')
end_time_check()
datProgramEnd = datetime.datetime.now()
diffSeconds = (datProgramEnd-datProgramStart).total_seconds()
diffMinutes = diffSeconds / 60.0
dfTimeCheck.loc[intTimeCheck + 1] = ["Overall", datProgramStart, datProgramEnd, diffSeconds, diffMinutes]
dfTimeCheck