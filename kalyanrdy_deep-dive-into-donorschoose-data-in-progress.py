import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, tqdm_pandas
import warnings
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
sns.set(color_codes=True)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

# Using error_bad_lines to skip the data row's which can cause error(in this case - ParserError: Error tokenizing data. C error).
#Look into https://stackoverflow.com/questions/18039057/python-pandas-error-tokenizing-data 
# Using warn_bad_lines=False to skip printing the warning that the row is skipped because of an error occured
print("Started Loading Data..")
print("Loading resources.csv ..")
resources_df = pd.read_csv('../input/Resources.csv',error_bad_lines=False,warn_bad_lines=False) 
print("Completed Loading resources.csv")
print("Loading Schools.csv ..")
schools_df = pd.read_csv('../input/Schools.csv',error_bad_lines=False,warn_bad_lines=False)
print("Completed Loading Schools.csv")
print("Loading Donors.csv ..")
donors_df = pd.read_csv('../input/Donors.csv',error_bad_lines=False,warn_bad_lines=False)
print("Completed Loading Donors.csv")
print("Loading Donations.csv ..")
donations_df = pd.read_csv('../input/Donations.csv',error_bad_lines=False,warn_bad_lines=False)
print("Completed Loading Donations.csv")
print("Loading Teachers.csv ..")
teachers_df = pd.read_csv('../input/Teachers.csv',error_bad_lines=False,warn_bad_lines=False)
print("Completed Loading Teachers.csv")
print("Loading Projects.csv ..")
projects_df = pd.read_csv('../input/Projects.csv',error_bad_lines=False,warn_bad_lines=False)
print("Completed Loading Projects.csv")

resources_df.head()
resources_df.describe()
schools_df.head()
schools_df.shape
donors_df.head()
donors_df.shape
donations_df.head()
donations_df.shape
teachers_df.head()
teachers_df.shape
projects_df.head()
projects_df.columns
resources_df.describe()

resources_df.fillna(0,inplace=True)
plt.figure(figsize=(15,10))
locs, labels = plt.xticks()

plt.setp(labels, rotation=90)
vendor_name_counts = resources_df["Resource Vendor Name"].value_counts()
sns.barplot(vendor_name_counts.index, vendor_name_counts.values)
trace = go.Pie(labels=vendor_name_counts.index, values=vendor_name_counts.values)

py.iplot([trace], filename='basic_pie_chart')
# # temp = resources_df['Resource Unit Price'].hist(bins=100)
# # temp.hist()
# plt.figure(figsize=(10,10))
# sns.set_style("whitegrid")
# ax = sns.boxplot(resources_df['Resource Unit Price'])
schools_df.head()

school_metro_type_counts = schools_df["School Metro Type"].value_counts()
trace = go.Pie(labels=school_metro_type_counts.index, values=school_metro_type_counts.values)
py.iplot([trace], filename='basic_pie_chart')
sns.set(rc={'figure.figsize':(20,10)})
schools_sub_df=schools_df[['School Metro Type','School Percentage Free Lunch']]
sns.boxplot(schools_sub_df['School Percentage Free Lunch'],schools_sub_df['School Metro Type'])
sns.violinplot(schools_sub_df['School Percentage Free Lunch'],schools_sub_df['School Metro Type'])
donors_df.head()
donor_is_teacher_counts = donors_df["Donor Is Teacher"].value_counts()
trace = go.Pie(labels=donor_is_teacher_counts.index, values=donor_is_teacher_counts.values)
py.iplot([trace], filename='donor_is_teacher_plt')
donors_df.dropna(inplace=True)
plt.figure(figsize=(15,10))
locs, labels = plt.xticks()

plt.setp(labels, rotation=90)
donor_state_counts = donors_df['Donor State'].value_counts().nlargest(30)
sns.barplot(donor_state_counts.index, donor_state_counts.values)
donors_df.dropna(inplace=True)
plt.figure(figsize=(20,10))
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
donor_city_counts = donors_df['Donor City'].value_counts().nlargest(30)
sns.barplot(donor_city_counts.index, donor_city_counts.values)
donors_df.groupby('Donor State')["Donor City"].value_counts().nlargest(5)
donations_df.head()
# # donations_df.plot.scatter("Donation Amount")
# donations_df.fillna(0,inplace=True)
# donation_bin = list(donations_df["Donation Amount"])
# plt.figure(figsize=(20,10))

# z = [0]*10
# for each in donation_bin:
#     z[int(each%10)]+=1
# sns.distplot(z)
# Conclusion :
# 1. Donation Amount is a Multi-Modal Distribution(read more at https://en.wikipedia.org/wiki/Multimodal_distribution)
temp = donations_df[["Donation Amount","Donation Received Date"]]
temp.set_index("Donation Received Date",inplace=True)
temp.plot()

donations_df['Donation Received Date']  = pd.to_datetime(donations_df["Donation Received Date"])
# df['col'] = pd.to_datetime(df['col'])

donations_df['year'] = donations_df['Donation Received Date'].dt.year
donations_df.head()
sns.countplot(x='year',data=donations_df)
sum_prices = donations_df.groupby("year")['Donation Amount'].sum()
# sns.histplot(sum_prices)
trace = go.Pie(labels=sum_prices.index, values=sum_prices.values)

py.iplot([trace], filename='basic_pie_chart')
