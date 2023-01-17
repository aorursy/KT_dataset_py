# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns #for Data visualisation

import matplotlib.pyplot as plt #for Data visualisation

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
xls = pd.ExcelFile('/kaggle/input/hospitalman/Hospital Management (1).xlsx')



# Now you can list all sheets in the file

xls.sheet_names

# ['house', 'house_extra', ...]



df1 = pd.read_excel(xls, 'Bed Management')

df2 = pd.read_excel(xls, 'Staff Management')



df3 = pd.read_excel(xls, 'Patient Information')

df4 = pd.read_excel(xls, 'Equipment Information')



df5 = pd.read_excel(xls, 'Population')

df1

#df2 = df2.rename({'Facility Name': 'Facility Name1'}, axis=1)  # new method
df2

df3



df4





df5
#concatinating datasets into one dataset

#df=pd.concat((df1,df2,df3,df4,df5),axis=1)
#df.head()
#df.isnull().sum()
#df.shape
data_frames = [df1,df2]
from functools import reduce

df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Facility Name'],

                                            how='inner'), data_frames)
df_merged.head()
df_merged.shape
df_merged.isnull().sum()
#result = pd.merge(df1, df_merged, on='Facility Name')

result = pd.merge(df3,df4, how='inner', on=['Facility Name'])
result.head()
result.isnull().sum()
#result = pd.merge(df1, df_merged, on='Facility Name')

df= pd.merge(df_merged,result, how='inner', on=['Facility Name'])
df.shape
df.isnull().sum()
# Remove two columns name is 'C' and 'D' 

df=df.drop(['Location_y_y', 'Province_y_y','Facility Type_y_y','Location_x_y','Province_x_y','Facility Type_x_y','Location_y_x','Province_y_x','Facility Type_y_x'], axis = 1)
df.shape
df.head()
df.T
df.shape
df.isnull().sum()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(df['Sum of Total Beds']);

plt.subplot(122) 

df['Sum of Total Beds'].plot.box(figsize=(16,5))

plt.show()
df['Sum of Total Beds'].describe()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(df['Patients']);

plt.subplot(122) 

df['Patients'].plot.box(figsize=(16,5))

plt.show()
df['Patients'].describe()
df['Patients'].mean()
df['Nurses assigned'].describe()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(df['Nurses assigned']);

plt.subplot(122) 

df['Nurses assigned'].plot.box(figsize=(16,5))

plt.show()
df['ICU Nurses'].describe()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(df['ICU Nurses']);

plt.subplot(122) 

df['ICU Nurses'].plot.box(figsize=(16,5))

plt.show()
df['Nurses to patients'].describe()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(df['Nurses to patients']);

plt.subplot(122) 

df['Nurses to patients'].plot.box(figsize=(16,5))

plt.show()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(df['COVID Patients']);

plt.subplot(122) 

df['COVID Patients'].plot.box(figsize=(16,5))

plt.show()
df['COVID Patients'].median()
df['COVID Patients'].describe()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(df['COVID Test pending']);

plt.subplot(122) 

df['COVID Test pending'].plot.box(figsize=(16,5))

plt.show()
df['COVID Test pending'].describe()
#indepent Variable(Numerical)

plt.figure(3) 

plt.subplot(121) 

sns.distplot(df['COVID POSITIVE']);

plt.subplot(122) 

df['COVID POSITIVE'].plot.box(figsize=(16,5))

plt.show()
df['COVID POSITIVE'].describe()
df=pd.concat([df,df5], axis=1)




df['Rank']= df['Rank'].replace(np.nan, '', regex=True)

df['2011 census']= df['2011 census'].replace(np.nan, '', regex=True)

df['Unnamed: 3']= df['Unnamed: 3'].replace(np.nan, '', regex=True)

df['2015 mid-year estimate']= df['2015 mid-year estimate'].replace(np.nan, '', regex=True)

df['2019 mid-year estimate']= df['2019 mid-year estimate'].replace(np.nan, '', regex=True)

df['2019 mid-year estimate']= df['2019 mid-year estimate'].replace(np.nan, '', regex=True)

df['Unnamed: 5']= df['Unnamed: 5'].replace(np.nan, '', regex=True)

df['Unnamed: 7']= df['Unnamed: 7'].replace(np.nan, '', regex=True)

 
#df['Admission Date1'] = pd.to_datetime(df['Admission Date'], format='%Y-%m-%d').apply(lambda dt: dt.replace(day=1)).dt.date
#df['Admission Date1']= df['Admission Date1'].replace(np.nan, 'Unknown', regex=True)
df.T
df['Province']= df['Province'].replace(np.nan, '', regex=True)
df.T
df.isnull().sum()
df=df.round()
df
#df=df[:382]
df=df[382:547]
output = pd.DataFrame({'Facility Name':df['Facility Name'],'Facility Type_x_x':df['Facility Type_x_x'],'Province_x_x':df['Province_x_x'],'Location_x_x':df['Location_x_x'],'Sum of Total Beds':df['Sum of Total Beds'],'Beds in Use':df['Beds in Use'],'Available Beds':df['Available Beds'],'Occupancy%':df['Occupancy%'],'Illness Type':df['Illness Type'],'Patients':df['Patients'],'Nurses assigned':df['Nurses assigned'],'COVID Patients':df['COVID Patients'],'COVID Test pending':df['COVID Test pending'],'COVID POSITIVE':df['COVID POSITIVE'],'Admission Date':df['Admission Date'],'Ventilators':df['Ventilators'],'Ventilators in Use':df['Ventilators in Use'],'Available Ventilators':df['Available Ventilators'],'Rank':df['Rank'],'Province':df['Province'],'2011 census':df['2011 census'],'Unnamed: 3':df['Unnamed: 3'],'2015 mid-year estimate':df['2015 mid-year estimate'],'Unnamed: 5':df['Unnamed: 5'],'2019 mid-year estimate':df['2019 mid-year estimate'],'Unnamed: 7':df['Unnamed: 7']})

output.to_csv('test.csv', index=False)

print("successfully saved!")
#output1 = pd.DataFrame({'Facility Name':train['Facility Name'],'Facility Type_x_x':train['Facility Type_x_x'],'Province_x_x':train['Province_x_x'],'Location_x_x':train['Location_x_x'],'Sum of Total Beds':train['Sum of Total Beds'],'Beds in Use':train['Beds in Use'],'Available Beds':train['Available Beds'],'Occupancy%':train['Occupancy%'],'Illness Type':train['Illness Type'],'Patients':train['Patients'],'Nurses assigned':train['Nurses assigned'],'COVID Patients':train['COVID Patients'],'COVID Test pending':train['COVID Test pending'],'COVID POSITIVE':train['COVID POSITIVE'],'Admission Date':train['Admission Date'],'Ventilators':train['Ventilators'],'Ventilators in Use':train['Ventilators in Use'],'Available Ventilators':train['Available Ventilators'],'Rank':train['Rank'],'Province':train['Province'],'2011 census':train['2011 census'],'Unnamed: 3':train['Unnamed: 3'],'2015 mid-year estimate':train['2015 mid-year estimate'],'Unnamed: 5':train['Unnamed: 5'],'2019 mid-year estimate':train['2019 mid-year estimate'],'Unnamed: 7':train['Unnamed: 7']})

#output1.to_csv('train.csv', index=False)

#print("successfully saved!")
#output = pd.DataFrame({'Facility Name':df['Facility Name'],'Facility Type_x_x':df['Facility Type_x_x'],'Province_x_x':df['Province_x_x'],'Location_x_x':df['Location_x_x'],'Sum of Total Beds':df['Sum of Total Beds'],'Beds in Use':df['Beds in Use'],'Available Beds':df['Available Beds'],'Occupancy%':df['Occupancy%'],'Illness Type':df['Illness Type'],'Patients':df['Patients'],'Nurses assigned':df['Nurses assigned'],'COVID Patients':df['COVID Patients'],'COVID Test pending':df['COVID Test pending'],'COVID POSITIVE':df['COVID POSITIVE'],'Admission Date':df['Admission Date'],'Ventilators':df['Ventilators'],'Ventilators in Use':df['Ventilators in Use'],'Available Ventilators':df['Available Ventilators'],'Rank':df['Rank'],'Province':df['Province'],'2011 census':df['2011 census'],'Unnamed: 3':df['Unnamed: 3'],'2015 mid-year estimate':df['2015 mid-year estimate'],'Unnamed: 5':df['Unnamed: 5'],'2019 mid-year estimate':df['2019 mid-year estimate'],'Unnamed: 7':df['Unnamed: 7']})

#output.to_csv('test.csv', index=False)

#print("successfully saved!")