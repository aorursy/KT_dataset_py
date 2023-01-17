# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/H1B-Data-2020-Q2.csv')

df.head()
#Using pandas built-in shape method to view the number of rows and columns

print("In the following data set, there are: \n%d rows and %d columns"%(df.shape[0], df.shape[1]))
#Viewing the data types for each columns to better understand what kind of values they have

for colName in df.columns:

    print("%s: %s"%(colName, df[colName].dtype))
#Selecting only the columns that would help us in visualization

h1b_df = df[['CASE_STATUS', 'SOC_TITLE', 'RECEIVED_DATE', 'DECISION_DATE', 'VISA_CLASS', 'JOB_TITLE', 'FULL_TIME_POSITION', 'BEGIN_DATE', 'END_DATE', 'TOTAL_WORKER_POSITIONS', 'NEW_EMPLOYMENT', 'CONTINUED_EMPLOYMENT', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'AGENT_REPRESENTING_EMPLOYER', 'PREVAILING_WAGE', 'PW_UNIT_OF_PAY', 'H-1B_DEPENDENT']]

print("After extracting useful features, there are: \n%d rows and %d columns"%(h1b_df.shape[0], h1b_df.shape[1]))

h1b_df.head()
print("Missing values by feature")

#Total number of missing values in each column

print(h1b_df.isna().sum())



#Total missing values in the entire dataset

print("\nTotal NaN values in the dataset: %d"%(h1b_df.isna().sum()).sum())



#Dropping the NaN values

clean_h1b_df = h1b_df.dropna(how='any')



#Comparing Shape

print("Old shape: %d rows, %d columns"%(h1b_df.shape[0], h1b_df.shape[1]))

print("New shape: %d rows, %d columns"%(clean_h1b_df.shape[0], clean_h1b_df.shape[1]))
#Finding duplicates in the cleaned data frame

clean_h1b_df.duplicated()
#Converting all possible dates from object to datetime format

clean_h1b_df['RECEIVED_DATE'] =  pd.to_datetime(clean_h1b_df['RECEIVED_DATE'])

clean_h1b_df['DECISION_DATE'] =  pd.to_datetime(clean_h1b_df['DECISION_DATE'])

clean_h1b_df['BEGIN_DATE'] =  pd.to_datetime(clean_h1b_df['BEGIN_DATE'])

clean_h1b_df['END_DATE'] =  pd.to_datetime(clean_h1b_df['END_DATE'])
#Changing binary valued columns to categorical data (unordered)

clean_h1b_df['FULL_TIME_POSITION'] = clean_h1b_df['FULL_TIME_POSITION'].astype('category')

clean_h1b_df['AGENT_REPRESENTING_EMPLOYER'] = clean_h1b_df['AGENT_REPRESENTING_EMPLOYER'].astype('category')

clean_h1b_df['H-1B_DEPENDENT'] = clean_h1b_df['H-1B_DEPENDENT'].astype('category')
# Viewing unique values in columns to find out their correct data type

clean_h1b_df['NEW_EMPLOYMENT'].unique()
#Since we have a mix of integers and characters, let's replace the characters with numbers

clean_h1b_df['NEW_EMPLOYMENT'].replace({'Y': 1, 'N': 0}, inplace=True)

clean_h1b_df['CONTINUED_EMPLOYMENT'].replace({'Y': 1, 'N': 0}, inplace=True)



#Once we have all numbers, make the final data type to be integer

clean_h1b_df['NEW_EMPLOYMENT'] = clean_h1b_df['NEW_EMPLOYMENT'].astype('int')

clean_h1b_df['CONTINUED_EMPLOYMENT'] = clean_h1b_df['CONTINUED_EMPLOYMENT'].astype('int')
#Removing extraneous characters like '$', ',' and converting it to a number type before conversion

clean_h1b_df['PREVAILING_WAGE'] = clean_h1b_df['PREVAILING_WAGE'].astype(str).str.replace("$", "")

clean_h1b_df['PREVAILING_WAGE'] = clean_h1b_df['PREVAILING_WAGE'].astype(str).str.replace(",", "").astype(float)
#Create an empty array to add all the outliers indexes.

indexToDrop = []



#Iterate over each wage type to view the top 3 largest wages

for i in clean_h1b_df[clean_h1b_df['PW_UNIT_OF_PAY'] == 'Hour'].nlargest(5, 'PREVAILING_WAGE').index[:2]:

    #Adding the outliers to the arrat

    indexToDrop.append(i)



#Viewing the dataframe

clean_h1b_df[clean_h1b_df['PW_UNIT_OF_PAY'] == 'Hour'].nlargest(3, 'PREVAILING_WAGE')
clean_h1b_df[clean_h1b_df['PW_UNIT_OF_PAY'] == 'Week'].nlargest(5, 'PREVAILING_WAGE')

#No outlier found
clean_h1b_df[clean_h1b_df['PW_UNIT_OF_PAY'] == 'Bi-Weekly'].nlargest(5, 'PREVAILING_WAGE')

#No outlier found
for i in clean_h1b_df[clean_h1b_df['PW_UNIT_OF_PAY'] == 'Month'].nlargest(5, 'PREVAILING_WAGE').index[:3]:

    indexToDrop.append(i)



clean_h1b_df[clean_h1b_df['PW_UNIT_OF_PAY'] == 'Month'].nlargest(5, 'PREVAILING_WAGE')

#First three entries are annual salaries disguised as monthly
#Dropping the outliers list from the dataframe

clean_h1b_df = clean_h1b_df.drop(indexToDrop,axis='index')
#Create a new column 'ANNUAL_WAGE' that combines the values from two columns into a yearly wage

#Using numpy select to perform column wise operation

clean_h1b_df['ANNUAL_INCOME'] = (

    np.select(

        condlist=[clean_h1b_df['PW_UNIT_OF_PAY'] == 'Week', clean_h1b_df['PW_UNIT_OF_PAY'] == 'Bi-Weekly', clean_h1b_df['PW_UNIT_OF_PAY'] == 'Month', clean_h1b_df['PW_UNIT_OF_PAY'] == 'Hour'], 

        choicelist=[clean_h1b_df['PREVAILING_WAGE']*52, clean_h1b_df['PREVAILING_WAGE']*26, clean_h1b_df['PREVAILING_WAGE']*12, clean_h1b_df['PREVAILING_WAGE']*40*52], 

        default=clean_h1b_df['PREVAILING_WAGE']))
#Dropping the old wage and wage type columns since we have a new combined 'ANNUAL_WAGE'

clean_h1b_df = clean_h1b_df.drop(['PREVAILING_WAGE', 'PW_UNIT_OF_PAY'], axis=1)
clean_h1b_df.head()
#Which is the busiest month for filing?

print("Month\t     Filed Cases")

print("________________________\n")



#Since we have our datetime format, we can extract the month name out of it

print(pd.DatetimeIndex(clean_h1b_df['RECEIVED_DATE']).month_name().value_counts())



print("\nYear\tFiled Cases")

print("___________________\n")

print(pd.DatetimeIndex(clean_h1b_df['RECEIVED_DATE']).year.value_counts())
#Plotting the number of cases against months

pd.DatetimeIndex(clean_h1b_df['RECEIVED_DATE']).month_name().value_counts().plot(style='^', figsize=(15,7))
print("Month\t    Decision")

print("____________________")

print(pd.DatetimeIndex(clean_h1b_df['DECISION_DATE']).month_name().value_counts())
clean_h1b_df['ANNUAL_INCOME'].plot(kind='box', figsize=(20,10))
clean_h1b_df['ANNUAL_INCOME'].describe().round()
#Viewing all the unique state names

clean_h1b_df['EMPLOYER_STATE'].unique()
#Using a dict of state names and their abbreviations to map and maintain consistency

states = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DC": "DISTRICT OF COLUMBIA", "DE":"Delaware","FL":"Florida","GA":"Georgia", "GU": "GUAM", "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MP": "NORTHERN MARIANA ISLANDS", "MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","PR":"PUERTO RICO", "RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"}



#Converting the values to uppercase to match our dataset

cap_states_us = {k.upper():v.upper() for k,v in states.items()}



#Replacing abbreviations with full names

clean_h1b_df['EMPLOYER_STATE'].replace(cap_states_us, inplace=True)
#Viewing our EMPLOYER_STATE column now

clean_h1b_df['EMPLOYER_STATE'].unique()
#Plotting the employer state column

clean_h1b_df['EMPLOYER_STATE'].value_counts()[:10].plot(kind='pie', figsize=(10,15))
clean_h1b_df['EMPLOYER_NAME'].value_counts()[:10].plot(kind='bar', figsize=(20,10))
#Importing regex to find a specific substring

import re



#Terms that are related to design related positions

terms = ['UX', 'UI', 'VISUAL', 'PRODUCT DESIGNER', 'PRODUCT DESIGN']

p = r'\b(?:{})\b'.format('|'.join(map(re.escape, terms)))



#Using our substring format to find designers who applied for H1B

clean_h1b_df[clean_h1b_df['JOB_TITLE'].str.contains(p)]['JOB_TITLE'].value_counts()[:10]
#Doing the same step as above for developers

terms = ['SOFTWARE', 'DEVELOPER', 'ENGINEER', 'PROGRAMMING', 'SDE', 'PROGRAMMER']

q = r'\b(?:{})\b'.format('|'.join(map(re.escape, terms)))



clean_h1b_df[clean_h1b_df['JOB_TITLE'].str.contains(q)]['JOB_TITLE'].value_counts()[:15]
#Create separate variables for designer and developer pandas series data

a = clean_h1b_df[clean_h1b_df['JOB_TITLE'].str.contains(p)]['JOB_TITLE'].value_counts()[:15]

b = clean_h1b_df[clean_h1b_df['JOB_TITLE'].str.contains(q)]['JOB_TITLE'].value_counts()[:15]



#Plotting both of the above

ax1 = a.plot(color='blue', grid=True, label='Designers', figsize=(10,10))

ax2 = b.plot(color='red', grid=True, label='Developers')



#Setting legend to labels

h1, l1 = ax1.get_legend_handles_labels()

h2, l2 = ax2.get_legend_handles_labels()

plt.legend(h1, l2)



#Since there are more than type of role for one profile (e.d Senior Developer, Software Developer, Systems Developer), we are only comparing all designers with all developers in terms of total case filings

plt.ylabel('No. of visa filings')



#Hiding multiple profiles (for detail refer above breakdown)

ax1.axes.get_xaxis().set_visible(False)



plt.show()
clean_h1b_df.to_csv('final.csv')