#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#find the dataset file name
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#let us create dataframe to read data set
df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
#let us see the data for first 5 records from dataframe
df.head()
#let us check the dataframe structure and its column type
df.info()
#let us find number of rows and column in given dataset
df.shape
# find missing values
features_na = [features for features in df.columns if df[features].isnull().sum() > 0]
for feature in features_na:
    print(feature, np.round(df[feature].isnull().mean(), 4),  ' % missing values and actual count is '+str(df[feature].isnull().sum()))

print('Total entries:{}'.format(len(df)))
# check what will be new shape after droping missing values
df.dropna(inplace=False).shape
#drop missing values
df.dropna(inplace=True)
#check shape
df.shape
for column in df.columns:
    print(column,df[column].nunique())
categorical_features=[feature for feature in df.columns if ((df[feature].dtypes=='O') & (feature not in ['Uniq Id','Crawl Timestamp']))]
categorical_features
for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(df[feature].unique())))
df.groupby('Industry',sort=True)['Industry'].count()[0:15]
from re import search

def get_comman_job_industry(x):
    x = x.replace(",", " /")
    if (search('it-software', x.lower())):
        return 'Software Services'
    elif (search('call ', x.lower())):
        return 'Call Centre'
    elif (search('banking', x.lower()) or search('insurance', x.lower()) or search('finance', x.lower())):
        return 'Financial Services'
    elif (search('recruitment', x.lower())): 
        return 'Recruitment'
    elif (search('pharma', x.lower())): 
        return 'Pharma'
    elif (search('isp', x.lower())): 
        return 'Telcom / ISP'
    elif (search('ecommerce', x.lower())): 
        return 'Ecommerce'
    elif (search('fmcg', x.lower())): 
        return 'FMCG'
    elif (search('ngo', x.lower())): 
        return 'NGO'
    elif (search('medical', x.lower())): 
        return 'Medical'
    elif (search('aviation', x.lower())): 
        return 'Aviation'
    elif (search('fresher ', x.lower())): 
        return 'Fresher'
    elif (search('education', x.lower())): 
        return 'Education'
    elif (search('construction', x.lower())): 
        return 'Construction'
    elif (search('consulting', x.lower())): 
        return 'Consulting'
    elif (search('automobile', x.lower())): 
        return 'Automobile'
    elif (search('travel', x.lower())): 
        return 'Travels'
    elif (search('advertising', x.lower()) or search('broadcasting', x.lower())): 
        return 'Advertising'
    elif (search('transportation', x.lower())): 
        return 'Transportation'
    elif (search('agriculture', x.lower())): 
        return 'Agriculture'
    elif (search('agriculture', x.lower())): 
        return 'Agriculture'
    elif (search('industrial', x.lower())): 
        return 'Industrial Products'
    elif (search('media', x.lower())): 
        return 'Entertainment'
    elif (search('teksystems', x.lower()) or search('allegis', x.lower()) or search('aston', x.lower())
         or search('solugenix', x.lower()) or search('laurus', x.lower()) ):
        return 'Other'
    else:
        return x.strip()
df['New_Industry']=df['Industry'].apply(get_comman_job_industry)
df.groupby('New_Industry',sort=True)['New_Industry'].count().sort_values(ascending=False)[0:15]
plt.figure(figsize=(18,6), facecolor='white')
df.groupby('New_Industry',sort=True)['New_Industry'].count().sort_values(ascending=False)[0:15].plot.bar(color='green')
plt.xlabel('Industry')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Industry')
plt.show()
df.groupby('Functional Area',sort=True)['Functional Area'].count()[0:20]
from re import search

def get_comman_func_area(x):
    x = x.replace(",", " /")
    if (search('beauty', x.lower())):
        return 'Beauty / Fitness'
    elif (search('teaching', x.lower())):
        return 'Teaching  / Education'
    elif (search('other', x.lower())):
        return 'Others'
    elif (search('teksystems', x.lower()) or search('allegis', x.lower()) or search('aston', x.lower())
         or search('solugenix', x.lower()) or search('laurus', x.lower()) ):
        return 'Other'
    else:
        return x.strip()
df['New_Functional_Area']=df['Functional Area'].apply(get_comman_func_area)
df.groupby('New_Functional_Area',sort=True)['New_Functional_Area'].count().sort_values(ascending=False)[0:15]
plt.figure(figsize=(18,6), facecolor='white')
df.groupby('New_Functional_Area',sort=True)['New_Functional_Area'].count().sort_values(ascending=False)[0:15].plot.bar()
plt.xlabel('Functional Area')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Functional Area')
plt.show()
df.groupby('Location',sort=True)['Location'].count()[0:30]
def get_location(df):
    df_new=pd.DataFrame()
    for index, row in df.iterrows():
        for loc in row['Location'].split(','):
            loc_df = pd.DataFrame([loc])
            df_new = pd.concat([df_new,loc_df],ignore_index=True)
    return df_new    
Location_df = get_location(df)
Location_df.columns = ['Location']
Location_df.groupby('Location',sort=True)['Location'].count().sort_values(ascending=False)[0:30]
from re import search
def get_comman_location(x):
    x = x.replace(",", " /")
    if (search('bengaluru', x.lower()) or search('bangalore', x.lower())):
        return 'Bengaluru'
    elif (search('ahmedabad', x.lower())):
        return 'Ahmedabad'
    elif (search('chennai', x.lower())):
        return 'Chennai'
    elif (search('coimbatore', x.lower())):
        return 'Coimbatore'
    elif (search('delhi', x.lower()) or search('noida', x.lower()) or search('gurgaon', x.lower())):
        return 'Delhi NCR'
    elif (search('hyderabad', x.lower())):
        return 'Hyderabad'
    elif (search('kolkata', x.lower())):
        return 'Kolkata'
    elif (search('mumbai', x.lower())):
        return 'Mumbai'
    elif (search('Pune', x.lower())):
        return 'pune'
    elif (search('other', x.lower())):
        return 'Others'
    else:
        return x.strip()
Location_df['New_Location']=Location_df['Location'].apply(get_comman_location)
Location_df.groupby('New_Location',sort=True)['New_Location'].count().sort_values(ascending=False)[0:15]
plt.figure(figsize=(18,6), facecolor='white')
Location_df.groupby('New_Location',sort=True)['New_Location'].count().sort_values(ascending=False)[0:15].plot.bar(color="red")
plt.xlabel('Location')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Locations')
plt.show()
df.groupby('Role Category',sort=True)['Role Category'].count().sort_values(ascending=False)[0:15]
plt.figure(figsize=(18,6), facecolor='white')
df.groupby('Role Category',sort=True)['Role Category'].count().sort_values(ascending=False)[0:15].plot.bar(color="yellow")
plt.xlabel('Role Category')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Role Category')
plt.show()
df.groupby('Role',sort=True)['Role'].count().sort_values(ascending=False)[0:15]
plt.figure(figsize=(18,6), facecolor='white')
df.groupby('Role',sort=True)['Role'].count().sort_values(ascending=False)[0:15].plot.bar(color="brown")
plt.xlabel('Role')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Role')
plt.show()
df.groupby('Job Experience Required',sort=True)['Job Experience Required'].count()[0:30]
df['New_Job_Exp']=df['Job Experience Required'].apply(lambda x: x.replace("Years", "yrs"))
df.groupby('New_Job_Exp',sort=True)['New_Job_Exp'].count().sort_values(ascending=False)[0:15]
plt.figure(figsize=(18,6), facecolor='white')
df.groupby('New_Job_Exp',sort=True)['New_Job_Exp'].count().sort_values(ascending=False)[0:15].plot.bar(color="orange")
plt.xlabel('Job Exp Required in Years')
plt.ylabel('Count')
plt.title('Distribution of Top 15 Job Experience Year Range Required')
plt.show()
import re
def get_exp_level(x):
    if re.findall('-',x):
        lst =x.replace('yrs','').strip().split('-')
        #print (x)
        lvl =(int(lst[0].strip())+int(lst[1].strip()))/2
        if (lvl >= 0 and lvl <= 2):
            return ('Freshers')
        elif (lvl >= 2 and lvl <= 5):
            return ('Intermediate')
        elif (lvl >= 5 and lvl <= 8):
            return ('Lead')
        elif (lvl >= 8 and lvl <= 12):
            return ('Manager')
        elif (lvl >= 12 and lvl <= 16):
            return ('Senior Manager')
        elif (lvl >= 16 and lvl <= 20):
            return ('Executive')
        elif (lvl >= 20):
            return ('Senior Executive')
        else:
            return('Others')
    else:
        return('Others')
df['New_Exp_Level']=df['New_Job_Exp'].apply(get_exp_level)
df.groupby('New_Exp_Level',sort=True)['New_Exp_Level'].count().sort_values(ascending=False)[0:30]
plt.figure(figsize=(18,6), facecolor='white')
df.groupby('New_Exp_Level',sort=True)['New_Exp_Level'].count().sort_values(ascending=False).plot.bar(color="purple")
plt.xlabel('Exp Level Required in Years')
plt.ylabel('Count')
plt.title('Distribution of Job Experience Level')
plt.show()
df.groupby('Job Salary',sort=True)['Job Salary'].count()[0:30]
import re
def get_salary(x):
    if re.findall('-',x):
        lst =x.replace('PA.','').replace(',','').replace('INR','').strip().split('-')        
        try:
            sal1 = int(lst[0].strip())
            sal2 = int(lst[1].strip())
            #print (sal1)
            if (sal1 <= 300000):
                return '0-3L PA'
            elif (sal1 >= 300000 & sal2 <= 800000 ):
                return '3-8L PA'
            elif (sal1 >= 800000 & sal2 <= 1500000 ):
                return '8 -15L PA'
            elif (sal1 >= 1500000 & sal2 <= 2200000 ):
                return '15 - 22L PA'
            elif (sal1 >= 2200000 & sal2 <= 3000000 ):
                return '22 - 30L PA'
            elif (sal1 >= 3000000 & sal2 <= 3800000 ):
                return '30 - 38L PA'
            if (sal1 >= 3800000):
                return '38L - Above PA'
        except:
            return('Others')
    else:
        return('Others')
df['New_Job_Salary']=df['Job Salary'].apply(get_salary)
df.groupby('New_Job_Salary',sort=True)['New_Job_Salary'].count().sort_values(ascending=False)[0:30]
df.groupby('Key Skills',sort=True)['Key Skills'].count()[0:30]
def get_skills(df):
    df_new=pd.DataFrame()
    for index, row in df.iterrows():
        for skill in row['Key Skills'].split('|'):
            skill_df = pd.DataFrame([skill])
            df_new = pd.concat([df_new,skill_df],ignore_index=True)
    return df_new    
key_skill_df = get_skills(df)
key_skill_df.columns = ['key_skills']
key_skill_df.groupby('key_skills',sort=True)['key_skills'].count().sort_values(ascending=False)[0:20]
plt.figure(figsize=(18,6), facecolor='white')
key_skill_df.groupby('key_skills',sort=True)['key_skills'].count().sort_values(ascending=False)[0:20].plot.bar(color="orange")
plt.xlabel('Key Skills')
plt.ylabel('Count')
plt.title('Distribution of Top 20 Key Skills')
plt.show()