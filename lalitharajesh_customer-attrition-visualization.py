#This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
print (df.shape)
print (df.columns)
print(df.describe())
df.isnull().sum()
df.info()
df['SeniorCitizen'] = df.SeniorCitizen.astype('object')
#Before that the empty data replace with np.nan
df['TotalCharges'] = df['TotalCharges'].replace(" ",np.nan)
#Now we see there are 11 missing values in Total Charges column.
df['TotalCharges'].isnull().sum()
# Displaying the rows with 11 missing values for our understanding purpose
df1 = df[df.isnull().any(axis=1)]
print (df1)
churn_yes = df[df['Churn'] == 'Yes']
print (len(churn_yes))
churn_no = df[df['Churn'] == 'No']
print (len(churn_no))
df2 = df.loc[df['Churn'] == 'Yes']
df2.head()
male_count = df2[df2['gender'] == 'Male']
print (len(male_count))
female_count = df2[df2['gender'] == 'Female']
print (len(female_count))
import matplotlib.pyplot as plt
# Data to plot
labels = 'Male', 'Female'
sizes = [930, 939]
colors = ['yellowgreen', 'lightcoral']
explode = (0.1, 0)  # explode 1st slice
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()
Senior_male_count = male_count[male_count['SeniorCitizen'] == 1]
print (len(Senior_male_count))
Junior_male_count = male_count[male_count['SeniorCitizen'] == 0]
print (len(Junior_male_count))
Senior_female_count = female_count[female_count['SeniorCitizen'] == 1]
print (len(Senior_female_count))
Junior_female_count = female_count[female_count['SeniorCitizen'] == 0]
print (len(Junior_female_count))
import seaborn as sns
plt.figure(figsize = (9,5))
sns.countplot(df2.PhoneService, hue = df2.Churn)
plt.figure(figsize=(12,6))
ax = sns.countplot(x="Churn", hue="Contract", data=df2);
ax.set_title('Contract Type vs Churn', fontsize=10)
ax.set_ylabel('Number of Customers', fontsize =10)
ax.set_xlabel('Churn', fontsize = 15)
Senior_male_count_services_1 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['PhoneService'] == 'Yes') & (Senior_male_count['Contract'] == 'Month-to-month')]
print ("Seniors (M) who availed phone service and left less than 10 months: ", len(Senior_male_count_services_1))
Senior_male_count_services_4 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['PhoneService'] == 'Yes') & (Senior_male_count['Contract'] == 'Month-to-month')]
print ("Seniors (M) who availed phone service and stayed more than 10 months: ",len(Senior_male_count_services_4))
Junior_male_count_services_1 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['PhoneService'] == 'Yes') & (Junior_male_count['Contract'] == 'Month-to-month')]
print ("Juniors (M) who availed phone service and left less than 10 months:" ,len(Junior_male_count_services_1))
Junior_male_count_services_4 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['PhoneService'] == 'Yes') & (Junior_male_count['Contract'] == 'Month-to-month')]
print ("Juniors (M) who availed phone service and stayed more than 10 months:",len(Junior_male_count_services_4))
Senior_female_count_services_1 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['PhoneService'] == 'Yes') & (Senior_female_count['Contract'] == 'Month-to-month')]
print ("Seniors (F) who availed phone service and left less than 10 months: ", len(Senior_female_count_services_1))
Senior_female_count_services_4 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['PhoneService'] == 'Yes') & (Senior_female_count['Contract'] == 'Month-to-month')]
print ("Seniors (F) who availed phone service and stayed more than 10 months: ",len(Senior_female_count_services_4))
Junior_female_count_services_1 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['PhoneService'] == 'Yes') & (Junior_female_count['Contract'] == 'Month-to-month')]
print ("Juniors (F) who availed phone service and left less than 10 months:" ,len(Junior_female_count_services_1))
Junior_female_count_services_4 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['PhoneService'] == 'Yes') & (Junior_female_count['Contract'] == 'Month-to-month')]
print ("Juniors (F) who availed phone service and stayed more than 10 months:",len(Junior_female_count_services_4))
# data to plot
n_groups = 2
PhService_usedby_MaleSeniors = [77, 115]
PhService_usedby_FemaleSeniors = [94, 105]
PhService_usedby_MaleJuniors = [313, 202]
PhService_usedby_FemaleJuniors = [337, 218]
# create plot
fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, PhService_usedby_MaleSeniors, bar_width,
                 alpha=opacity,
                 color='b',
                 label='ph.Ser used by Seniors(M)')
 
rects2 = plt.bar(index + bar_width, PhService_usedby_FemaleSeniors, bar_width,
                 alpha=opacity,
                 color='g',
                 label='ph.Ser used by Seniors(F)')
 
plt.xlabel('Female/Male customers (Sr)')
plt.ylabel('Phone services utilized')
plt.title('Phone Services used by Female/Male customers (Sr) only for short period and Long Period')
plt.xticks(index + bar_width, ('A', 'B'))
plt.legend()
 
plt.tight_layout()
plt.show()

# data to plot
n_groups = 2
PhService_usedby_MaleJuniors = [313, 202]
PhService_usedby_FemaleJuniors = [337, 218]
# create plot
fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, PhService_usedby_MaleJuniors, bar_width,
                 alpha=opacity,
                 color='yellow',
                 label='ph.Ser used by Juniors(M)')
 
rects2 = plt.bar(index + bar_width,PhService_usedby_FemaleJuniors, bar_width,
                 alpha=opacity,
                 color='R',
                 label='ph.Ser used by Juniors(F)')
 
plt.xlabel('Female/Male customers (Jr)')
plt.ylabel('Phone services utilized')
plt.title('Phone Services used by Female/Male customers (Jr) only for short period and Long Period')
plt.xticks(index + bar_width, ('C', 'D'))
plt.legend()
 
plt.tight_layout()
plt.show()
import seaborn as sns
plt.figure(figsize = (9,5))
sns.countplot(df2.InternetService, hue = df2.Churn)
Senior_male_count_services_7 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['Contract'] == 'Month-to-month')]
print ("Seniors (M) who availed DSL Internet service and left less than 10 months:", len(Senior_male_count_services_7))
Senior_male_count_services_10 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['Contract'] == 'Month-to-month')]
print ("Seniors (M) who availed Fiber Optic Internet service and left less than 10 months:", len(Senior_male_count_services_10))
Senior_male_count_services_13 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['Contract'] == 'Month-to-month')]
print ("Seniors (M) who availed DSL Internet service and stayed more than 10 months:", len(Senior_male_count_services_13))

Senior_male_count_services_16 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['Contract'] == 'Month-to-month')]
print ("Seniors (M) who availed Fiber optic Internet service and stayed more than 10 months:", len(Senior_male_count_services_16))
Senior_female_count_services_7 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['Contract'] == 'Month-to-month')]
print ("Seniors (F) who availed DSL Internet service and left less than 10 months:", len(Senior_female_count_services_7))
Senior_female_count_services_10 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['Contract'] == 'Month-to-month')]
print ("Seniors (F) who availed Fiber Optic Internet service and left less than 10 months:", len(Senior_female_count_services_10))
Senior_female_count_services_13 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['Contract'] == 'Month-to-month')]
print ("Seniors (F) who availed DSL Internet service and stayed more than 10 months:", len(Senior_female_count_services_13))
Senior_female_count_services_16 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['Contract'] == 'Month-to-month')]
print ("Seniors (F) who availed Fiber optic Internet service and stayed more than 10 months:", len(Senior_female_count_services_16))
Junior_male_count_services_7 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['Contract'] == 'Month-to-month')]
print ("Juniors (M) who availed DSL Internet service and left less than 10 months:", len(Junior_male_count_services_7))
Junior_male_count_services_8 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['Contract'] == 'Month-to-month')]
print ("Juniors (M) who availed Fiber Optic Internet service and left less than 10 months:", len(Junior_male_count_services_8))
Junior_male_count_services_9 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['Contract'] == 'Month-to-month')]
print ("Juniors (M) who availed DSL Internet service and stayed more than 10 months:", len(Junior_male_count_services_9))
Junior_male_count_services_10 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['Contract'] == 'Month-to-month')]
print ("Juniors (M) who availed Fiber optic Internet service and stayed more than 10 months:", len(Junior_male_count_services_10))
Junior_female_count_services_7 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['Contract'] == 'Month-to-month')]
print ("Juniors (F) who availed DSL Internet service and left less than 10 months:", len(Junior_female_count_services_7))
Junior_female_count_services_8 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['Contract'] == 'Month-to-month')]
print ("Juniors (F) who availed Fiber Optic Internet service and left less than 10 months:", len(Junior_female_count_services_8))
Junior_female_count_services_9 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['Contract'] == 'Month-to-month')]
print ("Juniors (F) who availed DSL Internet service and stayed more than 10 months:", len(Junior_female_count_services_9))
Junior_female_count_services_10 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['Contract'] == 'Month-to-month')]
print ("Juniors (F) who availed Fiber optic Internet service and stayed more than 10 months:", len(Junior_female_count_services_10))
# data to plot
n_groups = 2
DSLService_usedby_MaleSeniors = [20, 14]
DSLService_usedby_FemaleSeniors = [21, 11]

# create plot
fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, DSLService_usedby_MaleSeniors, bar_width,
                 alpha=opacity,
                 color='b',
                 label='DSLnetSer used by Seniors(M)')
 
rects2 = plt.bar(index + bar_width, DSLService_usedby_FemaleSeniors, bar_width,
                 alpha=opacity,
                 color='g',
                 label='DSLnetSer used by Seniors(F)')
 
plt.xlabel('Female/Male customers (Sr)')
plt.ylabel('DSLIntenet services utilized')
plt.title('DSLIntenet Services used by Female/Male customers (Sr) for short period and Long Period')
plt.xticks(index + bar_width, ('A', 'B'))
plt.legend()
 
plt.tight_layout()
plt.show()

# data to plot
n_groups = 2
DSLService_usedby_MaleJuniors = [126, 36]
DSLService_usedby_FemaleJuniors = [107, 51]
# create plot
fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, DSLService_usedby_MaleJuniors, bar_width,
                 alpha=opacity,
                 color='R',
                 label='DSLnetSer used by Juniors(M)')
 
rects2 = plt.bar(index + bar_width,DSLService_usedby_FemaleJuniors, bar_width,
                 alpha=opacity,
                 color='k',
                 label='DSLnetSer used by Juniors(F)')
 
plt.xlabel('Female/Male customers (Jr)')
plt.ylabel('DSLIntenet services utilized')
plt.title('DSLIntenet Services used by Female/Male customers (Jr) for short period and Long Period')
plt.xticks(index + bar_width, ('C', 'D'))
plt.legend()
 
plt.tight_layout()
plt.show()

# create plot
FibOptService_usedby_MaleSeniors = [66, 111]
FibOptService_usedby_FemaleSeniors = [82, 101]

fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, FibOptService_usedby_MaleSeniors, bar_width,
                 alpha=opacity,
                 color='b',
                 label='FibOptSer used by Seniors(M)')
 
rects2 = plt.bar(index + bar_width,FibOptService_usedby_FemaleSeniors, bar_width,
                 alpha=opacity,
                 color='y',
                 label='FibOptSer used by Seniors(F)')
 
plt.xlabel('Female/Male customers (Sr)')
plt.ylabel('FibOptIntenet services utilized')
plt.title('FibOptIntenet Services used by Female/Male customers (Sr) for short period and Long Period')
plt.xticks(index + bar_width, ('E', 'F'))
plt.legend()
 
plt.tight_layout()
plt.show()

# create plot

FibOptService_usedby_MaleJuniors = [187,175]
FibOptService_usedby_FemaleJuniors = [224, 179]

fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, FibOptService_usedby_MaleJuniors, bar_width,
                 alpha=opacity,
                 color='coral',
                 label='FibOptSer used by Juniors(M)')
 
rects2 = plt.bar(index + bar_width,FibOptService_usedby_FemaleJuniors, bar_width,
                 alpha=opacity,
                 color='k',
                 label='FibOptSer used by Juniors(F)')
 
plt.xlabel('Female/Male customers (Jr)')
plt.ylabel('FibOptIntenet services utilized')
plt.title('FibOptIntenet Services used by Female/Male customers (Jr) for short period and Long Period')
plt.xticks(index + bar_width, ('G', 'H'))
plt.legend()
 
plt.tight_layout()
plt.show()

Senior_male_count_services_19 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['Contract'] == 'Month-to-month') & (Senior_male_count['TechSupport'] == 'Yes')]
print ("Seniors (M) who availed DSL Internet service + Tech Services and left less than 10 months:", len(Senior_male_count_services_19))
Senior_male_count_services_20 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['Contract'] == 'Month-to-month')  & (Senior_male_count['TechSupport'] == 'Yes')]
print ("Seniors (M) who availed Fiber Optic Internet service + Tech Services and left less than 10 months:", len(Senior_male_count_services_20))
Senior_male_count_services_21 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['Contract'] == 'Month-to-month') & (Senior_male_count['TechSupport'] == 'Yes')]
print ("Seniors (M) who availed DSL Internet service + Tech Services and stayed more than 10 months:", len(Senior_male_count_services_21))
Senior_male_count_services_22 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['Contract'] == 'Month-to-month') & (Senior_male_count['TechSupport'] == 'Yes')]
print ("Seniors (M) who availed Fiber optic Internet service + Tech Services and stayed more than 10 months:", len(Senior_male_count_services_22))
Junior_male_count_services_11 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['Contract'] == 'Month-to-month') & (Junior_male_count['TechSupport'] == 'Yes')]
print ("Juniors (M) who availed DSL Internet service + Tech Services and left less than 10 months:", len(Junior_male_count_services_11))
Junior_male_count_services_12 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['Contract'] == 'Month-to-month') & (Junior_male_count['TechSupport'] == 'Yes')]
print ("Juniors (M) who availed Fiber Optic Internet service + Tech Services and left less than 10 months:", len(Junior_male_count_services_12))

Junior_male_count_services_13 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['Contract'] == 'Month-to-month') & (Junior_male_count['TechSupport'] == 'Yes')]
print ("Juniors (M) who availed DSL Internet service + Tech Services and stayed more than 10 months:", len(Junior_male_count_services_13))
Junior_male_count_services_14 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['Contract'] == 'Month-to-month') & (Junior_male_count['TechSupport'] == 'Yes')]
print ("Juniors (M) who availed Fiber optic Internet service + Tech Services and stayed more than 10 months:", len(Junior_male_count_services_14))
Senior_female_count_services_17 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['Contract'] == 'Month-to-month') & (Senior_female_count['TechSupport'] == 'Yes')]
print ("Seniors (F) who availed DSL Internet service + Tech Services and left less than 10 months:", len(Senior_female_count_services_17))
Senior_female_count_services_18 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['Contract'] == 'Month-to-month') & (Senior_female_count['TechSupport'] == 'Yes')]
print ("Seniors (F) who availed Fiber Optic Internet service + Tech Services and left less than 10 months:", len(Senior_female_count_services_18))
Senior_female_count_services_19 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['Contract'] == 'Month-to-month') & (Senior_female_count['TechSupport'] == 'Yes')]
print ("Seniors (F) who availed DSL Internet service + Tech Services and stayed more than 10 months:", len(Senior_female_count_services_19))
Senior_female_count_services_20 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['Contract'] == 'Month-to-month') & (Senior_female_count['TechSupport'] == 'Yes')]
print ("Seniors (F) who availed Fiber optic Internet service + Tech Services and stayed more than 10 months:", len(Senior_female_count_services_20))
Junior_female_count_services_11 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['Contract'] == 'Month-to-month') & (Junior_female_count['TechSupport'] == 'Yes')]
print ("Juniors (F) who availed DSL Internet service + Tech Services and left less than 10 months:", len(Junior_female_count_services_11))
Junior_female_count_services_12 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['Contract'] == 'Month-to-month') & (Junior_female_count['TechSupport'] == 'Yes')]
print ("Juniors (F) who availed Fiber Optic Internet service + Tech Services and left less than 10 months:", len(Junior_female_count_services_12))

Junior_female_count_services_13 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['Contract'] == 'Month-to-month') & (Junior_female_count['TechSupport'] == 'Yes')]
print ("Juniors (F) who availed DSL Internet service + Tech Services and stayed more than 10 months:", len(Junior_female_count_services_13))
Junior_female_count_services_14 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['Contract'] == 'Month-to-month') & (Junior_female_count['TechSupport'] == 'Yes')]
print ("Juniors (F) who availed Fiber optic Internet service + Tech Services and stayed more than 10 months:", len(Junior_female_count_services_14))
from matplotlib.pyplot import figure
# data to plot
n_groups = 2
DSL_IntTechService_usedby_JuniorsM = [18,13]
DSL_IntTechService_usedby_JuniorsF= [27,13]
FibOpt_IntTechService_usedby_JuniorsM = [14,31]
FibOpt_IntTechService_usedby_JuniorsF = [19,34]
# create plot
fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(n_groups)
index_1 = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, DSL_IntTechService_usedby_JuniorsM,bar_width,
                 alpha=opacity,
                 color='b',
                 label='DSL/Tech.Serv used by Jr(M)')
 
rects2 = plt.bar(index + bar_width, DSL_IntTechService_usedby_JuniorsF, bar_width,
                 alpha=opacity,
                 color='g',
                 label='DSL/Tech.Serv used by Jr(F)')
plt.xlabel('Male/Female customers (Jr)')
plt.ylabel('DSL net/Tech services utilized')
plt.title('DSL Internet + Tech Services used by Male/Female customers (Juniors count Only)')
plt.xticks(index + bar_width, ('A', 'B'))
plt.legend()

plt.tight_layout()
plt.show()


# data to plot
n_groups = 2
DSL_IntTechService_usedby_JuniorsM = [18,13]
DSL_IntTechService_usedby_JuniorsF= [27,13]
FibOpt_IntTechService_usedby_JuniorsM = [14,31]
FibOpt_IntTechService_usedby_JuniorsF = [19,34]
# create plot
fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(n_groups)
index_1 = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, FibOpt_IntTechService_usedby_JuniorsM,bar_width,
                 alpha=opacity,
                 color='b',
                 label='FibOpt/Tech.Serv used by Jr(M)')
 
rects2 = plt.bar(index + bar_width, FibOpt_IntTechService_usedby_JuniorsF, bar_width,
                 alpha=opacity,
                 color='g',
                 label='FibOpt/Tech.Serv used by Jr(F)')
plt.xlabel('Male/Female customers (Jr)')
plt.ylabel('FibOpt net/Tech services utilized')
plt.title('FibOpt Internet + Tech Services used by Male/Female customers (Juniors count Only)')
plt.xticks(index + bar_width, ('A', 'B'))
plt.legend()

plt.tight_layout()
plt.show()
Senior_male_count_services_1 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['PhoneService'] == 'Yes') & (Senior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (M) who availed phone service + Elec.Check Payment option and left less than 10 months: ", len(Senior_male_count_services_1))
Senior_male_count_services_4 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['PhoneService'] == 'Yes') & (Senior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (M) who availed phone service + Elec.Check Payment option and stayed more than 10 months: ",len(Senior_male_count_services_4))
Junior_male_count_services_1 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['PhoneService'] == 'Yes') & (Junior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (M) who availed phone service + Elec.Check Payment option and left less than 10 months:" ,len(Junior_male_count_services_1))
Junior_male_count_services_4 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['PhoneService'] == 'Yes') & (Junior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (M) who availed phone service + Elec.Check Payment option and stayed more than 10 months:",len(Junior_male_count_services_4))
Senior_female_count_services_1 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['PhoneService'] == 'Yes') & (Senior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (F) who availed phone service + Elec.Check Payment option and left less than 10 months: ", len(Senior_female_count_services_1))
Senior_female_count_services_4 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['PhoneService'] == 'Yes') & (Senior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (F) who availed phone service + Elec.Check Payment option and stayed more than 10 months: ",len(Senior_female_count_services_4))
Junior_female_count_services_1 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['PhoneService'] == 'Yes') & (Junior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (F) who availed phone service + Elec.Check Payment option and left less than 10 months:" ,len(Junior_female_count_services_1))
Junior_female_count_services_4 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['PhoneService'] == 'Yes') & (Junior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (F) who availed phone service + Elec.Check Payment option and stayed more than 10 months:",len(Junior_female_count_services_4))
Senior_male_count_services_1 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['PhoneService'] == 'Yes') & (Senior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (M) who availed phone service + Mailed Check Payment option and left less than 10 months: ", len(Senior_male_count_services_1))
Senior_male_count_services_4 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['PhoneService'] == 'Yes') & (Senior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (M) who availed phone service + Mailed Check Payment option and stayed more than 10 months: ",len(Senior_male_count_services_4))
Junior_male_count_services_1 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['PhoneService'] == 'Yes') & (Junior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (M) who availed phone service + Mailed Check Payment option and left less than 10 months:" ,len(Junior_male_count_services_1))
Junior_male_count_services_4 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['PhoneService'] == 'Yes') & (Junior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (M) who availed phone service + Mailed Check Payment option and stayed more than 10 months:",len(Junior_male_count_services_4))
Senior_female_count_services_1 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['PhoneService'] == 'Yes') & (Senior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (F) who availed phone service + Mailed Check Payment option and left less than 10 months: ", len(Senior_female_count_services_1))
Senior_female_count_services_4 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['PhoneService'] == 'Yes') & (Senior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (F) who availed phone service + Mailed Check Payment option and stayed more than 10 months: ",len(Senior_female_count_services_4))
Junior_female_count_services_1 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['PhoneService'] == 'Yes') & (Junior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (F) who availed phone service + Mailed Check Payment option and left less than 10 months:" ,len(Junior_female_count_services_1))
Junior_female_count_services_4 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['PhoneService'] == 'Yes') & (Junior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (F) who availed phone service + Mailed Check Payment option and stayed more than 10 months:",len(Junior_female_count_services_4))
PhService_ElecCheck_usedby_MaleSr = [54, 85]
PhService_ElecCheck_usedby_FemaleSr = [71, 71]
PhService_ElecCheck_usedby_MaleJr = [182, 159]
PhService_ElecCheck_usedby_FemaleJr = [177, 148]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [54., 85.]
z = [71., 71.]
a = [182., 159.]
b= [177., 148.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'ph.Ser/Elec check used by Sr(M)')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'ph.Ser/Elec check used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'ph.Ser/Elec check used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'ph.Ser/Elec check used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('Phone services + Elec.Check utilized')
plt.title('Phone Services + Elec.Check Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C', 'D'))

plt.legend()
plt.show()

PhService_MailCheck_usedby_MaleSr = [11, 7]
PhService_MailCheck_usedby_FemaleSr = [12, 6]
PhService_MailCheck_usedby_MaleJr = [93, 25]
PhService_MailCheck_usedby_FemaleJr = [89,28]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [11., 7.]
z = [12., 6.]
a = [93., 25.]
b= [89., 28.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'ph.Ser/Mail check used by Sr(M)')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'ph.Ser/Mail check used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'ph.Ser/Mail check used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'ph.Ser/Mail check used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('Phone services + Mail.Check utilized')
plt.title('Phone Services + Mail.Check Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C', 'D'))

plt.legend()
plt.show()
Senior_male_count_services_1 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['PhoneService'] == 'Yes') & (Senior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (M) who availed phone service + Bank transfer option and left less than 10 months: ", len(Senior_male_count_services_1))
Senior_male_count_services_4 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['PhoneService'] == 'Yes') & (Senior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (M) who availed phone service + Bank transfer option and stayed more than 10 months: ",len(Senior_male_count_services_4))
Junior_male_count_services_1 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['PhoneService'] == 'Yes') & (Junior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (M) who availed phone service + Bank transfer option and left less than 10 months:" ,len(Junior_male_count_services_1))
Junior_male_count_services_4 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['PhoneService'] == 'Yes') & (Junior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (M) who availed phone service + Bank transfer option and stayed more than 10 months:",len(Junior_male_count_services_4))
Senior_female_count_services_1 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['PhoneService'] == 'Yes') & (Senior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (F) who availed phone service + Bank transfer option and left less than 10 months: ", len(Senior_female_count_services_1))
Senior_female_count_services_4 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['PhoneService'] == 'Yes') & (Senior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (F) who availed phone service + EBank transfer option and stayed more than 10 months: ",len(Senior_female_count_services_4))
Junior_female_count_services_1 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['PhoneService'] == 'Yes') & (Junior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (F) who availed phone service + Bank transfer option and left less than 10 months:" ,len(Junior_female_count_services_1))
Junior_female_count_services_4 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['PhoneService'] == 'Yes') & (Junior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (F) who availed phone service + Bank transfer option and stayed more than 10 months:",len(Junior_female_count_services_4))
Senior_male_count_services_1 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['PhoneService'] == 'Yes') & (Senior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (M) who availed phone service + Credit card option and left less than 10 months: ", len(Senior_male_count_services_1))
Senior_male_count_services_4 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['PhoneService'] == 'Yes') & (Senior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (M) who availed phone service + Credit card option and stayed more than 10 months: ",len(Senior_male_count_services_4))
Junior_male_count_services_1 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['PhoneService'] == 'Yes') & (Junior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (M) who availed phone service + Credit card option and left less than 10 months:" ,len(Junior_male_count_services_1))
Junior_male_count_services_4 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['PhoneService'] == 'Yes') & (Junior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (M) who availed phone service + Credit card option and stayed more than 10 months:",len(Junior_male_count_services_4))
Senior_female_count_services_1 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['PhoneService'] == 'Yes') & (Senior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (F) who availed phone service + Credit card option and left less than 10 months: ", len(Senior_female_count_services_1))
Senior_female_count_services_4 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['PhoneService'] == 'Yes') & (Senior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (F) who availed phone service + Credit card option and stayed more than 10 months: ",len(Senior_female_count_services_4))
Junior_female_count_services_1 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['PhoneService'] == 'Yes') & (Junior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (F) who availed phone service + Credit card option and left less than 10 months:" ,len(Junior_female_count_services_1))
Junior_female_count_services_4 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['PhoneService'] == 'Yes') & (Junior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (F) who availed phone service + Credit card option and stayed more than 10 months:",len(Junior_female_count_services_4))
PhService_BankTransfer_usedby_MaleSr = [5, 18]
PhService_BankTransfer_usedby_FemaleSr = [4, 22]
PhService_BankTransfer_usedby_MaleJr = [24, 58]
PhService_BankTransfer_usedby_FemaleJr = [43, 59]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [5., 18.]
z = [4., 22.]
a = [24., 58.]
b= [43., 59.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'ph.Ser/Bank Trans. used by Sr(M)')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'ph.Ser/Bank Trans. used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'ph.Ser/Bank Trans. used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'ph.Ser/Bank Trans. used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('Phone services + BankTransfer utilized')
plt.title('Phone Services + BankTransfer Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C','D'))

plt.legend()
plt.show()

PhService_creditTrans_usedby_MaleSr = [7, 19]
PhService_creditTrans_usedby_FemaleSr = [7,25]
PhService_creditTrans_usedby_MaleJr = [18, 48]
PhService_creditTrans_usedby_FemaleJr = [30,53]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [7., 19.]
z = [7., 25.]
a = [18., 48.]
b= [30., 53.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'ph.Ser/CreditTrans used by Sr(M)')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'ph.Ser/CreditTrans used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'ph.Ser/CreditTrans used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'ph.Ser/CreditTrans used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('Phone services + CreditTransfer utilized')
plt.title('Phone Services + CreditTransfer Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C','D'))

plt.legend()
plt.show()

Senior_male_count_services_19 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (M) who availed DSL Internet service + Elec Check Payment Option and left less than 10 months:", len(Senior_male_count_services_19))
Senior_male_count_services_20 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (M) who availed Fiber Optic Internet service + Elec Check Payment Option and left less than 10 months:", len(Senior_male_count_services_20))
Senior_male_count_services_21 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (M) who availed DSL Internet service + Elec Check Payment Option and stayed more than 10 months:", len(Senior_male_count_services_21))
Senior_male_count_services_22 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (M) who availed Fiber optic Internet service + Elec Check Payment Option and stayed more than 10 months:", len(Senior_male_count_services_22))
Junior_male_count_services_11 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (M) who availed DSL Internet service + Elec Check Payment Option and left less than 10 months:", len(Junior_male_count_services_11))
Junior_male_count_services_12 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (M) who availed Fiber Optic Internet service + Elec Check Payment Option and left less than 10 months:", len(Junior_male_count_services_12))
Junior_male_count_services_13 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (M) who availed DSL Internet service + Elec Check Payment Option and stayed more than 10 months:", len(Junior_male_count_services_13))
Junior_male_count_services_14 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (M) who availed Fiber optic Internet service + Elec Check Payment Option and stayed more than 10 months:", len(Junior_male_count_services_14))
Senior_female_count_services_17 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (F) who availed DSL Internet service + Elec Check Payment Option and left less than 10 months:", len(Senior_female_count_services_17))
Senior_female_count_services_18 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (F) who availed Fiber Optic Internet service + Elec Check Payment Option and left less than 10 months:", len(Senior_female_count_services_18))
Senior_female_count_services_19 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (F) who availed DSL Internet service + Elec Check Payment Option and stayed more than 10 months:", len(Senior_female_count_services_19))
Senior_female_count_services_20 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Seniors (F) who availed Fiber optic Internet service + Elec Check Payment Option and stayed more than 10 months:", len(Senior_female_count_services_20))
Junior_female_count_services_11 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (F) who availed DSL Internet service + Elec Check Payment Option and left less than 10 months:", len(Junior_female_count_services_11))
Junior_female_count_services_12 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (F) who availed Fiber Optic Internet service + Elec Check Payment Option and left less than 10 months:", len(Junior_female_count_services_12))

Junior_female_count_services_13 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (F) who availed DSL Internet service + Elec Check Payment Option and stayed more than 10 months:", len(Junior_female_count_services_13))
Junior_female_count_services_14 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['PaymentMethod'] == 'Electronic check')]
print ("Juniors (F) who availed Fiber optic Internet service + Elec Check Payment Option and stayed more than 10 months:", len(Junior_female_count_services_14))
DSLserv_ElecCheck_usedby_MaleSr = [13, 9]
DSLserv_ElecCheck_usedby_FemaleSr = [14, 6]
DSLserv_ElecCheck_usedby_MaleJr = [61, 25]
DSLserv_ElecCheck_usedby_FemaleJr = [47, 29]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [13., 9.]
z = [14., 6.]
a = [61., 25.]
b= [47., 29.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'DSL.Ser/Elec Check used by Sr(M)')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'DSL.Ser/Elec Check used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'DSL.Ser/Elec Check used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'DSL.Ser/Elec Check used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('DSL.Service + Elec Check PM utilized')
plt.title('DSL Services + Elec Check Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C','D'))

plt.legend()
plt.show()

FibOptserv_ElecCheck_usedby_MaleSr = [48, 83]
FibOptserv_ElecCheck_usedby_FemaleSr = [64, 70]
FibOptserv_ElecCheck_usedby_MaleJr = [140, 141]
FibOptserv_ElecCheck_usedby_FemaleJr = [143, 133]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [48., 83.]
z = [64., 70.]
a = [140., 141.]
b= [142., 133.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'FibOpt.Ser/Elec Check used by Sr(M))')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'FibOpt.Ser/Elec Check used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'FibOpt.Ser/Elec Check used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'FibOpt.Ser/Elec Check used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('FibOpt services + Elec check utilized')
plt.title('FiberOptic Services + Electronic check Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C','D'))

plt.legend()
plt.show()
Senior_male_count_services_19 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (M) who availed DSL Internet service + Mailed Check Payment Option and left less than 10 months:", len(Senior_male_count_services_19))
Senior_male_count_services_20 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (M) who availed Fiber Optic Internet service + Mailed Check Payment Option and left less than 10 months:", len(Senior_male_count_services_20))
Senior_male_count_services_21 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (M) who availed DSL Internet service + Mailed Check Payment Option and stayed more than 10 months:", len(Senior_male_count_services_21))
Senior_male_count_services_22 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (M) who availed Fiber optic Internet service + Mailed Check Payment Option and stayed more than 10 months:", len(Senior_male_count_services_22))
Junior_male_count_services_11 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (M) who availed DSL Internet service + Mailed Check Payment Option and left less than 10 months:", len(Junior_male_count_services_11))
Junior_male_count_services_12 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (M) who availed Fiber Optic Internet service + Mailed Check Payment Option and left less than 10 months:", len(Junior_male_count_services_12))
Junior_male_count_services_13 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (M) who availed DSL Internet service + Mailed Check Payment Option and stayed more than 10 months:", len(Junior_male_count_services_13))
Junior_male_count_services_14 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (M) who availed Fiber optic Internet service + Mailed Check Payment Option and stayed more than 10 months:", len(Junior_male_count_services_14))
Senior_female_count_services_17 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (F) who availed DSL Internet service + Mailed Check Payment Option and left less than 10 months:", len(Senior_female_count_services_17))
Senior_female_count_services_18 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (F) who availed Fiber Optic Internet service + Mailed Check Payment Option and left less than 10 months:", len(Senior_female_count_services_18))
Senior_female_count_services_19 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (F) who availed DSL Internet service + Mailed Check Payment Option and stayed more than 10 months:", len(Senior_female_count_services_19))
Senior_female_count_services_20 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Seniors (F) who availed Fiber optic Internet service + Mailed Check Payment Option and stayed more than 10 months:", len(Senior_female_count_services_20))
Junior_female_count_services_11 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (F) who availed DSL Internet service + Mailed Check Payment Option and left less than 10 months:", len(Junior_female_count_services_11))
Junior_female_count_services_12 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (F) who availed Fiber Optic Internet service + Mailed Check Payment Option and left less than 10 months:", len(Junior_female_count_services_12))
Junior_female_count_services_13 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (F) who availed DSL Internet service + Mailed Check Payment Option and stayed more than 10 months:", len(Junior_female_count_services_13))
Junior_female_count_services_14 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['PaymentMethod'] == 'Mailed check')]
print ("Juniors (F) who availed Fiber optic Internet service + Mailed Check Payment Option and stayed more than 10 months:", len(Junior_female_count_services_14))
DSLserv_MailCheck_usedby_MaleSr = [4, 2]
DSLserv_MailCheck_usedby_FemaleSr = [6, 1]
DSLserv_MailCheck_usedby_MaleJr = [50, 12]
DSLserv_MailCheck_usedby_FemaleJr = [38, 12]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [4., 2.]
z = [6., 1.]
a = [49., 5.]
b= [38., 6.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'DSL.Ser/Mail Check used by Sr(M)')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'DSL.Ser/Mail Check used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'DSL.Ser/Mail Check used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'DSL.Ser/Mail Check used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('DSL.Service + Mail Check PM utilized')
plt.title('DSL Services + Mail Check Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C','D'))

plt.legend()
plt.show()

FibOptserv_MailCheck_usedby_MaleSr = [8, 6]
FibOptserv_MailCheck_usedby_FemaleSr = [8, 5]
FibOptserv_MailCheck_usedby_MaleJr = [22, 12]
FibOptserv_MailCheck_usedby_FemaleJr = [33, 13]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [8., 5.]
z = [8., 4.]
a = [22., 12.]
b= [33., 7.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'FibOpt.Ser/Mail Check used by Sr(M))')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'FibOpt.Ser/Mail Check used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'FibOpt.Ser/Mail Check used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'FibOpt.Ser/Mail Check used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('FibOpt services + Mailed check utilized')
plt.title('FiberOptic Services + Mailed check Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C','D'))

plt.legend()
plt.show()
Senior_male_count_services_19 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (M) who availed DSL Internet service + Bank Transfer Payment Option and left less than 10 months:", len(Senior_male_count_services_19))
Senior_male_count_services_20 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (M) who availed Fiber Optic Internet service + Bank Transfer Payment Option and left less than 10 months:", len(Senior_male_count_services_20))
Senior_male_count_services_21 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (M) who availed DSL Internet service + Bank Transfer Payment Option and stayed more than 10 months:", len(Senior_male_count_services_21))
Senior_male_count_services_22 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (M) who availed Fiber optic Internet service + Bank Transfer Payment Option and stayed more than 10 months:", len(Senior_male_count_services_22))
Junior_male_count_services_11 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (M) who availed DSL Internet service + Bank Transfer Payment Option and left less than 10 months:", len(Junior_male_count_services_11))
Junior_male_count_services_12 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (M) who availed Fiber Optic Internet service + Bank Transfer Payment Option and left less than 10 months:", len(Junior_male_count_services_12))
Junior_male_count_services_13 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (M) who availed DSL Internet service + Bank Transfer Payment Option and stayed more than 10 months:", len(Junior_male_count_services_13))
Junior_male_count_services_14 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (M) who availed Fiber optic Internet service + Bank Transfer Payment Option and stayed more than 10 months:", len(Junior_male_count_services_14))
Senior_female_count_services_17 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (F) who availed DSL Internet service + Bank Transfer Payment Option and left less than 10 months:", len(Senior_female_count_services_17))
Senior_female_count_services_18 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (F) who availed Fiber Optic Internet service + Bank Transfer Payment Option and left less than 10 months:", len(Senior_female_count_services_18))
Senior_female_count_services_19 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (F) who availed DSL Internet service + Bank Transfer Payment Option and stayed more than 10 months:", len(Senior_female_count_services_19))
Senior_female_count_services_20 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Seniors (F) who availed Fiber optic Internet service + Bank Transfer Payment Option and stayed more than 10 months:", len(Senior_female_count_services_20))
Junior_female_count_services_11 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (F) who availed DSL Internet service + Bank Transfer Payment Option and left less than 10 months:", len(Junior_female_count_services_11))
Junior_female_count_services_12 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (F) who availed Fiber Optic Internet service + Bank Transfer Payment Option and left less than 10 months:", len(Junior_female_count_services_12))
Junior_female_count_services_13 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (F) who availed DSL Internet service + Bank Transfer Payment Option and stayed more than 10 months:", len(Junior_female_count_services_13))
Junior_female_count_services_14 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['PaymentMethod'] == 'Bank transfer (automatic)')]
print ("Juniors (F) who availed Fiber optic Internet service + Bank Transfer Payment Option and stayed more than 10 months:", len(Junior_female_count_services_14))
DSLserv_BankTransfer_usedby_MaleSr = [2, 5]
DSLserv_BankTransfer_usedby_FemaleSr = [0, 2]
DSLserv_BankTransfer_usedby_MaleJr = [8, 15]
DSLserv_BankTransfer_usedby_FemaleJr = [10, 11]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [2., 5.]
z = [0., 2.]
a = [8., 15.]
b= [10., 11.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'DSL.Ser/BankTransfer used by Sr(M)')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'DSL.Ser/BankTransfer used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'DSL.Ser/BankTransfer used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'DSL.Ser/BankTransfer used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('DSL.Service + BankTransfer PM utilized')
plt.title('DSL Services + BankTransfer Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C','D'))

plt.legend()
plt.show()

FibOptserv_BankTransfer_usedby_MaleSr = [4, 16]
FibOptserv_BankTransfer_usedby_FemaleSr = [4, 20]
FibOptserv_BankTransfer_usedby_MaleJr = [16, 47]
FibOptserv_BankTransfer_usedby_FemaleJr = [30, 46]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [4., 16.]
z = [4., 20.]
a = [16., 47.]
b= [30., 46.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'FibOpt.Ser/BankTransfer used by Sr(M))')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'FibOpt.Ser/BankTransfer used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'FibOpt.Ser/BankTransfer used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'FibOpt.Ser/BankTransfer used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('FibOpt services + BankTransfer utilized')
plt.title('FiberOptic Services + BankTransfer Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C','D'))

plt.legend()
plt.show()
Senior_male_count_services_19 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (M) who availed DSL Internet service + Credit Card Transfer Payment Option and left less than 10 months:", len(Senior_male_count_services_19))
Senior_male_count_services_20 = Senior_male_count.loc[(Senior_male_count['tenure'] < 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (M) who availed Fiber Optic Internet service + Credit Card Payment Option and left less than 10 months:", len(Senior_male_count_services_20))
Senior_male_count_services_21 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'DSL') & (Senior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (M) who availed DSL Internet service + Credit Card Payment Option and stayed more than 10 months:", len(Senior_male_count_services_21))
Senior_male_count_services_22 = Senior_male_count.loc[(Senior_male_count['tenure'] > 10) & (Senior_male_count['InternetService'] == 'Fiber optic') & (Senior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (M) who availed Fiber optic Internet service + Credit Card Payment Option and stayed more than 10 months:", len(Senior_male_count_services_22))
Junior_male_count_services_11 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (M) who availed DSL Internet service + Credit Card Payment Option and left less than 10 months:", len(Junior_male_count_services_11))
Junior_male_count_services_12 = Junior_male_count.loc[(Junior_male_count['tenure'] < 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (M) who availed Fiber Optic Internet service + Credit Card Payment Option and left less than 10 months:", len(Junior_male_count_services_12))
Junior_male_count_services_13 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'DSL') & (Junior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (M) who availed DSL Internet service + Credit Card Payment Option and stayed more than 10 months:", len(Junior_male_count_services_13))
Junior_male_count_services_14 = Junior_male_count.loc[(Junior_male_count['tenure'] > 10) & (Junior_male_count['InternetService'] == 'Fiber optic') & (Junior_male_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (M) who availed Fiber optic Internet service + Credit Card Payment Option and stayed more than 10 months:", len(Junior_male_count_services_14))
Senior_female_count_services_17 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (F) who availed DSL Internet service + Credit Card Payment Option and left less than 10 months:", len(Senior_female_count_services_17))
Senior_female_count_services_18 = Senior_female_count.loc[(Senior_female_count['tenure'] < 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (F) who availed Fiber Optic Internet service + Credit Card Payment Option and left less than 10 months:", len(Senior_female_count_services_18))
Senior_female_count_services_19 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'DSL') & (Senior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (F) who availed DSL Internet service + Credit Card Payment Option and stayed more than 10 months:", len(Senior_female_count_services_19))
Senior_female_count_services_20 = Senior_female_count.loc[(Senior_female_count['tenure'] > 10) & (Senior_female_count['InternetService'] == 'Fiber optic') & (Senior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Seniors (F) who availed Fiber optic Internet service + Credit Card Payment Option and stayed more than 10 months:", len(Senior_female_count_services_20))
Junior_female_count_services_11 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (F) who availed DSL Internet service + Credit Card Payment Option and left less than 10 months:", len(Junior_female_count_services_11))
Junior_female_count_services_12 = Junior_female_count.loc[(Junior_female_count['tenure'] < 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (F) who availed Fiber Optic Internet service + Credit Card Payment Option and left less than 10 months:", len(Junior_female_count_services_12))
Junior_female_count_services_13 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'DSL') & (Junior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (F) who availed DSL Internet service + Credit Card Payment Option and stayed more than 10 months:", len(Junior_female_count_services_13))
Junior_female_count_services_14 = Junior_female_count.loc[(Junior_female_count['tenure'] > 10) & (Junior_female_count['InternetService'] == 'Fiber optic') & (Junior_female_count['PaymentMethod'] == 'Credit card (automatic)')]
print ("Juniors (F) who availed Fiber optic Internet service + Credit Card Payment Option and stayed more than 10 months:", len(Junior_female_count_services_14))
DSLserv_CreditTransfer_usedby_MaleSr = [1, 4]
DSLserv_CreditTransfer_usedby_FemaleSr = [0, 2]
DSLserv_CreditTransfer_usedby_MaleJr = [1, 6]
DSLserv_CreditTransfer_usedby_FemaleJr = [12, 21]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [1., 4.]
z = [0., 2.]
a = [1., 6.]
b= [12., 21.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'DSL.Ser/CreditTransfer used by Sr(M)')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'DSL.Ser/CreditTransfer used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'DSL.Ser/CreditTransfer used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'DSL.Ser/CreditTransfer used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('DSL.Service + CreditTransfer PM utilized')
plt.title('DSL Services + Credit Card Transfer Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C','D'))

plt.legend()
plt.show()

FibOptserv_CreditTransfer_usedby_MaleSr = [6, 16]
FibOptserv_CreditTransfer_usedby_FemaleSr = [6, 21]
FibOptserv_CreditTransfer_usedby_MaleJr = [9, 35]
FibOptserv_CreditTransfer_usedby_FemaleJr = [19, 36]

fig, ax = plt.subplots(figsize=(9, 5))
x= np.arange(2)
bar_width = 0.25

y = [6., 16.]
z = [6., 21.]
a = [9., 35.]
b= [19., 36.]

plt.bar(x+ 0.00,y,bar_width,color='b',label= 'FibOpt.Ser/CreditTransfer used by Sr(M))')
plt.bar(x+ 0.26,z,bar_width,color='g',label= 'FibOpt.Ser/CreditTransfer used by Sr(F)')
plt.bar(x+ 3.60,a,bar_width,color='y',label= 'FibOpt.Ser/CreditTransfer used by Jr(M)')
plt.bar(x+ 3.87,b,bar_width,color='k',label= 'FibOpt.Ser/CreditTransfer used by Jr(F)')

plt.xlabel('Female/Male customers')
plt.ylabel('FibOpt services + CreditTransfer utilized')
plt.title('FiberOptic Services + Credit card Transfer Payment Option used by Female/Male customers for short period and Long Period')
plt.xticks(x + bar_width, ('A','B','C','D'))

plt.legend()
plt.show()
'''
### checking distribution of MonthlyCharges and TotalCharges
from matplotlib import pyplot as plt    

fig = plt.figure(figsize=(8, 9))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

#Senior_male_count.MonthlyCharges.plot(kind = 'hist',color='c',figsize = (9,5))
n, bins, patches = ax1.hist(Senior_male_count.MonthlyCharges,color = 'r')
ax1.set_xlabel('Senior Male count)')
ax1.set_ylabel('MonthlyCharges')

n, bins, patches = ax2.hist(Senior_female_count.MonthlyCharges)
ax2.set_xlabel('Senior Female count')
ax2.set_ylabel('MonthlyCharges')

n, bins, patches = ax3.hist(Junior_male_count.MonthlyCharges,color = 'Coral')
ax3.set_xlabel('Junior Male count')
ax3.set_ylabel('MonthlyCharges')

n, bins, patches = ax4.hist(Junior_female_count.MonthlyCharges,color = 'green')
ax4.set_xlabel('Junior Female count')
ax4.set_ylabel('MonthlyCharges')
'''

'''telecom_churn_services = df2[['PhoneService','InternetService', 'Churn']]
telecom_churn_services = telecom_churn_services[(telecom_churn_services['PhoneService'] == 'Yes') & (telecom_churn_services['InternetService'] == 'DSL') or (telecom_churn_services['InternetService'] == 'Fiber optic')]    
agg = telecom_churn_services.groupby('Churn', as_index=False)[['PhoneService','InternetService']].sum()
plt.figure(figsize=(12,6))
ax = sns.barplot(y='churn', x='telecom_churn_services', data=agg, color = 'b')
ax.set_xlabel('Number of Online Services Availed', fontsize=15)
ax.set_ylabel('Average Monthly Charges',  fontsize=15)
ax.set_title('Avg Monthly Charges vs Number of Services', fontsize=20)'''
fig = plt.figure(figsize=(8, 9))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

#Senior_male_count.MonthlyCharges.plot(kind = 'hist',color='c',figsize = (9,5))
count_1 = df2['PhoneService']
y_pos_1 = df2['MonthlyCharges']

ax1.bar(count_1,y_pos_1,align='center',color = 'r')
ax1.set_xlabel('PhoneService)')
ax1.set_ylabel('MonthlyCharges')

count_2 = df2['InternetService']
y_pos_2 = df2['MonthlyCharges']
ax2.bar(count_2,y_pos_2,align='center')
ax2.set_xlabel('InternetService')
ax2.set_ylabel('MonthlyCharges')
'''import seaborn as sns

#df2['Services_1'] =df2.loc[(df2['InternetService'] == 'DSL') or (df2['InternetService'] == 'Fiber Optic')]
#df2['Services_2'] = df2.[(df2['PhoneService'] == 'Yes')]
#count of online services availed
df2['Services_1'] = (df2[['PhoneService']] == 'Yes')
# create plot
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(3)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(df2['Services_1'],df2['MonthlyCharges'],bar_width,
                 alpha=opacity,
                 color='b',
                 label='Ph.Service' + 'Monthly charges')
 
rects2 = plt.bar(df2['InternetService'],df2['MonthlyCharges'], bar_width,
                 alpha=opacity,
                 color='g',
                 label='Internet Service' + 'Monthly charges')
plt.xlabel('phone Service +Internet service)')
plt.ylabel('Monthly Charges')
plt.title('Whch component did increased the Monthly charges ?')
plt.legend()

plt.tight_layout()
plt.show()
'''
import seaborn as sns
#count of online services availed
df2['value_added_Services'] = (df2[['OnlineSecurity', 'DeviceProtection', 'StreamingMovies', 'TechSupport',
       'StreamingTV', 'OnlineBackup']] == 'Yes').sum(axis=1)
plt.figure(figsize=(12,6))
ax = sns.countplot(x='value_added_Services', hue='Churn', data=df2)
ax.set_title('Number of Services Availed by Churned customers', fontsize=20)
ax.set_ylabel('Number of churned Customers', fontsize=15)
ax.set_xlabel('Number of Online Services', fontsize=15)
#ax.set_xticklabels(labels, rotation=45 )
agg = df2.replace('Yes',1).groupby('value_added_Services', as_index=False)[['MonthlyCharges']].mean()
agg[['MonthlyCharges']] = np.round(agg[['MonthlyCharges']], 0)
plt.figure(figsize=(12,6))
ax = sns.barplot(y='MonthlyCharges', x='value_added_Services', data=agg)
ax.set_xlabel('Number of Online Services Availed', fontsize=15)
ax.set_ylabel('Average Monthly Charges',  fontsize=15)
ax.set_title('Avg Monthly Charges for services availed', fontsize=20)
from sklearn.metrics import roc_auc_score;
from sklearn.metrics import accuracy_score;
from sklearn.metrics import confusion_matrix;
from sklearn.model_selection import GridSearchCV;

df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
df[df['TotalCharges'].isna()==True] = 0
df['gender'].replace(['Male','Female'],[0,1],inplace=True)
df['Partner'].replace(['Yes','No'],[1,0],inplace=True)
df['Dependents'].replace(['Yes','No'],[1,0],inplace=True)
df['PhoneService'].replace(['Yes','No'],[1,0],inplace=True)
df['MultipleLines'].replace(['No phone service','No', 'Yes'],[0,0,1],inplace=True)
df['InternetService'].replace(['No','DSL','Fiber optic'],[0,1,2],inplace=True)
df['OnlineSecurity'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
df['OnlineBackup'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
df['DeviceProtection'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
df['TechSupport'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
df['StreamingTV'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
df['StreamingMovies'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
df['Contract'].replace(['Month-to-month', 'One year', 'Two year'],[0,1,2],inplace=True)
df['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)
df['PaymentMethod'].replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)
df['Churn'].replace(['Yes','No'],[1,0],inplace=True)

df.pop('customerID')
df.info()

import seaborn as sns
import matplotlib.pyplot as plt
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
df.pop('TotalCharges')
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.25)

train_y = train['Churn']
test_y = test['Churn']

train_x = train
train_x.pop('Churn')
test_x = test
test_x.pop('Churn')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

logisticRegr = LogisticRegression()
logisticRegr.fit(X=train_x, y=train_y)

test_y_pred = logisticRegr.predict(test_x)
confusion_matrix = confusion_matrix(test_y, test_y_pred)
print('Intercept: ' + str(logisticRegr.intercept_))
print('Regression: ' + str(logisticRegr.coef_))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr.score(test_x, test_y)))
print(classification_report(test_y, test_y_pred))

confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'Churn'), ('No churn', 'Churn'))
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)
df['Churn'].value_counts()
from sklearn.utils import resample
 
df_majority = df[df['Churn']==0]
df_minority = df[df['Churn']==1]
 
df_majority_upsampled = resample(df_majority,
replace=True,
n_samples=1869, #same number of samples as majority classe
random_state=1) #set the seed for random resampling
# Combine resampled results
df_upsampled = pd.concat([df_minority, df_majority_upsampled])
 
df_upsampled['Churn'].value_counts()
train_x_upsampled = train

train, test = train_test_split(df_upsampled, test_size = 0.25)
 
train_y_upsampled = train['Churn']
test_y_upsampled = test['Churn']
 
train_x_upsampled = train
train_x_upsampled.pop('Churn')
test_x_upsampled = test
test_x_upsampled.pop('Churn')
 
logisticRegr_balanced = LogisticRegression()
logisticRegr_balanced.fit(X=train_x_upsampled, y=train_y_upsampled)
 
test_y_pred_balanced = logisticRegr_balanced.predict(test_x_upsampled)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr_balanced.score(test_x_upsampled, test_y_upsampled)))
print(classification_report(test_y_upsampled, test_y_pred_balanced))
from sklearn import tree
# Create each decision tree (pruned and unpruned)
decisionTree_unpruned = tree.DecisionTreeClassifier()
decisionTree = tree.DecisionTreeClassifier(max_depth = 4)
 
# Fit each tree to our training data
decisionTree_unpruned = decisionTree_unpruned.fit(X=train_x, y=train_y)
decisionTree = decisionTree.fit(X=train_x, y=train_y)
test_y_pred_dt = decisionTree.predict(test_x)
print('Accuracy of unpruned decision tree classifier on training set: {:.2f}'.format(decisionTree_unpruned.score(train_x, train_y)))
print('Accuracy of unpruned decision tree classifier on test set: {:.2f}'.format(decisionTree_unpruned.score(test_x, test_y)))
print('Accuracy of decision tree classifier on training set: {:.2f}'.format(decisionTree.score(train_x, train_y)))
print('Accuracy of decision tree classifier on test set: {:.2f}'.format(decisionTree.score(test_x, test_y)))
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier()
randomForest.fit(train_x, train_y)
print('Accuracy of random forest classifier on test set: {:.2f}'.format(randomForest.score(test_x, test_y)))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
LogisticRegression(solver='lbfgs')
# Cross Validation
log_param = {'C':10**np.arange(-4, 2, .5)}
log = LogisticRegression()
reg_cv = GridSearchCV(log, log_param, cv=5, scoring='roc_auc')
reg_cv.fit(train_x, train_y)

# Selecting the best model
mod_log = LogisticRegression(C=reg_cv.best_params_['C'])
mod_log.fit(train_x, train_y)
y_pred_prob_log = mod_log.predict_proba(test_x)[:,1]
y_pred_log = mod_log.predict(test_x)
'''# Accuracy Measures
auc_log = roc_auc_score(test_y, y_pred_prob_log)
specificity_log = confusion_matrix(test_y, y_pred_log)[1,1] / (confusion_matrix(test_y, y_pred_log)[1,0] + confusion_matrix(test_y, y_pred_log)[1,1])
accuracy_log = accuracy_score(test_y, y_pred_log)
sensitivity_log = confusion_matrix(test_y, y_pred_log)[0,0]  / (confusion_matrix(test_y, y_pred_log)[0,1] + confusion_matrix(test_y, y_pred_log)[0,0])
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
tree_param = {'min_samples_leaf':np.arange(50, 300, 15), 'max_depth':np.arange(2, 12, 2)}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, tree_param, cv=5)
tree_cv.fit(train_x, train_y)
#Selecting the best model

mod_tree = DecisionTreeClassifier(max_depth=tree_cv.best_params_['max_depth'], 
                              min_samples_leaf=tree_cv.best_params_['min_samples_leaf'])
mod_tree.fit(train_x, train_y)
y_pred_prob_tree = mod_tree.predict_proba(test_x)[:,1]
y_pred_tree = mod_tree.predict(test_x)
# Accuracy Measures
from sklearn.metrics import roc_auc_score;
from sklearn.metrics import accuracy_score;
from sklearn.metrics import confusion_matrix;
auc_tree = roc_auc_score(test_y, y_pred_prob_tree)
specificity_tree = confusion_matrix(test_y, y_pred_tree)[1,1] / (confusion_matrix(test_y, y_pred_tree)[1,0] + confusion_matrix(test_y, y_pred_tree)[1,1])
accuracy_tree = accuracy_score(test_y, y_pred_tree)
sensitivity_tree = confusion_matrix(test_y, y_pred_tree)[0,0]  / (confusion_matrix(test_y, y_pred_tree)[0,1] + confusion_matrix(test_y, y_pred_tree)[0,0])
# Cross Validation

rf_param = {'max_features':np.arange(1, 5, 1)}
rf = RandomForestClassifier(n_estimators=500, max_depth=tree_cv.best_params_['max_depth'])
rf_cv = GridSearchCV(rf, rf_param, cv=5)
rf_cv.fit(train_x, train_y)

#Selecting the best model

mod_rf = RandomForestClassifier(n_estimators=500, max_depth=tree_cv.best_params_['max_depth'], 
                             max_features=rf_cv.best_params_['max_features'],
                             min_samples_leaf=tree_cv.best_params_['min_samples_leaf'])
mod_rf.fit(train_x, train_y)
y_pred_prob_rf = mod_rf.predict_proba(test_x)[:,1]
y_pred_rf = mod_rf.predict(test_x)
# Accuracy Measures
auc_rf = roc_auc_score(test_y, y_pred_prob_rf)
accuracy_rf = accuracy_score(test_y, y_pred_rf)
specificity_rf = confusion_matrix(test_y, y_pred_rf)[1,1] / (confusion_matrix(test_y, y_pred_rf)[1,0] + confusion_matrix(test_y, y_pred_rf)[1,1])
sensitivity_rf = confusion_matrix(test_y, y_pred_rf)[0,0]  / (confusion_matrix(test_y, y_pred_rf)[0,1] + confusion_matrix(test_y, y_pred_rf)[0,0])
#let x be 
x = np.arange(1, (len(y_pred_prob_rf)+1))

# Classification Trees
tree_df = pd.DataFrame()
tree_df['probability'] = [1-i for i in y_pred_prob_tree]
tree_df['actual'] = test_y
tree_df = tree_df.sort_values(by='probability', ascending=False)
tree_df['actual'] = np.where(tree_df['actual'] == 0, 1, 0)
tree_df['cumsum'] = np.cumsum(tree_df['actual'])
tree_df['index'] = x

# Random Forest
rf_df = pd.DataFrame()
rf_df['probability'] = [1-i for i in y_pred_prob_rf]
rf_df['actual'] = test_y
rf_df = rf_df.sort_values(by='probability', ascending=False)
rf_df['actual'] = np.where(rf_df['actual'] == 0, 1, 0)
rf_df['cumsum'] = np.cumsum(rf_df['actual'])
rf_df['index'] = x

# Logistic Regression
log_df = pd.DataFrame()
log_df['probability'] = y_pred_prob_log
log_df['actual'] = test_y
log_df = log_df.sort_values(by='probability', ascending=False)
log_df['cumsum'] = np.cumsum(log_df['actual'])
log_df['index'] = x
# Layout
fig = plt.figure(figsize = (15,10))
ax = plt.subplot(122)
ax1 = plt.subplot(121)

# Plot ax
ax.plot(rf_df['index'],rf_df['cumsum'], label = 'Random Forest', alpha = 0.4)
ax.plot(tree_df['index'],tree_df['cumsum'], label = 'Classification Trees', alpha = 0.4)
ax.plot(log_df['index'],log_df['cumsum'], label = 'Logistic Regression', alpha = 0.4)
ax.set_title('Zoomed In')
ax.set_ylabel('Number of Churn Observations')
ax.set_xlabel('Actual Outcome (In Ascending Order of Predicted Probability/Confidence)')

# Zoom ii
ax.set_ylim(200,100)
ax.set_xlim(1750,250)

# PLot ax1
ax1.plot(rf_df['index'],rf_df['cumsum'], label = 'Random Forest', alpha = 0.4)
ax1.plot(tree_df['index'],tree_df['cumsum'], label = 'Classification Trees', alpha = 0.4)
ax1.plot(log_df['index'],log_df['cumsum'], label = 'Logistic Regression', alpha = 0.4)
ax1.set_title('Zoomed Out')
ax1.set_ylabel('Number of Churn Observations')
ax1.set_xlabel('Actual Outcome (In Ascending Order of Predicted Probability/Confidence)')

plt.legend()
plt.tight_layout(pad = 4)
plt.show()
# Tabulate Results
result = pd.DataFrame({'Algorithm': ['Classification Tree', 
                                     'Random Forest'],#'Logistics Regression'],
                        'Accuracy':[accuracy_tree, 
                                    accuracy_rf],#,accuracy_log ],
                        'Specificity':[specificity_tree, 
                                     specificity_rf], #,specificity_log],
                      'Sensitivity':[sensitivity_tree, 
                                     sensitivity_rf]}) #,sensitivity_log]})
    
result.set_index('Algorithm').iloc[:, [0,1,2]].sort_values(['Sensitivity','Specificity'], ascending=[False, False])