# Import libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Reading the downloaded file yrbss2017.csv



dframe = pd.read_csv("../input/yrbss2017/yrbss2017.csv", na_values=' ')

dframe
# Print the first 5 rows



dframe.head()
# Import only the 13 mentioned columns



dframe = dframe[["Q1", "Q2", "Q3", "Q67", "Q6", "Q7", "Q5", "Q42", "Q32", "Q48", "Q10", "Q11", "Q28"]]

dframe.head()
# Label the columns with new names



dframe.columns = ['AGE','SEX', 'GRADE', 'SEXUAL IDENTITY', 'HEIGHT', 'WEIGHT', 'RACE/ETHINICITY', 'ALCOHOL USE per month', 'CIGARETTE USE per month', 'MARIJUANA USE per month', 'DRINKING & DRIVING', 'TEXTING & DRIVING', 'ATTEMPTED SUICIDE']

dframe.head()
# Number of rows x columns



dframe.shape
dframe.dtypes
dframe.info()
dframe.describe(include='all')
# Print the last 5 rows



dframe.tail()
# Print a random sample of 5 rows



dframe.sample(5)
# Dropping all NaN and replacing the data with the said data for each of the following columns



dframe = dframe.dropna()



age_dict = {1:12, 2.0:13, 3.0:14, 4.0:15, 5:16, 6.0:17, 7.0:18}

sex_dict = {1:'Female', 2.0:'Male'}

grade_dict = {1:'9th', 2.0:'10th', 3.0:'11th', 4.0:'12th', 5:'No Grade'}

sexual_identity_dict = {1:'Heterosexual(Straight)', 2.0:'Gay or Lesbian', 3.0:'Bisexual', 4.0:'Not Sure'}

dframe['RACE/ETHINICITY']= dframe['RACE/ETHINICITY'].str.strip()

race_dict = {'A':'American Indian or Alaska Native', 'B':'Asian', 'C':'Black or African American', 'D':'Native Hawaiian or Other Pacific Islander', 'E':'White'}

alcohol_dict = {1:'0 days', 2.0:'1 or 2 days', 3.0:'3 to 5 days', 4.0:'6 to 9 days', 5:'10 to 19 days', 6.0:'20 to 30 days', 7.0:'31 days/Whole Month'}

cigarette_dict = {1:'0 days', 2.0:'1 or 2 days', 3.0:'3 to 5 days', 4.0:'6 to 9 days', 5:'10 to 19 days', 6.0:'20 to 30 days', 7.0:'31 days/Whole Month'}

marijuana_dict = {1:'0 times', 2.0:'1 or 2 times', 3.0:'3 to 9 times', 4.0:'10 to 19 times', 5:'20 to 39 times', 6.0:'40 or more times'}

drunk_driving_dict = {1:'Do not drive', 2.0:'0 times', 3.0:'1 times', 4.0:'2 to 3 times', 5:'4 to 5 times', 6.0:'6 or more times'}

text_driving_dict = {1:'Do not drive', 2.0:'0 times', 3.0:'1 to 2 time', 4.0:'3 to 5 times', 5:'6 to 9 timess', 6.0:'10 to 19 times', 7.0:'20 to 29 times', 8.0:'All the time'}

suicide_dict = {1:'0 times', 2.0:'1 time', 3.0:'2 to 3 times', 4.0:'4 or 5 times', 5:'6 or more times'}



dframe['AGE'] = dframe["AGE"].map(age_dict)

dframe['SEX'] = dframe["SEX"].map(sex_dict)

dframe['GRADE'] = dframe["GRADE"].map(grade_dict)

dframe['SEXUAL IDENTITY'] = dframe["SEXUAL IDENTITY"].map(sexual_identity_dict)

dframe['RACE/ETHINICITY'] = dframe["RACE/ETHINICITY"].map(race_dict)

dframe['ALCOHOL USE per month'] = dframe["ALCOHOL USE per month"].map(alcohol_dict)

dframe['CIGARETTE USE per month'] = dframe["CIGARETTE USE per month"].map(cigarette_dict)

dframe['MARIJUANA USE per month'] = dframe["MARIJUANA USE per month"].map(marijuana_dict)

dframe['DRINKING & DRIVING'] = dframe["DRINKING & DRIVING"].map(drunk_driving_dict)

dframe['TEXTING & DRIVING'] = dframe["TEXTING & DRIVING"].map(text_driving_dict)

dframe['ATTEMPTED SUICIDE'] = dframe["ATTEMPTED SUICIDE"].map(suicide_dict)



dframe
# Dropping all NaN and replacing the data with the said data for each of the following columns



"""

dframe = dframe.dropna()

dframe['AGE']= dframe['AGE'].replace(1,12).replace(2,13).replace(3,14).replace(4,15).replace(5,16).replace(6,17).replace(7,18)

dframe['SEX']= dframe['SEX'].replace(1,'Female').replace(2,'Male')

dframe['GRADE']= dframe['GRADE'].replace(1,'9th').replace(2,'10th').replace(3,'11th').replace(4,'12th').replace(5,'No Grade')

dframe['SEXUAL IDENTITY']= dframe['SEXUAL IDENTITY'].replace(1,'Heterosexual(Straight)').replace(2,'Gay or Lesbian').replace(3,'Bisexual').replace(4,'Not Sure')

dframe['RACE/ETHINICITY']= dframe['RACE/ETHINICITY'].str.strip()

dframe['RACE/ETHINICITY']= dframe['RACE/ETHINICITY'].replace('A','American Indian or Alaska Native').replace('B','Asian').replace('C','Black or African American').replace('D','Native Hawaiian or Other Pacific Islander').replace('E','White')

dframe['ALCOHOL USE per month']= dframe['ALCOHOL USE per month'].replace(1,'0 days').replace(2,'1 or 2 days').replace(3,'3 to 5 days').replace(4,'6 to 9 days').replace(5,'10 to 19 days').replace(6,'20 to 30 days').replace(7,'31 days/Whole Month')

dframe['CIGARETTE USE per month']= dframe['CIGARETTE USE per month'].replace(1,'0 days').replace(2,'1 or 2 days').replace(3,'3 to 5 days').replace(4,'6 to 9 days').replace(5,'10 to 19 days').replace(6,'20 to 30 days').replace(7,'31 days/Whole Month')

dframe['MARIJUANA USE per month']= dframe['MARIJUANA USE per month'].replace(1,'0 times').replace(2,'1 or 2 times').replace(3,'3 to 9 times').replace(4,'10 to 19 times').replace(5,'20 to 39 times').replace(6,'40 or more times')

dframe['DRINKING & DRIVING']= dframe['DRINKING & DRIVING'].replace(1,'Do not drive').replace(2,'0 times').replace(3,'1 time').replace(4,'2 to 3 times').replace(5,'4 to 5 times').replace(6,'6 or more times')

dframe['TEXTING & DRIVING']= dframe['TEXTING & DRIVING'].replace(1,'Do not drive').replace(2,'0 times').replace(3,'1 to 2 time').replace(4,'3 to 5 times').replace(5,'6 to 9 times').replace(6,'10 to 19 times').replace(7,'20 to 29 times').replace(8,'All the time')

dframe['ATTEMPTED SUICIDE']= dframe['ATTEMPTED SUICIDE'].replace(1,'0 times').replace(2,'1 time').replace(3,'2 or 3 times').replace(4,'4 or 5 times').replace(5,'6 or more times')

dframe"""
# Import library



import seaborn as sns
# since the data is a survey collection of answers, a count plot to get the total count of variables



sns.countplot(dframe['AGE'])
sns.countplot(dframe['SEX'])
# Analysis of Cigarette use in comaprison to Age



fig,ax=plt.subplots(figsize=(12,6))

sns.countplot(x='CIGARETTE USE per month', hue='AGE', data=dframe, palette=sns.cubehelix_palette(8))
# Since the above graph shows that 17 year olds consume the most amount of cigarette

# the below graph shows the cigarette consumption of the 17 year olds specifically



fig,ax=plt.subplots(figsize=(12,6))

df = dframe.loc[dframe['AGE']==17]

sns.countplot(df['CIGARETTE USE per month'])
# Analysis of Marijuana use in comaprison to Age



fig,ax=plt.subplots(figsize=(12,6))

sns.countplot(x='MARIJUANA USE per month', hue='AGE', data=dframe, palette="Paired")
# Since the above graph shows that 17 year olds consume the most amount of Marijuana

# the below graph shows the alcohol consumption of the 17 year olds specifically



fig,ax=plt.subplots(figsize=(12,6))

df = dframe.loc[dframe['AGE']==17]

sns.countplot(df['MARIJUANA USE per month'])
# Analysis of Alcohol use in comaprison to Age



fig,ax=plt.subplots(figsize=(12,6))

sns.countplot(x='ALCOHOL USE per month', hue='AGE', data=dframe, palette="Greens_d")
# Since the above graph shows that 17 year olds consume the most amount of alcohol

# the below graph shows the alcohol consumption of the 17 year olds specifically



fig,ax=plt.subplots(figsize=(12,6))

df = dframe.loc[dframe['AGE']==17]

sns.countplot(df['ALCOHOL USE per month'])
# Analysis of the risk of Texting & Driving by Age



fig,ax=plt.subplots(figsize=(12,6))

sns.countplot(x='TEXTING & DRIVING', hue='AGE', data=dframe, palette=sns.color_palette("BrBG", 7))
# Since the above graph shows that 17 year olds mostly text and drive

# the below graph shows the analysis of specifically the 17 year olds



fig,ax=plt.subplots(figsize=(12,6))

df = dframe.loc[dframe['AGE']==17]

sns.countplot(df['TEXTING & DRIVING'])
# Analysis of the risk of Drinking & Driving by Age



fig,ax=plt.subplots(figsize=(12,6))

sns.countplot(x='DRINKING & DRIVING', hue='AGE', data=dframe, palette=sns.diverging_palette(10, 220, sep=80, n=7))
# Analysis of Attempted Suicide by Age



fig,ax=plt.subplots(figsize=(12,6))

sns.countplot(x='ATTEMPTED SUICIDE', hue='AGE', data=dframe, palette=sns.color_palette("hls", 8))
# Since the above graph shows that 16 year olds are the ones that have attempted suicide the maximum number of times

# the below graph shows the count/analysis of specifically the 16 year olds



fig,ax=plt.subplots(figsize=(12,6))

df = dframe.loc[dframe['AGE']==16]

sns.countplot(df['ATTEMPTED SUICIDE'])
# Analysis of Height vs Weight 



fig,ax=plt.subplots(1,2,figsize=(12,5))

ax[0].scatter(dframe['WEIGHT'],dframe['HEIGHT'])

ax[0].set_xlabel('Weight')

ax[0].set_ylabel('Height')

sns.kdeplot(dframe['WEIGHT'],dframe['HEIGHT'], cmap=sns.cubehelix_palette(light=1, as_cmap=True), shade=True)

plt.title('Height vs Weight')