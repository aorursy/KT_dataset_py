# importing neccessary libraries

import pandas as pd

import numpy as np
# import the data set and create the data frame



fb_ads=pd.read_csv('../input/fb_conversion_data.csv')

fb_ads.head(5)
# review to see if there are any missing data

missing_data=fb_ads.isnull()

missing_data.head(3)
for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print("")
# review the data types

fb_ads.dtypes
# review full information on the data set

fb_ads.info()
fb_ads.shape
fb_ads.columns
# Review the age and gender distribution briefly



age_count= fb_ads['age'].value_counts()

age_count= age_count.to_frame()
age_count.head()
# rename column to age count and index age

age_count.rename(columns={'age': 'Age Count'}, inplace=True)

age_count.index.name='Age'

age_count.head(5)
# visualize the age count

# import neccessary libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
age_count.plot(kind='bar',

              figsize=(15,10),

              color='orange')



plt.title('Age Count Frequency')

plt.xlabel('Age Count')

plt.ylabel('Frequency')



plt.show()
# similarly review the gender distribution

gender_count=fb_ads['gender'].value_counts()

gender_count=gender_count.to_frame()

gender_count.rename(columns={'gender': 'Gender Count'}, inplace=True)

gender_count.index.name='Gender Count'

gender_count.head(5)
# visualize gender count

gender_count.plot(kind='bar', figsize=(15,10), color='red')



plt.title('Gender Frequency Distribution')

plt.xlabel('Gender Count')

plt.ylabel('Frequency')



plt.show()
# Convert Age to average age and numeric variable

fb_ads['age'][fb_ads['age']=='30-34']=32

fb_ads['age'][fb_ads['age']=='35-39']=37

fb_ads['age'][fb_ads['age']=='40-44']=42

fb_ads['age'][fb_ads['age']=='45-49']=47

fb_ads[['age']]=fb_ads[['age']].astype('int')
# convert Gender to 0 and 1 values and numeric variable

fb_ads['gender'][fb_ads['gender']=='M']=0

fb_ads['gender'][fb_ads['gender']=='F']=1

fb_ads[['gender']]=fb_ads[['gender']].astype('int')
fb_ads.head(5)
fb_ads.dtypes
fb_ads['age'].unique()

fb_ads['interest'].unique()
### Review the Spent distribution on gender



plt.figure(figsize=(15,10))

sns.boxplot(x='age', y='Spent', data=fb_ads)



plt.show()
# Review the Spent distribution on Gender

plt.figure(figsize=(15,10))

sns.boxplot(x='gender', y='Spent', data=fb_ads)



plt.show()
# Review the Spent distribution on specific campaigns

plt.figure(figsize=(15,10))

sns.boxplot(x='xyz_campaign_id', y='Spent', data=fb_ads)



plt.show()
# Review the Spent distribution on interests.

plt.figure(figsize=(15,10))

sns.boxplot(x='interest', y='Spent', data=fb_ads)



plt.show()

# Creating CTR and CPC as new feautures and adding them to the dataframe



fb_ads['CTR']=(fb_ads['Clicks']/fb_ads['Impressions'])*100

fb_ads['CPC']=fb_ads['Spent']/fb_ads['Clicks']
fb_ads.head(5)
# Review of overall correlation between variables



plt.figure(figsize=(15,10))

sns.heatmap(fb_ads.corr())



plt.show()
# detail correlation between Impressions and Clicks



fb_ads[['Impressions', 'Clicks']].corr()
plt.figure(figsize=(15,10))

sns.regplot(x='Impressions', y='Clicks', data=fb_ads)

plt.ylim(0,)
# look at pearson correlation and pvalue

# import neccesary library

from scipy import stats
pearson_1, p_value_1 = stats.pearsonr(fb_ads['Impressions'], fb_ads['Clicks'])

print(pearson_1);

print(p_value_1)
# detail correlation between Clicks and Spent



fb_ads[['Spent', 'Clicks']].corr()
plt.figure(figsize=(15,10))

sns.regplot(x='Spent', y='Clicks', data=fb_ads)

plt.ylim(0,)
# detail distribution of Campaign and CTR



plt.figure(figsize=(15,10))

sns.boxplot(x='xyz_campaign_id', y='CTR', data=fb_ads)

plt.show()
# filter campaign 1178 data and review in detail



is_1178=fb_ads['xyz_campaign_id']==1178

campaign_1178=fb_ads[is_1178]

campaign_1178.head(5)
# review the correlation between Spent and CTR on Campaign 1178

campaign_1178[['Spent', 'CTR']].corr()
plt.figure(figsize=(15,10))

sns.regplot(x='CTR', y='Spent', data=campaign_1178)

plt.ylim(0,)
# select feature set X



X = fb_ads[['Impressions', 'Clicks', 'Spent', 'CTR']]

X.head(5)
# select target feature y



y=fb_ads[['Total_Conversion']]

y.head(5)
# create training and test data and split the feature set data

# import neccessary libraries

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=1)
X_train.head(5)
y_train.head(5)
X_train.isnull().sum()
y_train.isnull().sum()
# create the model with Random Forest

# import neccessary libraries

from sklearn.ensemble import RandomForestRegressor
fb_ads_model=RandomForestRegressor(random_state=1)

fb_ads_model.fit(X_train, y_train)
# create prediction with the model

fb_ads_pred=fb_ads_model.predict(X_test)
print(fb_ads_pred[0:5])
print(y_test[0:5])