# Load libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.style as style

%matplotlib inline
# Load data 

df = pd.read_excel('../input/PRNS_Activities 2013-2015.xlsx', sheet_name=None)

df = pd.concat((df[i] for i in df.keys()), sort=False)
# Explore data

df.head()
# Over 11,000 observations and 19 features.

df.shape
# A few of the features are duplicates so we'll have to consolidate and remove any remaining missing values. 

df.isnull().sum()
# WeekDays and Day(s) can be combined.

df.WeekDays.unique()
df['Day(s)'].unique()
df['WeekDays'] = df['WeekDays'].fillna(df['Day(s)'])
# So can Fee and Fees.

df['Fee'] = df['Fee'].fillna(df['Fees'])
# The only missing values are in features we consolidated so those can be removed.

df.isnull().sum()
df.pop('Fees')
df.pop('Day(s)')
# No null observations left

df.isnull().sum()
# Create a feature for year.

year = df["SeasonName"].str.split(" ", n = 1, expand = True)
year.head()
df["SeasonName"] = year[0]

df["Year"] = year[1]
df.head()
df.describe()
df.shape
df.columns
df.info()
# Checking out all the different categories of activies, Sports, Camps, and Senior programs have the most activities.

style.use('fivethirtyeight')

df['CategoryName'].value_counts().plot(kind='barh')
# Activity locations are mostly at community centers, spacing is causing some duplication we'll remove that next. 

df['Community Center'].value_counts()
# Remove spacing at the beginning or end of each string to remove duplicates in our value counting.

df['Community Center'] = df['Community Center'].str.strip()

df['Community Center'].value_counts()
# Activities mostly take place in Summer and Fall. 

df['SeasonName'].value_counts().plot(kind='barh')
# Age groups for activities seem to be for seniors and young children. 

df['Minimum Age'].value_counts().plot(kind='bar')
# Activities are divided by the 4 seasons each year. 

df.SeasonName.unique()
# Adding a revenue category to show how much money the city generates per activity

df['Revenue'] = df['NumberEnrolled'] * df['Fee']
df.head()
# Activities with 1 or less enrolled will be considered "low attendance".
df.corr()
# Copying the data frame for analysis

df_copy = df.copy()
df_copy.head()
df_copy.shape
# Group each activity by avg enrollment and revenue.

avg_enrolled = df_copy.groupby('CategoryName')['NumberEnrolled'].mean().sort_values(ascending=False)

avg_revenue = df_copy.groupby('CategoryName')['Revenue'].mean().sort_values(ascending=False)
X = avg_enrolled

Y = avg_revenue
# Once enrollment gets around 10 the revenue jumps, more popular acitvities can probably charge more fees.

plt.plot(X,Y)
# Group by average enrollment per year to see if there are any meaningful differences between years. 

avg_yearly_enrollment = df_copy.groupby('Year')['NumberEnrolled'].mean()
# Average enrollment seems steady over three years between 10-11. 

age_plot = plt.plot(avg_yearly_enrollment)

plt.xlabel("Year")

plt.ylabel("Average Activity Enrollment")

plt.yticks(range(8,14))
# Setting the target for low attendance for activities with less than 1 person enrolled. 

low_attendance = df_copy['NumberEnrolled'] <= 1
df_low = df_copy[low_attendance]
# About 1/4th of activities meet the low enrollment threshold.

df_low.shape
df_copy.shape
df_low.head()
# Seasons with low enrollment are also the seasons with the most activities. 

low_seasons = df_low['SeasonName'].value_counts().plot(kind='barh', title="Low Attendance by Season")

low_seasons.set_xlabel("Number of Low Attendance Classes")
# Total activities for each season.

df['SeasonName'].value_counts().plot(kind='barh')
# Activities that are offered the most, Camps and Sports, also have the highest number of low attendance. 

low_cat = df_low['CategoryName'].value_counts().plot(kind='barh', title="Low Attendance by Category")

low_cat.set_xlabel("Number of Low Attendance Classes")
# Total activities for each category.

df['CategoryName'].value_counts().plot(kind='barh',)
# Most activities with low attendance do get cancelled.

df_low.Status.value_counts()
df_low['Minimum Age'].unique()
# Looking at activity by age groups.

Senior = df['Minimum Age'] >= 50

Child = df['Minimum Age'] < 18

Adult = df['Minimum Age'].between(17, 51, inclusive=True)
# Adding age groups based on the minimum age for an activity.

bins = [-1, 12, 17, 49, np.inf]

names = ['Children', 'Teens', 'Adults', 'Seniors']



df['AgeRange'] = pd.cut(df['Minimum Age'], bins, labels=names)
# Most activities are for Children and Seniors .

df.AgeRange.value_counts()
# Creating attendance feature based on enrollment

bins = [-1, 1, 12, np.inf]

names = ['low', 'normal', 'high']



df['Enrollment'] = pd.cut(df['NumberEnrolled'], bins, labels=names)
df.head(10)
df.Enrollment.value_counts()
# Conversion to a Classification Task...

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
clean_data = df.copy()
# Creating a categorical feature for activities that are "low attendance".

clean_data['enrollment_label'] = (clean_data['NumberEnrolled'] < 2)*1

print(clean_data['enrollment_label'])
# Creating a target variable for the model. 

y=clean_data[['enrollment_label']].copy()
clean_data['NumberEnrolled'].head(10)
y.head(10)
clean_data.describe()
clean_data.columns
# Converting the data types so they can be used in the model. 

clean_data['BeginningDate'] = clean_data['BeginningDate'].astype(np.int64)

clean_data['EndingDate'] = clean_data['EndingDate'].astype(np.int64)

clean_data['NumberOfHours'] = clean_data['NumberOfHours'].astype(np.int64)

#clean_data['StartingTime'] = clean_data['StartingTime'].astype(np.int64)

#clean_data['EndingTime'] = clean_data['EndingTime'].astype(np.int64)

clean_data.info()
# Enrollment features that will be used in the model. 

enrollment_features = ['BeginningDate','EndingDate','Minimum Age','Maximum Age',

                       'NumberOfHours','NumberOfDates','Fee']
X = clean_data[enrollment_features].copy()
X.columns
y.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=456)
# Creating the classifier then fitting it to our training data.

enrollment_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

enrollment_classifier.fit(X_train, y_train)
type(enrollment_classifier)
# Using testing data to get the model predictions. 

predictions = enrollment_classifier.predict(X_test)
predictions[:10]
y_test['enrollment_label'][:10]
# How well does our model predict the target, 88.6% prediction accuracy. 

accuracy_score(y_true = y_test, y_pred = predictions)
enrollment_classifier
# Creating a graph of the classification tree.

from sklearn import tree

#tree.export_graphviz(enrollment_classifier, out_file='tree.dot')
clean_data.shape
clean_data.columns
# Checking proportion of low attendance activities by community center.

community_centers = clean_data['Community Center'].unique().tolist()
print(community_centers)
centers = pd.DataFrame(columns=['Community Center', 'total activities', 

                                'total low attendance','% low attendance', ])



for i in community_centers:

    

    center = i

    

    center_df = clean_data.loc[clean_data['Community Center'].str.contains(center)]

    total_centers = len(center_df.index)

    low_attend = center_df['enrollment_label'].isin([1]).sum()

    

    proportion_low = (low_attend/total_centers)*100

    

    centers = centers.append({'Community Center':i, 

                              'total activities': total_centers, 

                              'total low attendance': low_attend, 

                              '% low attendance': proportion_low }, ignore_index=True)

    
centers.sort_values(by='% low attendance', ascending=False).head(5)
# Checking proportion of low attendance activities by season. 

seasonNames = clean_data['SeasonName'].unique().tolist()
seasonNames
seasons = pd.DataFrame(columns=['Season', 'total activities', 

                                'total low attendance','% low attendance'])



for i in seasonNames:

    

    season = i

    

    season_df = clean_data.loc[clean_data['SeasonName'].str.contains(season)]

    total_seasons = len(season_df.index)

    low_attend = season_df['enrollment_label'].isin([1]).sum()

    

    proportion_low = (low_attend/total_seasons)*100

    

    seasons = seasons.append({'Season':i, 

                              'total activities': total_seasons, 

                              'total low attendance': low_attend, 

                              '% low attendance': proportion_low, }, ignore_index=True)

    
seasons.sort_values(by='% low attendance', ascending=False).head(5)
spring = clean_data.loc[clean_data['SeasonName'].str.contains('Spring 2015')]
spring['enrollment_label'].value_counts()
# Checking proportion of low attendance activities by age 

ageGroups = clean_data['AgeRange'].unique().tolist()
ageGroups
ages = pd.DataFrame(columns=['Age Groups', 'total activities', 

                                'total low attendance','% low attendance'])



for i in ageGroups:

    

    age = i

    

    age_df = clean_data.loc[clean_data['AgeRange'].str.contains(age)]

    total_ages = len(age_df.index)

    low_attend = age_df['enrollment_label'].isin([1]).sum()

    

    proportion_low = (low_attend/total_ages)*100

    

    ages = ages.append({'Age Groups':i, 

                              'total activities': total_ages, 

                              'total low attendance': low_attend, 

                              '% low attendance': proportion_low, }, ignore_index=True)

    
ages.sort_values(by='% low attendance', ascending=False).head(5)