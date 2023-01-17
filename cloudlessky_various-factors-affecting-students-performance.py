

import pandas as pd    # for data analysis 

import numpy as np     # to work with array

import matplotlib.pyplot as plt  # used for visualization

import seaborn as sns            # based on matplotlib provides more features for visualization

%matplotlib inline 



import os
StudentsPerformance = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
StudentsPerformance #lets display our dataset
StudentsPerformance.columns  #displaying columns
StudentsPerformance.head(5) #displaying the top 5 rows
StudentsPerformance.describe() #providing the statistical overview of all the numerical columns in our dataset
StudentsPerformance['race/ethnicity'].describe() #describing race/ethnicity column
StudentsPerformance.sample(5) #five random rows from the dataset, helpful to get an idea about our dataset
StudentsPerformance.isnull().sum() #counting no of null/missing values
StudentsPerformance.info() #information about count of values present and their datatype
#information about data shape

print("No. of rows: {}".format(StudentsPerformance.shape[0]))

print("No. of columns: {}".format(StudentsPerformance.shape[1]))
StudentsPerformance['percentage'] = (StudentsPerformance['math score']+StudentsPerformance['reading score'] + StudentsPerformance['writing score'])/3

StudentsPerformance['percentage']=StudentsPerformance.percentage.round(decimals=2) #rounding up the values in percentage to two decimal place
StudentsPerformance.sample(10) #viewing ten random rows
StudentsPerformance['percentage'].describe() #describing percentage column
TopTenPrcntgs = StudentsPerformance.sort_values('percentage',ascending=False).head(10)

TopTenPrcntgs #displaying ten rows with highest percentage
BottomTenPrcntgs = StudentsPerformance.sort_values('percentage', ascending=True).head(10)

BottomTenPrcntgs #displaying ten rows with lowest percentage
StudentsPerformance.gender.value_counts()     # no. of male and female students
StudentsPerformance['race/ethnicity'].value_counts()  # No. of people belonging to each race/ethnicity
StudentsPerformance.mean() #Calculating mean of each column with numeric value
gender_details = StudentsPerformance.groupby('gender')[['math score','reading score', 'writing score','percentage']].mean()

gender_details.sort_values('percentage',ascending=False)
parental_lvl_of_edu_details = StudentsPerformance.groupby('parental level of education')[['math score','writing score','reading score','percentage']].mean()

parental_lvl_of_edu_details.sort_values('percentage',ascending=False)
lunch_details = StudentsPerformance.groupby('lunch')[['math score','writing score','reading score','percentage']].mean()

lunch_details.sort_values('percentage',ascending=False)
sns.set_style("whitegrid") #set theme to whitegrid
sns.scatterplot(x='math score',y='writing score',data=StudentsPerformance, hue= 'lunch',alpha=0.6,s=50); # alpha defines opacity and s defines size of the dots, you can play change them.

plt.title("Scatter plot example"); #giving title to graph

#we didn't need to provide label for x and y axis in seaborn 
#using matplotlib to plot scatter plot

fig, ax=plt.subplots() # plt.subplots() return tuple containing figure and axes object that are stored in fig and ax.



ax.scatter(StudentsPerformance['percentage'],StudentsPerformance['race/ethnicity']);

plt.xlabel('Percentage Scored');

plt.ylabel('Race/Ethnicity');
plt.hist(StudentsPerformance['reading score'],alpha=0.4);

plt.hist(StudentsPerformance.percentage,alpha=0.4);

plt.legend(['reading score','percentage']);
StudentsPerformance['parental level of education'].value_counts().plot(kind='bar');
StudentsPerformance['race/ethnicity'].value_counts().plot(kind='pie');
sns.pairplot(StudentsPerformance, hue='gender');
sns.pairplot(StudentsPerformance,hue='lunch');
plt.hist([StudentsPerformance['reading score'],StudentsPerformance['writing score'],StudentsPerformance['math score']], stacked=True);

plt.legend(['reading score','writing score','math score']);
sns.barplot(x='race/ethnicity',y='percentage', data=StudentsPerformance);
sns.barplot(x='gender',y='percentage', data=StudentsPerformance);
sns.barplot(x='lunch',y='percentage',data=StudentsPerformance);
sns.barplot(x='test preparation course',y='percentage',data=StudentsPerformance);
plt.figure(figsize=(10,4)); #define figure size

sns.barplot(x='parental level of education',y='percentage',data=StudentsPerformance);