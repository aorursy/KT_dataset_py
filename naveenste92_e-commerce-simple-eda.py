#reading Required packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import *
import wordcloud
import os
#reading data
data = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')
data.head(10)
#dropping the unused colm.
data=data.drop(['Unnamed: 0'],axis=1)
print('No of columns:',data.shape[0])
print('No of rows:',data.shape[1])
data.dtypes
#check for missing values
data.isnull().values.any()
#finding the missing value count and percentage of all columns.
mss=data.isnull().sum()
columns = data.columns
percent_missing = data.isnull().sum() * 100 / len(data)
missing_value_data = pd.DataFrame({'missing_cnt':mss,
                                 'percent_missing': percent_missing})
missing_value_data
#since the missing value percentage is less in variables, we are replacing it with mean and mode.
for column in data.columns:
    data[column].fillna(data[column].mode()[0], inplace=True)
data.isnull().values.any()
#binning the Age
data['AGE'] = pd.cut(data['Age'], [0,10,20,30,40,50,60], labels=['0-10','11-20','21-30','31-40','41-50','Abv 50'])
#plot for age group.
sns.countplot(x ='AGE', data = data)
#plot for rating
sns.countplot(x = 'Rating', data = data)
#plot for division categories
sns.countplot(x = 'Division Name', data = data)
#plot for department categories
sns.countplot(x = 'Department Name', data = data)
#plot for age vs division
A=data.groupby(['AGE','Division Name'])['AGE'].count().unstack('Division Name')
A.plot(kind='bar', stacked=True)
#plot for division n department
B= data.groupby(['Division Name', 'Department Name'])['Division Name'].count().unstack('Department Name')
B.plot(kind='bar', stacked=True)
#boxplot for rating on various departments
sns.boxplot(x="Department Name", y="Rating", data=data,whis="range", palette="vlag")
# wordcloud for Title
A=data['Title'].str.cat(sep=' ')
# Create the wordcloud object
wordcloud = WordCloud(width=800, height=480, margin=0).generate(A)
 
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
# wordcloud for Review Text
A1=data['Review Text'].str.cat(sep=' ')
# Create the wordcloud object
wordcloud = WordCloud(width=800, height=480, margin=0).generate(A1)
 
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()