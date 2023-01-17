#Importing the required libraries

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import os

import seaborn as sns
multi_choice_data = pd.read_csv('../input/multipleChoiceResponses.csv')

#multi_choice_data.head() 

multi_choice_data = multi_choice_data.iloc[1:]

multi_choice_data.head(10)
sns.set(rc={'figure.figsize':(10,6)})

sns.countplot(y='Q1',data = multi_choice_data,

              order = multi_choice_data['Q1'].value_counts().index)

plt.ylabel('Gender', size = 15, fontweight='bold', labelpad = 15)

plt.xlabel('No of Repondents',size = 15, fontweight='bold', labelpad = 15)

plt.title('Gender Distribution', size = 20, color = 'blue', fontweight='bold', pad = 20)

sns.set(rc = {'figure.figsize' : (10,6)})

sns.countplot(y='Q2',data = multi_choice_data,

              order = multi_choice_data['Q2'].value_counts().index)

plt.ylabel('Age Group', size = 15, fontweight='bold', labelpad = 15)

plt.xlabel('No of Repondents', size = 15, fontweight='bold', labelpad = 15)

plt.title('Age Distribution', size = 20, color = 'blue', fontweight='bold', pad = 20)

sns.set(rc={'figure.figsize':(12,12)})

sns.countplot(y='Q3',data = multi_choice_data,

              order = multi_choice_data['Q3'].value_counts().index)

plt.ylabel('Country', size = 15, fontweight='bold', labelpad = 15)

plt.xlabel('No of Respondents', size = 15, fontweight='bold', labelpad = 15)

plt.title('Country wise frequency distribution of Respondents',  size = 20, color = 'blue', fontweight='bold', pad = 20)

Gender_Vs_Age = pd.pivot_table(multi_choice_data, values = 'Time from Start to Finish (seconds)',

                              index = ['Q2','Q1'],

                              aggfunc = len).reset_index().rename(columns = {'Time from Start to Finish (seconds)' : 'count'})



sns.set(rc = {'figure.figsize': (10,7)})

p = sns.barplot(x="Q2", y="count", hue="Q1", data=Gender_Vs_Age)

_ = plt.setp(p.get_xticklabels(), rotation= 90)  # Rotate labels

plt.title('Gender Vs Age Distribution',  size = 20, color = 'blue', fontweight='bold', pad = 20)

plt.xlabel('Age Group', size = 15, fontweight='bold', labelpad = 15)

plt.ylabel('No of Respondent', size = 15, fontweight='bold', labelpad = 15)
sns.set(rc = {'figure.figsize' : (10,6)})

sns.countplot(y='Q4',data = multi_choice_data,

              order = multi_choice_data['Q4'].value_counts().index)

plt.ylabel('Education', size = 15, fontweight='bold', labelpad = 15)

plt.xlabel('No of Repondents', size = 15, fontweight='bold', labelpad = 15)

plt.title('Education Distribution', size = 25, color = 'blue', fontweight='bold', pad = 20)



Education = pd.DataFrame(multi_choice_data['Q4'].value_counts(normalize = True)*100).reset_index().rename(columns = {'index':'Education'})

Education.head(10)
sns.set(rc = {'figure.figsize' : (10,7)})

sns.countplot(y='Q6',data = multi_choice_data,

              order = multi_choice_data['Q6'].value_counts().index)

plt.ylabel('Current Role', size = 15, fontweight='bold', labelpad = 15)

plt.xlabel('No of Repondents', size = 15, fontweight='bold', labelpad = 15)

plt.title('Different Current Roles & its Frequency Distribution', size = 15, fontweight='bold', pad = 15, color = 'blue')
sns.set(rc = {'figure.figsize' : (10,7)})

sns.countplot(y='Q7',data = multi_choice_data,

              order = multi_choice_data['Q7'].value_counts().index)

plt.ylabel('Industry Type', size = 15, fontweight='bold', labelpad = 15)

plt.xlabel('No of Repondents', size = 15, fontweight='bold', labelpad = 15)

plt.title('Different Industries where survey respondents are employed', size = 15, fontweight='bold', pad = 15, color = 'blue')
sns.set(rc = {'figure.figsize' : (10,7)})

sns.countplot(y='Q8',data = multi_choice_data,

              order = multi_choice_data['Q8'].value_counts().index)

plt.ylabel('Experience', size = 15, fontweight='bold', labelpad = 15)

plt.xlabel('No of Repondents',size = 15, fontweight='bold', labelpad = 15)

plt.title('Experience of Kagglers', size = 20, fontweight='bold', pad = 15, color = 'blue')
sns.set(rc = {'figure.figsize' : (10,7)})

sns.countplot(y='Q9',data = multi_choice_data,

              order = multi_choice_data['Q9'].value_counts().index)

plt.ylabel('Salary Brackets (Approximate $USD)', size = 15, fontweight='bold', labelpad = 15)

plt.xlabel('No of Repondents', size = 15, fontweight='bold', labelpad = 15)

plt.title('Salary yearly ($USD) of Kagglers who are in DS & ML', size = 18, fontweight='bold', pad = 15, color = 'blue')
sns.set(rc = {'figure.figsize' : (5,4)})

sns.countplot(y='Q10',data = multi_choice_data,

              order = multi_choice_data['Q10'].value_counts().index)

plt.ylabel('ML Usage', size = 15, fontweight='bold', labelpad = 15)

plt.xlabel('No of Repondents', size = 15, fontweight='bold', labelpad = 15)

plt.title('Machine Learning Usage', size = 15, fontweight='bold', pad = 15, color = 'blue')
Industry_Vs_MLusage = pd.pivot_table(multi_choice_data, values = 'Time from Start to Finish (seconds)',

                              index = ['Q7','Q10'],

                              aggfunc = len).reset_index().rename(columns = {'Time from Start to Finish (seconds)' : 'count'})



sns.set(rc = {'figure.figsize': (20,10)})

p = sns.barplot(x="Q7", y="count", hue="Q10", data=Industry_Vs_MLusage)

_ = plt.setp(p.get_xticklabels(), rotation= 90)  # Rotate labels

plt.title('Industry_Vs_ML usage', size = 15, fontweight='bold', pad = 15, color = 'blue')

plt.xlabel('Industry', size = 15, fontweight='bold', labelpad = 15)

plt.ylabel('No of Respondent', size = 15, fontweight='bold', labelpad = 15)