import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
hr_dataSet = pd.read_csv('../input/HR_comma_sep.csv')
hr_dataSet.head(10)
print('Display the number of features associated with the HR Dataframe: {} \n'.format(hr_dataSet.columns.value_counts().sum()))



print('Display the features associated with HR Dataframe:')

print('satisfaction_level \nlast_evaluation \nnumber_project \naverage_montly_hours \ntime_spend_company \nWork_accident \nleft \npromotion_last_5years \nsales \nsalary')

print('\n')



print('Check the dimensionality of HR Dataframe: {}\n'.format(hr_dataSet.shape))

print('Display each missing values for all features:')

print(hr_dataSet.isnull().sum())

print('\n')



print('Display each data type for all the above features:')

print(hr_dataSet.dtypes)

print('\n')



print('Display the number distinct observations for each feature:')

print(hr_dataSet.nunique())

print('\n')
hr_dataSet.columns
hr_dataSet.reindex(columns=['left','satisfaction_level', 'last_evaluation', 'number_project',

       'average_montly_hours', 'time_spend_company', 'Work_accident', 

       'promotion_last_5years', 'sales', 'salary']).head()
hr_dictionary = pd.DataFrame(data=hr_dataSet.nunique(), columns=['Unique Values'])

hr_dictionary['Missing Values'] = hr_dataSet.isnull().sum()

hr_dictionary['Data Types'] = hr_dataSet.dtypes

def data_type_info(x):

    if x <=10:

        return  'Categorical'

    else:

        return'Continuous'



hr_dictionary['Data_Type_Info'] = hr_dictionary['Unique Values'].map(data_type_info)
hr_dictionary
hr_dataSet.describe()
hr_dataSet.describe(include=['object'])
print('Display normalized values for salary:')

print(hr_dataSet.salary.value_counts(normalize=True)*100)

print('\n')



print('Display normalized values for Sales:')

print(hr_dataSet.sales.value_counts(normalize=True)*100)

print('\n')



print('Display normalized values for promotion in last five years(Not Promoted = 0, Promoted = 1):')

print(hr_dataSet.promotion_last_5years.value_counts(normalize=True)*100)

print('\n')



print('Display normalized values for people who stayed and left(stayed = 0,left = 1):')

print(hr_dataSet.left.value_counts(normalize=True)*100)

print('\n')



hr_dataSet_time_spent = hr_dataSet.time_spend_company.value_counts(normalize=True,sort=True)*100

hr_dataSet_time_spent.sort_index()

print('Display normalized values for the time spent at the company:')

print(hr_dataSet_time_spent.sort_index())

print('\n')



hr_dataSet_number_projects = hr_dataSet.number_project.value_counts(normalize=True)*100

hr_dataSet_number_projects.sort_index()

print('Display normalized values for number of projects:')

print(hr_dataSet_number_projects.sort_index())

print('\n')
fig4,axes4 = plt.subplots(nrows=3,ncols=3, figsize=(16,12))



sns.boxplot(hr_dataSet['satisfaction_level'],color='red', ax=axes4[0,0] )

sns.boxplot(hr_dataSet['average_montly_hours'],color='white',ax=axes4[0,1])

sns.boxplot(hr_dataSet['last_evaluation'],color='blue',ax=axes4[0,2])

sns.violinplot(y='satisfaction_level',color='red', data=hr_dataSet,ax=axes4[1,0])

sns.violinplot(y='average_montly_hours',color='white', data=hr_dataSet,ax=axes4[1,1])

sns.violinplot(y='last_evaluation',color='blue', data=hr_dataSet,ax=axes4[1,2]);

sns.distplot(hr_dataSet['satisfaction_level'],color='red', ax=axes4[2,0] )

sns.distplot(hr_dataSet['average_montly_hours'],color='white',ax=axes4[2,1])

sns.distplot(hr_dataSet['last_evaluation'],color='blue',ax=axes4[2,2]);
#left left and stay

hr_dataSet_left = hr_dataSet[hr_dataSet['left']==1]

hr_dataSet_stay = hr_dataSet[hr_dataSet['left']==0]
#renamed column for legend

hr_dataSet_left_legend_sat=hr_dataSet_left.rename_axis({'satisfaction_level':'satisfaction_level_left'},axis=1)

hr_dataSet_stay_legend_sat=hr_dataSet_stay.rename_axis({'satisfaction_level':'satisfaction_level_stay'},axis=1)



hr_dataSet_left_legend_amh = hr_dataSet_left.rename_axis({'average_montly_hours':'average_montly_hours_left'},axis=1)

hr_dataSet_stay_legend_amh = hr_dataSet_stay.rename_axis({'average_montly_hours':'average_montly_hours_stay'},axis=1)



hr_dataSet_left_legend_LE = hr_dataSet_left.rename_axis({'last_evaluation':'last_evaluation_left'},axis=1)

hr_dataSet_stay_legend_LE = hr_dataSet_stay.rename_axis({'last_evaluation':'last_evaluation_stay'},axis=1)
hr_dataSet_left_legend_sat.satisfaction_level_left.plot(kind= 'kde',legend=True,linestyle ='--',linewidth=5,figsize=(16,6));

hr_dataSet_stay_legend_sat.satisfaction_level_stay.plot(kind='kde',legend=True,linestyle ='-.',linewidth=5,figsize=(16,6));

plt.title('Distribution Comparing Satisfaction levels for Individuals who left and stayed',fontsize=(18));

plt.xlabel('Satisfaction levels',fontsize=14);
#satisfaction_level low and high

hr_dataSet_sat_low = hr_dataSet[hr_dataSet['satisfaction_level']<.5]

hr_dataSet_sat_high =hr_dataSet[hr_dataSet['satisfaction_level'] >.5]
hr_dataSet_sat_high.left.value_counts(normalize=True)
hr_dataSet_sat_low.left.value_counts(normalize=True)
hr_dataSet_left_legend_amh.average_montly_hours_left.plot(kind= 'kde',legend=True,linestyle ='--',linewidth=5,figsize=(16,6));

hr_dataSet_stay_legend_amh.average_montly_hours_stay.plot(kind='kde',legend=True,linestyle ='-.',linewidth=5,figsize=(16,6));

plt.title('Distribution Comparing Average Monthly Hours for Individuals who left and stayed',fontsize=(18));

plt.xlabel('Average Monthly Hours Worked',fontsize=14);
hr_dataSet_left_legend_LE.last_evaluation_left.plot(kind= 'kde',legend=True,linestyle ='--',linewidth=5,figsize=(16,6));

hr_dataSet_stay_legend_LE.last_evaluation_stay.plot(kind= 'kde',legend=True,linestyle ='--',linewidth=5,figsize=(16,6));

plt.title('Distribution Comparing Last Evaluations for Individuals who left and stayed',fontsize=(18));

plt.xlabel('Last Evaluation',fontsize=14);
left_salary = pd.crosstab(columns=hr_dataSet['salary'], index=hr_dataSet['left'],normalize='columns')*100

left_salary
sns.heatmap(left_salary);
left_sales = pd.crosstab(columns=hr_dataSet['sales'], index=hr_dataSet['left'],normalize='columns').T

left_sales
fig5,axes4 =plt.subplots(nrows=2, ncols=2, figsize=(16,10))



sns.countplot(x='time_spend_company', hue='left', data=hr_dataSet, ax=axes4[0,0])

sns.countplot(x='left', hue='salary', data=hr_dataSet,ax=axes4[0,1])

sns.countplot(x='promotion_last_5years', hue='left', data=hr_dataSet,ax=axes4[1,0])

sns.countplot('Work_accident',hue='left',data=hr_dataSet,ax=axes4[1,1]);
fig1,axes2 = plt.subplots(nrows=2, ncols=3, figsize=(15,13))



sns.boxplot(x=hr_dataSet.left,y=hr_dataSet.satisfaction_level,data=hr_dataSet, ax=axes2[0,0])

sns.boxplot(x=hr_dataSet.left,y=hr_dataSet.time_spend_company,data=hr_dataSet, ax = axes2[0,1])

sns.boxplot(x=hr_dataSet.left,y=hr_dataSet.average_montly_hours,data=hr_dataSet, ax = axes2[0,2])

sns.boxplot(x=hr_dataSet.left,y=hr_dataSet.last_evaluation,data=hr_dataSet, ax = axes2[1,2])

sns.boxplot(x=hr_dataSet.left,y=hr_dataSet.number_project,data=hr_dataSet,ax = axes2[1,1]);

sns.violinplot(x=hr_dataSet.left,y=hr_dataSet.satisfaction_level,data=hr_dataSet, ax=axes2[1,0]);

hr_dataSet_num =hr_dataSet.copy()



def salary_object_to_int64(x):

    if x == 'low':

        return 1

    elif x == 'medium':

        return 2

    else :

        return 3

    

hr_dataSet_num['salary'] = hr_dataSet_num.salary.map(salary_object_to_int64)



hr_dataSet_num['salary'] = hr_dataSet_num.salary.astype(dtype='int64')

hr_dataSet_num.dtypes
cmap = sns.diverging_palette(100, 115, as_cmap=True)



hr_dataSet_corr = hr_dataSet_num.corr()

sns.heatmap(hr_dataSet_corr,annot=True,cmap=cmap);
sns.lmplot(x='satisfaction_level',y='average_montly_hours',data=hr_dataSet,col='left',fit_reg=False,sharey=True);
sns.lmplot(x='number_project',y='average_montly_hours',data=hr_dataSet,col='left',fit_reg=False,sharey=True);
sns.lmplot(x='last_evaluation',y='average_montly_hours',data=hr_dataSet,col='left',fit_reg=False,sharey=True);
hr_dataSet.left.value_counts(normalize=True)
#target value

y = hr_dataSet['left'].values  

y.ndim
#features

X1= hr_dataSet.drop(labels='left',axis=1)

X2=pd.get_dummies(X1)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
lr = LogisticRegression()
lr.fit(X2,y)
new_scores = cross_val_score(lr,X2,y,cv=5)

print(new_scores)

new_scores.mean()
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
dtc = tree.DecisionTreeClassifier(max_depth=5)
dtc = dtc.fit(X2, y)
scores = cross_val_score(dtc, X2, y, cv=5)

print(scores)

scores.mean()
importance2 = pd.DataFrame({'feature':X2.columns,'importance':dtc.feature_importances_})

importance2.set_index('feature',inplace=True)

importance2.sort_values(by='importance').plot(kind='barh',figsize=(14,13));



plt.title('Feature Importance for Decision Tree', fontsize=20);