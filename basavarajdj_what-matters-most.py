# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb



% matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.DataFrame.from_csv('../input/HR_comma_sep.csv', index_col=None)

# Any results you write to the current directory are saved as output.
data.shape
data.head(5)
data['salary'].value_counts()
#plotting above counts 

fig, ax = plt.subplots(1,2, figsize=(17,5))

sb.countplot(y = data['salary'], ax = ax[0])

sb.countplot(y = data['left'], ax = ax[1])
low_salary = data.loc[data['salary'] == 'low', ['left']]

print(low_salary['left'].value_counts())



med_salary = data.loc[data['salary'] == 'medium', ['left']]

print(med_salary['left'].value_counts())



high_salary = data.loc[data['salary'] == 'high', ['left']]

print(high_salary['left'].value_counts())



sb.factorplot("left", data= data, col = 'salary', kind = 'count')

sb.factorplot("satisfaction_level", data = data, col = 'left', kind = 'violin', size = 6, color = 'g')
sb.factorplot("last_evaluation", data = data, col = 'left', kind = 'violin', size = 6)
sb.factorplot("number_project", data = data, col = 'left' , kind = 'count', size = 6)
sb.factorplot("average_montly_hours", data = data, col = 'left', size = 6, kind = 'violin', color = '#FFC300')
sb.factorplot("Work_accident", data = data, col = 'left', size = 6, kind = 'count')
sb.factorplot("promotion_last_5years", data = data, col = 'left', size = 6, kind = 'count')
promo_left = data.loc[data['left']==1, ['promotion_last_5years','time_spend_company']]

print(promo_left.shape)

#print(promo_left)

spend_5year = promo_left.loc[promo_left['time_spend_company']>=5, ['promotion_last_5years','time_spend_company']]

print(spend_5year.shape)



promo_true = spend_5year.loc[spend_5year['promotion_last_5years']==1,['promotion_last_5years','time_spend_company']]

print(promo_true.shape)
sb.factorplot('time_spend_company', data = data , col = 'left', kind = 'count', size = 6)
sb.factorplot('sales', data = data, col = 'left', kind = 'count', size = 6)
dept_left = data.loc[data['left']==1,['sales']]

dept_stayed = data.loc[data['left']==0,['sales']]



#print(dept_left['sales'].sort_values().value_counts(sort = False))

#print(dept_stayed['sales'].sort_values().value_counts(sort = False))



temp_df_left = {"dept": ['sales','technical','support','IT','hr','accounting','marketing','product_mng','RandD','management'],

                         "counts": [1014,697,555,273,215,204,203,198,121,91]}



temp_df_stayed = {"dept": ['sales','technical','support','IT','hr','accounting','marketing','product_mng','RandD','management'],

                         "counts": [3126,2023,1674,954,524,563,655,704,666,539]}



for x,y,z in zip(temp_df_left['counts'], temp_df_stayed['counts'],temp_df_left['dept']):

    print(str(round(x/y,2))+" : "+str(y/y)+" - "+z)

    

data_for_heatmap = data.drop(['sales','salary'],1)

print(data_for_heatmap.shape)

data_for_heatmap.head(5)
fig, ax = plt.subplots(1,1,figsize=(12,10))

sb.heatmap(data_for_heatmap.corr(), annot=True, linewidth = 2, cmap="YlGnBu", square=True, ax= ax)
data.describe()
print(data['sales'].value_counts())

encoded_sales = data['sales'].astype('category')

data['sales_cd'] = encoded_sales.cat.codes



print(data['salary'].value_counts())

encoded_salary = data['salary'].astype('category')

data['salary_cd'] = encoded_salary.cat.codes

data.head(4)
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from sklearn.model_selection import train_test_split

import graphviz

import os



decTree = DecisionTreeClassifier()



#remove categorical attributes

data_for_model = data.drop(['sales','salary'],1)

data_for_model.head(5)
#take out target variable 

target = data_for_model.pop('left')
#split the train and test data

train_X, test_X, train_Y, test_Y = train_test_split(data_for_model, target, test_size = 0.20, random_state = 42)



decTree = decTree.fit(train_X, train_Y)
decTree.score(test_X, test_Y)
#code to export decision tree to an image file



#import pydotplus



#dot_data = export_graphviz(decTree, out_file=None, feature_names= data_for_model.columns, class_names = ['Stayed','Left'])

#graph = pydotplus.graph_from_dot_data(dot_data)  

#graph.write_png("Decision_Tree.png")