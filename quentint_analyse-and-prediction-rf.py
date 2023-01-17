# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/HR_comma_sep.csv')



data.head()

data.info()
#Print correlation between features

correlation = data.corr()

plt.figure(figsize=(12,6.5))

sns.heatmap(correlation, vmax=1, square=True, annot=True)
##Distribution of employes by sales

count_by_sales = pd.DataFrame(data.groupby('sales').size().rename('counts'))

y = np.arange(len(data['sales'].unique()))

plt.barh(y, count_by_sales.get_values())

plt.yticks(y, count_by_sales.index.tolist())

plt.xlabel('Number of employees')

plt.title('Number of employees by sales')
#Satisfaction of employees

plt.hist(data['satisfaction_level'], bins='auto')

plt.xlabel('Satisfaction')

plt.ylabel('Number of employees')

plt.title('Satisfaction level')

#Analyse of 4 features with high correlation



sns.pairplot(data, hue="left", vars=['number_project', 'last_evaluation', 'average_montly_hours'])

plt.show()
#Salary



salary_dummies = pd.get_dummies(data['salary'])

salary_dummies.columns = ['low', 'high', 'medium']

salary_dummies.drop(['medium'], axis=1, inplace = True)

data.drop(['salary'], axis=1, inplace=True)

data = data.join(salary_dummies)
#plot sales column

data['sales'] = data['sales'].map({'sales': 0, 'IT': 1, 'product_mng': 2, 'management' : 3, 'marketing' : 4, 'support' : 5, 'technical' : 6, 'accounting' : 7, 'RandD' : 8, 'hr' : 9})

data['sales'] = data['sales'].astype(int)







#Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

target = data["left"].values

features_forest = data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'low', 'high']]

my_forest = random_forest.fit(features_forest, target)

print(my_forest.score(features_forest, target))