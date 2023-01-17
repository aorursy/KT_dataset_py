# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="darkgrid")





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Importing dataset for analysis

dataset = pd.read_csv('../input/HR.csv')

# Let's describe the data to get a summary statistics of the dataset

dataset.describe()
dataset.info()

# We can see that there are no missing rows in the dataset
#Let's rename the columns for better readability

dataset = dataset.rename(columns = {'satisfaction_level' :'satisfaction',

                                   'last_evaluation' : 'evaluation',

                                   'number_project':'projectCount',

                                   'average_montly_hours':'avgMonthlyHours',

                                   'time_spend_company': 'yearsAtComp',

                                   'left' : 'Resigned',

                                   'promotion_last_5years':'promotion',

                                   'sales':'department'})
dataset.head()
# Let's see what variables (independent) affect our dependent variable(Resigned)

# Let's see the correlation of the variables on the Resigned variable

corr = dataset.corr()

sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values)

# From the below heatmap we can notice two observations

# 1) There is a positive correlation between projectCount,evaluation and average_monthly_hours. Therefore, 

# those who worked on more number of projects and worked for more hours got good evaluations

# 2) There is a negative correlation between satisfaction and Resigned. Therefore, we can say that people

# who were less satisfied were more prone to resign! 
# From the above graph we can see that the majority of the employees who left had either low or medium salaries

# Therefore, salary plays an important part in the employee attrition rate. 

sns.countplot(x = 'salary', hue ='Resigned',data= dataset).set_title('Employee Salary vs Attrition')
# Let's see which department in the company has the most people resigning

# We can see that Sales , Technical and Suppport has the highest number of Resignations

# Therefore, the company should concentrate on these departments and take up measures to reduce the Attrition rate

# We can also see, that management has the lowest attrition rate among all the other departments. 



sns.countplot(x = 'department', hue ='Resigned',data= dataset,palette="Set1").set_title('Department vs Attrition')

plt.xticks(rotation=-45)
# Let's check if the projectCount has any impact on the Attrition rate

# Here we can see that if the projectCount is less i.e 2 people have resigned, and if the project count is 

# extermely more (i.e 5,6 and 7) people have resigned. 

# People working on 3 and 4 projects at a time seem to be more happy and thus are less prone to resign.

# All the people working on 7 projects at a time left the company. 

# Therefore, overworked and underworked employees are prone to leave the company sooner or later!

sns.countplot(x = 'projectCount', hue ='Resigned',data= dataset,palette="Set2").set_title('ProjectCount vs Attrition')

# Let's check out the impact of evaluation of the employee on his or her decision to stay in the company

# Does a person who has received lower evaluation leave the company ? 

# From the graph , we can see that people having lower and higher evaluation have left the company

# While the people having medium evaluations between 0.5-0.8 have stayed in the company

ax=sns.kdeplot(dataset.loc[(dataset['Resigned'] == 0),'evaluation'] , color='green',shade=True,label='Stay')

ax=sns.kdeplot(dataset.loc[(dataset['Resigned'] == 1),'evaluation'] , color='black',shade=True, label='Leave')

ax.set(xlabel='', ylabel='')

plt.title('Evaluation vs Resignation')
# Does the time spend in the company working affect the employee's decision to leave or stay in the company?

ax=sns.kdeplot(dataset.loc[(dataset['Resigned'] == 0),'avgMonthlyHours'] , color='green',shade=True,label='Stay')

ax=sns.kdeplot(dataset.loc[(dataset['Resigned'] == 1),'avgMonthlyHours'] , color='black',shade=True, label='Leave')

ax.set(xlabel='', ylabel='')

plt.title('AvgMonthlyHours vs Resignation')
# Let's see if employees who have spent more number of years at the company are more likely to resign or stay

sns.countplot(x='yearsAtComp', hue = 'Resigned', data=dataset).set_title('yearAtcompany vs Resignation');

# Here we can see that people who are mostly 3-6 years of experience are mostly leaving the company.

# People with 7+ years of experience do not leave the company as they tend to get comfortable with the 

# environment
# It is very clear from the below plot that people who are less satisfied with their work tend to leave

# the company

# Employees with higher satisfaction level tend to stay 

ax=sns.kdeplot(dataset.loc[(dataset['Resigned'] == 0),'satisfaction'] , color='green',shade=True,label='Stay')

ax=sns.kdeplot(dataset.loc[(dataset['Resigned'] == 1),'satisfaction'] , color='black',shade=True, label='Leave')

ax.set(xlabel='', ylabel='')

plt.title('Satisfaction vs Resignation')
X = dataset.iloc[:,[0,1]].values
#Using the elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

	kmeans = KMeans(n_clusters = i, init ='k-means++',max_iter = 300,n_init = 10, random_state = 0)

	kmeans.fit(X)

	wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()

#Applying k-means to the dataset

kmeans = KMeans(n_clusters = 4,init ='k-means++',max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(X)

#Visualizing the clusters

plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],s = 10, c ='red',label = 'Cluster1')

plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],s = 10, c ='blue',label = 'Cluster2')

plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],s = 10, c ='green',label = 'Cluster3')

plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1],s = 10, c ='cyan',label = 'Cluster4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c ='Yellow',label = 'Centroids')

plt.title('Cluster of employees')

plt.xlabel('Satisfaction')	

plt.ylabel('Evaluation')

plt.legend()

plt.show()




