# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import sklearn

import warnings

warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"] = [10,5]
# Import data set

dataset = pd.read_csv('../input/HR_comma_sep.csv')
# Determine number of rows and columns

dataset.shape
# View first few rows of data

dataset.head()
# Data types

dataset.info()
# Statistical Summary

dataset.describe().transpose()
# Convert to categorical

categories = ['sales','salary','Work_accident','left','promotion_last_5years']

for cat in categories:

    dataset[cat] = dataset[cat].astype('category')

# Confirm changes

dataset.info()
sns.set_style('whitegrid')

plt.figure(figsize = (10,5))

sns.countplot(dataset['left'],palette = ['grey','red'], alpha =.80).set_title('Quit vs Did Not Quit')

plt.show()
print('Employees That Left:', len(dataset[dataset['left']==1]))

print('Employees That Have Not Left:', len(dataset[dataset['left']==0]))
dataset.describe().transpose()
# Identify numeric features

dataset.select_dtypes(['float64','int64']).columns
# Subplots of Numeric Variables

sns.set_style('darkgrid')

fig = plt.figure(figsize = (16,10))



ax1 = fig.add_subplot(321)

ax1.hist(dataset['satisfaction_level'], bins = 15,color = 'teal',edgecolor= 'black',alpha = .70)

ax1.set_title('Satisfaction Level All Employees')



ax2 = fig.add_subplot(323)

ax2.hist(dataset['last_evaluation'], bins = 15,color = 'teal',edgecolor= 'black',alpha = .70)

ax2.set_title('Last Evaluation of All Employees')



ax3 = fig.add_subplot(325)

ax3.hist(dataset['average_montly_hours'], bins = 15,color = 'teal',edgecolor= 'black',alpha = .70)

ax3.set_title('Average Monthly Hours of All Employees')



ax4 = fig.add_subplot(222)

ax4.hist(dataset['number_project'], bins = 15, color = 'teal',edgecolor= 'black', alpha = .70)

ax4.set_title('Number of Projects of All Employees')



ax5 = fig.add_subplot(224)

ax5.hist(dataset['time_spend_company'], bins = 15,color = 'teal', edgecolor= 'black', alpha = .70)

ax5.set_title('Time Spent With Company of All Employees')



plt.show()
#sns.pairplot(dataset, vars = ['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours', 'time_spend_company'], palette = 'viridis')



# Histogram

#dataset.groupby('left').hist()



# Create facet grid of histogram

# p1 = sns.FacetGrid(data = dataset, col = 'left')

# p1.map(plt.hist, 'satisfaction_level')



# Subplots

# sns.set_style('darkgrid') # pre-defined options are white, whitegrid, dark, darkgrid, ticks so need to pass keyword argument

# f, axes = plt.subplots(3,2, figsize = (15,15))



# axes[0,0].hist(dataset[dataset['left'] ==0].satisfaction_level, bins = 15, label ='Not left', alpha = .50,edgecolor= 'black')

# axes[0,0].hist(dataset[dataset['left']==1].satisfaction_level, bins = 15, label = 'Left', alpha = .50, edgecolor = 'black')

# axes[0,0].set_title('Satisfaction Level of Employees that Left vs Did Not Leave')

# axes[0,0].legend()



# axes[0,1].hist(dataset[dataset['left']==0].last_evaluation, bins = 15, label = 'Not Left', alpha = .50, edgecolor ='black')

# axes[0,1].hist(dataset[dataset['left']==1].last_evaluation, bins = 15, label = 'Left', alpha = .50, edgecolor = 'black')

# axes[0,1].set_title('Last Evaluation of Employees that Left vs Did Not Leave')

# axes[0,1].legend()



# axes[1,0].hist(dataset[dataset['left']==0].number_project, bins = 15, label = 'Not Left', alpha = .50, edgecolor = 'black')

# axes[1,0].hist(dataset[dataset['left']==1].number_project, bins = 15, label = 'Left', alpha = .50, edgecolor = 'black')

# axes[1,0].set_title('Number of Projects by Employees that Left vs Did Not Leave')

# axes[1,0].legend()



# axes[1,1].hist(dataset[dataset['left']==0].average_montly_hours, bins = 15, label = 'Not Left', alpha = .50, edgecolor = 'black')

# axes[1,1].hist(dataset[dataset['left']==1].average_montly_hours, bins= 15, label = 'Left', alpha = .50, edgecolor = 'black')

# axes[1,1].set_title('Average Monthly Hours of Employees that Left vs Did Not Leave')

# axes[1,1].legend()



# axes[2,0].hist(dataset[dataset['left']==0].time_spend_company, bins = 15, label = 'Not Left', alpha = .50, edgecolor = 'black')

# axes[2,0].hist(dataset[dataset['left']==1].time_spend_company, bins = 15, label = 'Not Left', alpha = .50, edgecolor = 'black')

# axes[2,0].set_title('Time At Company of Employees that Left vs Did Not Leave')

# plt.show()
# Statistical summary of employees that did not quit

dataset[dataset['left']==0].describe().transpose().head()
# Stat summary of employees that quit

dataset[dataset['left']== 1].describe().transpose().head()
# Subplots of Numeric Features

sns.set_style('darkgrid')

fig = plt.figure(figsize = (16,10))



ax1 = fig.add_subplot(321)

ax1.hist(dataset[dataset['left'] ==0].satisfaction_level, bins = 15, label ='Not Quit', alpha = .50,edgecolor= 'black',color ='grey')

ax1.hist(dataset[dataset['left']==1].satisfaction_level, bins = 15, label = 'Quit', alpha = .50, edgecolor = 'black',color = 'red')

ax1.set_title('Satisfaction Level of Employees that Quit vs Did Not Quit')

ax1.legend(loc = 'upper left')



ax2 = fig.add_subplot(323)

ax2.hist(dataset[dataset['left']==0].last_evaluation, bins = 15, label = 'Not Quit', alpha = .50, edgecolor ='black', color = 'grey')

ax2.hist(dataset[dataset['left']==1].last_evaluation, bins = 15, label = 'Quit', alpha = .50, edgecolor = 'black',color ='red')

ax2.set_title('Last Evaluation of Employees that Quit vs Not Quit')

ax2.legend(loc = 'upper left')



ax3 = fig.add_subplot(325)

ax3.hist(dataset[dataset['left']==0].average_montly_hours, bins = 15, label = 'Not Quit', alpha = .50, edgecolor = 'black', color = 'grey')

ax3.hist(dataset[dataset['left']==1].average_montly_hours, bins= 15, label = 'Quit', alpha = .50, edgecolor = 'black', color ='red')

ax3.set_title('Average Monthly Hours of Employees that Quit vs Did Not Quit')

ax3.legend(loc = 'upper left')



ax4 = fig.add_subplot(222)

ax4.hist(dataset[dataset['left']==0].number_project, bins = 15, label = 'Not Quit', alpha = .50, edgecolor = 'black', color = 'grey')

ax4.hist(dataset[dataset['left']==1].number_project, bins = 15, label = 'Quit', alpha = .50, edgecolor = 'black', color = 'red')

ax4.set_title('Number of Projects by Employees that Quit vs Not Quit')

ax4.legend(loc = 'upper right')



ax5 = fig.add_subplot(224)

ax5.hist(dataset[dataset['left']==0].time_spend_company, bins = 15, label = 'Not Quit', alpha = .50, edgecolor = 'black', color = 'grey')

ax5.hist(dataset[dataset['left']==1].time_spend_company, bins = 15, label = 'Quit', alpha = .50, edgecolor = 'black', color = 'red')

ax5.set_title('Time At Company of Employees that Quit vs Not Quit')

ax5.legend(loc = 'upper right')

plt.show()
# Identify categorical features  

dataset.select_dtypes(['category']).columns
accident = dataset.groupby(['Work_accident','left']).Work_accident.count().unstack()

p1 = accident.plot(kind = 'bar', stacked = True, 

                   title = 'Work Accidents Of Employees That Quit vs Did Not Quit', 

                   color = ['grey','red'], alpha = .70)

p1.set_xlabel('Work Accident')

p1.set_ylabel('# Employees')

p1.legend(['Not Quit','Quit'])

plt.show()
promotion = dataset.groupby(['promotion_last_5years','left']).promotion_last_5years.count().unstack()

p2 = promotion.plot(kind = 'bar', stacked = True, 

                    title = 'Promotions Of Employees That Quit vs Did Not Quit', 

                    color = ['grey','red'], alpha = .70)

p2.set_xlabel('Promotion Last 5 years')

p2.set_ylabel('# Employees')

p2.legend(['Not Quit','Quit'])

plt.show()
dataset['sales'].unique()
# Abbreviate sales categories

dataset['sales'].unique()

dataset['sales'] = dataset['sales'].map({'sales': 'S','accounting':'ACC','hr':'HR','technical':'TECH', 

                                         'support':'SUP', 'management':'MGMT', 'IT':'IT','product_mng':'PM',

                                         'marketing':'MKT', 'RandD':'RD'})
salesdept = dataset.groupby(['sales','left']).sales.count().unstack()

p3 = salesdept.plot(kind = 'bar', stacked = True, 

                    title = 'Departments of Employees that Quit vs Did Not Quit', 

                    color = ['grey','red'], alpha = .70)

p3.set_xlabel('Department')

p3.set_ylabel('# Employees')

p2.legend(['Not Quit', 'Quit'])

plt.show()
salary = dataset.groupby(['salary','left']).salary.count().unstack()

p4 = salary.plot(kind = 'bar', stacked = True,

                 title = 'Salary Level of Employees that Quit vs Did Not Quit', 

                 color = ['grey','red'], alpha = .70)

p4.set_xlabel('Salary Level')

p4.set_ylabel('# Employees')

plt.show()
# Identify categorical features to be converted into 'dummy' or indicator variables

dataset.select_dtypes(['category']).columns
# Convert dummies

sales = pd.get_dummies(dataset['sales'], drop_first = True) #drop first prevents multi-collinearity

salary = pd.get_dummies(dataset['salary'], drop_first = True)
# Add new dummy columns to data frame

dataset = pd.concat([dataset,salary,sales],axis = 1)

dataset.head().transpose()



# Drop unnecessary columns

dataset = dataset.drop(['sales','salary'], axis = 1)

dataset.head()
# Split

# Create matrix of features

X = dataset.drop('left', axis = 1)



# Create dependent variable vector

y = dataset['left']



# Use x and y variables to split the data into training and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 101)
# Fit

# Import model

from sklearn.linear_model import LogisticRegression



# Create instance of model

classifier = LogisticRegression()



# Fit to training set

classifier.fit(X_train,y_train) 

# Predict

y_pred = classifier.predict(X_test)
# Score It

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score



# Create confusion matrix

print(confusion_matrix(y_test,y_pred))  



# Create classification report

print(classification_report(y_test,y_pred))



# Accuracy score

print('Accuracy',accuracy_score(y_test, y_pred)*100,'%')

  