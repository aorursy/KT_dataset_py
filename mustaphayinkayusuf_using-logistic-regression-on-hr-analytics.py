# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

import seaborn as sns
df = pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')

df.head()
df.count()
df.isna().sum()
#The minimum and maximum satisaction level

df.satisfaction_level.min(), df.satisfaction_level.max()
#Status of employees with satisfaction level of 1.0



df.loc[df.satisfaction_level == 1.0, 'left'].value_counts()
#Status of employees with satisfaction level of 0.9

df.loc[df.satisfaction_level == 0.9, 'left'].value_counts()
#Barcharts showing the status of employees with satisfaction level less than or equal to 0.5 and greater than 0.5 in percentage

plt.subplot(1,2,1)

df.loc[df.satisfaction_level <=  0.5, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar',

                                                                                              grid= True)

plt.title('satisfacttion level <= 0.5')

plt.xlabel('left')

plt.ylabel('%age')

plt.subplot(1,2,2)

df.loc[df.satisfaction_level > 0.5, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar',

                                                                                            grid = True)

plt.title('satisfaction level >0.5')

plt.xlabel('left')

plt.ylabel('%age')

plt.tight_layout()

plt.show()
#Minimum and maxamim evaluation

df.last_evaluation.min(), df.last_evaluation.max()
#Status of employees with last evaluation of 1.0

df.loc[df.last_evaluation == 1.0, 'left'].value_counts()
#Status of employees with last evaluation of 0.36

df.loc[df.last_evaluation == 0.36, 'left'].value_counts()
df.last_evaluation.median()
df.loc[df.last_evaluation >=0.72, 'left'].value_counts()
df.loc[df.last_evaluation < 0.72, 'left'].value_counts()
#Barcharts showing the status of employees with last_evaluation less than and greater than or equal to 0.72 in percentage

plt.subplot(1,2,1)

df.loc[df.last_evaluation < 0.72, 'left'].value_counts(normalize = True).sort_index().plot(kind ='bar',

                                                                                          grid = True)

plt.title('Evaluation less than 0.72')

plt.xlabel('Left')

plt.ylabel('%age')

plt.subplot(1,2,2)

df.loc[df.last_evaluation >= 0.72, 'left'].value_counts(normalize = True).sort_index().plot(kind ='bar', 

                                                                                          grid = True)

plt.title('Evaluation more than 0.72')

plt.xlabel('Left')

plt.ylabel('%age')

plt.tight_layout()

plt.show()
#The minimum and maximum number of projects

df.number_project.min(), df.number_project.max()
#Status of employees with two(2) projects

df.loc[df.number_project ==2, 'left'].value_counts()
#Status of employees with seven(7) projects

df.loc[df.number_project ==7, 'left'].value_counts()
#Barcharts showing status of employees with less than or equal to four(4) projects  and more than four(4) projects

plt.subplot(1,2,1)

df.loc[df.number_project <=4, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar', grid = True)

plt.title('Projects <=4')

plt.xlabel('Left')

plt.ylabel('%age')

plt.subplot(1,2,2)

df.loc[df.number_project >4, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar', grid = True)

plt.title('Projects > 4')

plt.xlabel('Left')

plt.ylabel('%age')

plt.tight_layout()

plt.show()
#The minimun and maximum average monthly hours

df.average_montly_hours.min(), df.average_montly_hours.max()
#The mean of avarage monthly hours

df.average_montly_hours.mean()
#Status of employees who work for 310 hours

df.loc[df.average_montly_hours == 310, 'left'].value_counts()
#Status of employees who work for 96 hours

df.loc[df.average_montly_hours == 96, 'left'].value_counts()
df.loc[df.average_montly_hours >= 160, 'left'].value_counts()
df.loc[df.average_montly_hours < 160, 'left'].value_counts()
#Barcharts showing ths status of workers with less than 160 hrs and workers with more than or equal to 160 hrs 

plt.subplot(1,2,1)

df.loc[df.average_montly_hours < 160, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar',

                                                                                              grid = True)

plt.title('Monthly hours < 160')

plt.xlabel('Left')

plt.ylabel('%age')

plt.subplot(1,2,2)

df.loc[df.average_montly_hours >= 160, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar',

                                                                                               grid = True)

plt.title('Monthly hours >= 160')

plt.xlabel('Left')

plt.ylabel('%age')

plt.tight_layout()

plt.show()

#The years spent and number of employees

df.time_spend_company.value_counts().sort_index()
#Status of number of employees who spent 10 years

df.loc[df.time_spend_company == 10, 'left'].value_counts()
#Status of number of employees who spent two(2) years

df.loc[df.time_spend_company ==2, 'left'].value_counts()
#Barcharts showing status of employees with time_spend_company less than or equal to 5 and more than 5 in percentage

plt.subplot(1,2,1)

df.loc[df.time_spend_company <=5, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar',

                                                                                          grid = True)

plt.title('5 years and below')

plt.xlabel('Left')

plt.ylabel('%age')

plt.subplot(1,2,2)

df.loc[df.time_spend_company >5, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar', 

                                                                                         grid = True)

plt.title('Spent more than 5 years')

plt.xlabel('Left')

plt.ylabel('%age')

plt.tight_layout()

plt.show()
#Status of employees with work accident

df.Work_accident.value_counts()
#Status of employees with no work accident

df.loc[df.Work_accident == 0, 'left'].value_counts()
#Status of employees with work accident

df.loc[df.Work_accident == 1, 'left'].value_counts()
#Barcharts showing status of employees without accident and those with accident in percentage

plt.subplot(1,2,1)

df.loc[df.Work_accident == 0, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar', grid = True)

plt.title('Employees with no accident')

plt.xlabel('left')

plt.ylabel('%age')

plt.subplot(1,2,2)

df.loc[df.Work_accident == 1, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar', grid = True)

plt.title('Employees with accident')

plt.xlabel('left')

plt.ylabel('%age')

plt.tight_layout()

plt.show()
#Status of employees without and with promotion last five (5) years

df.promotion_last_5years.value_counts()
#Status of employees with no promotion last five(5) years

df.loc[df.promotion_last_5years == 0, 'left'].value_counts()
#Status of employees with promotion last five(5) years

df.loc[df.promotion_last_5years == 1, 'left'].value_counts()
#Barcharts showing status of employees without and with promotion last five (5) years in percentage

plt.subplot(1,2,1)

df.loc[df.promotion_last_5years == 0, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar', grid = True)

plt.title('No promotion last five years')

plt.xlabel('Left')

plt.ylabel('%age')

plt.subplot(1,2,2)

df.loc[df.promotion_last_5years == 1, 'left'].value_counts(normalize = True).sort_index().plot(kind = 'bar', grid = True)

plt.title('Promotion last five years')

plt.xlabel('Left')

plt.ylabel('%age')

plt.tight_layout()

plt.show()
#Departments and number of employees

df.Department.value_counts()
#Grouping by department and salary, the status of employees

df.groupby(['Department', 'salary'])[ 'left'].value_counts().head()
#Status of employees with low salary

df.loc[df.salary == 'low', 'left'].value_counts()
#Status of employees with medium salary

df.loc[df.salary == 'medium', 'left'].value_counts()
#Status of employees with high salary

df.loc[df.salary == 'high', 'left'].value_counts()
#Barcharts showing the status of employees with low, medium and high salaries in percentage

plt.subplot(1,2,1)

df.loc[df.salary == 'low', 'left'].value_counts(normalize = True).plot(kind = 'bar', grid = True)

plt.title('Employees with low salaries')

plt.xlabel('Left')

plt.ylabel('%age')

plt.subplot(1,2,2)

df.loc[df.salary == 'medium', 'left'].value_counts(normalize = True).plot(kind = 'bar', grid = True)

plt.title('Employees with medium salaries')

plt.xlabel('Left')

plt.ylabel('%age')

plt.tight_layout()

plt.show()

plt.subplot(1,2,1)

df.loc[df.salary == 'high', 'left'].value_counts(normalize = True).plot(kind = 'bar', grid = True)

plt.title('Employees with high salaries')

plt.xlabel('Left')

plt.ylabel('%age')

plt.show()
#Transform salary into numbers

df.loc[df.salary == 'low', 'salary'] = 0

df.loc[df.salary == 'medium', 'salary'] = 1

df.loc[df.salary == 'high', 'salary'] = 2
corr = df.corr()
plt.figure(figsize = (10, 7))

sns.heatmap(corr, annot = True, color = 'yellow')

plt.title('Heatmap (Correlation coefficients)')

plt.show()
X = df[['satisfaction_level', 'average_montly_hours', 'number_project', 'salary']]

y = df['left']
#Split the data set into training and testing set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
#Using logistic regression to fit the training data  

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)
#The accuracy of the model on the test data set

model.score(x_test,y_test)
#Predicting the test set data

y_pred = model.predict(x_test)
#The length of predicted set data

len(y_pred)
#First five (5) rows on the test data set

y_test[:5]
#First five (5) predictions on the test data set

y_pred[:5]
#Checking the probability of the employees (first five(5)) whether they stay or leave

model.predict_proba(x_test[:5])
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm, annot = True)

plt.title('Confusion matrix')

plt.xlabel('Predicted')

plt.ylabel('Truth')

plt.show()