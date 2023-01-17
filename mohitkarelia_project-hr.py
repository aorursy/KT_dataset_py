# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

sb.set_palette('pastel')

sb.set_style('whitegrid')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

data.head()
data.dropna(1,inplace=True)

data.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],1,inplace=True)
from sklearn.preprocessing import LabelBinarizer

le = LabelBinarizer()

data['Attrition'] = le.fit_transform(data['Attrition']) #yes:1 No:0

data.select_dtypes(['object']).head(1)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['BusinessTravel'] = le.fit_transform(data['BusinessTravel']) #'Travel_Rarely':2, 'Travel_Frequently':1, 'Non-Travel':0

data['Department'] = le.fit_transform(data['Department'])#'Sales':2, 'Research & Development':1, 'Human Resources':0

data['EducationField'] = le.fit_transform(data['EducationField']) #'Human Resources':0,'Life Sciences':1,'Marketing': 2,'Medical': 3, 'Other':4,'TrchnicalDegree':5

data['Gender'] = le.fit_transform(data['Gender']) # 'Female': 0 ,'Male': 1

data['JobRole'] = le.fit_transform(data['JobRole']) #'Sales Executive':7, 'Research Scientist':6, 'Laboratory Technician':2,'Manufacturing Director':4, 'Healthcare Representative':0, 'Manager':3,'Sales Representative':8, 'Research Director':5, 'Human Resources:1'

data['MaritalStatus'] = le.fit_transform(data['MaritalStatus']) #'Single':2, 'Married':1, 'Divorced':0

data['OverTime'] = le.fit_transform(data['OverTime']) #'Yes':1, 'No':0
data.describe()
plt.hist(x =[data[data['Attrition']==1]['Age'],data[data['Attrition']==0]['Age']],stacked=True , color=['darkorange','pink'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('Age')

plt.ylabel('Employee Count')

plt.title('Attrition Chances at different age')
plt.hist(x =[data[data['Attrition']==1]['BusinessTravel'],data[data['Attrition']==0]['BusinessTravel']],stacked=True , color=['darkorange','pink'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('Business Trips')

plt.ylabel('Employee Count')

plt.title('Attrition Chances with respect to Business Trips')

print("Non-Travel = 0 Travels Frequently = 1 Travels Rarely = 2")

80/250
plt.hist(x =[data[data['Attrition']==1]['DailyRate'],data[data['Attrition']==0]['DailyRate']],stacked=True , color=['violet','pink'],label = ['Attrition Present','Attrition Absent'])

plt.legend()



plt.xlabel('DailyRate')

plt.ylabel('Employee Count')

plt.title('Attrition Chances with respect to EmployeeRates')

plt.hist(x =[data[data['Attrition']==1]['Department'],data[data['Attrition']==0]['Department']],stacked=True , color=['violet','pink'],label = ['Attrition Present','Attrition Absent'])

plt.legend()



plt.xlabel('Department')

plt.ylabel('Employee Count')

plt.title('Attrition Chances in Department')

print("'Human Resources':0 'Research & Development':1 'Sales':2, ")
plt.hist(x =[data[data['Attrition']==1]['DistanceFromHome'],data[data['Attrition']==0]['DistanceFromHome']],stacked=True , color=['violet','pink'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('Distance From Home')

plt.ylabel('Employee')

plt.title('Attrition and Distance From hoem')
plt.hist(x =[data[data['Attrition']==1]['EducationField'],data[data['Attrition']==0]['EducationField']],stacked=True , color=['violet','pink'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('EducationFeild')

plt.ylabel('Employee')

plt.title('Education and attrition')

print("#'Human Resources':0,'Life Sciences':1,'Marketing': 2,'Medical': 3, 'Other':4,'TrchnicalDegree':5")
plt.hist(x =[data[data['Attrition']==1]['EnvironmentSatisfaction'],data[data['Attrition']==0]['EnvironmentSatisfaction']],stacked=True , color=['violet','pink'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('Environment Satisfaction')

plt.ylabel('Employee')

plt.title('Environment Satisfaction vs Attrition')


plt.hist(x =[data[data['Attrition']==1]['Gender'],data[data['Attrition']==0]['Gender']],stacked=True , color=['blue','cyan'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('Gender')

plt.ylabel('Employee')

plt.title('Gender vs Attrition')
plt.hist(x =[data[data['Attrition']==1]['JobInvolvement'],data[data['Attrition']==0]['JobInvolvement']],stacked=True , color=['orange','red'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('Involvement')

plt.ylabel('Employee')

plt.title('Involvement vs Attrition')
fig,ax = plt.subplots(1,2,figsize = (12,5))

ax[0].hist(x =[data[data['Attrition']==1]['JobLevel'],data[data['Attrition']==0]['JobLevel']],stacked=True , color=['magenta','orange'],label = ['Attrition Present','Attrition Absent'])

ax[0].legend()

ax[0].set_xlabel('JobLevel')

ax[0].set_ylabel('Employee')

ax[0].set_title('JobLevel vs Attrition')

ax[1].hist(x =[data[data['Attrition']==1]['JobRole'],data[data['Attrition']==0]['JobRole']],stacked=True , color=['magenta','orange'],label = ['Attrition Present','Attrition Absent'])

ax[1].legend()

ax[1].set_xlabel('JobRole')

ax[1].set_ylabel('Employee')

ax[1].set_title('JobRole vs Attrition')
plt.hist(x =[data[data['Attrition']==1]['JobSatisfaction'],data[data['Attrition']==0]['JobSatisfaction']],stacked=True , color=['orange','red'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('Satisfaction')

plt.ylabel('Employee')

plt.title('Satisfaction vs Attrition')
plt.hist(x =[data[data['Attrition']==1]['MaritalStatus'],data[data['Attrition']==0]['MaritalStatus']],stacked=True , color=['orange','magenta'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('MaritalStatus')

plt.ylabel('Employee')

plt.title('MaritalStatus vs Attrition')

print("Single':2, 'Married':1, 'Divorced':0")
plt.hist(x =[data[data['Attrition']==1]['MonthlyIncome'],data[data['Attrition']==0]['MonthlyIncome']],stacked=True , color=['darkorange','magenta'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('MonthlyIncome')

plt.ylabel('Employee')

plt.title('Income vs Attrition')


plt.hist(x =[data[data['Attrition']==1]['NumCompaniesWorked'],data[data['Attrition']==0]['NumCompaniesWorked']],stacked=True , color=['darkorange','magenta'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('No of companies')

plt.ylabel('Employee')

plt.title('Number of Companies vs Attrition')

plt.hist(x =[data[data['Attrition']==1]['OverTime'],data[data['Attrition']==0]['OverTime']],stacked=True , color=['cyan','magenta'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('Overtime')

plt.ylabel('Employee')

plt.title('Overtime and attrition')


sb.distplot(data[data['Attrition']==1]['PercentSalaryHike'],color='red')

plt.legend()

plt.xlabel('%Salary Hike')

plt.ylabel('Employee')

plt.title('SalaryHike and attrition')


plt.hist(x =[data[data['Attrition']==1]['PerformanceRating'],data[data['Attrition']==0]['PerformanceRating']],stacked=True , color=['pink','red'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('PerformanceRating')

plt.ylabel('Employee')

plt.title('PerformanceRating and attrition')
plt.hist(x =[data[data['Attrition']==1]['RelationshipSatisfaction'],data[data['Attrition']==0]['RelationshipSatisfaction']],stacked=True , color=['blue','darkorange'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('RelationshipSatisfaction')

plt.ylabel('Employee')

plt.title('RelationshipSatisfaction and attrition')    
plt.hist(x =[data[data['Attrition']==1]['StockOptionLevel'],data[data['Attrition']==0]['StockOptionLevel']],stacked=True , color=['orange','black'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('StockOptionLevel')

plt.ylabel('Employee')

plt.title('Stock Option Level and attrition')   

fig,ax = plt.subplots(1,5,figsize = (25,7))

ax[0].hist(x =[data[data['Attrition']==1]['TotalWorkingYears'],data[data['Attrition']==0]['TotalWorkingYears']],stacked=True , color=['blue','black'],label = ['Attrition Present','Attrition Absent'])

ax[0].legend()

ax[0].set_xlabel('TotalWorkingYears')

ax[0].set_ylabel('Employee')

ax[0].set_title('Working Years and attrition')  

ax[1].hist(x =[data[data['Attrition']==1]['YearsAtCompany'],data[data['Attrition']==0]['YearsAtCompany']],stacked=True , color=['blue','black'],label = ['Attrition Present','Attrition Absent'])

ax[1].legend()

ax[1].set_xlabel('Years in company')

ax[1].set_ylabel('Employee')

ax[1].set_title('Years in company and attrition')

ax[2].hist(x =[data[data['Attrition']==1]['YearsInCurrentRole'],data[data['Attrition']==0]['YearsInCurrentRole']],stacked=True , color=['blue','black'],label = ['Attrition Present','Attrition Absent'])

ax[2].legend()

ax[2].set_xlabel('Years In Current Role')

ax[2].set_ylabel('Employee')

ax[2].set_title('Years in current role and attrition')  

ax[3].hist(x =[data[data['Attrition']==1][ 'YearsSinceLastPromotion'],data[data['Attrition']==0][ 'YearsSinceLastPromotion']],stacked=True , color=['blue','black'],label = ['Attrition Present','Attrition Absent'])

ax[3].legend()

ax[3].set_xlabel('Years Since Last Promotion')

ax[3].set_ylabel('Employee')

ax[3].set_title('Promotion and attrition')

ax[4].hist(x =[data[data['Attrition']==1]['YearsWithCurrManager'],data[data['Attrition']==0]['YearsWithCurrManager']],stacked=True , color=['blue','black'],label = ['Attrition Present','Attrition Absent'])

ax[4].legend()

ax[4].set_xlabel('Years with current manager')

ax[4].set_ylabel('Employee')

ax[4].set_title('Working with manager and attrition')  





plt.hist(x =[data[data['Attrition']==1]['TrainingTimesLastYear'],data[data['Attrition']==0][ 'TrainingTimesLastYear']],stacked=True , color=['green','black'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('Training Times')

plt.ylabel('Employee')

plt.title('Training Times and attrition')  
plt.hist(x =[data[data['Attrition']==1]['WorkLifeBalance'],data[data['Attrition']==0][ 'WorkLifeBalance']],stacked=True , color=['magenta','black'],label = ['Attrition Present','Attrition Absent'])

plt.legend()

plt.xlabel('WorkLifeBalance')

plt.ylabel('Employee')

plt.title('Work-Life balance and attrition')  