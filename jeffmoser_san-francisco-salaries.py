%matplotlib inline
from pandas import Series,DataFrame
import pandas as pd
%matplotlib inline
import nltk as nl
salaries=pd.read_csv("../input/Salaries.csv")
salaries.head()
# There are many features that won't be very useful in our prediction. We can remove this.
salaries=salaries.drop(['Status','Benefits','Notes','Agency','EmployeeName'],axis=1)
#if we observe the dataset, there are rows where no job title is provided. We need to remove this.
#We also need to remove rows where TotalPayment values do not exist -( They contain 0 values )
salaries=salaries[salaries['JobTitle']!='Not provided']
salaries=salaries[salaries['TotalPayBenefits']!=0.00]
salaries['TotalPayBenefits']=salaries['TotalPayBenefits']-salaries['TotalPay']
salaries=salaries[salaries['TotalPay']>=0]       
#removing negative salaries

# How many job titles are there? Can we use these key words in job titles to predict their salaries? 
salaries['JobTitle'].value_counts()[:20].plot(kind='bar')
# Ignore case
salaries['JobTitle'].str.lower().value_counts()[:20].plot(kind='bar')