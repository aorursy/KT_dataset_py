#import the warnings.

import warnings

warnings.filterwarnings('ignore')
#import the useful libraries.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#read the data set of "bank telemarketing campaign" in inp0.

inp0= pd.read_csv(r'/kaggle/input/bank_marketing_updated_v1.csv')

inp0
#Print the head of the data frame.

inp0.head()
#read the file in inp0 without first two rows as it is of no use.

inp0= pd.read_csv(r'/kaggle/input/bank_marketing_updated_v1.csv',skiprows=2)

inp0
#print the head of the data frame.

inp0.head()
#print the information of variables to check their data types.

print(type(inp0.info()))
#convert the age variable data type from float to integer.



inp0['age']=inp0['age'].astype('Int64')

inp0
#print the average age of customers.

avg=np.mean(inp0['age'])

print(round(avg,2))
#drop the customer id as it is of no use.

inp0=inp0.drop(['customerid'],axis=1)

inp0
#Extract job in newly created 'job' column from "jobedu" column.

jobedu=inp0['jobedu'].apply(lambda x: pd.Series(x.split(',')))

inp0['job']=jobedu[0]

inp0
#Extract education in newly created 'education' column from "jobedu" column.

inp0['education']= jobedu[1]

inp0
#drop the "jobedu" column from the dataframe.

inp0=inp0.drop(['jobedu'],axis=1)

inp0
#count the missing values in age column.

inp0.age.isnull().sum(axis=0)
#pring the shape of dataframe inp0

inp0.shape
#calculate the percentage of missing values in age column.

round(100*(inp0.age.isnull().sum(axis=0)/len(inp0.index)),2)
#drop the records with age missing in inp0 and copy in inp1 dataframe.

inp1= inp0[~inp0.age.isnull()].copy()

inp1
#count the missing values in month column in inp1.

inp1.month.isnull().sum(axis=0)
#print the percentage of each month in the data frame inp1.

inp1.month.value_counts(normalize=True) *100
#find the mode of month in inp1

mode_month=inp1.month.mode()[0]

mode_month
# fill the missing values with mode value of month in inp1.

inp1.month.fillna(mode_month,inplace=True)
#let's see the null values in the month column.

inp1.month.isna().sum()
#count the missing values in response column in inp1.

inp1.response.isna().sum()
#calculate the percentage of missing values in response column. 

round(100*(inp1.response.isna().sum())/len(inp1.index),3)
#drop the records with response missings in inp1.

inp1=inp1[~inp1.response.isna()]

inp1
#calculate the missing values in each column of data frame: inp1.

inp1.isna().sum()
#describe the pdays column of inp1.

inp1.pdays.describe()
#describe the pdays column with considering the -1 values.

inp1.loc[inp1.pdays<0,"pdays"]= np.NaN

inp1.pdays.describe()
#describe the age variable in inp1.

inp1.age.describe()
#plot the histogram of age variable.

import matplotlib.pyplot as plt

inp1.age.plot.hist()

plt.show()
#plot the boxplot of age variable.

import seaborn as sns

sns.boxplot(inp1.age)

plt.show()
#describe the salary variable of inp1.

inp0.salary.describe()
#plot the boxplot of salary variable.

sns.boxplot(inp0.salary)
#describe the balance variable of inp1.

inp0.balance.describe()
#plot the boxplot of balance variable.

sns.boxplot(inp0.balance)
#plot the boxplot of balance variable after scaling in 8:2.

plt.figure(figsize=(8,2))

sns.boxplot(inp0.balance)
#print the quantile (0.5, 0.7, 0.9, 0.95 and 0.99) of balance variable

inp1.balance.quantile([0.5, 0.7, 0.9, 0.95,0.99])

#describe the duration variable of inp1

inp1.duration.describe()

#convert the duration variable into single unit i.e. minutes. and remove the sec or min prefix.

inp1.duration=inp1.duration.apply(lambda x: float(x.split()[0])/60 if x.find("sec")>0 else float(x.split()[0]))

#describe the duration variable

inp1.duration.describe()
#calculate the percentage of each marital status category. 

marital=inp1.marital.value_counts(normalize=True)*100

marital
#plot the bar graph of percentage marital status categories

marital.plot.barh(color='r')

plt.show()
#calculate the percentage of each job status category.

job=inp1.job.value_counts(normalize=True)*100

job
#plot the bar graph of percentage job categories

job.plot.barh(color='g')

plt.show()
#calculate the percentage of each education category.

education=inp1.education.value_counts(normalize=True)*100

education
#plot the pie chart of education categories

education.plot.pie()

plt.show()

#calculate the percentage of each poutcome category.

poutcome=inp1.poutcome.value_counts(normalize=True)*100

poutcome                                                                               
poutcome.plot.pie()

plt.show()
poutcometarget=inp1[~(inp1.poutcome=='unknown')].poutcome.value_counts(normalize=True)*100

poutcometarget
poutcometarget.plot.bar()

plt.show()
#calculate the percentage of each response category.

response=inp1.response.value_counts(normalize=True)*100

response
#plot the pie chart of response categories

response.plot.pie()

plt.show()
#plot the scatter plot of balance and salary variable in inp0

inp0.plot.scatter(x='salary',y='balance')

plt.show()
#plot the scatter plot of balance and age variable in inp1

plt.scatter(inp1.age,inp1.balance)

plt.show()
#plot the pair plot of salary, balance and age in inp1 dataframe.

sns.pairplot(data=inp1,vars=['salary','balance','age'])

plt.show()
#plot the correlation matrix of salary, balance and age in inp1 dataframe.

inp1[['salary','balance','age']].corr()
sns.heatmap(inp1[['salary','balance','age']].corr(),annot=True,cmap='Reds')



plt.show()
#groupby the response to find the mean of the salary with response no & yes seperatly.

inp1.groupby("response")["salary"].mean()
#groupby the response to find the median of the salary with response no & yes seperatly.

inp1.groupby("response")["salary"].median()
#plot the box plot of salary for yes & no responses.

sns.boxplot(data=inp1,x='response',y='salary')

plt.show()
#plot the box plot of balance for yes & no responses.

sns.boxplot(data=inp1,x='response',y='balance')

plt.show()
#groupby the response to find the mean of the balance with response no & yes seperatly.

inp1.groupby("response")['balance'].mean()
#groupby the response to find the median of the balance with response no & yes seperatly.

inp1.groupby("response")['balance'].median()
#function to find the 75th percentile.

def p75(x):

    return np.quantile(x, 0.75)
#calculate the mean, median and 75th percentile of balance with response

inp1.groupby("response")['balance'].aggregate(["mean","median",p75])
#plot the bar graph of balance's mean an median with response.

inp1.groupby("response")['balance'].aggregate(["mean","median"]).plot.bar()

plt.show()
#groupby the education to find the mean of the salary education category.

inp1.groupby("education")["salary"].mean()
#groupby the education to find the median of the salary for each education category.

inp1.groupby("education")["salary"].median()
#groupby the job to find the mean of the salary for each job category.

inp1.groupby("job")["salary"].mean()
#create response_flag of numerical data type where response "yes"= 1, "no"= 0

inp1['response_flag']=np.where(inp1.response=='yes',1,0)

inp1.response_flag.value_counts()
#calculate the mean of response_flag with different education categories.

inp1.groupby("education")["response_flag"].mean()
#calculate the mean of response_flag with different marital status categories.

inp1.groupby("marital")["response_flag"].mean()
#plot the bar graph of marital status with average value of response_flag

(inp1.groupby("marital")["response_flag"].mean()).plot.barh()
#plot the bar graph of personal loan status with average value of response_flag

(inp1.groupby("loan")["response_flag"].mean()*100).plot.bar()

plt.show()
#plot the bar graph of housing loan status with average value of response_flag

(inp1.groupby("housing")["response_flag"].mean()*100).plot.bar()

#plot the boxplot of age with response_flag

sns.boxplot(data=inp1,x='response_flag',y='age')
#create the buckets of <30, 30-40, 40-50 50-60 and 60+ from age column.

inp1["age_group"]= pd.cut(inp1.age,[0,30,40,50,60,120],labels=["<30", "30-40", "40-50","50-60","60+"])
#plot the percentage of each buckets and average values of response_flag in each buckets. plot in subplots.

plt.figure(figsize=[10,4])

plt.subplot(1,2,1)

(inp1.age_group.value_counts(normalize=True)*100).plot.bar()

plt.subplot(1,2,2)

(inp1.groupby(['age_group'])['response_flag'].mean()*100).plot.bar()

plt.show()
#plot the bar graph of job categories with response_flag mean value.

(inp1.groupby(['job'])['response_flag'].mean()*100).plot.barh()

plt.show()
#create heat map of education vs marital vs response_flag



ax=pd.pivot_table(data=inp1,index="education",columns='marital',values='response_flag')

sns.heatmap(ax,annot=True,cmap='PiYG')

plt.show()

#create the heat map of Job vs marital vs response_flag.

ax=pd.pivot_table(data=inp1,index='job',columns='marital',values='response_flag')

sns.heatmap(ax,annot=True,cmap='rainbow')

plt.show()
#create the heat map of education vs poutcome vs response_flag.

ax=pd.pivot_table(data=inp1,index='education',columns='poutcome',values='response_flag')

sns.heatmap(ax,annot=True,cmap='plasma_r')

plt.show()