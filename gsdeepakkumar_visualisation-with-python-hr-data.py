#loading the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

sns.set(context="notebook",style="white",palette="dark")

plt.style.use('fivethirtyeight')
# read the datafile

HR=pd.read_csv("../input/HR_comma_sep.csv")
# glimpse of the data

HR.head()
#Getting to know about the data types

HR.info()
#Check for anymissing values

HR.isnull().sum()
plt.figure(figsize=(12,7))

sns.distplot(HR.average_montly_hours,bins=30,kde=False)

plt.title("Distribution of Average Monthly Hours")

plt.xlabel("Average Monthly Hours",fontsize=12)

plt.ylabel("Count",fontsize=12)

plt.show()
plt.figure(figsize=(12,7))

ax=sns.kdeplot(HR.loc[(HR.left==0),'average_montly_hours'],color="g",shade=True,label="Stays in company")

ax=sns.kdeplot(HR.loc[(HR.left==1),'average_montly_hours'],color="r",shade=True,label="Left the company")

ax.set(xlabel='Average monthly hours',ylabel="Frequency")

plt.title("Employee Turnover with Average Monthly Hours",fontsize=16)
plt.figure(figsize=(12,5))

sns.distplot(HR.satisfaction_level,kde=False)

plt.title("Distribution of Satisfaction Level",fontsize=16)

plt.xlabel("Satisfaction Level",fontsize=12)

plt.ylabel("Count",fontsize=12)
fig,ax=plt.subplots(ncols=2,figsize=(12,5))

left=HR[HR.left==0]

stay=HR[HR.left==1]

sns.kdeplot(left.satisfaction_level,shade=True,color="r",ax=ax[0],legend=False)

ax[0].set_xlabel("Satisfaction Level")

ax[0].set_ylabel("Density")

ax[0].set_title("Those who Leave")

sns.kdeplot(stay.satisfaction_level,shade=True,color="g",ax=ax[1],legend=False)

ax[1].set_xlabel("Satisfaction Level")

ax[1].set_ylabel('Density')

ax[1].set_title('Those who stay')

plt.suptitle("Satisfaction level Vs Turnover",fontsize=16)
fig=plt.figure(figsize=(12,8))

g=sns.factorplot(x="number_project",hue="left",data=HR,kind="count",legend_out=True,size=8,aspect=0.7)

g._legend.set_title("Turnover")

plt.xlabel("Number of Projects",fontsize=12)

plt.ylabel("Count",fontsize=12)

plt.title("Number of Projects Vs Turnover",fontsize=16)

fig=plt.figure(figsize=(10,10))

sns.boxplot(x="number_project",y="satisfaction_level",hue="left",data=HR,palette='viridis',linewidth=2.5)

plt.xlabel("Number of Projects",fontsize=12)

plt.ylabel("Satisfaction Level",fontsize=12)

plt.title("Boxplot of Satisfaction Level with Number of Projects",fontsize=16)
HR.last_evaluation.describe()
print("There are {} people having evaluation score greater than 0.7".format(len(HR[HR.last_evaluation>0.7])))

print("There are {} people having evaluation score lesser than 0.7".format(len(HR[HR.last_evaluation<0.7])))
plt.figure(figsize=(7,5))

sns.distplot(HR.last_evaluation,bins=30,color="r")

plt.xlabel("Evaluation",fontsize=12)

plt.ylabel("Frequency",fontsize=12)

plt.title("Distribution of Evaluation",fontsize=16)
ax=plt.figure(figsize=(7,8))

ax=sns.factorplot(x="left",y="last_evaluation",col="number_project",data=HR,kind="box",size=4,aspect=0.6)

#ax.set(xlabel="Turnover",ylabel="EvaluationScore",title="Trend of Turnover with Number of Projects and Evaluation")

ax.set_xlabels("Turnover")

ax.set_ylabels('Evaluation Score')

ax.fig.suptitle("Trend of Turnover with Number of Projects and Evaluation",x=0.5,y=1.2)
pd.crosstab(HR["time_spend_company"],HR["left"],margins=False).apply(lambda x: (x/x.sum())*100).round()
ax=plt.figure(figsize=(7,5))

ax=sns.barplot(x="time_spend_company",y="time_spend_company",data=HR,hue="left",estimator=lambda x: len(x) / len(HR) * 100)

ax.set(ylabel="Percentage")

ax.set(title="Years In Company Vs Turnover")

ax.set_ylim(0,50)
ax=plt.figure(figsize=(10,8))

ax=sns.countplot(x="promotion_last_5years",hue="left",palette="inferno",data=HR)

ax.set_xlabel("Promotion")

ax.set_ylabel("Count")

ax.set_title("Last 5 Year Promotion Count")
(HR.salary.value_counts(normalize=True))*100
plt.figure(figsize=(10,8))

ax=sns.countplot(x="salary",hue="left",data=HR,order=HR.salary.value_counts().iloc[:].index,palette="viridis")

ax.set_title("Salary Range Vs Turnover",fontsize=20)

ax.set_xlabel("Salary Range",fontsize=12)

ax.set_ylabel("Count",fontsize=12)
plt.figure(figsize=(10,8))

ax=sns.boxplot(x="salary",y="satisfaction_level",hue="left",data=HR,palette="PRGn")

ax.set_title("Satifaction Levels with Salary",fontsize=18)

ax.set_xlabel("Salary",fontsize=12)

ax.set_ylabel("Satisfaction Level",fontsize=12)

sal=pd.DataFrame(HR[HR.salary=="low"].sales.value_counts())

sal.reset_index(level=0,inplace=True)

sal=sal.sort_values(by='sales',ascending=False)

sal.columns=("Department","Count")

#col=["3A0EE8","3A7EE8","3A7E0E","6BE81A","FF1C31","FF0DB5","394873","FFC202"]

plt.figure(figsize=(10,5))

sns.barplot(x="Department",y="Count",data=sal,palette="RdBu")

plt.xlabel("Department")

plt.ylabel("Count")

plt.title("Low Salary:Department Count")
sal=pd.DataFrame(HR[HR.salary=="medium"].sales.value_counts())

sal.reset_index(level=0,inplace=True)

sal=sal.sort_values(by='sales',ascending=False)

sal.columns=("Department","Count")



plt.figure(figsize=(10,5))

sns.barplot(x="Department",y="Count",data=sal,palette="Set1_r")

plt.xlabel("Department")

plt.ylabel("Count")

plt.title("Medium Salary:Department Count")
sal=pd.DataFrame(HR[HR.salary=="high"].sales.value_counts())

sal.reset_index(level=0,inplace=True)

sal=sal.sort_values(by='sales',ascending=False)

sal.columns=("Department","Count")



plt.figure(figsize=(10,5))

sns.barplot(x="Department",y="Count",data=sal,palette="viridis")

plt.xlabel("Department")

plt.ylabel("Count")

plt.title("High Salary:Department Count")
HR.sales.value_counts()
ax=plt.figure(figsize=(10,8))

left=pd.DataFrame(HR[HR.left==1].sales.value_counts())

left.reset_index(level=0,inplace=True)

left=left.sort_values(by='sales',ascending=False)

left.columns=("Department","count")

plt.figure(figsize=(17,20))

sns.barplot(x = 'Department', y = 'count', data = left)

plt.title('Department Count - Those who left ', fontsize = 30)

plt.ylabel('Count', fontsize = 25)

plt.xlabel('Department', fontsize = 25)

ax=plt.figure(figsize=(10,8))

stay=pd.DataFrame(HR[HR.left==0].sales.value_counts())

stay.reset_index(level=0,inplace=True)

stay=stay.sort_values(by='sales',ascending=False)

stay.columns=("Department","count")

plt.figure(figsize=(17,20))

sns.barplot(x = 'Department', y = 'count', data = stay)

plt.title('Department Count - Those who stay ', fontsize = 30)

plt.ylabel('Count', fontsize = 25)

plt.xlabel('Department', fontsize = 25)