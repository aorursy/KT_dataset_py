import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.patches import Patch

from matplotlib.lines import Line2D
#Importing the Employee Survey Data 

df1=pd.read_csv('../input/employee_survey_data.csv')

print(df1.columns)

df1.head()

#Importing the Employee General Data 

df2=pd.read_csv('../input/general_data.csv')

print(df2.columns)

df2.head()
#Importing the Manager Survey Data

df3=pd.read_csv('../input/manager_survey_data.csv')

print(df3.columns)

df3.head()
df=df1.merge(df2,on='EmployeeID',how='inner')
#Creating a master dataset by merging all 3 datasets

df=df.merge(df3,on='EmployeeID',how='inner')

df.head()
#Finding out about the data types and null values present in the dataset across columns

df.info()
atrate=len(df[df['Attrition']=='Yes'])/4000

print('Attrition Rate: ',atrate*100,'%')
#Plotting the Age Distribution of the Employees for overall company and those who left

plt.figure(figsize=[8,8])

#Overall Company

plt.hist(x=df['Age'])

#Those who left

plt.hist(x=df[df['Attrition']=='Yes']['Age']);

#Legend Elements

legend_elements = [Line2D([0], [0], color='b', lw=4, label='Overall Company'),

                   Line2D([0], [0], lw=4, color='orange', label='Ex-Employees')]

plt.legend(handles=legend_elements, loc='best');

#Polishing The Graph

plt.title('Age Distibution Of Employees at XYZ',size=14,fontweight='bold');

plt.ylabel('Number of Employees',size=12);

plt.xlabel('Age of Employees',size=12);
#Creating a pie plot to visualize the gender distribution

plt.figure(figsize=[8,8])

src=df.Gender.value_counts()

plt.pie(src,labels=src.index,startangle=90,counterclock=False,colors=['#8db3ef','#f9e5f9'],wedgeprops={'width':0.4},autopct='%1.0f%%', pctdistance=0.75);
df.columns
#Removing Redundant columns which have same value throughout

df.drop(['Over18','EmployeeCount','StandardHours'],axis=1,inplace=True)
df.columns
#Creating a Function to seperate employees into their corresponding genration

def Gen(row):

    if row['Age']<=37:

        row['Gen']='Millenials'

        return 'Millenials'

    elif ((row['Age']>37)&(row['Age']<=54)):

        row['Gen']='Generation X'

        return 'Generation X'

    elif ((row['Age']>54)&(row['Age']<74)):

        row['Gen']='Boomers'

        return 'Boomers'

    else:

        row['Gen']='Silent'

        return 'Silent'

df['Gen']=''
#Applying the function

df['Gen']=df.apply(Gen,axis=1)
df.head()
#Visualising the number of companies each genration has worked for with respect to their current status in the company

plt.figure(figsize=[8,6])

#Those Who Left

sns.pointplot(data=df[df['Attrition']=='Yes'],x='Gen',y='NumCompaniesWorked',order=['Millenials','Generation X','Boomers'],color='Red',linestyles=["--"]);

#Those who did not leave

sns.pointplot(data=df[df['Attrition']=='No'],x='Gen',y='NumCompaniesWorked',order=['Millenials','Generation X','Boomers'],color='Green',linestyles=["--"]);

legend_elements = [Line2D([0], [0], color='r', lw=4, label='Left'),

                   Line2D([0], [0], lw=4, color='g', label='Stayed')]

#Polishing the graph

plt.legend(handles=legend_elements, loc='best')

plt.xlabel('Genration of Employee',fontweight='bold')

plt.ylabel('Number of Companies Worked At Before',fontweight='bold')

plt.title('Generation vs Number of Companies they have worked at with respect to Attrition',fontweight='bold',size=12);
#Creating a function to elaborate on the education levels with help of values given in the data dictionary

def Edu(row):

    if row['Education']==1:

        row['Education']='Without College Degree'

        return 'Without College Degree'

    elif (row['Education']==2):

        row['Education']='College'

        return 'College'

    elif (row['Education']==3):

        row['Education']='Bachelor'

        return 'Bachelor'

    elif (row['Education']==4):

        row['Education']='Master'

        return 'Master'

    else:

        row['Education']='Doctor'

        return 'Doctor'
#Applying function

df['Education']=df.apply(Edu,axis=1)
#Understanding the Department-wise distribution

plt.figure(figsize=[8,8])

src=df.Department.value_counts()

plt.pie(src,labels=src.index,startangle=90,counterclock=False,colors=['#21dd08','#118202','#73f26f'],wedgeprops={'width':0.4},autopct='%1.0f%%', pctdistance=0.75);
#Plotting the percentage of attrition across dept

plt.figure(figsize=[8,8])

x=df.groupby(['Department']).Attrition.count()

y=df[df['Attrition']=='Yes'].groupby(['Department']).Age.count()

#Calculated Percentage

z=(y/x)*100

#Plotting it

z.plot(kind='bar',color=[sns.color_palette()[0]]);

#Polishing

plt.xlabel('Department',size=10);

plt.ylabel('Percentage of Employees Left');

plt.title('Percentage of Attrition Across Every Department ',fontweight='bold');

plt.xticks(rotation=45);
#plotting the salaries accross department 

g = sns.catplot(x="Department", y="MonthlyIncome", hue="Attrition", data=df,

                height=6, kind="bar", palette="muted")

#Polishing 

plt.ylabel('Monthly Income ($)');
#Creating a function to classify the employee according to the number of years they have been serving under the same manager

def Curr(row):

    if row['YearsWithCurrManager']<2:

        row['Curr']='Fresh Manager'

        return 'Fresh Manager'

    elif ((row['YearsWithCurrManager']>=2)&(row['YearsWithCurrManager']<=4)):

        row['Curr']='2-4 Years'

        return '2-4 Years'

    elif ((row['YearsWithCurrManager']>4)):

        row['Curr']='Old Manager'

        return 'Old Manager'

    

df['Curr']=df.apply(Curr,axis=1)
#Creating the Order via transforming it into a Categorical Datatype

order=['Fresh Manager','2-4 Years','Old Manager']

order=pd.api.types.CategoricalDtype(ordered=True,categories=order)

df['Curr']=df['Curr'].astype(order)
#Plotting the Job Satisfaction of employees under varying years served under the same manager

g = sns.catplot(x="JobSatisfaction", y="Curr" ,data=df,height=6, kind="bar", palette="muted")

ax = plt.gca()

#polishing

#Adding the values inside the bars

for p in ax.patches:

    

    ax.text(1.4 , p.get_height()-0.75, '2.76', 

            fontsize=12, color='Black', ha='center', va='bottom')

    ax.text(1.4 , p.get_height()+0.25, '2.73', 

            fontsize=12, color='Black', ha='center', va='bottom')

    ax.text(1.4 , p.get_height()+1.25, '2.71', 

            fontsize=12, color='Black', ha='center', va='bottom')

plt.title('Job Satisfaction vs Years under same Manager',fontweight="bold")

plt.xlabel('Job Satisfaction')

plt.ylabel('Years under Same Manager')
#plotting the attrition accross employees varying with the number of years they have served at the company

plt.figure(figsize=[8,8])

g = sns.countplot(x="YearsAtCompany",hue='Attrition',data=df);

plt.ylabel('Number Of Employees');

plt.title('Employees vs Number of Years Worked at XYZ',fontweight='bold');
#Understanding the gender distribution of employees who left early

x=df[df['YearsAtCompany']<=1]

sns.countplot(data=x,x='Gender',hue='Attrition');