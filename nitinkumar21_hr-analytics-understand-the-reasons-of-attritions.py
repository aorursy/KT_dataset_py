import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sns

# the commonly used alias for seaborn is sns



# set a seaborn style of your taste

sns.set_style("whitegrid")
df=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df.tail()
df.shape
print(df.info())
# Check Missing Values

round(100*(df.isnull().sum()/len(df.index)), 2)
# Employee Number is Unique field Check for Duplicates

print(any(df['EmployeeNumber'].duplicated())) 
df.loc[df['Attrition']=="Yes",'attr']=1

df.loc[df['Attrition']=="No",'attr']=0

df.attr.value_counts()
df.columns
# Importing matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Let's see the correlation matrix 

plt.figure(figsize = (16,10))     # Size of the figure

sns.heatmap(df.corr(),annot = True)
#Assign Label 

#Education 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'

#EnvironmentSatisfaction 1 'Low' 2 'Medium' 3 'High' 4 'Very High'

#JobInvolvement 1 'Low' 2 'Medium' 3 'High' 4 'Very High'

#JobSatisfaction 1 'Low' 2 'Medium' 3 'High' 4 'Very High'

#PerformanceRating 1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'

#RelationshipSatisfaction 1 'Low' 2 'Medium' 3 'High' 4 'Very High'

#WorkLifeBalance 1 'Bad' 2 'Good' 3 'Better' 4 'Best'



#Education

df.loc[df['Education'] == 1,'Education_type'] = "Below College"

df.loc[df['Education'] == 2,'Education_type'] = "College"

df.loc[df['Education'] == 3,'Education_type'] = "Bachelor"

df.loc[df['Education'] == 4,'Education_type'] = "Master"

df.loc[df['Education'] == 5,'Education_type'] = "Doctor"

#EnvironmentSatisfaction

df.loc[df['EnvironmentSatisfaction'] == 1,'Envnt_Satisfctn_type'] = "Low"

df.loc[df['EnvironmentSatisfaction'] == 2,'Envnt_Satisfctn_type'] = "Medium"

df.loc[df['EnvironmentSatisfaction'] == 3,'Envnt_Satisfctn_type'] = "High"

df.loc[df['EnvironmentSatisfaction'] == 4,'Envnt_Satisfctn_type'] = "Very High"



#JobInvolvement



df.loc[df['JobInvolvement'] == 1,'JobInvolvement_type'] = "Low"

df.loc[df['JobInvolvement'] == 2,'JobInvolvement_type'] = "Medium"

df.loc[df['JobInvolvement'] == 3,'JobInvolvement_type'] = "High"

df.loc[df['JobInvolvement'] == 4,'JobInvolvement_type'] = "Very High"



#JobSatisfaction



df.loc[df['JobSatisfaction'] == 1,'JobSatisfaction_type'] = "Low"

df.loc[df['JobSatisfaction'] == 2,'JobSatisfaction_type'] = "Medium"

df.loc[df['JobSatisfaction'] == 3,'JobSatisfaction_type'] = "High"

df.loc[df['JobSatisfaction'] == 4,'JobSatisfaction_type'] = "Very High"



#PerformanceRating

df.loc[df['PerformanceRating']==1,'Perfrm_rating_type']="Low"

df.loc[df['PerformanceRating'] == 2,'Perfrm_rating_type'] = "Medium"

df.loc[df['PerformanceRating'] == 3,'Perfrm_rating_type'] = "High"

df.loc[df['PerformanceRating'] == 4,'Perfrm_rating_type'] = "Very High"

#RelationshipSatisfaction



df.loc[df['RelationshipSatisfaction']==1,'Rel_satisfctn_type']="Low"

df.loc[df['RelationshipSatisfaction'] == 2,'Rel_satisfctn_type'] = "Medium"

df.loc[df['RelationshipSatisfaction'] == 3,'Rel_satisfctn_type'] = "High"

df.loc[df['RelationshipSatisfaction'] == 4,'Rel_satisfctn_type'] = "Very High"



#WorkLifeBalance

df.loc[df['WorkLifeBalance']==1,'WorkLifeBal_type']="Bad"

df.loc[df['WorkLifeBalance'] == 2,'WorkLifeBal_type'] = "Good"

df.loc[df['WorkLifeBalance'] == 3,'WorkLifeBal_type'] = "Better"

df.loc[df['WorkLifeBalance'] == 4,'WorkLifeBal_type'] = "Best"



df['WorkLifeBal_type'].value_counts()



#Create age band

df.Age.max() #60

df.Age.min() #18

df.Age.mean() #36



df.loc[((df['Age'] >= 18) & (df['Age'] <= 25)),'Age_band']='18_ge_to_le_25'

df.loc[((df['Age'] > 25) & (df['Age'] <= 35)),'Age_band']='25_gr_to_le_35'

df.loc[((df['Age'] > 35) & (df['Age'] <= 45)),'Age_band']='35_gr_to_le_45'

df.loc[((df['Age'] > 45) & (df['Age'] <= 55)),'Age_band']='45_gr_to_le_55'

df.loc[(df['Age'] > 55),'Age_band']='gr_55'

df['Age_band'].value_counts()
plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

sns.boxplot(x='Gender',y="Age", hue="Attrition", data=df)

plt.show()
plt.figure(figsize=(20, 6))

sns.boxplot(x='Gender', y='MonthlyIncome', hue="Attrition", data=df)

plt.title('Gender wise avg. Salaries')

plt.show()
plt.figure(figsize=(20, 4))

sns.boxplot(x='Gender',y='JobSatisfaction', data=df, hue='Attrition')

plt.title("Genderwise Job Satisfaction")

plt.ylabel("Jobsatisfaction Unit")

plt.show()
plt.figure(figsize=(20, 4))

sns.countplot(x='Age_band', data=df, hue='Attrition')

plt.title("Attrition Age Bandwise")

plt.ylabel("Number of Clients")

plt.show()
# Check Attrition Percentage for each band



group_by = df.groupby('Age_band')



df_t1=df.groupby('Age_band')['EmployeeNumber'].count().reset_index()

df_t2=df[df['Attrition'] == "Yes"].groupby('Age_band')['EmployeeNumber'].count().reset_index()

df_t1['tot_count']=df_t1['EmployeeNumber']

df_t1=df_t1.drop(columns = ['EmployeeNumber'])

df_t2['attr_cnt']=df_t2['EmployeeNumber']

df_t2=df_t2.drop(columns = ['EmployeeNumber'])

df_t1

df_t2

#Merge these two datasets

final = pd.merge(df_t1, df_t2, how='inner', on = 'Age_band')

final['attrition_per_total']=round(final['attr_cnt']/(final['tot_count'])*100)

final

#Plot on bar chart

plt.figure(figsize=(15, 4))

sns.barplot(x='Age_band', y='attrition_per_total', data=final)

plt.title("No of Clients Age Bandwise")

plt.ylabel("Attrition(%) wrt. total")

plt.show()
# subplot 1

#distplot for not attrited clients

plt.figure(figsize=(24, 6))

df_dis1=df.loc[df['Attrition']=="Yes"]

plt.subplot(2, 2, 1)

plt.title('DistanceFromHome for Attrited Employees')

sns.distplot(df_dis1['DistanceFromHome'])



# subplot 2

plt.figure(figsize=(24, 6))

df_dis2=df.loc[df['Attrition']=="No"]

plt.subplot(2, 2, 2)

plt.title('DistanceFromHome for Not attrited Employees')

sns.distplot(df_dis2['DistanceFromHome'])
plt.figure(figsize=(20, 6))

sns.boxplot(x='JobRole', y='DistanceFromHome', data=df, hue="Attrition")
# subplot 1

#distplot for not attrited clients

plt.figure(figsize=(24, 6))

df_dis1=df.loc[df['Attrition']=="Yes"]

plt.subplot(2, 2, 1)

plt.title('MonthlyIncome for Attrited Employees')

sns.distplot(df_dis1['MonthlyIncome'])



# subplot 2

plt.figure(figsize=(24, 6))

df_dis2=df.loc[df['Attrition']=="No"]

plt.subplot(2, 2, 2)

plt.title('MonthlyIncome for Not attrited Employees')

sns.distplot(df_dis2['MonthlyIncome'])
plt.figure(figsize=(20, 6))

sns.boxplot(x='Department', y='MonthlyIncome', hue="Attrition", data=df)

plt.title('Department  wise Salaries')

plt.show()
group_by = df.groupby('Department')



df_t1=df.groupby('Department')['EmployeeNumber'].count().reset_index()

df_t2=df[df['Attrition'] == "Yes"].groupby('Department')['EmployeeNumber'].count().reset_index()

df_t1['tot_count']=df_t1['EmployeeNumber']

df_t1=df_t1.drop(columns = ['EmployeeNumber'])

df_t2['attr_cnt']=df_t2['EmployeeNumber']

df_t2=df_t2.drop(columns = ['EmployeeNumber'])

df_t1

df_t2

#Merge these two datasets

final = pd.merge(df_t1, df_t2, how='inner', on = 'Department')

final['attrition_per_total']=round(final['attr_cnt']/(final['tot_count'])*100)

final

#Plot on bar chart

plt.figure(figsize=(15, 4))

sns.barplot(x='Department', y='attrition_per_total', data=final)

plt.title("No of Clients Department wise")

plt.ylabel("Attrition(%) wrt. total")

plt.show()
plt.figure(figsize=(20, 6))

sns.boxplot(x='JobSatisfaction_type', y='MonthlyIncome', hue="Attrition", data=df)

plt.title('Jobsatisfaction Level and Salaries')

plt.show()
plt.figure(figsize=(20, 6))

sns.boxplot(x='Education_type', y='MonthlyIncome', hue="Attrition", data=df)

plt.show()
plt.figure(figsize=(20, 4))

sns.boxplot(x='JobLevel', y='MonthlyIncome', hue="Attrition", data=df)

plt.show()
plt.figure(figsize=(20, 4))

sns.boxplot(x='JobRole', y='DailyRate', hue="Attrition", data=df)

plt.show()
plt.figure(figsize=(15, 4))

sns.boxplot(x='Perfrm_rating_type', y='PercentSalaryHike', hue="Attrition", data=df)

plt.title("Avg. Salary Hike by performance ratingwise")

plt.show()
plt.figure(figsize=(20, 4))

sns.boxplot(x='JobRole', y='EnvironmentSatisfaction', hue="Attrition", data=df)

plt.title("Enivironment Satisfaction JobRole wise")

plt.show()
plt.figure(figsize=(20, 4))

sns.boxplot(x='Department', y='WorkLifeBalance', hue="Attrition", data=df)

plt.title("Work life balance department wise")

plt.show()
plt.figure(figsize=(20, 4))

sns.countplot(x='OverTime', hue="Attrition", data=df)

plt.title("Attriton Overtime wise")

plt.show()
#Overtime vs Jobrole

df.loc[df['OverTime']=="Yes",'OverTime_1']=1

df.loc[df['OverTime'] =="No",'OverTime_1'] =0



plt.figure(figsize=(20, 4))

sns.boxplot(x='JobRole', y='OverTime_1', hue="Attrition", data=df)

plt.title("OverTime JobRole wise")

plt.show()

plt.figure(figsize=(20, 4))

sns.boxplot(x='JobRole', y='YearsAtCompany', hue="Attrition", data=df)

plt.title("YearsAtCompany JobRole wise")

plt.show()
plt.figure(figsize=(20, 4))

sns.barplot(x='JobRole', y='YearsInCurrentRole', hue="Attrition", data=df)

plt.title("YearsInCurrentRole JobRole wise")

plt.show()
plt.figure(figsize=(20, 4))

sns.barplot(x='JobLevel', y='YearsSinceLastPromotion', hue="Attrition", data=df)

plt.title("YearsSinceLastPromotion JobLevel wise")

plt.show()
group_by = df.groupby('Rel_satisfctn_type')



df_t1=df.groupby('Rel_satisfctn_type')['EmployeeNumber'].count().reset_index()

df_t2=df[df['Attrition'] == "Yes"].groupby('Rel_satisfctn_type')['EmployeeNumber'].count().reset_index()

df_t1['tot_count']=df_t1['EmployeeNumber']

df_t1=df_t1.drop(columns = ['EmployeeNumber'])

df_t2['attr_cnt']=df_t2['EmployeeNumber']

df_t2=df_t2.drop(columns = ['EmployeeNumber'])

df_t1

df_t2

#Merge these two datasets

final = pd.merge(df_t1, df_t2, how='inner', on = 'Rel_satisfctn_type')

final['attrition_per_total']=round(final['attr_cnt']/(final['tot_count'])*100)

final

#Plot on bar chart

plt.figure(figsize=(15, 4))

sns.barplot(x='Rel_satisfctn_type', y='attrition_per_total', data=final)

plt.title("No of Employees Rel_satisfctn_type Bandwise")

plt.ylabel("Attrition(%) wrt. total")

plt.show()
group_by = df.groupby(['MaritalStatus','Gender'])

df_t1=group_by['EmployeeNumber'].count().reset_index()

df_t2=df[df['Attrition'] == "Yes"].groupby(['MaritalStatus','Gender'])['EmployeeNumber'].count().reset_index()

df_t1['tot_count']=df_t1['EmployeeNumber']

df_t1=df_t1.drop(columns = ['EmployeeNumber'])

df_t2['attr_cnt']=df_t2['EmployeeNumber']

df_t2=df_t2.drop(columns = ['EmployeeNumber'])

df_t1

final = pd.merge(df_t1, df_t2, how='inner', on = ['MaritalStatus','Gender'])

final['attrition_per_total']=round(final['attr_cnt']/(final['tot_count'])*100)

final

#Plot on bar chart

plt.figure(figsize=(15, 4))

sns.barplot(x='MaritalStatus', y='attrition_per_total',hue='Gender', data=final)

plt.title("No of Employees MaritalStatus Bandwise")

plt.ylabel("Attrition(%) wrt. total")

plt.show()

final
df['BusinessTravel'].value_counts()
group_by = df.groupby(['MaritalStatus','BusinessTravel'])

df_t1=group_by['EmployeeNumber'].count().reset_index()

df_t2=df[df['Attrition'] == "Yes"].groupby(['MaritalStatus','BusinessTravel'])['EmployeeNumber'].count().reset_index()

df_t1['tot_count']=df_t1['EmployeeNumber']

df_t1=df_t1.drop(columns = ['EmployeeNumber'])

df_t2['attr_cnt']=df_t2['EmployeeNumber']

df_t2=df_t2.drop(columns = ['EmployeeNumber'])

df_t1

final = pd.merge(df_t1, df_t2, how='inner', on = ['MaritalStatus','BusinessTravel'])

final['attrition_per_total']=round(final['attr_cnt']/(final['tot_count'])*100)

final

#Plot on bar chart

plt.figure(figsize=(15, 4))

sns.barplot(x='MaritalStatus', y='attrition_per_total',hue='BusinessTravel', data=final)

plt.title("")

plt.ylabel("Attrition(%) wrt. total")

plt.show()
final