import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("../input/suicides-in-india/Suicides in India 2001-2012.csv")

# Type Code  categorically seperates the counts into various sections

# We will select Means_adopted here since it has relevant data for all the years for all the States and UTs

dataFrame1 = df[(df.Age_group!='0-100+') & (df.Type_code=='Means_adopted') ]

dataFrame2 = df[(df.Age_group=='0-100+')]

dataFrame1.info()
df_StateOTY = pd.DataFrame(dataFrame1.groupby(["State"])["Total"].sum()).reset_index().sort_values(by='Total')

plt.figure(figsize=(25, 25))
plt.xticks(rotation=90,fontsize=30)
plt.yticks(rotation=90,fontsize=30)

ax1 = sns.barplot(x='State', y='Total', data=df_StateOTY)
ax1.set_xlabel("State",fontsize=30)
ax1.set_ylabel("Number of Cases",fontsize=30)
ax1 = plt.title("State vs Number of Cases Over The Years",fontsize=40)

df_StateVsGender = pd.pivot_table(pd.DataFrame(dataFrame1.groupby(["State","Gender"])["Total"].sum()).reset_index().sort_values(by='Total')
                                    , values ='Total', index =['State'], 
                         columns =['Gender'],aggfunc=np.sum ) 


df_StateVsGender=df_StateVsGender.sort_values(by='Male')

plt.figure(figsize=(25, 25))
plt.xticks(rotation=90,fontsize=30)
plt.yticks(rotation=90,fontsize=30)


axF = sns.barplot(x=df_StateVsGender.index, y="Male",data=df_StateVsGender.sort_values(by='Male') );
axF.set_xlabel("State",fontsize=30)
axF.set_ylabel("Number of Cases , Men Only",fontsize=30)
axF = plt.title("State vs Number of Cases Over The Years for Men",fontsize=40)




df_StateVsGender=df_StateVsGender.sort_values(by='Female')

plt.figure(figsize=(25, 25))
plt.xticks(rotation=90,fontsize=30)
plt.yticks(rotation=90,fontsize=30)



axF = sns.barplot(x=df_StateVsGender.index, y="Female",data=df_StateVsGender );
axF.set_xlabel("State",fontsize=30)
axF.set_ylabel("Number of Cases , Women Only",fontsize=30)
axF = plt.title("State vs Number of Cases Over The Years for Women",fontsize=40)


plt.rcParams['figure.figsize']=(20,20)
sns.heatmap(df_StateVsGender, annot=True,annot_kws={'size':14}, cmap='Greens',fmt="0.1f")
df_StateVsAge_group = pd.pivot_table(pd.DataFrame(dataFrame1.groupby(["State","Age_group"])["Total"].sum()).reset_index().sort_values(by='Total')
                                    , values ='Total', index =['State'], 
                         columns =['Age_group'],aggfunc=np.sum ) 


df_StateVsAge_group

df_StateVsAge_groupPCT = df_StateVsAge_group/df_StateVsAge_group[df_StateVsAge_group.columns].sum()*100
df_StateVsAge_groupPCT
df_StateVsYear = pd.pivot_table(pd.DataFrame(dataFrame1.groupby(["State","Year"])["Total"].sum()).reset_index().sort_values(by='Total')
                                    , values ='Total', index =['State'], 
                         columns =['Year'],aggfunc=np.sum ) 
df_StateVsYear
# Converting share of states per year in percentage.
df_StateVsYearPCT = df_StateVsYear/df_StateVsYear[df_StateVsYear.columns].sum()*100
df_StateVsYearPCT
# Taking transpose of data created earlier
df_YearVsState = df_StateVsYear.T

df_YearVsGender = pd.pivot_table(pd.DataFrame(dataFrame1.groupby(["Gender","Year"])["Total"].sum()).reset_index().sort_values(by='Total')
                                    , values ='Total', index =['Year'], 
                         columns =['Gender'],aggfunc=np.sum ) 

df_YearVsGender
df_YearVsAge_group = pd.pivot_table(pd.DataFrame(dataFrame1.groupby(["Age_group","Year"])["Total"].sum()).reset_index().sort_values(by='Total')
                                    , values ='Total', index =['Year'], 
                         columns =['Age_group'],aggfunc=np.sum ) 
df_YearVsAge_group

# Distribution of age group count according to year in  perecentage 
df_YearVsAge_groupPCT = df_YearVsAge_group/df_YearVsAge_group[df_YearVsAge_group.columns].sum()*100
plt.rcParams['figure.figsize']=(20,10)
sns.heatmap(df_YearVsAge_groupPCT,cmap='Greens', annot=True,annot_kws={'size':16}, fmt="0.1f")
df_Age_groupVsYear = df_YearVsAge_group.T
df_Age_groupVsYear
# Converting share of Age Groups per year in percentage 
df_Age_groupVsYearPCT = df_Age_groupVsYear/df_Age_groupVsYear[df_Age_groupVsYear.columns].sum()*100
plt.rcParams['figure.figsize']=(20,10)
sns.heatmap(df_Age_groupVsYearPCT,cmap='Greens', annot=True,annot_kws={'size':16}, fmt="0.1f")

df_Age_groupVsState=df_StateVsAge_group.T
df_Age_groupVsState

# Converting data to percentage according to column
df_Age_groupVsStatePCT = df_Age_groupVsState/df_Age_groupVsState[df_Age_groupVsState.columns].sum()*100
df_Age_groupVsStatePCT

df_Age_groupVsStatePCT
#plt.rcParams['figure.figsize']=(20,10)
#sns.heatmap(df_Age_groupVsStatePCT,cmap='Greens', fmt="0.1f")

df_Age_groupVsStatePCT.T

df_Age_groupVsGender = pd.pivot_table(pd.DataFrame(dataFrame1.groupby(["Age_group","Gender"])["Total"].sum()).reset_index().sort_values(by='Total')
                                        , values ='Total', index =['Age_group'], 
                                          columns =['Gender'],aggfunc=np.sum ) 

df_Age_groupVsGenderPCT = df_Age_groupVsGender/df_Age_groupVsGender[df_Age_groupVsGender.columns].sum()*100

plt.rcParams['figure.figsize']=(20,10)
sns.heatmap(df_Age_groupVsGenderPCT,cmap='Greens', annot=True,annot_kws={'size':20},linecolor='Black',linewidths=0.01 , fmt="0.01f")



df_GenderVsAge_Group = df_Age_groupVsGender.T 
df_GenderVsAge_Group
df_GenderVsYear = df_YearVsGender.T
df_GenderVsYearPCT = df_GenderVsYear/df_GenderVsYear[df_GenderVsYear.columns].sum()*100

plt.rcParams['figure.figsize']=(20,10)
sns.heatmap(df_GenderVsYearPCT,cmap='Greens', annot=True,annot_kws={'size':20},linecolor='Black',linewidths=0.004, fmt="0.01f")

df_GenderVsState = df_StateVsGender.T
df_GenderVsStatePCT = df_GenderVsState/df_GenderVsState[df_GenderVsState.columns].sum()*100
df_GenderVsStatePCT

for state in (dataFrame1.State.unique()):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    category = ['Female', 'Male']
    count = [df_GenderVsState[state].Female,df_GenderVsState[state].Male]
    ax.pie(count, labels =category ,autopct='%1.2f%%' )
    plt.title(state,fontsize=18)
    plt.show()
df_StateYearVsGender = pd.pivot_table(pd.DataFrame(dataFrame1.groupby(["State","Gender","Year"])["Total"].sum()).reset_index().sort_values(by='Total')
                                    , values ='Total', index =['State','Year'], 
                         columns =['Gender'],aggfunc=np.sum ) 
 
df_StateYearVsGender
df_StateYearVsAgeGroup = pd.pivot_table(pd.DataFrame(dataFrame1.groupby(["State","Age_group","Year"])["Total"].sum()).reset_index().sort_values(by='Total')
                                    , values ='Total', index =['State','Year'], 
                         columns =['Age_group'],aggfunc=np.sum ) 
df_StateYearVsAgeGroup
