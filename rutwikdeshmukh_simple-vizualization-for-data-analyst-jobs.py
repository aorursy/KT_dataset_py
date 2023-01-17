import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set()
df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
df.head()
df.drop(['Unnamed: 0'],axis = 1,inplace =True)
df1 = df.copy()
df1['Job Title'],df1['Department']= df1['Job Title'].str.split(',',1).str
df1['Job Title']
df1.head()
df1['Company Name'],_ = df1['Company Name'].str.split('\n',1).str
df1.head()
df1['Salary Estimate'],_ = df['Salary Estimate'].str.split('(',1).str
df1.head()
df1['Min Salary'],df1['Max Salary'] = df1['Salary Estimate'].str.split("-").str
df1.head()
df1['Min Salary'] = df1['Min Salary'].replace('',np.nan)
df1['Min Salary']=df1['Min Salary'].str.strip().str.lstrip('$').str.rstrip('K').fillna(0).astype(int)
df1['Max Salary'] = df1['Max Salary'].str.strip().str.lstrip('$').str.rstrip('K').fillna(0).astype(int)
df1
df1 = df1.drop(['Salary Estimate'], axis=1)
df1['Easy Apply'].value_counts()
df1['Easy Apply'] = df['Easy Apply'].map({'-1':0,'True':1})
df1.head()
df_easy_apply = df1[df1['Easy Apply']==1]
df_easy_apply.reset_index()
df_easy_apply_1 = df_easy_apply.groupby('Company Name')['Easy Apply'].count().reset_index()
df_easy = df_easy_apply_1.sort_values('Easy Apply', ascending=False).head(10).reset_index().drop('index',axis=1)
df_easy.head()
plt.figure(figsize=(12,5))

chart = sns.barplot(data = df_easy,x='Company Name',y='Easy Apply')

chart = chart.set_xticklabels(
        chart.get_xticklabels(),
        rotation=60,
        horizontalalignment='right',
        fontweight='light'
        )
df_employees = df1['Size'].value_counts().to_frame().reset_index()
df_employees.rename(columns = {'index':'No. of Employees','Size':'No. of Companies'},inplace=True)
df_employees
df_employees = df_employees.drop(6)
plt.figure(figsize=(12,6))
plot=sns.barplot(
       data=df_employees,
    x='No. of Companies',
    y='No. of Employees',
    )
plot = plot.set_xticklabels(
        plot.get_xticklabels(),
        rotation=65,
        horizontalalignment = 'right',
        fontweight ='light'
        )
ratingData = df1.groupby('Rating')['Company Name'].count().to_frame().reset_index()
ratingData = ratingData.sort_values('Rating').reset_index()
ratingData = ratingData.loc[16:,:] #For considering companies with rating > 3.0
ratingData = ratingData.rename(columns={'Company Name':'No. of Companies'})
plt.figure(figsize=(12,6))

plot = sns.barplot(
    data=ratingData,
    x='Rating',
    y='No. of Companies',
)

plot.set_xticklabels(
    plot.get_xticklabels(), 
    rotation=60, 
    horizontalalignment='right',
    fontweight='light',
 
)

plot.axes.yaxis.label.set_text("No. of companies")

dataOwnership = df1['Type of ownership'].value_counts().to_frame()
dataOwnership = dataOwnership.drop(['-1','Unknown'])
dataOwnership = dataOwnership.reset_index()
dataOwnership = dataOwnership.rename(columns = {'Type of ownership':'No. of companies','index':'Type of Ownership'})
dataOwnership
plt.figure(figsize=(10,6))
plot = sns.barplot(
    data = dataOwnership,
    x='Type of Ownership',
    y='No. of companies'
)

plot.set_xticklabels(
    plot.get_xticklabels(),
    rotation = 60,
    horizontalalignment='right',
)
plot.axes.yaxis.label.set_text("No. of companies")
max_sal = df1.groupby(['Sector']).mean()
max_sal = max_sal.reset_index()
max_sal = max_sal[1:]
max_sal
max_sal = max_sal.drop([19,14,15,24,8,13],axis=0) # To avoid overlapping values 
max_sal = max_sal.sort_values('Max Salary',ascending=False)
plt.figure(figsize=(12,5))
plt.scatter(max_sal['Sector'],max_sal['Max Salary'])
plt.xlabel('Sector')
plt.ylabel('Max Salary')
plt.xticks(rotation=45)