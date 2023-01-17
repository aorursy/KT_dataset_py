import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
df=pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
df.head()
sns.heatmap(df.isnull(),cmap='gnuplot')
df.isna().any()
df['Company Name'].isna().value_counts()
df.info()
unn_col=['Unnamed: 0','Job Description','Company Name']
for cols in unn_col:
    df.drop(cols,axis=1,inplace=True)
df.head()
df['Salary Estimate'].mode()[0]
df['Salary Estimate']=df['Salary Estimate'].replace('-1',df['Salary Estimate'].mode()[0])
df['Salary Estimate'],_=df['Salary Estimate'].str.split('(',1).str
df['Salary Estimate']
df['Min_Salary'],df['Max_Salary']=df['Salary Estimate'].str.split('-').str
df.head()
df['Min_Salary']=df['Min_Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype('int')
df['Max_Salary']=df['Max_Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype('int')
df.head()
df.info()
df.replace('-1',np.nan,inplace=True)
df.head()
sns.heatmap(df.isna(),cmap='viridis')
miss_values=[]
def check_null(df):
    for i in range(df.columns.shape[0]):
        if df.iloc[:,i].isnull().any():
            print('Missing values in {} : {} '.format(df.columns[i],df.iloc[:,i].isna().value_counts()[1]))
            miss_values.append(df.iloc[:,i].isna().value_counts()[1])
            i+=1

miss_val_arr=np.array(miss_values)
miss_val_arr
check_null(df)
null_cols=[]
for i in range(df.columns.shape[0]):
    if df.iloc[:,i].isnull().any():
        null_cols.append(df.columns[i])
null_arr=np.array(null_cols)
miss_val_arr
miss_val=pd.DataFrame(null_arr)
miss_val.rename(columns={0:'Column name'},inplace=True)
miss_val['Missing values']=miss_val_arr
miss_val['Percentage missing (%)']=np.round(100* miss_val['Missing values']/df.shape[0],1)
miss_val
df.drop('Competitors',axis=1,inplace=True)
df['Easy Apply']=df['Easy Apply'].fillna('False')
df['Count']=1
df_salaries=df.groupby('Salary Estimate')['Count'].sum().reset_index()
df_salaries.sort_values(by='Count',ascending=False,inplace=True)
df_salaries.head()

sns.catplot('Salary Estimate','Count',data=df_salaries,height=5,aspect=3)
plt.xticks(rotation=90)
df_salaries_top=df_salaries.head(10)
plt.figure(figsize=(10,8))
plt.bar(df_salaries_top['Salary Estimate'],df_salaries_top['Count'],color=['red','blue','green','orange','brown','purple'])
plt.xticks(rotation=45)
plt.xlabel('Salary estimates',size=15)
plt.ylabel('Number of jobs',size=15)
plt.title('Top 10 salary estimates',size=20)

fig1=plt.figure(figsize=(10,5))
ax1=fig1.add_subplot(121)
g=sns.distplot(df['Min_Salary'],ax=ax1,color='green')
ax1.set_xlabel('Minimum Salary \n Median min salary:$ {0:.1f} K'.format(df['Min_Salary'].median()))
l1=g.axvline(df['Min_Salary'].median(),color='red')

ax2=fig1.add_subplot(122)
h=sns.distplot(df['Max_Salary'],ax=ax2,color='Red')
ax2.set_xlabel('Maximum_Salary \n Median min salary:$ {0:.1f} K'.format(df['Max_Salary'].median()))
l2=h.axvline(df['Max_Salary'].median(),color='Blue')

fig2=plt.figure(figsize=(20,10))
ax1=fig2.add_subplot(111)

#Plots


g=sns.distplot(df['Min_Salary'],ax=ax1,color='green',label='Minimum salary')
h=sns.distplot(df['Max_Salary'],ax=ax1,color='Red',label='Maximum salary')

# Vertical median lines
l1=g.axvline(df['Min_Salary'].median(),color='black',label='Median min salary')
l2=h.axvline(df['Max_Salary'].median(),color='Blue',label='Median max salary')

#Font descriptions

plt.xlabel('Salary distribution',size=20)
plt.title('Min/Max salary distribution',size=20)

#Legend box
plt.legend(fontsize='x-large', title_fontsize='40')
df_ea_ind=df[df['Easy Apply']=='True']
df_ea_ind_grouped=df_ea_ind.groupby('Industry')['Easy Apply'].count().reset_index()
df_ea_ind_grouped.sort_values(by='Easy Apply',ascending=False,inplace=True)
df_ea_ind_grouped

sns.catplot('Industry','Easy Apply',data=df_ea_ind_grouped,kind='bar',height=10,aspect=2)
plt.xticks(rotation=90,size=15)
plt.ylabel('Job openings',size=20)
plt.xlabel('Industry',size=20)
plt.title('Current job openings in the industry',size=25)
ticks=np.arange(20)
plt.yticks(ticks,fontsize=15)
df_ea_loc=df_ea_ind.groupby('Location')['Easy Apply'].count().reset_index()
df_ea_loc
df_ea_loc.sort_values('Easy Apply',ascending=False,inplace=True)
sns.catplot('Location','Easy Apply',data=df_ea_loc,kind='bar',height=10,aspect=2,palette='summer')
plt.xticks(rotation=90,size=15)
plt.ylabel('Job openings',size=20)
plt.xlabel('Location',size=20)
plt.title('Current job openings in different locations',size=25)
ticks=np.arange(20)
plt.yticks(ticks,fontsize=15)
df_salaries=df.groupby('Location')[['Min_Salary','Max_Salary']].mean()
df_salaries.sort_values(by='Max_Salary',ascending=False,inplace=True)
df_salaries=df_salaries.head(10)
df_salaries
fig3=go.Figure()
fig3.add_trace(go.Bar(x=df_salaries.index,y=df_salaries['Min_Salary'],name='Minimum Salary'))
fig3.add_trace(go.Bar(x=df_salaries.index,y=df_salaries['Max_Salary'],name='Maximum Salary'))

fig3.update_layout(title='Top 10 cities with mean minimum and maximum salaries',barmode='stack')
df_sal_ind=df.groupby('Industry')[['Min_Salary','Max_Salary']].mean()

df_sal_ind=df_sal_ind.sort_values('Max_Salary',ascending=False)
df_sal_ind=df_sal_ind.head(10)
fig4=go.Figure()
fig4.add_trace(go.Bar(x=df_sal_ind.index,y=df_sal_ind['Min_Salary'],name='Minimum Salary'))
fig4.add_trace(go.Bar(x=df_sal_ind.index,y=df_sal_ind['Max_Salary'],name='Maximum Salary'))

fig4.update_layout(title='Top 10 Industries with mean minimum and maximum salaries in $',barmode='stack')
df_rate=df.groupby('Rating')['Count'].sum().reset_index()
df_rate.sort_values(by='Count',ascending=False,inplace=True)
df_rate=df_rate.iloc[1:,:].head(10)  #Since we are discounting the null values given by -1

sns.catplot('Rating','Count',data=df_rate,kind='bar',palette='winter',height=5,aspect=2)