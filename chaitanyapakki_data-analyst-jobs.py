#PROBLEM STATEMENT



#1)Find best jobs by Salary, Company Rating and Location

#2)Is there correlation between rating and salary offered by the company ?(in Python)

#3)Show correlation between various features using a heatmap.

#4)Find the best jobs by salary and company rating

#5)Explore skills required in job descriptions

#6)Predict salary based on industry, location, company revenue

# Importing Pandas and NumPy

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

%matplotlib inline

import seaborn as sns

import matplotlib

matplotlib.rcParams["figure.figsize"] = (20,10)
data=pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
data.head(3)
data['Easy Apply'].value_counts()

data['Competitors'].value_counts()[0:4]

data['Rating'].value_counts()[:5]

#we can see there are number of values such as  -1, -1.0 in the data frame. this is kind of null values. we need to clean this
df1 = data.copy()
df1.columns
df1 = df1.drop(['Unnamed: 0'], axis = 'columns')
df1.columns
df2 = df1.dropna()

df2.isnull().sum()
#we can see there are number of values such as  -1, -1.0 in the data frame. this is kind of null values. we need to clean this
df2=df2.replace(-1,np.nan)

df2=df2.replace(-1.0,np.nan)

df2['Competitors'].value_counts()[:5]

df2=df2.replace('-1',np.nan)

df2['Competitors'].value_counts()[:5]

df2.columns
df2['Company Name'].unique()
#df2['Company Name'] = df2['Company Name'].apply(lambda x: str(x.strip('\n', 1).str))
df2['Company Name'],_=df2['Company Name'].str.split('\n', 1).str

df2.head(3)
df2['Salary Estimate'],_=df2['Salary Estimate'].str.split('(', 1).str

df2.head(3)
#here in the below code we have created extra columns, separating information from 'job title'
df2['Job Title'],df2['Department']=df2['Job Title'].str.split(',', 1).str
df2.head()
df2['Salary Estimate'].unique()
df3 =df2.copy()
df3['Min_Salary'], df3['Max_Salary'] = df3['Salary Estimate'].str.split('-').str



df3.columns
df3['Min_Salary'].unique()



df3['Min_Salary'] = df3['Min_Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype('int')





df3['Min_Salary'].unique()

df3['Max_Salary'].unique()

df3['Max_Salary'] = df3['Max_Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype('int')
df3['Max_Salary'].unique()

df4 = df3.copy()

df4.head(3)
df4 = df4.drop(['Salary Estimate'], axis = 'columns')
df4.columns
#current openings

df4['Easy Apply']=df4['Easy Apply'].fillna(False).astype('bool')

df4['Easy Apply'].unique()
df4_easy_apply =  df4[df4['Easy Apply'] == True]



df5 = df4_easy_apply.groupby('Company Name')['Easy Apply'].count().reset_index()



current_openings = df5.sort_values('Easy Apply',ascending=False).head(10)
current_openings.head(5)
df5.head(5)
df4_easy_apply.head(5)
plt.figure(figsize=(10,5))

chart = sns.barplot(

    data=current_openings,

    x='Company Name',

    y='Easy Apply',

    palette='Set1'

)

chart=chart.set_xticklabels(

    chart.get_xticklabels(), 

    rotation=65, 

    horizontalalignment='right',

    fontweight='light',

    

)

 
df4.head(3)
#salary distribution of data analyst
data_analyst = df4[df4['Job Title']=='Data Analyst']





sns.set(style="white", palette="muted", color_codes=True)





f, axes = plt.subplots(1, 2, figsize=(15, 8), sharex=True)

sns.despine(left=True)



#Plot a histogram and kernel density estimate



sns.distplot(data_analyst['Min_Salary'], color="b", ax=axes[0])



sns.distplot(data_analyst['Max_Salary'], color="r",ax=axes[1])



plt.setp(axes, yticks=[])

plt.tight_layout()
#top 20 cities of maximum and minimum salary
df5=df4.groupby('Location')[['Max_Salary','Min_Salary']].mean().sort_values(['Max_Salary','Min_Salary'],ascending=False).head(20)
df5.head(5)
fig = go.Figure()



fig.add_trace(go.Bar(x=df5.index,y=df5['Min_Salary'],name='Minimum salary'))

fig.add_trace(go.Bar(x=df5.index,y=df5['Max_Salary'],name='Maximum Salary'))



fig.update_layout(title='Top 20 cities with their minimum and maximum salaries',barmode='stack')



fig.show()
# top 20 roles
df6=df4.groupby('Job Title')[['Max_Salary','Min_Salary']].mean().sort_values(['Max_Salary','Min_Salary'],ascending=False).head(20)
fig = go.Figure()



fig.add_trace(go.Bar(x=df6.index,y=df6['Min_Salary'],name='Minimum salary'))

fig.add_trace(go.Bar(x=df6.index,y=df6['Max_Salary'],name='Maximum Salary'))



fig.update_layout(title='Top 20 roles with their minimum and maximum salaries',barmode='stack')



fig.show()
#revenue
df4.head(3)

df4['Revenue'].unique()
def filter_revenue(x):

    revenue=0

    if(x== 'Unknown / Non-Applicable' or type(x)==float):

        revenue=0

    elif(('million' in x) and ('billion' not in x)):

        maxRev = x.replace('(USD)','').replace("million",'').replace('$','').strip().split('to')

        if('Less than' in maxRev[0]):

            revenue = float(maxRev[0].replace('Less than','').strip())

        else:

            if(len(maxRev)==2):

                revenue = float(maxRev[1])

            elif(len(maxRev)<2):

                revenue = float(maxRev[0])

    elif(('billion'in x)):

        maxRev = x.replace('(USD)','').replace("billion",'').replace('$','').strip().split('to')

        if('+' in maxRev[0]):

            revenue = float(maxRev[0].replace('+','').strip())*1000

        else:

            if(len(maxRev)==2):

                revenue = float(maxRev[1])*1000

            elif(len(maxRev)<2):

                revenue = float(maxRev[0])*1000

    return revenue
# applying the above function to REVENUE
df4['Max_revenue']=df4['Revenue'].apply(lambda x: filter_revenue(x))
df4.head()
df7=df4.groupby('Sector')[['Max_revenue']].mean().sort_values(['Max_revenue'],ascending=False).head(20)
df7.head()
df7.reset_index(inplace = True)

df7.head()
df7
plt.figure(figsize=(10,5))

chart = sns.barplot(

    data=df7,

    x='Sector',

    y='Max_revenue'

)

chart.set_xticklabels(

    chart.get_xticklabels(), 

    rotation=65, 

    horizontalalignment='right',

    fontweight='light',

 

)

chart.axes.yaxis.label.set_text("Revenue(Million dollars)")
#jobs with openings
df9=pd.DataFrame(df4[df4['Easy Apply']==True]['Job Title'].value_counts()).rename(columns={'Job Title':'No_of_openings'})
df9
df9=df9.reset_index().rename(columns={'index':'Job Title'})
df9.head()
plt.figure(figsize=(10,5))

chart = sns.barplot(

    data=df9,

    x='Job Title',

    y='No_of_openings',

    palette='Set1'

)

chart=chart.set_xticklabels(

    chart.get_xticklabels(), 

    rotation=65, 

    horizontalalignment='right',

    fontweight='light',

 

)
#heat map
# all numeric (float and int) variables in the dataset

df4_numeric = df4.select_dtypes(include=['float64', 'int64'])

df4_numeric.head()
# correlation matrix

cor = df4_numeric.corr()

cor
# plotting correlations on a heatmap



# figure size

plt.figure(figsize=(16,8))



# heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()
