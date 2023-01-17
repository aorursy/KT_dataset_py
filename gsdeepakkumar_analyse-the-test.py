import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

school = pd.read_csv('../input/2016 School Explorer.csv')
test = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')

print('Number of rows in school explorer:',school.shape[0],'Number of columns in school explorer',school.shape[1],end='\n')
print('Number of rows SHSAT Data:',test.shape[0],'Number of columns in SHSAT Data',test.shape[1],end='\n')
school.head(3)
test.head(4)
school['School Name'].nunique()
school.groupby('City')['School Name'].nunique().sort_values(ascending=False)
grade=school['Grades'].str.split(',').apply(pd.Series ,1).stack()
grade.index=grade.index.droplevel(-1)
grade=grade.to_frame()
grade.columns=['Grade']
grade.head()
plt.figure(figsize=(10,8))
g=sns.countplot(x=grade['Grade'],data=grade,order=['PK','0K','1K','2K','3K','4K','5K','6K','7K','8K','9K','10K','11K','12K'],palette=sns.color_palette('Set3'))
g.set_title('Grade Distribution across the school',fontsize=16)
g.set_xlabel('Grade',fontsize=10)
g.set_ylabel('Count',fontsize=10)

plt.figure(figsize=(8,8))
sns.distplot(school['Economic Need Index'].dropna(),kde=False,color='green').set_title('Distribution of the Economic Need Index across the schools',fontsize=16)
school['Community School?'].value_counts()
plt.figure(figsize=(10,8))
ax=sns.kdeplot(school.loc[(school['Community School?']=='Yes'),'Economic Need Index'].dropna(),color="green",shade=True,label="Community School")
ax=sns.kdeplot(school.loc[(school['Community School?']=='No'),'Economic Need Index'].dropna(),color="pink",shade=True,label=" Non Community School")
ax.set_title("Economic Need Index and Community School")
school['School Income Estimate']=school['School Income Estimate'].str.replace(",","")
school['School Income Estimate']=school['School Income Estimate'].str.replace("$","")
school['School Income Estimate']=school['School Income Estimate'].astype(float)
plt.figure(figsize=(8,8))
ax=sns.kdeplot(school.loc[(school['Community School?']=='Yes'),'School Income Estimate'].dropna(),color="green",shade=True,label="Community School")
ax=sns.kdeplot(school.loc[(school['Community School?']=='No'),'School Income Estimate'].dropna(),color="pink",shade=True,label=" Non Community School")
ax.set_title("School Income Estimate and Community School")
plt.figure(figsize=(10,15))
sns.set_style("whitegrid")
my_order = school.groupby(by=["City"])["School Income Estimate"].max().iloc[::-1].index
g=sns.boxplot(x=school['School Income Estimate'],y=school['City'],data=school,hue=school['Community School?'],palette=sns.color_palette(palette="dark"),order=my_order)
g.set_title("Boxplot of City Vs Income estimate",fontsize=20)
g.set_xlabel("School Income Estimate",fontsize=16)
g.set_ylabel("City",fontsize=16)
sns.set(font_scale=2)
g=sns.FacetGrid(school,col="Community School?",size=8)
g.map(sns.regplot,"Economic Need Index","School Income Estimate",fit_reg=True)
test['School name']=test['School name'].str.upper()
school_test = pd.merge(school,test,how="left",left_on=['School Name'],right_on=['School name'])
school_test.shape
trend=test.groupby(['Year of SHST'])['Enrollment on 10/31','Number of students who registered for the SHSAT','Number of students who took the SHSAT'].sum().reset_index()
trend['Year of SHST']=pd.to_datetime(trend['Year of SHST'],format="%Y")
trend['Year of SHST']=trend['Year of SHST'].dt.year
trend.head()
trace1 = go.Bar(
    x=trend['Year of SHST'],
    y=trend['Number of students who registered for the SHSAT'],
    orientation = 'v',
    name = "Number of students who registered"
)
trace2 = go.Bar(
    x=trend['Year of SHST'],
    y=trend['Number of students who took the SHSAT'],
    orientation = 'v',
    name = "Number of students who took the test")
layout = go.Layout(
    title='SHSAT:Students who enrolled Vs Students who wrote',
    barmode='group'
    #width = 1000,
    #height = 4500
    #yaxis=dict(tickangle=-45),
)

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="trend")
