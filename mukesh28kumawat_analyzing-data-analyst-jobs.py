import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly
import seaborn as sns
%matplotlib inline
import plotly.graph_objects as go
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv")

pd.options.display.max_columns=None
df.head()
df.isnull().sum()
#In the dataset their are very few null values but their are many rows with value '-1' so we need replace these values with null values first.
df.replace({'-1': None},inplace =True, regex= True)      #to replace string values
df=df.replace({-1:None})                                 #to replace integer values
#Now we will get the actual number of null values
df.isnull().sum()
#Deleting the unnessecary columns
df.drop(['Competitors','Easy Apply'], axis=1,inplace=True)    #Most of values in these columns are null
df.drop(['Unnamed: 0','Job Description'], axis=1,inplace=True)
# First of we will check the number of null values in each row and delete the rows with more then 2 null values
a=df.isnull().sum(axis=1).to_frame('Number of null values')

a

b=[]
for i in range(2253):
    if a['Number of null values'].iloc[i]>2 :
        b.append(i)
    i=i+1

# b contains index number of rows with more than two null values      
#Now we will delete rows with more than two null values
df.drop( b, inplace=True)
df.shape                           # We have removed around 190 rows
for i in range(1912) :
    if df['Size'].iloc[i]== '1 to 50 employees' :
        df['Size'].iloc[i]='Early stage startup'
    elif df['Size'].iloc[i]== '51 to 200 employees' or df['Size'].iloc[i]== '201 to 500 employees' or df['Size'].iloc[i]== 'Unknown' :
        df['Size'].iloc[i]='Micro Enterprise'
    elif df['Size'].iloc[i]== '501 to 1000 employees' :
        df['Size'].iloc[i]='Small Enterprise'
    elif df['Size'].iloc[i]== '1001 to 5000 employees' :
        df['Size'].iloc[i]='Medium Enterprise'
    elif df['Size'].iloc[i]== '5001 to 10000 employees' or df['Size'].iloc[i]== '10000+ employees' :
        df['Size'].iloc[i]='Large Enterprise'
    i=i+1
df['Size'].value_counts()
df["Salary Estimate"].fillna('$42K-$76K (Glassdoor est.)', inplace = True) 
#Function to find out the mean salary
def salary(a) :
    a = a.replace("K", "")
    a = a.replace("$", "")
    a=a[0:-16]
    a=a.split("-")
    b=int(a[0])
    c=int(a[1])
    return (b+c)*1000/2
df['Salary Estimate']=df['Salary Estimate'].apply(salary)
df.rename(columns = {'Salary Estimate':'Salary Estimate in USD'}, inplace = True)
df["Rating"].fillna(3.7 , inplace = True) 
df["Type of ownership"].replace({"Unknown": "Company - Private"}, inplace=True)
df["Industry"].fillna('IT Services', inplace = True)
df["Sector"].fillna('Information Technology', inplace = True) 
for i in range(1912):
    if type(df['Founded'].iloc[i])== int :
        if df['Founded'].iloc[i]>2000 :
            df['Founded'].iloc[i]= 'After 2000'
        elif  1950 <df['Founded'].iloc[i]<=2000 :
            df['Founded'].iloc[i]= 'Between 1951-2000'
        elif  1900 <df['Founded'].iloc[i]<=1950 :
            df['Founded'].iloc[i]= 'Between 1901-1950'
        elif df['Founded'].iloc[i]<1900 :
            df['Founded'].iloc[i]= 'Before 1900'
    else :
        df['Founded'].iloc[i]=None
    i=i+1
    
df['Founded'].value_counts()
fig = px.histogram(df, x="Founded",nbins=20)
fig.show()
df["Founded"].fillna("unknown", inplace = True)
for i in range(1912) :
    if df['Founded'].iloc[i]=='unknown' :
        if df['Size'].iloc[i]=='Small Enterprise' or df['Size'].iloc[i]=='Micro Enterprise' or df['Size'].iloc[i]=='Early stage startup' :
            df['Founded'].iloc[i]='After 2000'
        elif df['Size'].iloc[i]=='Medium Enterprise' or df['Size'].iloc[i]=='Large Enterprise' :
            df['Founded'].iloc[i]='1951-2000'
    i=i+1
#First we will add new column that will contain state abbrevation from the column Location.
df['Location_State']=0*df['Rating']
for i in range(1912) :
    df['Location_State'].iloc[i]=df['Location'].iloc[i].split(',')[1]
    i=i+1
for i in range(1912):
    df['Location_State'].iloc[i]=df['Location_State'].iloc[i].replace(" ","")
    i=i+1

df.Location_State.unique()
#the file df1 contain the list on state in USA and their abbrivations so we will merge df1 and df to get the name of state from the abbrevation code
url='https://drive.google.com/file/d/1_m3xn7Q0U2melCCSFz-ZX_wQWob85_OJ/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df1 = pd.read_excel(path)
df1.head()
i = pd.merge(df, df1,  on ='Location_State', how ='left')
df=i.copy()

df['Location_State']=df['State']
df.drop(['State','Latitude','Longitude'], axis=1,inplace=True)
df.head()
df.isnull().sum()
df.dropna(axis=0,inplace=True)
a=df['Location_State'].value_counts()
a=a.reset_index()
a.columns=['Location_State','Number of openings']
fig = px.pie(a, values='Number of openings', names='Location_State', title='State wise distribution of Jobs')
fig.show()
b=a.copy()
b.columns=['State','Number of openings']
c = pd.merge(b, df1,  on ='State', how ='left')
df3=c.copy()
df3['Location_State']=df3['State']
df3.drop(['State'], axis=1,inplace=True)

fig = px.scatter_geo(df3,lat='Latitude',lon='Longitude',color='Location_State',size_max=40,
                     locationmode = 'USA-states',size='Number of openings',scope='usa',width=1000,height=700)
fig.show()
g=df.groupby(['Location_State'],as_index=False)[['Salary Estimate in USD']].mean()
g.columns=['State','Average salary offered in USD']
d = pd.merge(g, df1,  on ='State', how ='left')
df4=d.copy()
df4['Location_State']=df4['State']
df4.drop(['State'], axis=1,inplace=True)
fig = px.scatter_geo(df4,lat='Latitude',lon='Longitude',color='Location_State',size_max=40,
                     locationmode = 'USA-states',size='Average salary offered in USD',scope='usa',width=1000,height=700)
fig.show()
g=df['Company Name'].value_counts().nlargest(10)
g=g.reset_index()
g.columns=['Company Name','Number of openings']
g
fig=px.bar(g,x='Company Name',y='Number of openings',color='Number of openings',labels={
                     "Company Name": "Company Name",
                     "Number of openings": "Number of openings"
                     
                 },
                 title='Companies with highest number of openings')
fig.show()

g1=df.groupby(['Industry'],as_index=False)[['Salary Estimate in USD']].mean()
g1.columns=['Industry','Average salary offered in USD']
g1=g1.sort_values(by=['Average salary offered in USD'],ascending=False)
g1=g1.head(10)
g1
fig=px.bar(g1,x='Industry',y='Average salary offered in USD',color='Average salary offered in USD',labels={
                     "Industry": "Industry",
                     "Average salary offered in USD": "Average salary offered in USD"
                     
                 },
                 title='Top Paying Industries')
fig.show()

fig = px.box(df,x="Salary Estimate in USD", y="Sector",height=1000,color='Sector')
fig.show()

g2=df['Sector'].value_counts().nlargest(10)
g2=g2.reset_index()
g2.columns=['Sector','Number of openings']
g2
fig=px.bar(g2,x='Sector',y='Number of openings',color='Number of openings',labels={
                     "Sector": "Sector",
                     "Number of openings": "Number of openings"
                     
                 },
                 title='Sectors with highest number of openings')
fig.show()

g3=df['Founded'].value_counts()
g3=g3.reset_index()
g3.columns=['Founded','Number of companies']
fig = px.pie(g3, values='Number of companies', names='Founded', title='Age wise distribution of companies')
fig.show()

g4=df.groupby(['Founded'],as_index=False)[['Salary Estimate in USD']].mean()
g4.columns=['Founded','Average salary offered in USD']


fig=px.bar(g4,x='Founded',y='Average salary offered in USD',color='Average salary offered in USD',labels={
                     "Founded": "Founded",
                     "Average salary offered in USD": "Average salary offered in USD"
                     
                 },
                 title='Company age Vs Average salary')
fig.show()
g5=df.groupby(['Rating'],as_index=False)[['Salary Estimate in USD']].mean()
g5.columns=['Rating','Average salary offered in USD']
fig=px.bar(g5,x='Rating',y='Average salary offered in USD',color='Average salary offered in USD',labels={
                     "Rating": "Rating",
                     "Average salary offered in USD": "Average salary offered in USD"
                     
                 },
                 title='Rating Vs Average salary')
fig.show()
#from the bar plot we can see that their is no direct relation between rating and salary offered so we can conclude that the rating depends on other factors too.
g6=df.groupby(['Size'],as_index=False)[['Salary Estimate in USD']].count()
g6.columns=['Size','Number of Openings']
fig = px.pie(g6, values='Number of Openings', names='Size', title='Job openings as per company size')
fig.show()