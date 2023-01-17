# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
data.head(10)
data['Role'].value_counts()
#There are 649 different Role categories let's generalize it a bit 

data['Role'] = data['Role'].str.lower()

data_job=data['Role'].astype(str) 

#Remove Special Chars from Role Category and splitting each word 

data_job=data_job.str.replace('[+-/\,.()@:|;&_~]', ' ')    

data_job1=data_job.str.split(expand=True)



#Getting the splitted words in one column for finding the frequencies for each unique word

y=1

for x in range(0, 10):

    dj1=data_job1[x].dropna()

    dj1=dj1.to_frame(name="A") 

    dj2=data_job1[y].dropna()

    dj2=dj2.to_frame(name="A") 

    dj3=dj1.append(dj2,ignore_index=True)

    if x == 0:

          dj4=dj3  

    else:

          dj4=dj4.append(dj3,ignore_index=True)

    y=y+1



#Getting the frequency of each unique word and sorting based on the frequencies.

dj5 = dj4['A'].value_counts().rename_axis('unique_values').reset_index(name='counts')

dj5["Rank"] = dj5["counts"]. rank(method='first',ascending=False) 





#Considering only the words which have the frequency 1000  or greater.

djX=dj5.loc[dj5['counts'] >= 1000]

#making the list of considered words only 

searchfor=djX.unique_values.tolist()



#Creating a generalized job category with the most occouring categories

#The loop is reversed because if any category has two words the word with highest frequency will be considered

data['Job_Cat']='other'

indices = []    

for x in reversed(searchfor):

    indices = []    

    for i, elem in enumerate(data['Role'].astype(str)):

        if x in elem:

            indices.append(i)

    data.iloc[indices,data.columns.get_loc('Job_Cat')]=x

            

#data['Job_Cat'].value_counts()



less_words=data['Job_Cat'].value_counts()

less_words=less_words.reset_index()

less_words=less_words.loc[less_words['Job_Cat'] < 100]

less_words=less_words['index'].tolist()



for x in less_words:

    indices = []    

    for i, elem in enumerate(data['Job_Cat'].astype(str)):

        if x == elem:

            indices.append(i)

    data.iloc[indices,data.columns.get_loc('Job_Cat')]='other'

    

data.loc[data['Job_Cat']=='nan', 'Job_Cat']= 'other'



data['Job_Cat'].value_counts()
data['Role Category'].value_counts()
#data['Role Category'].value_counts().head(10)



dj5=data['Role Category'].value_counts().head(30)

dj5=dj5.index.tolist()



data['New_Role_Category']='Other'

for x in dj5:

    indices = []    

    for i, elem in enumerate(data['Role Category'].astype(str)):

        if x in elem:

            indices.append(i)

    data.iloc[indices,data.columns.get_loc('New_Role_Category')]=x    



data['New_Role_Category'].value_counts()
#Removing the Blankspaces from start and end of the location and making it consistent using UpperCase

data['New_Location']=data['Location'].str.strip()

data['New_Location'] = data['New_Location'].str.upper()



#Getting Indexes of locations where there are multiple locations (i.e. if Comma(,) is present)

indices=[]

for i, elem in enumerate(data['New_Location'].astype(str)):

        if ',' in elem:

            indices.append(i)

#Updaing the new location values in those rows with 'Multi' category

data.iloc[indices,data.columns.get_loc('New_Location')]='Multi'

data['New_Location'].value_counts()



#taking the top 10 locations along with multi, so top 11 locations

locations=data['New_Location'].value_counts().head(11)

locations=locations.index.tolist()



# using the top 10 locations to genralize the locations. 

#Here we have used in opreator for cases like 'MUMBAI (WADALA)' will be considered as 'MUMBAI'

data['final_location']='OTHER'

for x in locations:

    indices = []    

    for i, elem in enumerate(data['New_Location'].astype(str)):

        if x in elem:

            indices.append(i)

    data.iloc[indices,data.columns.get_loc('final_location')]=x





data['final_location'].value_counts()
data['New_Job_Salary']=data['Job Salary'].str.strip()

data['New_Job_Salary'] = data['New_Job_Salary'].str.upper()



data.loc[(data['New_Job_Salary']=='OPENINGS: 1') | (data['New_Job_Salary']=='NOT DISCLOSED')

,'New_Job_Salary']= 'NOT DISCLOSED BY RECRUITER'



data.loc[(data['New_Job_Salary']=='BEST IN THE INDUSTRY') 

| (data['New_Job_Salary']=='OPENINGS: 2')

| (data['New_Job_Salary']=='BEST IN INDUSTRY')

,'New_Job_Salary']= 'NOT DISCLOSED BY RECRUITER'



data.loc[data['New_Job_Salary']==',', 'New_Job_Salary']= ''

data.loc[data['New_Job_Salary']=='PA.', 'New_Job_Salary']= ''



data['New_Job_Salary']=data['New_Job_Salary'].str.replace(',', '')    

data['New_Job_Salary']=data['New_Job_Salary'].str.replace('PA.', '')    

data['New_Job_Salary']=data['New_Job_Salary'].str.replace('INR.', '')    



data['New_Job_Salary'] =data['New_Job_Salary'].str.extract(r'(\d+ - \d+)')



data[['Min_Salary','Max_Salary']]=data['New_Job_Salary'].str.split('-',1,expand=True)

data['Min_Salary']=data['Min_Salary'].str.strip()

data['Max_Salary']=data['Max_Salary'].str.strip()



data[['New_Job_Salary','Min_Salary','Max_Salary']].head(10)
data['Industry_new']=data['Industry'].str.replace(',\s',',')

data[['Industry_new','X']]=data['Industry_new'].str.split(',',1,expand=True)

data['Industry_new']=data['Industry_new'].str.strip()





R=data['Industry_new'].value_counts().head(20)

R.reset_index()

R=R.drop(columns=['Industry_new'])    

    

data['Industry_Final']='Other'

indices = []    

for x in reversed(R.index):

    indices = []    

    for i, elem in enumerate(data['Industry'].astype(str)):

        if x in elem:

            indices.append(i)

    data.iloc[indices,data.columns.get_loc('Industry_Final')]=x

    

    

data['Industry_Final'].value_counts()

data['Experience']=data['Job Experience Required'].str.strip()

data['Experience'] =data['Experience'].str.extract(r'(\d+ - \d+)')



data['Experience'].value_counts().sum()



data[['Min_Ex','Max_Ex']]=data['Experience'].str.split('-',1,expand=True)

data['Min_Ex']=data['Min_Ex'].str.strip()

data['Max_Ex']=data['Max_Ex'].str.strip()
import plotly.express as px



data['Key Skills']=data['Key Skills'].str.strip()

data['Key Skills'] = data['Key Skills'].str.upper()

data_skill=data['Key Skills'].str.split('|',expand=True)



data_skill=data['Key Skills'].str.split('|',expand=True)

data_skill['Industry']=data['Industry_Final']

Industries= data['Industry_Final'].value_counts().rename_axis('industry').reset_index(name='counts').head(5)

for K in Industries['industry']:

    data_IT=data_skill.iloc[np.where(data_skill['Industry']==K)]

    data_IT=data_IT.reset_index(drop=True)

    y=1

    length=len(data_IT.columns)

    length=length-1

    for x in range(length):

        djj1=data_IT[x].dropna()

        djj1=djj1.to_frame(name="A") 

        djj2=data_IT[y].dropna()

        djj2=djj2.to_frame(name="A") 

        djj3=djj1.append(djj2,ignore_index=True)

        if x == 0:

            djj4=djj3  

        else:

            djj4=djj4.append(djj3,ignore_index=True)

            y=y+1

    djj4['A'] = djj4['A'].str.upper()

    djj4['A'] = djj4['A'].str.strip()

    djj6= djj4['A'].value_counts().rename_axis('unique_values').reset_index(name='counts').head(20)

    djj6=djj6.sort_values(by='unique_values')

    fig3 = px.line_polar(djj6,r='counts', theta='unique_values', line_close=True)

    fig3.update_traces(fill='toself')  

    fig3.update_layout(

    height=400,

    title_text=K)

    fig3.show()

    
data1=data[['Uniq Id', 'Job_Cat', 'New_Role_Category','final_location',

            'Industry_Final', 'Min_Salary', 'Max_Salary', 'Min_Ex','Max_Ex']]

data1
data1.head(5)
import matplotlib.pyplot as plt



data_loc=data1['final_location'].value_counts()

data_loc = data1['final_location'].value_counts().rename_axis('unique_values').reset_index(name='counts')

x=data_loc['unique_values']

y=data_loc['counts']

plt.bar(x,y)

plt.xticks(rotation=90)

data_loc=data1['Industry_Final'].value_counts()

data_loc = data1['Industry_Final'].value_counts().rename_axis('unique_values').reset_index(name='counts')

x=data_loc['unique_values']

y=data_loc['counts']

plt.bar(x,y)

plt.xticks(rotation=90)
data_loc=data1['Job_Cat'].value_counts()

data_loc = data1['Job_Cat'].value_counts().rename_axis('unique_values').reset_index(name='counts')

x=data_loc['unique_values']

y=data_loc['counts']

plt.bar(x,y)

plt.xticks(rotation=90)
table = pd.crosstab(data1['Job_Cat'], data1['final_location'], dropna=False,margins=True,margins_name='Total')

#table = pd.crosstab(data1['Job_Cat'], data1['final_location'], dropna=False)

table = table.drop('Total')

table=table.sort_values(by=['Total'],ascending=False)

table
table1 = pd.crosstab(data1['Industry_Final'], data1['final_location'], dropna=False,margins=True,margins_name='Total')

#table = pd.crosstab(data1['Job_Cat'], data1['final_location'], dropna=False)

table1 = table1.drop('Total')

table1=table1.sort_values(by=['Total'],ascending=False)

table1
table2 = pd.crosstab(data1['Job_Cat'], data1['Industry_Final'], dropna=False,margins=True,margins_name='Total')

table2 = table2.drop('Total')

table2=table2.sort_values(by=['Total'],ascending=False)

table2
dataSaL=data1.dropna(subset=['Min_Salary','Max_Salary'])



dataSaL.S_Max_sal=0

dataSaL.S_Min_sal=0

dataSaL=dataSaL.reset_index(drop=True)



dataSaL['Min_Salary']= dataSaL["Min_Salary"].astype(int) 

dataSaL['Max_Salary']= dataSaL["Max_Salary"].astype(int)





dataSaL.loc[dataSaL["Min_Salary"].between(0, 250000), 'S_Min_sal']= 'A. 0L - 2.5L'

dataSaL.loc[dataSaL["Min_Salary"].between(250001, 500000), 'S_Min_sal']= 'B. 2.5L - 5L'

dataSaL.loc[dataSaL["Min_Salary"].between(500001, 750000), 'S_Min_sal']= 'C. 5L - 7.5L'

dataSaL.loc[dataSaL["Min_Salary"].between(750001, 1000000), 'S_Min_sal']= 'D. 7.5L - 10L'

dataSaL.loc[dataSaL["Min_Salary"].between(1000001, 1250000), 'S_Min_sal']= 'E. 10L - 12.5L'

dataSaL.loc[dataSaL["Min_Salary"].between(1250001, 1500000), 'S_Min_sal']= 'F. 12.5L - 15L'

dataSaL.loc[dataSaL["Min_Salary"].between(1500001, 2000000), 'S_Min_sal']= 'G. 15L - 20L'

dataSaL.loc[dataSaL["Min_Salary"].between(2000001, 3000000), 'S_Min_sal']= 'H. 20L - 30L'

dataSaL.loc[dataSaL["Min_Salary"].between(3000001, 5000000), 'S_Min_sal']= 'I. 30L - 50L'

dataSaL.loc[dataSaL["Min_Salary"].between(5000001, 10000000), 'S_Min_sal']='J. 50L - 1CR'



dataSaL.loc[dataSaL["Max_Salary"].between(0, 250000), 'S_Max_sal']= 'A. 0L - 2.5L'

dataSaL.loc[dataSaL["Max_Salary"].between(250001, 500000), 'S_Max_sal']= 'B. 2.5L - 5L'

dataSaL.loc[dataSaL["Max_Salary"].between(500001, 750000), 'S_Max_sal']= 'C. 5L - 7.5L'

dataSaL.loc[dataSaL["Max_Salary"].between(750001, 1000000), 'S_Max_sal']= 'D. 7.5L - 10L'

dataSaL.loc[dataSaL["Max_Salary"].between(1000001, 1250000), 'S_Max_sal']= 'E. 10L - 12.5L'

dataSaL.loc[dataSaL["Max_Salary"].between(1250001, 1500000), 'S_Max_sal']= 'F. 12.5L - 15L'

dataSaL.loc[dataSaL["Max_Salary"].between(1500001, 2000000), 'S_Max_sal']= 'G. 15L - 20L'

dataSaL.loc[dataSaL["Max_Salary"].between(2000001, 3000000), 'S_Max_sal']= 'H. 20L - 30L'

dataSaL.loc[dataSaL["Max_Salary"].between(3000001, 5000000), 'S_Max_sal']= 'I. 30L - 50L'

dataSaL.loc[dataSaL["Max_Salary"].between(5000001, 10000000), 'S_Max_sal']='J. 50L - 1CR'



dataSaL['S_Max_sal'].value_counts()

FL=dataSaL.groupby('final_location').median()

FL.sort_values(by='Min_Salary',ascending=False)
IF=dataSaL.groupby('Industry_Final').median()

IF.sort_values(by='Min_Salary',ascending=False)
JC=dataSaL.groupby('Job_Cat').median()

JC.sort_values(by='Min_Salary',ascending=False)
tablesal = pd.crosstab(dataSaL["S_Max_sal"], dataSaL["S_Min_sal"], dropna=False,margins=True,margins_name='Total')

tablesal
dataSaL2=dataSaL.dropna(subset=['Min_Ex','Max_Ex'])

dataSaL2=dataSaL2.reset_index(drop=True)





dataSaL2['Min_Ex']= dataSaL2["Min_Ex"].astype(int) 



dataSaL2.loc[dataSaL2["Min_Ex"]== 0 ,'S_Min_Ex']= 'A. 0 Year'

dataSaL2.loc[dataSaL2["Min_Ex"]== 1 , 'S_Min_Ex']= 'B. 1 Year'

dataSaL2.loc[dataSaL2["Min_Ex"]==2, 'S_Min_Ex']= 'C. 2 Years'

dataSaL2.loc[dataSaL2["Min_Ex"]==3, 'S_Min_Ex']= 'D. 3 Years'

dataSaL2.loc[dataSaL2["Min_Ex"]==4, 'S_Min_Ex']= 'E. 4 Years'

dataSaL2.loc[dataSaL2["Min_Ex"]==5, 'S_Min_Ex']= 'F. 5 Years'

dataSaL2.loc[dataSaL2["Min_Ex"].between(6, 7), 'S_Min_Ex']='G. 6-7 Years'

dataSaL2.loc[dataSaL2["Min_Ex"].between(8, 10), 'S_Min_Ex']='H. 8-10 Years'

dataSaL2.loc[dataSaL2["Min_Ex"].between(11, 12), 'S_Min_Ex']='I. 11-12 Years'

dataSaL2.loc[dataSaL2["Min_Ex"].between(13, 15), 'S_Min_Ex']='J. 13-15 Years'

dataSaL2.loc[dataSaL2["Min_Ex"] > 15, 'S_Min_Ex']='K. 15+ Years'



dataSaL2['S_Min_Ex'].value_counts()

tablesalEx = pd.crosstab(dataSaL2["Job_Cat"], dataSaL2["S_Min_Ex"],values=dataSaL2['Min_Salary'],aggfunc=np.median, dropna=False)

tablesalEx
tablesalEx = pd.crosstab(dataSaL2["final_location"], dataSaL2["S_Min_Ex"],values=dataSaL2['Min_Salary'],aggfunc=np.median, dropna=False)

tablesalEx
tablesalEx = pd.crosstab(dataSaL2["Industry_Final"], dataSaL2["S_Min_Ex"],values=dataSaL2['Min_Salary'],aggfunc=np.median, dropna=False)

tablesalEx