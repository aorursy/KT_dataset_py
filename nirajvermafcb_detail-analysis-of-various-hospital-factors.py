# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Loading all the necessary libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for visualisation

import seaborn as sns #for visualisation

%matplotlib inline 
hospital_data=pd.read_csv("../input/HospInfo.csv")

hospital_data.head()
hospital_data.info()
def num_missing(x):

  return sum(x.isnull())



#Applying per column:

print ("Missing values per column:")

print (hospital_data.apply(num_missing, axis=0) )#axis=0 defines that function is to be applied on each column
hospital_data.drop('Location',axis=1,inplace='True')
hospital_data.shape
hospital_data.describe()
hospital_data.columns.tolist()
unique_hospital_ownership=hospital_data['Hospital Ownership'].unique()

unique_hospital_ownership
dummy_data=pd.get_dummies(hospital_data['Hospital Ownership'])

dummy_data.head()

#dummy_data.info()
a=dummy_data['Government - Federal'].sum()

b=dummy_data['Government - Hospital District or Authority'].sum()

c=dummy_data['Government - Local'].sum()

d=dummy_data['Government - State'].sum()

e=dummy_data['Physician'].sum()

f=dummy_data['Proprietary'].sum()

g=dummy_data['Tribal'].sum()

h=dummy_data['Voluntary non-profit - Church'].sum()

i=dummy_data['Voluntary non-profit - Other'].sum()

j=dummy_data['Voluntary non-profit - Private'].sum()

list=[a,b,c,d,e,f,g,h,i,j]

list
ax=sns.barplot(y=unique_hospital_ownership,x=list,data=hospital_data)

ax.set(xlabel='Number of  hospitals', ylabel='Ownership')
a= pd.pivot_table(hospital_data,values=['Hospital overall rating'],index=['Hospital Ownership'],columns=['Hospital Type'],aggfunc='count',margins=False)



plt.figure(figsize=(10,10))

sns.heatmap(a['Hospital overall rating'],linewidths=.5,annot=True,vmin=0.01,cmap='YlGnBu')

plt.title('Total rating of the types of hospitals under the ownership of various community')
hospital_data['Hospital overall rating'].unique()
AvailableRating_data=hospital_data.drop(hospital_data[hospital_data['Hospital overall rating']=='Not Available'].index)

#AvailableRating_data.info()
sorted_rating=AvailableRating_data.sort_values(['Hospital overall rating'], ascending=False)

sorted_rating['Hospital overall rating'].head()

sorted_rating[['Hospital Name','Hospital overall rating']].head()
Unique_sorted_rating=sorted_rating['Hospital overall rating'].unique()

Unique_sorted_rating
rating_with_5=sorted_rating.loc[sorted_rating['Hospital overall rating'] =='5']

Rating_5=rating_with_5['Provider ID'].count()

#rating_with_5[['Hospital Name','Hospital overall rating']].head()

rating_with_4=sorted_rating.loc[sorted_rating['Hospital overall rating'] =='4']

Rating_4=rating_with_4['Provider ID'].count()

rating_with_3=sorted_rating.loc[sorted_rating['Hospital overall rating'] =='3']

Rating_3=rating_with_3['Provider ID'].count()

rating_with_2=sorted_rating.loc[sorted_rating['Hospital overall rating'] =='2']

Rating_2=rating_with_2['Provider ID'].count()

rating_with_1=sorted_rating.loc[sorted_rating['Hospital overall rating'] =='1']

Rating_1=rating_with_1['Provider ID'].count()

#Rating_5

#Rating_4

#Rating_3

#Rating_2

#Rating_1

list=[Rating_5,Rating_4,Rating_3,Rating_2,Rating_1]

list

print(Rating_5,Rating_4,Rating_3,Rating_2,Rating_1)
ax=sns.barplot(x=Unique_sorted_rating,y=list,data=hospital_data,palette='pastel')

ax.set(xlabel='Rating out of 5', ylabel='Number of  hospitals')
hospital_data['Hospital Type'].unique()
State_acute_5=hospital_data.loc[(hospital_data["Hospital Type"]=="Acute Care Hospitals") & (hospital_data["Hospital overall rating"]=="5"),["State"]]

State_acute_5.head()

#State_acute_5['State'].unique()
S_A_5=State_acute_5['State'].value_counts()

index=S_A_5.index

values=S_A_5.values

values
dims = (8, 10)

fig, ax = plt.subplots(figsize=dims)



ax=sns.barplot(y=index,x=values,palette='GnBu_d')

ax.set(xlabel='Total number of Acute Care hospitals with 5 rating', ylabel='States')
Critical_access_5=hospital_data.loc[(hospital_data["Hospital Type"]=="Critical Access Hospitals") & (hospital_data["Hospital overall rating"]=="5"),["State"]]

C_A_5=Critical_access_5['State'].value_counts()

C_A_5

index=C_A_5.index

values=C_A_5.values

values
dims = (8, 2)

fig, ax = plt.subplots(figsize=dims)



ax=sns.barplot(y=index,x=values,palette='YlOrBr')

ax.set(xlabel='Total number of Critical Care hospitals with 5 rating', ylabel='States')
Chidrens_5=hospital_data.loc[(hospital_data["Hospital Type"]=="Childrens") & (hospital_data["Hospital overall rating"]=="5"),["State"]]

C_5=Chidrens_5['State'].value_counts()

C_5

index=C_5.index

values=C_5.values

values

index
State_acute_1=hospital_data.loc[(hospital_data["Hospital Type"]=="Acute Care Hospitals") & (hospital_data["Hospital overall rating"]=="1"),["State"]]

State_acute_1.head()

#State_acute_1['State'].unique()

S_A_1=State_acute_1['State'].value_counts()

index=S_A_1.index

values=S_A_1.values

values
dims = (8, 10)

fig, ax = plt.subplots(figsize=dims)



ax=sns.barplot(y=index,x=values,palette='cubehelix')

ax.set(xlabel='Total number of Acute Care hospitals with 1 rating', ylabel='States')
Critical_access_1=hospital_data.loc[(hospital_data["Hospital Type"]=="Critical Access Hospitals") & (hospital_data["Hospital overall rating"]=="1"),["State"]]

C_A_1=Critical_access_1['State'].value_counts()

C_A_1

index=C_A_1.index

values=C_A_1.values

values
dims = (8, 1)

fig, ax = plt.subplots(figsize=dims)



ax=sns.barplot(y=index,x=values,palette='Spectral')

ax.set(xlabel='Total number of Critical Acess hospitals with 1 rating', ylabel='States')
Chidrens_1=hospital_data.loc[(hospital_data["Hospital Type"]=="Childrens") & (hospital_data["Hospital overall rating"]=="1"),["State"]]

C_1=Chidrens_1['State'].value_counts()

C_1

index=C_1.index

values=C_1.values

values

index
unique_hospital_type=hospital_data['Hospital Type'].unique()

#hospital_data['Hospital Type'].count()
hospital_type=hospital_data.loc[hospital_data['Hospital Type']=='Acute Care Hospitals']

Acute_care=hospital_type['Hospital Type'].count()



hospital_type=hospital_data.loc[hospital_data['Hospital Type']=='Critical Access Hospitals']

Critical_Acess=hospital_type['Hospital Type'].count()



hospital_type=hospital_data.loc[hospital_data['Hospital Type']=='Childrens']

Childrens=hospital_type['Hospital Type'].count()

list=[Acute_care,Critical_Acess,Childrens]

list
ax=sns.barplot(x=unique_hospital_type,y=list,data=hospital_data,palette='colorblind')

ax.set(xlabel='Types of hospitals', ylabel='Number of  hospitals')
hospital_data['Hospital overall rating'].unique()
clean_hospital_data=hospital_data.drop(hospital_data[hospital_data['Hospital overall rating']=='Not Available'].index)

#clean_hospital_data['Hospital overall rating'].astype(float)

clean_hospital_data['Hospital overall rating'].unique()
clean_hospital_data['Hospital overall rating']=clean_hospital_data['Hospital overall rating'].astype(float)
clean_hospital_data['Hospital overall rating'].mean()

clean_hospital_data['Hospital overall rating'].count()
Statewise_avarage_rating=clean_hospital_data.groupby('State')['Hospital overall rating'].mean()

#Statewise_avarage_rating.sort_values(ascending=False)
index=Statewise_avarage_rating.sort_values(ascending=False).index

values=Statewise_avarage_rating.sort_values(ascending=False).values

#index

#values
a4_dims = (8, 10)

fig, ax = plt.subplots(figsize=a4_dims)



ax=sns.barplot(y=index,x=values)

ax.set(xlabel='Average rating of the hospitals', ylabel='State')
Mortality_NotAvailable=hospital_data.loc[hospital_data['Mortality national comparison']=='Not Available']

Mortality_NotAvailable['Mortality national comparison'].count()
Non_available_data=Mortality_NotAvailable.groupby('Hospital Type')['Mortality national comparison'].count()

#Non_available_data

Non_available_data.sort_values(ascending=False)
index=Non_available_data.sort_values(ascending=False).index

values=Non_available_data.sort_values(ascending=False).values

#index

#values
dims = (6, 6)

fig, ax = plt.subplots(figsize=dims)



ax=sns.barplot(y=values,x=index,palette='PiYG')

ax.set(xlabel='Hospitals types', ylabel='Count of Mortality data Non-Availabilty') 
SafetyOfCare_NotAvailable=hospital_data.loc[hospital_data['Safety of care national comparison']=='Not Available']

SafetyOfCare_NotAvailable['Safety of care national comparison'].count()
SafetyOfCare_NotAvailable=hospital_data.loc[hospital_data['Safety of care national comparison']=='Not Available']

SafetyOfCare_NotAvailable['Safety of care national comparison'].count()

Non_available_data=SafetyOfCare_NotAvailable.groupby('Hospital Type')['Safety of care national comparison'].count()

#Non_available_data

Non_available_data.sort_values(ascending=False)

index=Non_available_data.sort_values(ascending=False).index

values=Non_available_data.sort_values(ascending=False).values
dims = (6, 6)

fig, ax = plt.subplots(figsize=dims)



ax=sns.barplot(y=values,x=index,palette='BrBG')

ax.set(xlabel='Hospital Types ', ylabel='Count of Safety of care data Non-Availabilty')
Readmission_NotAvailable=hospital_data.loc[hospital_data['Readmission national comparison']=='Not Available']

Readmission_NotAvailable['Readmission national comparison'].count()

Non_available_data=Readmission_NotAvailable.groupby('Hospital Type')['Readmission national comparison'].count()

#Non_available_data

Non_available_data.sort_values(ascending=False)

index=Non_available_data.sort_values(ascending=False).index

values=Non_available_data.sort_values(ascending=False).values

#index

#values
dims = (6, 7)

fig, ax = plt.subplots(figsize=dims)



ax=sns.barplot(y=values,x=index,palette='RdYlGn')

ax.set(xlabel='Hospital Types ', ylabel='Count of Readmission data Non-Availabilty')
#Still Working