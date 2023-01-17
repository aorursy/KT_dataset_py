#Data processing packages

import numpy as np 

import pandas as pd 



#Visualization packages

import matplotlib.pyplot as plt 

import seaborn as sns 



import warnings

warnings.filterwarnings('ignore')
# + Add Data set - Suicides in India

#data = pd.read_csv('____________________________')

data = pd.read_csv('../input/Suicides in India 2001-2012.csv')
#Find the size of the data Rows x Columns

#data.____

data.shape
#Display first 5 rows of Suicides data

#data.______

data.head()
#Find Basic Statistics like count, mean, standard deviation, min, max etc.

#data.___________(include='all')

data.describe(include='all')
#Find the the information about the fields, field datatypes and Null values

#data._______

data.info()
#Segregate the data based on Type_code

#eduDf = data[data['Type_code']=='__________'] #Education_Status

#causesDf = data[data['Type_code']=='__________'] #Causes

#meansDf = data[data['Type_code']=='__________'] #Means_adopted

#profDf = data[data['Type_code']=='__________'] #Professional_Profile

#socialDf = data[data['Type_code']=='__________'] #Social_Status

eduDf = data[data['Type_code']=='Education_Status'] 

causesDf = data[data['Type_code']=='Causes'] 

meansDf = data[data['Type_code']=='Means_adopted'] 

profDf = data[data['Type_code']=='Professional_Profile']

socialDf = data[data['Type_code']=='Social_Status']
#Extract Type, Gender and Total for each type

#eduDf = eduDf[['Type','Gender','Total']]       #Extract Type, Gender and Total  in eduDf

#causesDf = causesDf[['Type','Gender','Total']] #Extract Type, Gender and Total  in causesDf

#meansDf = meansDf[['Type','Gender','Total']]   #Extract Type, Gender and Total  in meansDf

#socialDf = socialDf[['Type','Gender','Total']] #Extract Type, Gender and Total  in socialDf

#profDf = profDf[['Type','Gender','Total']]     #Extract Type, Gender and Total  in profDf



eduDf = eduDf[['Type','Gender','Total']] #Extract Type, Gender and Total  in eduDf

causesDf = causesDf[['Type','Gender','Total']] #Extract Type, Gender and Total  in causesDf

meansDf = meansDf[['Type','Gender','Total']] #Extract Type, Gender and Total  in meansDf

socialDf = socialDf[['Type','Gender','Total']] #Extract Type, Gender and Total  in socialDf

profDf = profDf[['Type','Gender','Total']]  #Extract Type, Gender and Total  in profDf



#eduSort = eduDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False) #Groupby Type and Gender for eduDf

#causesSort = causesDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False) #Groupby Type and Gender for causesDf

#meansSort = meansDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False) #Groupby Type and Gender for meansDf

#socialSort = socialDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False) #Groupby Type and Gender for socialDf

#profSort = profDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False) #Groupby Type and Gender for profDf



eduSort = eduDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False) #Groupby Type and Gender for eduDf

causesSort = causesDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False) #Groupby Type and Gender for causesDf

meansSort = meansDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False) #Groupby Type and Gender for meansDf

socialSort = socialDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False) #Groupby Type and Gender for socialDf

profSort = profDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False) #Groupby Type and Gender for profDf
plt.figure(figsize=(12,6))

sns.barplot(x='Type',y='Total',hue='Gender',data=eduSort,palette='viridis')

plt.xticks(rotation=45,ha='right') #rotation=inclination of the text (xlabel)

plt.title('(Educational background)   vs   (Total suicides)   vs   (Gender)') 

plt.tight_layout()
causesDf.is_copy = False

causesDf.loc[causesDf['Type']=='Bankruptcy or Sudden change in Economic','Type'] = 'Change in Economic Status'

causesDf.loc[causesDf['Type']=='Bankruptcy or Sudden change in Economic Status','Type'] = 'Change in Economic Status'

causesDf.loc[causesDf['Type']=='Other Causes (Please Specity)','Type'] = 'Causes Not known'

causesDf.loc[causesDf['Type']=='Not having Children (Barrenness/Impotency','Type'] = 'Not having Children(Barrenness/Impotency'
plt.figure(figsize=(12,6))

sns.barplot(x='Type',y='Total',hue='Gender',data=causesSort,palette='viridis')

plt.xticks(rotation=45,ha='right') #rotation=inclination of the text (xlabel)

plt.title('(Causes of suicide)   vs   (Total suicides)   vs   (Gender)') 

plt.tight_layout()
plt.figure(figsize=(12,6))

sns.barplot(x='Type',y='Total',hue='Gender',data=meansSort,palette='viridis')

plt.xticks(rotation=45,ha='right') #rotation=inclination of the text (xlabel)

plt.title('(Means of suicide)   vs   (Total suicides)   vs   (Gender)') 

plt.tight_layout()
plt.figure(figsize=(12,6))

sns.barplot(x='Type',y='Total',hue='Gender',data=socialSort,palette='viridis')

plt.xticks(rotation=45,ha='right') #rotation=inclination of the text (xlabel)

plt.title('(Marriage status of suicide victim)   vs   (Total suicides)   vs   (Gender)') 

plt.tight_layout()
plt.figure(figsize=(12,6))

sns.barplot(x='Type',y='Total',hue='Gender',data=profSort,palette='viridis')

plt.xticks(rotation=45,ha='right') #rotation=inclination of the text (xlabel)

plt.title('(Professional status of suicide victim)   vs   (Total suicides)   vs   (Gender)') 

plt.tight_layout()
causes = data[data['Type_code']=='Causes']

causesGrp = causes.groupby(['State','Age_group'],as_index=False).sum()

causesGrpPvt = causesGrp.pivot(index='Age_group',columns='State',values='Total')
plt.figure(figsize=(14,6))

plt.xticks(rotation=45,ha='right')

sns.heatmap(causesGrpPvt,cmap='YlGnBu')

plt.tight_layout()
edu = data[data['Type_code']=='Education_Status']

st = edu.groupby(['State','Gender'],as_index=False).sum().sort_values('Total',ascending=False)

#Removing entries like "Total (States)", "(All India)" etc from the output

st = st[(st['State']!='Total (States)') & (st['State']!='Total (All India)') & (st['State']!='Total (Uts)')]

# values for areas are taken from wikipedia

statesArea = {'Maharashtra':307713,'West Bengal':88752,'Tamil Nadu':130058,'Andhra Pradesh':275045,'Karnataka':191791,'Kerala':38863,'Madhya Pradesh':308350,'Gujarat':196024,'Chhattisgarh':135191,'Odisha':155707,'Rajasthan':342239,'Uttar Pradesh':243290,'Assam':78438,'Haryana':44212,'Delhi (Ut)':1484,'Jharkhand':79714,'Punjab':50362,'Bihar':94163,'Tripura':10486,'Puducherry':562,'Himachal Pradesh':55673,'Uttarakhand':53483,'Goa':3702,'Jammu & Kashmir':222236,'Sikkim':7096,'A & N Islands':8249,'Arunachal Pradesh':83743,'Meghalaya':22429,'Chandigarh':114,'Mizoram':21081,'D & N Haveli':491,'Manipur':22327,'Nagaland':16579,'Daman & Diu':112,'Lakshadweep':32}

for state in statesArea.keys():

    st.loc[st['State']==state,'Area'] = statesArea[state]

st['Suicides_per_squareKm'] = st['Total']/st['Area']

sortedStates = st.sort_values('Suicides_per_squareKm',ascending=False)
plt.figure(figsize=(12,6))

sns.barplot(x='State',y='Suicides_per_squareKm',data=sortedStates,hue='Gender',palette='viridis')

plt.xticks(rotation=45,ha='right')

plt.tight_layout()
indiaOverall = data[(data['Type_code']=='Education_Status') & (data['State']=='Total (All India)')]

overall = indiaOverall.groupby(['Year'],as_index=False).sum()
plt.figure(figsize=(12,6))

plt.xticks(rotation=45,ha='right')

sns.barplot(x='Year',y='Total',data=overall,palette='viridis').set_title('Suicides in India overall')

plt.tight_layout()
Suicides_in_2012=int(overall[overall['Year']==2012]['Total'])

Suicides_in_2001=int(overall[overall['Year']==2001]['Total'])

(Suicides_in_2012 - Suicides_in_2001)*100/Suicides_in_2012