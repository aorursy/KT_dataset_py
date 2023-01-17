# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore') 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Suicides in India 2001-2012.csv")
causesDf = df[df['Type_code']=='Causes']

causesDf.loc[causesDf['Type'] == 'Bankruptcy or Sudden change in Economic','Type'] = 'Bankruptcy or Sudden change in Economic Status'
causesDf.loc[causesDf['Type'] == 'Not having Children(Barrenness/Impotency','Type'] = 'Not having Children (Barrenness/Impotency)' 
causesDf.loc[causesDf['Type'] == 'Not having Children (Barrenness/Impotency','Type'] = 'Not having Children (Barrenness/Impotency)' 
causesDf.loc[causesDf['Type'] == 'Other Causes (Please Specity)','Type'] = 'Causes Not known'
diff_causes = causesDf.groupby('Type',as_index=False).agg({'Total':np.sum})
diff_causes = diff_causes.sort_values('Total',ascending=False)
#print(diff_causes)
plt.figure(figsize=(12,6))
sns.barplot(x='Type',y='Total',data=diff_causes)
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
causesDf_age_wise = causesDf.groupby('Age_group',as_index=False).agg({'Total':np.sum}).sort_values('Age_group',ascending=True)
sns.barplot(x='Age_group',y='Total',data = causesDf_age_wise)
plt.show()
#Studying the causes of death of children
children_causes_Df = causesDf[causesDf['Age_group']=='0-14']
children_reason = children_causes_Df.groupby('Type',as_index=False).agg({'Total':np.sum})
children_reason = children_reason.sort_values('Total',ascending=False)
plt.figure(figsize = (12,6))
sns.barplot(x='Type',y='Total',data=children_reason)
plt.xticks(Rotation=45,ha='right')
plt.tight_layout()
#Studying the causes of death of elder
elder_causes_Df = causesDf[causesDf['Age_group']=='60+']
elder_reason = elder_causes_Df.groupby('Type',as_index=False).agg({'Total':np.sum})
elder_reason = elder_reason.sort_values('Total',ascending=False)
plt.figure(figsize = (12,6))
sns.barplot(x='Type',y='Total',data=elder_reason)
plt.xticks(Rotation=45,ha='right')
plt.tight_layout()
gender_Df = df.groupby('Gender',as_index=False).agg({'Total':np.sum})
total_suicides = df.Total.sum()
gender_Df['Percent'] = gender_Df.Total*100.0/total_suicides
sns.barplot(x='Gender',y='Percent',data=gender_Df)
plt.show()
causes_gender_Df = causesDf[['Gender','Type','Total']].groupby(['Type','Gender'],as_index=False).agg({'Total':np.sum}).sort_values('Total',ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x='Type',y='Total',hue='Gender',data=causes_gender_Df)
plt.xticks(rotation=45,ha='right')
plt.show()
#Suicides in 2001 and 2011 are:
suicides_2001 = df[df['Year'] == 2001].Total.values.sum()
suicides_2011 = df[df['Year'] == 2011].Total.values.sum()
#Suicides per person in 2001 and 2011
suicides_per_person_2001 = suicides_2001/1028737000
suicides_per_person_2011 = suicides_2011/1210855000
increase_percent = (suicides_per_person_2011 - suicides_per_person_2001)*100/suicides_per_person_2001
print('Suicides per person in 2001:',suicides_per_person_2001)
print('Suicides per person in 2011:',suicides_per_person_2011)
print('Increase in percent of suicides from 2001 to 2011:',increase_percent)
population_state = [{'Population_2001': 356000,  'Population_2011': 381000,  'State': 'A & N Islands'}, {'Population_2001': 76210000,  'Population_2011': 84581000,  'State': 'Andhra Pradesh'}, {'Population_2001': 1098000,  'Population_2011': 1384000,  'State': 'Arunachal Pradesh'}, {'Population_2001': 26656000, 'Population_2011': 31206000, 'State': 'Assam'}, {'Population_2001': 82999000, 'Population_2011': 104099000, 'State': 'Bihar'}, {'Population_2001': 901000,  'Population_2011': 1055000,  'State': 'Chandigarh'}, {'Population_2001': 20834000,  'Population_2011': 25545000,  'State': 'Chhattisgarh'}, {'Population_2001': 220000,  'Population_2011': 344000,  'State': 'D & N Haveli'}, {'Population_2001': 158000,  'Population_2011': 243000,  'State': 'Daman & Diu'}, {'Population_2001': 13851000,  'Population_2011': 16788000,  'State': 'Delhi (Ut)'}, {'Population_2001': 1348000, 'Population_2011': 1459000, 'State': 'Goa'}, {'Population_2001': 50671000,  'Population_2011': 60440000,  'State': 'Gujarat'}, {'Population_2001': 21145000,  'Population_2011': 25351000,  'State': 'Haryana'}, {'Population_2001': 6078000,  'Population_2011': 6865000,  'State': 'Himachal Pradesh'}, {'Population_2001': 10144000,  'Population_2011': 12541000,  'State': 'Jammu & Kashmir'}, {'Population_2001': 26946000,  'Population_2011': 32988000,  'State': 'Jharkhand'}, {'Population_2001': 52851000,  'Population_2011': 61095000,  'State': 'Karnataka'}, {'Population_2001': 31841000, 'Population_2011': 33406000, 'State': 'Kerala'}, {'Population_2001': 61000, 'Population_2011': 64000, 'State': 'Lakshadweep'}, {'Population_2001': 60348000,  'Population_2011': 72627000,  'State': 'Madhya Pradesh'}, {'Population_2001': 96879000,  'Population_2011': 112374000,  'State': 'Maharashtra'}, {'Population_2001': 2294000, 'Population_2011': 2856000, 'State': 'Manipur'}, {'Population_2001': 2319000,  'Population_2011': 2967000,  'State': 'Meghalaya'}, {'Population_2001': 889000, 'Population_2011': 1097000, 'State': 'Mizoram'},{'State':'Nagaland','Population_2001':1990000,'Population_2011':1979000} ,{'Population_2001': 36805000, 'Population_2011': 41974000, 'State': 'Odisha'}, {'Population_2001': 974000,  'Population_2011': 1248000,  'State': 'Puducherry'}, {'Population_2001': 24359000, 'Population_2011': 27743000, 'State': 'Punjab'}, {'Population_2001': 56507000,  'Population_2011': 68548000,  'State': 'Rajasthan'}, {'Population_2001': 541000, 'Population_2011': 611000, 'State': 'Sikkim'}, {'Population_2001': 62406000,  'Population_2011': 72147000,  'State': 'Tamil Nadu'}, {'Population_2001': 3199000, 'Population_2011': 3674000, 'State': 'Tripura'}, {'Population_2001': 166198000,  'Population_2011': 199812000,  'State': 'Uttar Pradesh'}, {'Population_2001': 8489000,  'Population_2011': 10086000,  'State': 'Uttarakhand'}, {'Population_2001': 80176000,  'Population_2011': 91276000,  'State': 'West Bengal'}]
population_state_Df = pd.DataFrame(population_state)
suicides_state = df[df['Year']==2001].groupby('State',as_index=False).agg({'Total':np.sum})
suicides_state = suicides_state.rename(columns = {'Total':'Suicides_2001'})
suicides_state['Suicides_2011'] = df[df['Year']==2011].groupby('State',as_index=False).agg({'Total':np.sum}).Total.values
final = pd.merge(population_state_Df,suicides_state,on='State')
final['Suicides_percapita_2001'] = final['Suicides_2001']/final['Population_2001']
final['Suicides_percapita_2011'] = final['Suicides_2011']/final['Population_2011']
final = final.sort_values(['Suicides_percapita_2011','Suicides_percapita_2001'],ascending=[False,False])
new1 = pd.DataFrame([])
new1['State'] = final.State.values
new1['Year'] = ['2001']*len(final)
new1['Suicides_percapita'] = final.Suicides_percapita_2001.values
new2 = pd.DataFrame([])
new2['State'] = final.State.values
new2['Year'] = ['2011']*len(final)
new2['Suicides_percapita'] = final.Suicides_percapita_2011.values
suicides_percapita = pd.concat([new1,new2])
plt.figure(figsize=(14,8))
sns.barplot(x='State',y='Suicides_percapita',hue='Year',data=suicides_percapita)
plt.xticks(Rotation=45,ha='right')
plt.show()