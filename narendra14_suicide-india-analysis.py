import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

suicideData = pd.read_csv("../input/Suicides in India 2001-2012.csv")

suicideData.head(5)
union_territory = {'D & N Haveli', 'Chandigarh','Lakshadweep','Daman & Diu','Puducherry','Delhi (Ut)','A & N Islands'}

states = set(suicideData['State']) - {'Total (All India)', 'Total (Uts)', 'Total (States)'} - union_territory

status_total = suicideData[suicideData['State'] == 'Total (All India)'] 

status_states = suicideData[suicideData['State'] == 'Total (Uts)']

status_uts = suicideData[suicideData['State'] == 'Total (States)']
# State/UT wise suicide Number from 2001-2012



no_suicide_state = []

total = states.union(union_territory)

for i in total:

    temp = suicideData[suicideData['State'] == i]

    no_suicide_state.append(sum(temp['Total']))



df = pd.DataFrame(no_suicide_state,index=total,columns=pd.Index(['suicides']))

df.sort_values(by='suicides', inplace=True,ascending=False)

df.plot(kind='bar')

plt.show()

# No. of Female and Male suicide number

index_ = df.index

grp_ = suicideData.groupby(['Gender','State']).sum()



for i in index_:

    female = grp_['Total']['Female'][i]

    male = grp_['Total']['Male'][i]

    df.loc[i,'Female'] = female

    df.loc[i,'Male'] = male



df.loc[:,['Female','Male']].plot(kind='bar', stacked=True)

plt.show()

typecode = set(suicideData['Type_code'])

typecode
education_states = suicideData[suicideData['Type_code'] == 'Education_Status']

education_level = set(education_states['Type'])



res_ = education_states.groupby(['Type','Gender']).sum()

education_plt = pd.DataFrame([0]*len(education_level),index=list(education_level), columns=pd.Index(['dummy']))



for i in education_level:

    female = res_['Total'][i]['Female']

    male = res_['Total'][i]['Male']

    education_plt.loc[i,'Female'] = female

    education_plt.loc[i,'Male'] = male



education_plt.sort_values(by='Male', inplace=True, ascending=False)

education_plt.loc[:,['Female','Male']].plot(kind='bar',stacked=True)

plt.show()

Social_states = suicideData[suicideData['Type_code'] == 'Social_Status']

Social_level = set(Social_states['Type'])



res_ = Social_states.groupby(['Type','Gender']).sum()

Social_plt = pd.DataFrame([0]*len(Social_level),index=list(Social_level), columns=pd.Index(['dummy']))



for i in Social_level:

    female = res_['Total'][i]['Female']

    male = res_['Total'][i]['Male']

    Social_plt.loc[i,'Female'] = female

    Social_plt.loc[i,'Male'] = male



Social_plt.sort_values(by='Male', inplace=True, ascending=False)

Social_plt.loc[:,['Female','Male']].plot(kind='bar',stacked=True)

plt.show()

Professional_states = suicideData[suicideData['Type_code'] == 'Professional_Profile']

Professional_level = set(Professional_states['Type'])



Professional_ = []

for i in Professional_level:

    tmp = Professional_states[Professional_states['Type'] == i]

    Professional_.append(sum(tmp['Total']))



Professional_plt = pd.DataFrame(Professional_,index=Professional_level,columns=pd.Index(['suicides']))

Professional_plt.sort_values(by='suicides', inplace=True, ascending=False)

Professional_plt.plot(kind='bar')

plt.show()
Means_adopted_states = suicideData[suicideData['Type_code'] == 'Means_adopted']

Means_adopted_level = set(Means_adopted_states['Type'])



res_ = Means_adopted_states.groupby(['Type','Gender']).sum()

Means_adopted_plt = pd.DataFrame([0]*len(Means_adopted_level),index=list(Means_adopted_level), columns=pd.Index(['dummy']))



for i in Means_adopted_level:

    female = res_['Total'][i]['Female']

    male = res_['Total'][i]['Male']

    Means_adopted_plt.loc[i,'Female'] = female

    Means_adopted_plt.loc[i,'Male'] = male



Means_adopted_plt.sort_values(by='Male', inplace=True, ascending=False)

Means_adopted_plt.loc[:,['Female','Male']].plot(kind='bar',stacked=True)

plt.show()

Causes_states = suicideData[suicideData['Type_code'] == 'Causes']

Causes_level = set(Causes_states['Type'])



res_ = Causes_states.groupby(['Type','Gender']).sum()

Causes_plt = pd.DataFrame([0]*len(Causes_level),index=list(Causes_level), columns=pd.Index(['dummy']))



for i in Causes_level:

    female = res_['Total'][i]['Female']

    male = res_['Total'][i]['Male']

    Causes_plt.loc[i,'Female'] = female

    Causes_plt.loc[i,'Male'] = male



Causes_plt.sort_values(by='Male', inplace=True, ascending=False)

Causes_plt.loc[:,['Female','Male']].plot(kind='bar',stacked=True)

plt.show()

# Age group

res_ = suicideData.groupby(['Age_group','Gender']).sum()



age_ = {'0-14','15-29','30-44','45-59','60+'}

df = pd.DataFrame([0]*len(age_),index=list(age_), columns=pd.Index(['dummy']))



for i in age_:

    female = res_['Total'][i]['Female']

    male = res_['Total'][i]['Male']

    df.loc[i,'Female'] = female

    df.loc[i,'Male'] = male



df.sort_values(by='Male', inplace=True, ascending=False)

df.loc[:,['Female','Male']].plot(kind='bar',stacked=True)

plt.show()