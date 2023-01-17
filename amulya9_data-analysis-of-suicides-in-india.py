import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/Suicides in India 2001-2012.csv')

eduDf = df[df['Type_code']=='Education_Status']

causesDf = df[df['Type_code']=='Causes']

meansDf = df[df['Type_code']=='Means_adopted']

profDf = df[df['Type_code']=='Professional_Profile']

socialDf = df[df['Type_code']=='Social_Status']
plt.figure(figsize=(12,6))

eduDf = eduDf[['Type','Gender','Total']]

edSort = eduDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False)

sns.barplot(x='Type',y='Total',hue='Gender',data=edSort,palette='viridis')

plt.xticks(rotation=45,ha='right')

plt.tight_layout()
plt.figure(figsize=(9,6))

socialDf = socialDf[['Type','Gender','Total']]

socialSort = socialDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False)

sns.barplot(x='Type',y='Total',data=socialSort,hue='Gender',palette='viridis')

plt.xticks(rotation=45,ha='right')

plt.tight_layout()
causesDf.is_copy = False

causesDf.loc[causesDf['Type']=='Bankruptcy or Sudden change in Economic','Type'] = 'Change in Economic Status'

causesDf.loc[causesDf['Type']=='Bankruptcy or Sudden change in Economic Status','Type'] = 'Change in Economic Status'

causesDf.loc[causesDf['Type']=='Other Causes (Please Specity)','Type'] = 'Causes Not known'

causesDf.loc[causesDf['Type']=='Not having Children (Barrenness/Impotency','Type'] = 'Not having Children(Barrenness/Impotency'

plt.figure(figsize=(12,6))

causesDf = causesDf[['Type','Gender','Total']]

causesSort = causesDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False)

sns.barplot(x='Type',y='Total',data=causesSort,hue='Gender',palette='viridis')

plt.xticks(rotation=45,ha='right')

plt.tight_layout()
plt.figure(figsize=(12,6))

profDf = profDf[['Type','Gender','Total']]

profSort = profDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False)

sns.barplot(x='Type',y='Total',data=profSort,hue='Gender',palette='viridis')

plt.xticks(rotation=45,ha='right')

plt.tight_layout()
causes = df[df['Type_code']=='Causes']

causesGrp = causes.groupby(['State','Age_group'],as_index=False).sum()

causesGrpPvt = causesGrp.pivot(index='Age_group',columns='State',values='Total')

plt.figure(figsize=(14,6))

plt.xticks(rotation=45,ha='right')

sns.heatmap(causesGrpPvt,cmap='YlGnBu')

plt.tight_layout()
edu = df[df['Type_code']=='Education_Status']

plt.figure(figsize=(12,6))

st = edu.groupby(['State','Gender'],as_index=False).sum().sort_values('Total',ascending=False)

st = st[(st['State']!='Total (States)') & (st['State']!='Total (All India)') & (st['State']!='Total (Uts)')]

# values for areas are taken from wikipedia

statesArea = {'Maharashtra':307713,'West Bengal':88752,'Tamil Nadu':130058,'Andhra Pradesh':275045,'Karnataka':191791,'Kerala':38863,'Madhya Pradesh':308350,'Gujarat':196024,'Chhattisgarh':135191,'Odisha':155707,'Rajasthan':342239,'Uttar Pradesh':243290,'Assam':78438,'Haryana':44212,'Delhi (Ut)':1484,'Jharkhand':79714,'Punjab':50362,'Bihar':94163,'Tripura':10486,'Puducherry':562,'Himachal Pradesh':55673,'Uttarakhand':53483,'Goa':3702,'Jammu & Kashmir':222236,'Sikkim':7096,'A & N Islands':8249,'Arunachal Pradesh':83743,'Meghalaya':22429,'Chandigarh':114,'Mizoram':21081,'D & N Haveli':491,'Manipur':22327,'Nagaland':16579,'Daman & Diu':112,'Lakshadweep':32}

for state in statesArea.keys():

    st.loc[st['State']==state,'Area'] = statesArea[state]

st['Suicides_per_squareKm'] = st['Total']/st['Area']

sortedStates = st.sort_values('Suicides_per_squareKm',ascending=False)

sns.barplot(x='State',y='Suicides_per_squareKm',data=sortedStates,hue='Gender',palette='viridis')

plt.xticks(rotation=45,ha='right')

plt.tight_layout()
indiaOverall = df[(df['Type_code']=='Education_Status') & (df['State']=='Total (All India)')]

overall = indiaOverall.groupby(['Year'],as_index=False).sum()

print(overall)

plt.figure(figsize=(9,4))

plt.xticks(rotation=45,ha='right')

sns.barplot(x='Year',y='Total',data=overall,palette='viridis').set_title('Suicides in India overall')

plt.tight_layout()