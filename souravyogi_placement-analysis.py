import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt 
data=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
placed=pd.DataFrame(data[data['status']=='Placed'])
data.info()
data['status'].value_counts()
data['salary']=data['salary'].fillna(0.0)
sns.scatterplot(x=data['mba_p'],y=data['etest_p'])
sns.regplot(x=data['mba_p'],y=data['etest_p'])
sns.lmplot(x='mba_p',y='etest_p',hue='status',data=data)
sns.swarmplot(x=data['status'],y=data['mba_p'])
sns.swarmplot(x=data['status'],y=data['etest_p'])
sns.swarmplot(x=data['status'],y=data['degree_p'])
m_data=placed[placed['gender']=='M']

f_data=placed[placed['gender']=='F']
sns.distplot(a=m_data['salary'],kde=False,label='Salary for Men')

sns.distplot(a=f_data['salary'],kde=False,label='Salary for Women')



plt.legend()
sns.barplot(y=m_data['salary'],x=data['gender'])

sns.barplot(y=f_data['salary'],x=data['gender'])
sns.barplot(x=placed['gender'],y=placed['mba_p'])
data.head()
sns.barplot(x=data['specialisation'],y=data['salary'])
sns.barplot(x=data['specialisation'],y=data['mba_p'])
ndata=placed.groupby(['gender','specialisation'])[['salary']].mean().reset_index()
ndata
ndata['ss']=['female finance','female HR','male finance','male HR']
sns.barplot(x=ndata['ss'],y=ndata['salary'])
sns.countplot(x='workex',data=data)
sns.barplot(x=placed['workex'],y=placed['salary'])
sns.barplot(x=placed['ssc_b'],y=placed['salary'])
ismale=(data['gender']=='M')
isfemale=(data['gender']=='F')
isplaced=(data['status']=='Placed')
ismale.value_counts()#There are 139 boys and 76 girls
isplaced.value_counts()#148 students are placed and 67 arent
(ismale & isplaced).value_counts()#Out of 148 boys 100 got placed
100/148 # Probability of a boy getting placed is 0.675
(isfemale&isplaced).value_counts()#Out of 76 girls 48 got placed
48/76 #Probability of a girl geting placed is 0.63

#Therefore boys have a higher chance of getting placed
sns.regplot(x=data['ssc_p'],y=data['hsc_p'])
placed
sns.regplot(x=placed['degree_p'],y=placed['salary'])
sns.countplot(x='status',data=data)
values = [(data['ssc_p'].mean()),(data['hsc_p'].mean()),(data['mba_p'].mean()),(data['degree_p'].mean()),(data['etest_p'].mean())]

names=['ssc_p','hsc_p','mba_p','degree_p','etest_p']
sns.barplot(x=names,y=values)
data['workex'].replace(to_replace='Yes',value=1,inplace=True)

data['workex'].replace(to_replace='No',value=0,inplace=True)