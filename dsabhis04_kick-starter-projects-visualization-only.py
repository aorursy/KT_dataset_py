import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

d = pd.read_csv('../input/ks-projects-201801.csv',encoding='latin1')

df = pd.DataFrame(d)
df.head()
#mapping the state of kick starter project to 5 classes
df['state']=df['state'].map({'failed':0,'successful':1,'canceled':2,'live':3,'undefined':4,'suspended':5})
sns.countplot(x='state',data=df)
# as classes 3,4,5 are very low in count model might not train well so i am droping class 2,3,4,5. 
# Analysing only sucess & failure projects 
dfx=df[df['state']<2]
sns.countplot(x='state',data=dfx)
dfx.head()
sns.jointplot(x=dfx['state'],y=dfx['backers'],data=dfx)

sns.barplot(x='state',y='backers',data=dfx,hue='state')
sns.barplot(x='state',y='backers',data=dfx,hue='state',estimator=np.std)
sns.barplot(x='state',y='pledged',data=dfx,hue='state')
sns.barplot(x='state',y='goal',data=dfx,hue='state')
sns.barplot(x='state',y='goal',data=dfx,hue='state',estimator=np.std)
sns.countplot(x='state',data=dfx)
ax=sns.countplot(x='state',data=dfx)
ax
sns.boxplot(x='state',y='goal',data=dfx)
sns.boxplot(x='state',y='pledged',data=dfx,hue='state')
dfx.head()
sns.stripplot(x='state',y='backers',data=dfx,jitter=True)
sns.stripplot(x='state',y='goal',data=dfx,jitter=True)
sns.stripplot(x='state',y='pledged',data=dfx,jitter=True)
sns.stripplot(x='state',y='usd_goal_real',data=dfx,jitter=True)
sns.stripplot(x='state',y='usd_pledged_real',data=dfx,jitter=True)
heatt=dfx.corr()
sns.heatmap(heatt,annot=True)
cco=dfx.pivot_table(index='main_category',columns='currency',values='pledged',)
sns.heatmap(cco)

#ccd=dfx.pivot_table(index='pledged',columns='currency',values='pledged',)
#sns.heatmap(ccd)
dfx.pivot_table(index='main_category',columns='category',values='pledged')
cmp=dfx.pivot_table(index='category',columns='main_category',values='pledged')
cm=dfx.pivot_table(index='category',columns='main_category',values='state')
sns.heatmap(cm,cmap='coolwarm')
sns.heatmap(cmp,cmap='coolwarm',linecolor='black')
#dfx.head()
#dfx.head()

sns.countplot(x='country',data=dfx,hue='state',)
dfx['state'].count()


sns.countplot(x='currency',data=dfx,hue='state',)
sns.countplot(x='main_category',data=dfx,hue='state',)
