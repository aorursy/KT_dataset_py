import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
battles = pd.read_csv('../input/battles.csv',index_col=0)
battles.head()
chardeath = pd.read_csv('../input/character-deaths.csv')
chardeath.head()
chardeath.shape
battles.shape
charpred = pd.read_csv('../input/character-predictions.csv',index_col=0)
charpred.head()
charpred.shape
plt.figure(figsize = (10,6))
sns.heatmap(battles.isna(),cmap='coolwarm')
battles1 = battles.drop(['attacker_2','attacker_3','attacker_4','defender_2','defender_3','defender_4','note','attacker_size','defender_size'],axis=1)
plt.figure(figsize = (10,6))
sns.heatmap(battles1.isna(),cmap='coolwarm')
sns.set_style('darkgrid')
sns.countplot(battles1['year'])
battles1['attacker_king'].value_counts()
attacker_outcome = pd.get_dummies(battles1['attacker_outcome'],drop_first=True,dummy_na=False)
battles1 = pd.concat([battles1,attacker_outcome['win']],axis=1)
battles1.head()
plt.figure(figsize=(15,5))
sns.barplot(x='attacker_1',y='win',data= battles1)
sns.heatmap(chardeath.isna())
chardeath1 = chardeath.drop(['Death Year','Book of Death','Death Chapter'],axis=1)
chardeath1.head()
sns.countplot(x='Gender',hue= 'Nobility',data=chardeath1)
plt.figure(figsize = (18,6))
sns.countplot(y='Allegiances',data=chardeath1)
