#Packages

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

!pip install factor_analyzer  

from factor_analyzer import FactorAnalyzer
#Import dataset

df = pd.read_csv("../input/airline-passenger-satisfaction/train.csv")
df.head()
df.drop(['Unnamed: 0', 'id', ], axis=1, inplace=True)
df.describe()
sns.heatmap(df.isnull(), cbar=False)

df.isnull().sum()
plt.figure(figsize=(20,10))

c= df.corr()

sns.heatmap(c)
df.drop(['Arrival Delay in Minutes'], axis=1, inplace=True)
print(df[df.duplicated()])
df['satisfaction'].describe()
df.satisfaction.replace(['satisfied', 'neutral or dissatisfied'], [1,0], inplace=True)
eco = df[df['Class']=='Eco'][df.columns[6:20]].mean().mean()

eco_plus = df[df['Class']=='Eco Plus'][df.columns[6:20]].mean().mean()

business = df[df['Class']=='Business'][df.columns[6:20]].mean().mean()

print(eco, eco_plus, business)
df.groupby('Class')[df.columns[6:20]].mean()
plt.subplot(1,2,1)

df.Class.value_counts().plot(kind='bar', figsize=(10,5))

plt.title('Observations per class')

plt.subplot(1,2,2)

df[df['satisfaction']==0].Class.value_counts().plot(kind='bar', figsize=(10,5))

plt.title('Neutral or dissatisfied per class')
eco_proportion = len(df[df['Class']=='Eco'])/len(df)

bad_proportion = len(df[df['Class']=='Eco']['satisfaction']==0)/len(df[df['satisfaction']==0])

print(eco_proportion, bad_proportion)
df[df['Class']=='Eco'][df.columns[6:20]].mean()
#Subset of the data

x =df[df.columns[6:20]] 



fa = FactorAnalyzer()

fa.fit(x, 10)



#Get Eigen values and plot

ev, v = fa.get_eigenvalues()

ev

plt.plot(range(1,x.shape[1]+1),ev)
fa = FactorAnalyzer(3, rotation='varimax')

fa.fit(x)

loads = fa.loadings_

print(loads)
!pip install pingouin

import pingouin as pg
#Create factors

factor1 = df[['Food and drink', 'Seat comfort', 'Inflight entertainment', 'Cleanliness']]

factor2 = df[['On-board service', 'Baggage handling', 'Inflight service']]

factor3 = df[['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location']]



#Get cronbach alpha

factor1_alpha = pg.cronbach_alpha(factor1)

factor2_alpha = pg.cronbach_alpha(factor2)

factor3_alpha = pg.cronbach_alpha(factor3)



print(factor1_alpha, factor2_alpha, factor3_alpha)