import pandas as pd

import numpy as np





# visualization

import seaborn as sns           

import matplotlib.pyplot as plt

%matplotlib inline









#baselie model libraries 

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf 

from keras.models import Sequential

from keras.layers import Dense,Flatten







# warning ignore

import warnings 

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
data.head()
data.isna().sum()
sum(data.duplicated())
data.info()
f,ax=plt.subplots(1,2,figsize=(18,8))

data['Geography'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number of customer by countries')

ax[0].set_ylabel('count')

sns.countplot(data=data,x='Geography',hue='Exited',ax=ax[1])

ax[1].set_title('Countries:Exited vs Non Exited')

ax[1].set_ylabel('count');
f,ax=plt.subplots(1,2,figsize=(18,8))

data['Gender'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number of customer by gender')

ax[0].set_ylabel('count')

sns.countplot(data=data,x='Gender',hue='Exited',ax=ax[1])

ax[1].set_title('Gender:Exited vs Non Exited')

ax[1].set_ylabel('count');
Non_Exited = data[data['Exited']==0]

Exited = data[data['Exited']==1]
plt.subplots(figsize=(18,8))

sns.distplot(Non_Exited['Age'])

sns.distplot(Exited['Age'])

plt.title('Age:Exited vs Non Exited')

plt.legend([0,1],title='Exited')

plt.ylabel('percentage');
pd.crosstab(data.NumOfProducts,data.Exited,margins=True).style.background_gradient(cmap='OrRd')
f,ax = plt.subplots(1,2,figsize=(18,8))

data['NumOfProducts'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number of customer by Number of Product')

ax[0].set_ylabel('count')

sns.countplot(data=data,x='NumOfProducts',hue='Exited',ax=ax[1])

ax[1].set_title('Number of Product:Exited vs Non Exited')

ax[1].set_ylabel('count');
plt.figure(figsize=(18,8))

plt.hist(x='CreditScore',bins=100,data=Non_Exited,edgecolor='black',color='red')

plt.hist(x='CreditScore',bins=100,data=Exited,edgecolor='black',color='blue')

plt.title('Credit score: Exited vs Non-Exited')

plt.legend([0,1],title='Exited');
plt.figure(figsize=(18,8))

p1=sns.kdeplot(Non_Exited['Balance'], shade=True, color="r")

p1=sns.kdeplot(Exited['Balance'], shade=True, color="b");

plt.title('Account Balance: Exited vs Non-Exited')

plt.legend([0,1],title='Exited');
sns.factorplot('IsActiveMember','Exited',col='Tenure',col_wrap=4,data=data);
f,ax = plt.subplots(1,3,figsize=(18,8))

data['HasCrCard'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number of customer by credit card')

ax[0].set_xlabel('Credit card')



sns.countplot(data=data,x='HasCrCard',hue='Exited',ax=ax[1])

ax[1].set_title('Number of Product: Exited vs Non Exited')

ax[1].set_ylabel('count');



sns.boxplot(data=data,y='CreditScore',x='HasCrCard',hue='Exited',ax=ax[2])

ax[2].set_title('Credit card & score: Exited vs Non Exited');
plt.figure(figsize=(18,8))

plt.hist(x='EstimatedSalary',bins=100,data=Non_Exited,edgecolor='black',color='red')

plt.hist(x='EstimatedSalary',bins=100,data=Exited,edgecolor='black',color='blue')

plt.title('Estimated salary: Exited vs Non-Exited')

plt.legend([0,1],title='Exited');
plt.title("features correlation matrics".title(),

          fontsize=20,weight="bold")



sns.heatmap(data.corr(),annot=True,cmap='RdYlBu',linewidths=0.2, vmin=-1, vmax=1,linecolor = 'black') 

fig=plt.gcf()

fig.set_size_inches(10,8);