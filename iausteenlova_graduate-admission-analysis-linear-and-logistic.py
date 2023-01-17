import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
DF  = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df = DF.copy()
df.isnull().sum()
df.info()
df.iloc[:,-1].head()
# Let's take the 'Chance of Admit' column into output variable

output = df.iloc[:,-1].copy()
df = df.set_index('Serial No.')
df.columns
corr = df.corr()['Chance of Admit '].sort_values()
corr
sns.set_style('darkgrid')

sns.heatmap(df.corr())
sns.scatterplot(y = 'CGPA',x='Chance of Admit ',data=df)

plt.title('CGPA impact on chance of admission')

plt.show()
sns.distplot(df['Chance of Admit '])

plt.show()
plt.figure(figsize=(20,12))

sns.countplot(df['Chance of Admit '])

plt.show()
sns.scatterplot(y = 'GRE Score',x='Chance of Admit ',data=df)

plt.title('GRE Score impact on chance of admission')

plt.show()
pp = sns.pairplot(df)

plt.show()
sns.barplot(df.Research,df['Chance of Admit '])
def fun2class(x):

    if x >= 0.75 :

        return 2

    elif x > 0.4 and x < 0.75:

        return 1

    elif x < 0.4:

        return 0
df['COAC'] = df['Chance of Admit '].apply(fun2class)
plt.figure(figsize=(10,6))



plt.subplot(1,2,1)

k1 = sns.boxplot(x=df['COAC'],y=df['SOP'])

k1.axes.set_title('SOP strenth to get Admission')



plt.subplot(1,2,2)

k1 = sns.boxplot(x=df['COAC'],y=df['LOR '])

k1.axes.set_title('LOR strenth to get Admission')

plt.show()
sns.boxplot(x=df['COAC'],y=df['University Rating'])
plt.figure(figsize=(10,6))

sns.boxplot(x = df['COAC'],y=df['CGPA'],hue=df['University Rating'])
df.head()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score
xt,XT,yt,YT = train_test_split(df.iloc[:,:-2],df.iloc[:,-2],test_size=0.25)
LRmod = LinearRegression()
LRmod.fit(xt,yt)

YTP = LRmod.predict(XT)

print(f'MSD : {round(mean_squared_error(YT,YTP),5)}')

print(f'MAE : {round(mean_absolute_error(YT,YTP),5)}')
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
Pmod = PCA(2)
x = StandardScaler().fit_transform(df.iloc[:,:-2])

x = pd.DataFrame(x)
# Pmod.fit_transform(x)

X = pd.DataFrame(Pmod.fit_transform(x))
plt.bar(range(1,3),Pmod.explained_variance_ratio_)
xt,XT,yt,YT = train_test_split(X,df.iloc[:,-2],test_size = 0.25)
LRmod.fit(xt,yt)

YTP = LRmod.predict(XT)

print(f'MSD : {round(mean_squared_error(YT,YTP),5)}')

print(f'MAE : {round(mean_absolute_error(YT,YTP),5)}')
from sklearn.linear_model import LogisticRegression
xt,XT,yt,YT = train_test_split(df.iloc[:,:-2],df.iloc[:,-1],test_size = 0.25)
Logmod = LogisticRegression()
Logmod.fit(xt,yt)

YTP = Logmod.predict(XT)

print(f'Accuracy : {accuracy_score(YT,YTP)}')
df.corr()['COAC'].sort_values(ascending = False)
xt,XT,yt,YT = train_test_split(df.iloc[:,[0,1,5]],df.iloc[:,-1],test_size = 0.25)
Logmod.fit(xt,yt)

YTP = Logmod.predict(XT)

print(f'Accuracy : {accuracy_score(YT,YTP)}')