import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

import warnings

import statistics as st

warnings.filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.head()
print('Number of Rows & Columns: ' , df.shape)
#Label 1 if x>0.80 and 0 if x<=0.80



df['ChanceAdmit'] = df['Chance of Admit '].map(lambda x : 1 if x>0.80 else 0)
df.describe().T[1:7]
fig,ax = plt.subplots(1,2,figsize=(10,5))

sns.scatterplot(df['GRE Score'] , df['Chance of Admit ']  , hue=df['ChanceAdmit'], ax=ax[0])

sns.scatterplot(df['TOEFL Score'] , df['Chance of Admit '] ,hue=df['ChanceAdmit'] , ax=ax[1])

ax[0].set_title('GRE Score vs Chance of Admit')

ax[1].set_title('TOEFL Score vs Chance of Admit')

plt.show()
fig,ax = plt.subplots(1,2,figsize=(10,5))

sns.barplot(df['Research'] , df['GRE Score'] ,ax=ax[0])

sns.barplot(df['Research'] , df['GRE Score'] , ax=ax[1])

ax[0].set_title('GRE Score vs Research experience')

ax[1].set_title('TOEFL Score vs Research experience')

plt.show()
sns.scatterplot(df['CGPA'] , df['Chance of Admit '] ,hue=df['ChanceAdmit'])

plt.axvline(df['CGPA'].mean())

plt.title('CGPA vs Chance of Admit')

plt.show()
df[(df['CGPA']<8.5) & (df['ChanceAdmit']==1)].T
fig,ax = plt.subplots(1,2,figsize=(10,5))

df.groupby(['Research'])['Research'].count().plot.pie(autopct='%.f%%' ,ax=ax[0] ,shadow=True )

ax[0].set_title('% of students having Research experience')

df.groupby(['University Rating'])['University Rating'].count().plot.pie(autopct='%.f%%' , ax=ax[1],shadow=True)

ax[1].set_title('University Rating of students')

plt.show()
(df['ChanceAdmit'][df['Research']==1].value_counts().sort_index()/df['Research'].value_counts().sort_index()).plot(kind='bar',\

                                                                                                            color=['r','g'])

plt.title('% of people admitted based on research experience')

plt.xlabel('Admitted or Not')

plt.ylabel('%')

plt.show()
fig,ax = plt.subplots(1,2,figsize=(10,5))

sns.barplot(df['University Rating'] , df['SOP'] ,ax=ax[0])

sns.barplot(df['University Rating'] , df['LOR '] , ax=ax[1])

ax[0].set_title('SOP vs University Rating')

ax[1].set_title('LOR vs University Rating')

plt.show()
sns.barplot(df['University Rating']  , df['CGPA'])

plt.title('CGPA vs University Rating')

plt.show()
fig,ax = plt.subplots(1,2,figsize=(10,5))

sns.barplot(df['ChanceAdmit'] , df['SOP'] ,ax=ax[0])

sns.barplot(df['ChanceAdmit'] , df['LOR '] , ax=ax[1])

ax[0].set_title('SOP vs ChanceAdmit')

ax[1].set_title('LOR vs ChanceAdmit')

plt.show()
sns.pointplot(df['University Rating'] , df['Chance of Admit '])

plt.title('Chance of Admit vs University Rating')

plt.show()
df.groupby(['ChanceAdmit'])['ChanceAdmit'].count().plot.pie(autopct='%.f%%' , shadow=True)

plt.title('% of Students admitted')

plt.legend(['Not Admitted','Admitted'])

plt.show()
from IPython.display import Image 

Image("/kaggle/input/graduate-admission-dashboard/Graduate_Admission_Dashboard.JPG")