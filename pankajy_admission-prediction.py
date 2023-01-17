# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Admission_Predict.csv')
df.info()
df = df.rename(columns={'GRE Score': 'GRE_Score'})

df = df.rename(columns={'TOEFL Score': 'TOEFL_Score'})

df = df.rename(columns={'University Rating': 'University_Rating'})

df = df.rename(columns={'Chance of Admit ': 'Chance_Of_Admit'})

df = df.rename(columns={'LOR ': 'LOR'})
df.isnull().sum()
df.head()
df.tail()
df.describe()
df = df.drop(['Serial No.'], axis=1)
fig,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
cor=df.corr()['Chance_Of_Admit'].sort_values(ascending=False)

# Print the correlations

print(cor)
print('Not Having Research:',len(df[df.Research == 0]))

print('Having Research:',len(df[df.Research == 1]))

y = len(df[df.Research == 0]),len(df[df.Research == 1])

x = ['Not Having Research','Having Research']

plt.bar(x,y)

plt.title('Research Experience')

plt.show()

explode = (0.1,0)  

fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df['Research'].value_counts(), explode=explode,labels=['having_research','no_research'], autopct='%1.1f%%',

        shadow=True)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
y = ([df['TOEFL_Score'].min(),df['TOEFL_Score'].mean(),df['TOEFL_Score'].max()])

x = ['Worst','Average','Best']

plt.bar(x,y)

plt.title('TOEFL Scores')

plt.xlabel='Level'

plt.ylabel='TOEFL Score'

plt.show()
sns.scatterplot(data=df,x='GRE_Score',y='TOEFL_Score',hue='Research')
y = df.GRE_Score.min(), df.GRE_Score.mean(), df.GRE_Score.max()

x = ['Worst', 'Average', 'Best']

plt.bar(x,y)

plt.title('GRE_Scores')

plt.ylabel='GRE_Score'

plt.show()
df.GRE_Score.plot(kind = 'hist', bins=200, figsize=(7,6))

plt.title('GRE Score')

plt.show()
plt.scatter(df.University_Rating, df.CGPA)

plt.title('CGPA Scores for University Rating')

plt.xlabel=('University Rating')

plt.ylabel=('CGPA')

plt.show()
sns.scatterplot(data=df,x='GRE_Score', y='CGPA' , hue='CGPA',size='CGPA')
sns.scatterplot(data=df, x='GRE_Score', y='TOEFL_Score',size='CGPA')
x = df[df['Chance_Of_Admit'] >= 0.75]['University_Rating'].value_counts().head(5)

plt.title('University Ratings of Candidates with an 75% acceptance chance')

x.plot(kind='bar',figsize=(15, 10))

plt.xlabel=('University Rating')

plt.ylabel=('Candidates')

plt.show()
sns.scatterplot(x=df.CGPA, y=df.SOP,hue=df.SOP)
sns.scatterplot(x=df.GRE_Score, y=df.SOP,hue=df.SOP)
df[(df['Chance_Of_Admit']>0.90)].mean().reset_index()
y = df["Chance_Of_Admit"].values

x = df.drop(["Chance_Of_Admit"],axis=1)



# separating train (80%) and test (%20) sets

from sklearn.model_selection import train_test_split



x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 20)
from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(x_train,y_train)

y_head_linear = linear.predict(x_test)



print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(linear.predict(x_test.iloc[[1],:])))

print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(linear.predict(x_test.iloc[[2],:])))



from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test,y_head_linear))



y_head_linear_train = linear.predict(x_train)

print("r_square score (train dataset): ", r2_score(y_train,y_head_linear_train))
Score=['337','118','4','4.5','4.5','9.65','1']

Score=pd.DataFrame(Score).T

chance=linear.predict(Score)

chance

from sklearn.ensemble import RandomForestRegressor

randomforest = RandomForestRegressor(n_estimators = 100, random_state = 20)

randomforest.fit(x_train,y_train)

y_head_randomforest = randomforest.predict(x_test) 



from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test,y_head_randomforest))

print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(randomforest.predict(x_test.iloc[[1],:])))

print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(randomforest.predict(x_test.iloc[[2],:])))





y_head_randomforest_train = randomforest.predict(x_train)

print("r_square score (train dataset): ", r2_score(y_train,y_head_randomforest_train))