# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/cities_r2.csv')
df['Grad_%'] = df['total_graduates']/df['population_total']

df['Grad_Percentage'] = df['Grad_%']

df['M_Pop'] = df.population_male/df.population_total 

df['F_Pop'] = df.population_female/df.population_total 

df['Child_Pop'] = df['0-6_population_total']/df.population_total 

df['MGrad_Perc'] = df['male_graduates']/df.population_total 

df['FGrad_Perc'] = df['female_graduates']/df.population_total 

df['GradGap'] = df['MGrad_Perc']-df['FGrad_Perc']

df['Literate_Perc'] = df['literates_total']/df.population_total 

df['Literate_MP'] = df['literates_male']/df.population_male 

df['Literate_FP'] = df['literates_female']/df.population_female 

df['Literacy_Gap'] = df['Literate_MP']-df['Literate_FP']
df = df.sort_values('population_total', ascending = False)

fig, ax = plt.subplots(figsize = (10,10))

sns.stripplot(y="state_name", x="population_total", data=df, jitter=True)
gendertotal = df['population_male'].sum()/df['population_total'].sum()

gendertotal
sorted_df = df.sort_values(['M_Pop'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).M_Pop)

plt.yticks(range(10),sorted_df.tail(10).name_of_city, fontsize = 10)

plt.plot([.5,.5],[-1,10], '--',color = 'r')

plt.title('Most Male')

plt.xlim([0,1])



plt.subplot(2,1,2)

plt.style.use('fivethirtyeight')

plt.barh(range(10),sorted_df.head(10).M_Pop)

plt.yticks(range(10),sorted_df.head(10).name_of_city, fontsize = 10)

plt.plot([.5,.5],[-1,10], '--',color = 'r')

plt.title('Most Female')

plt.xlim([0,1])

plt.xlabel('Male Population Percent by City')
sorted_df = df.sort_values(['population_total'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).M_Pop)

plt.yticks(range(10),sorted_df.tail(10).name_of_city, fontsize = 10)

plt.plot([.5,.5],[-1,10], '--',color = 'r')

plt.title('Male Percentage in Largest Cities')

plt.xlim([0,1])
childtotal = df['0-6_population_male'].sum()/df['0-6_population_total'].sum()

childtotal
df['Child_Pop'] = df['0-6_population_total']/df.population_total 

df['Child_Pop'].mean()

sorted_df = df.sort_values(['Child_Pop'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).Child_Pop)

plt.yticks(range(10),sorted_df.tail(10).name_of_city, fontsize = 10)

plt.plot([.107,.107],[-1,10], '--',color = 'r')

plt.title('Most Children')

plt.xlim([0,.5])



plt.subplot(2,1,2)

plt.style.use('fivethirtyeight')

plt.barh(range(10),sorted_df.head(10).Child_Pop)

plt.yticks(range(10),sorted_df.head(10).name_of_city, fontsize = 10)

plt.plot([.107,.107],[-1,10], '--',color = 'r')

plt.title('Least Children')

plt.xlim([0,.5])

plt.xlabel('Child Population Percent by City')
sorted_df = df.sort_values(['population_total'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).Child_Pop)

plt.yticks(range(10),sorted_df.tail(10).name_of_city, fontsize = 10)

plt.plot([.107,.107],[-1,10], '--',color = 'r')

plt.title('Child Percentage in Largest Cities')

plt.xlim([0,.5])



plt.subplot(2,1,2)

plt.barh(range(10),sorted_df.head(10).Child_Pop)

plt.yticks(range(10),sorted_df.head(10).name_of_city, fontsize = 10)

plt.plot([.107,.107],[-1,10], '--',color = 'r')

plt.title('Child Percentage in Smallest Cities')

plt.xlim([0,.5])

lit_total = df['literates_total'].sum()/df['population_total'].sum()

lit_m = df['literates_male'].sum()/df['population_male'].sum()

lit_f = df['literates_female'].sum()/df['population_female'].sum()
#Overall Literacy Rate

lit_total 
#Male Literacy Rate

lit_m 
#Female Literacy Rate 

lit_f 
sorted_df = df.sort_values(['Literate_Perc'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).Literate_Perc)

plt.yticks(range(10),sorted_df.tail(10).name_of_city, fontsize = 10)

plt.plot([.808,.808],[-1,10], '--',color = 'r')

plt.title('Most Literate')

plt.xlim([0,1])



plt.subplot(2,1,2)

plt.style.use('fivethirtyeight')

plt.barh(range(10),sorted_df.head(10).Literate_Perc)

plt.yticks(range(10),sorted_df.head(10).name_of_city, fontsize = 10)

plt.plot([.808,.808],[-1,10], '--',color = 'r')

plt.title('Least Literate')

plt.xlim([0,1])

plt.xlabel('Literacy Rate by City')
sorted_df = df.sort_values(['population_total'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).Literate_Perc)

plt.yticks(range(10),sorted_df.tail(10).name_of_city, fontsize = 10)

plt.plot([.808,.808],[-1,10], '--',color = 'r')

plt.title('Literacy Rate in Largest Cities')

plt.xlim([0,1]) 



plt.subplot(2,1,2)

plt.barh(range(10),sorted_df.head(10).Literate_Perc)

plt.yticks(range(10),sorted_df.head(10).name_of_city, fontsize = 10)

plt.plot([.808,.808],[-1,10], '--',color = 'r')

plt.title('Literacy Rate in Smallest Cities')

plt.xlim([0,1])
sns.jointplot(y='Literate_Perc',x='population_total',data=df,kind='reg', color = 'b')
sorted_df = df.sort_values(['Literacy_Gap'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).Literacy_Gap)

plt.yticks(range(10),sorted_df.tail(10).name_of_city, fontsize = 10)

plt.plot([.073,.073],[-1,10], '--',color = 'r')

plt.title('Largest Gender Literacy Gap')

plt.xlim([-.05,1])



plt.subplot(2,1,2)

plt.style.use('fivethirtyeight')

plt.barh(range(10),sorted_df.head(10).Literacy_Gap)

plt.yticks(range(10),sorted_df.head(10).name_of_city, fontsize = 10)

plt.plot([.073,.073],[-1,10], '--',color = 'r')

plt.title('Smallest Gender Literacy Gap')

plt.xlim([-.05,1])

plt.xlabel('Gender Literacy Gaps')
fig, ax = plt.subplots(figsize = (10,10))

sns.stripplot(y="state_name", x="Literate_Perc", data=df, jitter=True)
fig, ax = plt.subplots(figsize = (10,10))

sns.stripplot(y="state_name", x="Literacy_Gap", data=df, jitter=True)
sns.jointplot(y='population_total',x='Literacy_Gap',data=df,kind='reg', color = 'b')
sns.jointplot(y='Literate_Perc',x='Literacy_Gap',data=df,kind='reg', color = 'b')
sns.jointplot(y='Literacy_Gap',x='Child_Pop',data=df,kind='reg', color = 'b') 
sns.jointplot(y='Literate_Perc',x='M_Pop',data=df,kind='reg', color = 'b') 
sns.jointplot(y='Child_Pop',x='M_Pop',data=df,kind='reg', color = 'b') 
sns.jointplot(x='Literate_Perc',y='Grad_%',data=df,kind='reg', color = 'b')
plt.plot(df.Child_Pop, df.Literate_MP, 'o')

plt.plot(df.Child_Pop, df.Literate_FP, 'o')

plt.legend(['Males','Females'])

plt.xlabel('Child Population')

plt.ylabel('Literacy %')

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
females_df = pd.DataFrame()

males_df = pd.DataFrame()



females_df['Gender'] = np.ones(len(df))

males_df['Gender'] = np.zeros(len(df))



females_df['Literacy_Percentage'] = df.Literate_FP

males_df['Literacy_Percentage'] = df.Literate_MP



females_df['F_share'] = df.F_Pop

males_df['F_share'] = df.F_Pop



females_df['non_weighted_all_weekly'] = df['Literate_Perc']

males_df['non_weighted_all_weekly'] = df['Literate_Perc']



regression_df = males_df.append(females_df)



model = LinearRegression()

columns = ['F_share','Gender','non_weighted_all_weekly']

X = regression_df[columns]



X_std = StandardScaler().fit_transform(X)

y = regression_df['Literacy_Percentage']



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)



model.fit(X_train,y_train)



plt.barh([0,1,2],model.coef_)

plt.yticks(range(3),['Share of women','Gender','Literacy Percentage'], fontsize = 10)

plt.title('Regression Coefficients')



print('R^2 on training...',model.score(X_train,y_train))

print('R^2 on test...',model.score(X_test,y_test))
df['Grad_Percentage'].mean()
sorted_df = df.sort_values(['Grad_Percentage'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).Grad_Percentage)

plt.yticks(range(10),sorted_df.tail(10).name_of_city, fontsize = 10)

plt.plot([.132,.132],[-1,10], '--',color = 'r')

plt.title('Highest Grad Rate')

plt.xlim([0,1])



plt.subplot(2,1,2)

plt.style.use('fivethirtyeight')

plt.barh(range(10),sorted_df.head(10).Grad_Percentage)

plt.yticks(range(10),sorted_df.head(10).name_of_city, fontsize = 10)

plt.plot([.132,.132],[-1,10], '--',color = 'r')

plt.title('Lowest Grad Rate')

plt.xlim([0,1])

plt.xlabel('Graduation Rate Percent by City')
sorted_df = df.sort_values(['population_total'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).Grad_Percentage)

plt.yticks(range(10),sorted_df.tail(10).name_of_city, fontsize = 10)

plt.plot([.132,.132],[-1,10], '--',color = 'r')

plt.title('Graduation Rate in Largest Cities')

plt.xlim([0,1]) 



plt.subplot(2,1,2)

plt.barh(range(10),sorted_df.head(10).Grad_Percentage)

plt.yticks(range(10),sorted_df.head(10).name_of_city, fontsize = 10)

plt.plot([.132,.132],[-1,10], '--',color = 'r')

plt.title('Graduation Rate in Smallest Cities')

plt.xlim([0,1])
df['GradGap'].mean()
sorted_df = df.sort_values(['GradGap'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).GradGap)

plt.yticks(range(10),sorted_df.tail(10).name_of_city, fontsize = 10)

plt.plot([.022,.022],[-1,10], '--',color = 'r')

plt.title('Largest Gender Graduation Gap')

plt.xlim([-.05,1])



plt.subplot(2,1,2)

plt.style.use('fivethirtyeight')

plt.barh(range(10),sorted_df.head(10).GradGap)

plt.yticks(range(10),sorted_df.head(10).name_of_city, fontsize = 10)

plt.plot([.022,.022],[-1,10], '--',color = 'r')

plt.title('Smallest Gender Graduation Gap')

plt.xlim([-.05,1])

plt.xlabel('Gender Graduation Gaps')
LitGap = df['Literacy_Gap'].mean()/df['Literate_Perc'].mean()

LitGap
GGap = df['GradGap'].mean()/df['Grad_%'].mean()

GGap
model = LinearRegression()

columns = ['Gender']

X = regression_df[columns]



X_std = StandardScaler().fit_transform(X)

y = regression_df['Literacy_Percentage']



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)



model.fit(X_train,y_train)



print('R^2 on training...',model.score(X_train,y_train))

print('R^2 on test...',model.score(X_test,y_test))
sns.jointplot(y='population_total',x='GradGap',data=df,kind='reg', color = 'b')
sns.jointplot(x='population_total',y='Grad_%',data=df,kind='reg', color = 'b')
sns.jointplot(x='Child_Pop',y='Grad_%',data=df,kind='reg', color = 'b')
fig, ax = plt.subplots(figsize = (10,10))

sns.stripplot(y="state_name", x="Grad_%", data=df, jitter=True)
fig, ax = plt.subplots(figsize = (10,10))

sns.stripplot(y="state_name", x="Literate_Perc", data=df, jitter=True)
females_df = pd.DataFrame()

males_df = pd.DataFrame()



females_df['Gender'] = np.ones(len(df))

males_df['Gender'] = np.zeros(len(df))



females_df['Graduation_Percentage'] = df.FGrad_Perc

males_df['Graduation_Percentage'] = df.MGrad_Perc



females_df['F_share'] = df.F_Pop

males_df['F_share'] = df.F_Pop



females_df['non_weighted_all_weekly'] = df['Grad_%']

males_df['non_weighted_all_weekly'] = df['Grad_%']



regression_df = males_df.append(females_df)



model = LinearRegression()

columns = ['F_share','Gender','non_weighted_all_weekly']

X = regression_df[columns]



X_std = StandardScaler().fit_transform(X)

y = regression_df['Graduation_Percentage']



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)



model.fit(X_train,y_train)



plt.barh([0,1,2],model.coef_)

plt.yticks(range(3),['Share of women','Gender','Graduation Percentage'], fontsize = 10)

plt.title('Regression Coefficients')



print('R^2 on training...',model.score(X_train,y_train))

print('R^2 on test...',model.score(X_test,y_test))
model = LinearRegression()

columns = ['Gender']

X = regression_df[columns]



X_std = StandardScaler().fit_transform(X)

y = regression_df['Graduation_Percentage']



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)



model.fit(X_train,y_train)



print('R^2 on training...',model.score(X_train,y_train))

print('R^2 on test...',model.score(X_test,y_test))