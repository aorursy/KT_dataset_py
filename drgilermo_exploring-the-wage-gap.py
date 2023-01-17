import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

from subprocess import check_output

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/inc_occ_gender.csv')



df = df[~(df.M_weekly == 'Na')]

df = df[~(df.F_weekly == 'Na')]

       

df['M_weekly'] = df.M_weekly.apply(lambda x: int(x))

df['F_weekly'] = df.F_weekly.apply(lambda x: int(x))

df['M_workers'] = df.M_workers.apply(lambda x: int(x))

df['F_workers'] = df.F_workers.apply(lambda x: int(x))

df['All_weekly'] = df.All_weekly.apply(lambda x: int(x))

df['All_workers'] = df.All_workers.apply(lambda x: int(x))

df['M_share'] = df.M_workers/df.All_workers 

df['F_share'] = df.F_workers/df.All_workers 

df['non_weighted_all_weekly'] = (df.M_weekly + df.F_weekly)/2

df['Gap'] = df.M_weekly - df.F_weekly

df['Ratio'] = df.F_weekly/df.M_weekly

df['Ratio_of_workers'] = df.F_workers/df.M_workers



df = df.reset_index(drop = True)
sorted_df = df.sort(['Ratio'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).Ratio)

plt.yticks(range(10),sorted_df.tail(10).Occupation, fontsize = 10)

plt.plot([1,1],[0,10], '--',color = 'r')

plt.title('Most Equal Fields')

plt.xlim([0,1.2])



plt.subplot(2,1,2)

plt.style.use('fivethirtyeight')

plt.barh(range(10),sorted_df.head(10).Ratio)

plt.yticks(range(10),sorted_df.head(10).Occupation, fontsize = 10)

plt.plot([1,1],[0,10], '--',color = 'r')

plt.title('Most unequal Fields')

plt.xlim([0,1.2])

plt.xlabel('Female/Male wage ratio')



sorted_df = df.sort(['F_share'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).F_share)

plt.yticks(range(10),sorted_df.tail(10).Occupation, fontsize = 10)

plt.plot([0.5,0.5],[0,10], '--',color = 'r')

plt.xlim([0,1])

plt.title('Fields with largerst share of women')



plt.subplot(2,1,2)

plt.style.use('fivethirtyeight')

plt.barh(range(10),sorted_df.head(10).F_share)

plt.yticks(range(10),sorted_df.head(10).Occupation, fontsize = 10)

plt.plot([0.5,0.5],[0,10], '--',color = 'r')

plt.title('Fields with smallest share of women')

plt.xlim([0,1])



sorted_df = df.sort(['non_weighted_all_weekly'], ascending = [True])



plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).non_weighted_all_weekly)

plt.yticks(range(10),sorted_df.tail(10).Occupation, fontsize = 10)

plt.xlim([0,2000])

plt.xlabel('Weekly Income[$]')

plt.title('Most paying fields')



plt.subplot(2,1,2)

plt.style.use('fivethirtyeight')

plt.barh(range(10),sorted_df.head(10).non_weighted_all_weekly)

plt.yticks(range(10),sorted_df.head(10).Occupation, fontsize = 10)

plt.xlim([0,2000])

plt.title('Least paying fields')
plt.figure(figsize = (10,10))



plt.subplot(2,1,1)

plt.barh(range(10),sorted_df.tail(10).F_share)

plt.yticks(range(10),sorted_df.tail(10).Occupation, fontsize = 10)

plt.xlim([0,1])

plt.title('Share of women in the nost paying fields')



plt.subplot(2,1,2)

plt.barh(range(10),sorted_df.head(10).F_share)

plt.yticks(range(10),sorted_df.head(10).Occupation, fontsize = 10)

plt.xlim([0,1])

plt.title('Share of women in the least paying fields')

plt.xlabel('Shere of Women')
sns.distplot(df.Ratio, bins = np.linspace(0.4,1.2,28))

plt.title('Median Wage Ratio Distribution')



np.mean(df.Ratio)
plt.plot(df.non_weighted_all_weekly, df.Ratio,'o',markersize = 10, alpha = 0.8)

plt.xlabel('Non Weighted Weekly Salary [$]')

plt.ylabel('Female/Male Wage Ratio')

plt.title('The gap is larger at higher salaries')



x = df.non_weighted_all_weekly

y = df.Ratio

fn = np.polyfit(x,y,1)

fit_fn = np.poly1d(fn) 

plt.plot(x,fit_fn(x))

plt.plot(df['F_share'], df.Ratio,'o', markersize = 10, alpha = 0.8)



x = df['F_share']

y = df.Ratio

fn = np.polyfit(x,y,1)

fit_fn = np.poly1d(fn) 

plt.plot(x,fit_fn(x))

plt.title('The Gap slightly decreases with the share of females')

plt.plot(df.non_weighted_all_weekly, df.M_weekly,'o')

plt.plot(df.non_weighted_all_weekly, df.F_weekly,'o')

plt.legend(['Males','Females'])

plt.xlabel('Field Median Salary')

plt.ylabel('Salary')

plt.show()

females_df = pd.DataFrame()

males_df = pd.DataFrame()



females_df['Gender'] = np.ones(len(df))

males_df['Gender'] = np.zeros(len(df))



females_df['Salary'] = df.F_weekly

males_df['Salary'] = df.M_weekly



females_df['F_share'] = df.F_share

males_df['F_share'] = df.F_share



females_df['non_weighted_all_weekly'] = df['non_weighted_all_weekly']

males_df['non_weighted_all_weekly'] = df['non_weighted_all_weekly']



regression_df = males_df.append(females_df)



model = LinearRegression()

columns = ['F_share','Gender','non_weighted_all_weekly']

X = regression_df[columns]



X_std = StandardScaler().fit_transform(X)

y = regression_df['Salary']



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)



model.fit(X_train,y_train)



plt.barh([0,1,2],model.coef_)

plt.yticks(range(3),['Share of women','Gender','Salary in Field'], fontsize = 10)

plt.title('Regression Coefficients')



print('R^2 on training...',model.score(X_train,y_train))

print('R^2 on test...',model.score(X_test,y_test))
model = LinearRegression()

columns = ['Gender']

X = regression_df[columns]



X_std = StandardScaler().fit_transform(X)

y = regression_df['Salary']



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)



model.fit(X_train,y_train)



print('R^2 on training...',model.score(X_train,y_train))

print('R^2 on test...',model.score(X_test,y_test))
