%matplotlib inline
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics



import statsmodels.api as sm
import os

print(os.listdir('../input/biol165'))

os.chdir('../input/biol165')
litter_census_geoid = pd.read_csv('litter_with_census_geoid.csv')

litter_census = pd.read_csv('census_blocks_with_litter.csv')
litter_census_geoid.head()
litter_census.head()
income_census = pd.read_csv('census_income.csv')

income_census.head(1)
income_census = income_census[1:]

income_census['GEO.id2'] = income_census['GEO.id2'].astype(int)

income_census.head()
len(np.unique(income_census['GEO.id2'])), len(np.unique(litter_census['GEOID10']))
litter_income = pd.merge(income_census, litter_census, 

         left_on='GEO.id2', right_on='GEOID10')[['GEOID10', 'HD01_VD01', 'HD02_VD01', 'LITTER_AVERAGE']]
litter_income = litter_income.astype(float)
(litter_income.HD01_VD01).isna().sum()
litter_income = litter_income.dropna()
litter_income.corr()
sns.pairplot(litter_income[litter_income.columns[1:]], kind='reg', plot_kws={'line_kws':{'color':'red'}})

plt.show()
litter_income.plot(x='HD01_VD01', y='LITTER_AVERAGE', style='o')  



plt.title('Median Income vs. Average Litter Level')  

plt.xlabel('Median Income')  

plt.ylabel('Average Litter Level');
sns.jointplot(litter_income.HD01_VD01, litter_income.LITTER_AVERAGE, kind='kde');
plt.figure(figsize=(10,6))

plt.tight_layout()

sns.distplot(litter_income['LITTER_AVERAGE'])

plt.title('Distribution of (Mean) Litter by Census Tract in Philadelphia');
X = litter_income['HD01_VD01']

y = litter_income['LITTER_AVERAGE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(np.array(X_train).reshape(-1, 1), y_train)
X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
y_pred = regressor.predict(np.array(X_test).reshape(-1,1))
plt.scatter(X_test, y_test,  color='gray')

plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.title('Income & Litter: Predicted Trend & Actual Census Tracts')

plt.xlabel('Median Income')

plt.ylabel('Litter')

plt.show()
print('Train Score (R-Squared):', regressor.score(np.array(X_train).reshape(-1, 1), y_train))  

print('Test Score (R-Squared):', regressor.score(np.array(X_test).reshape(-1, 1), y_test))  



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
compiled_df = pd.read_csv('litter_income_tree_ipd.csv')

compiled_df.head()
sns.pairplot(compiled_df[compiled_df.columns[5:]], kind='reg', palette="husl", plot_kws={'line_kws':{'color':'red'}})

plt.show()
compiled_df.columns
y = compiled_df['LITTER_AVE']

X = compiled_df[['Median Household Income', 'TREE_DENSITY', 'IPD_SCORE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(np.array(X_train), y_train)
X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
y_pred = regressor.predict(np.array(X_test))
print('Train Score (R-Squared):', regressor.score(np.array(X_train), y_train))  

print('Test Score (R-Squared):', regressor.score(np.array(X_test), y_test))  



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
compiled_df.columns



y = compiled_df['Median Household Income']

X = compiled_df[['TREE_DENSITY', 'LITTER_AVE', 'IPD_SCORE']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



regressor = LinearRegression()  

regressor.fit(np.array(X_train), y_train)



X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())



y_pred = regressor.predict(np.array(X_test))
print('Train Score (R-Squared):', regressor.score(np.array(X_train), y_train))  

print('Test Score (R-Squared):', regressor.score(np.array(X_test), y_test))  



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
compiled_df.columns



y = compiled_df['TREE_DENSITY']

X = compiled_df[['Median Household Income', 'LITTER_AVE', 'IPD_SCORE']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



regressor = LinearRegression()  

regressor.fit(np.array(X_train), y_train)



X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())



y_pred = regressor.predict(np.array(X_test))
print('Train Score (R-Squared):', regressor.score(np.array(X_train), y_train))  

print('Test Score (R-Squared):', regressor.score(np.array(X_test), y_test))  



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
df_gsi = pd.read_csv('litter_income_tree_ipd_gsi.csv')
df_gsi.corr()
sns.pairplot(df_gsi[df_gsi.columns[5:]], kind='reg', plot_kws={'line_kws':{'color':'red'}});
df_gsi.plot(x='Median Hou', y='GSI_KDE', style='o')  



plt.title('Median Income vs. GSI Projects Density')  

plt.xlabel('Median Income')  

plt.ylabel('GSI Projects Density');
sns.jointplot(df_gsi['Median Hou'], df_gsi['GSI_KDE'], kind='kde')

plt.title('GSI KDE by Median Income');
sns.jointplot(df_gsi['Median Hou'], df_gsi['GSI_KDE'], kind='reg', color='darkred')
X = df_gsi['Median Hou']

y = df_gsi['GSI_KDE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(np.array(X_train).reshape(-1, 1), y_train)
X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
y_pred = regressor.predict(np.array(X_test).reshape(-1,1))
plt.scatter(X_test, y_test,  color='gray')

plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.title('GSI Projects & Income w/ Prediction Line (Simple Linear Regression)')

plt.xlabel('Median Income')

plt.ylabel('GSI Projects KDI');
print('Train Score (R-Squared):', regressor.score(np.array(X_train).reshape(-1, 1), y_train))  

print('Test Score (R-Squared):', regressor.score(np.array(X_test).reshape(-1, 1), y_test))  



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
df_gsi.head()
X = df_gsi[['LITTER_AVE', 'TREE_DENSI', 'IPD_SCORE', 'GSI_KDE']]

y = df_gsi['Median Hou']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(np.array(X_train), y_train)
X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
y_pred = regressor.predict(np.array(X_test))
print('Train Score (R-Squared):', regressor.score(np.array(X_train), y_train))  

print('Test Score (R-Squared):', regressor.score(np.array(X_test), y_test))  



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))