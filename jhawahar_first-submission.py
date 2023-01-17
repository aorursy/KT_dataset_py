import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from statsmodels.stats.outliers_influence import variance_inflation_factor
raw_data = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
raw_data.head()
raw_data.describe(include = 'all')
raw_data.isnull().sum()
data_cleaned = raw_data.drop(['Serial No.'], axis = 1)
data_cleaned.describe(include = 'all')
sns.distplot(data_cleaned['GRE Score'])
q = data_cleaned['GRE Score'].quantile(0.01)

data_1 = data_cleaned[data_cleaned['GRE Score']>q]

data_1.describe(include = 'all')
sns.distplot(data_1['GRE Score'])
sns.distplot(data_1['TOEFL Score'])
q = data_1['TOEFL Score'].quantile(0.01)

data_2 = data_1[data_1['TOEFL Score']>q]

data_2.describe(include = 'all')
sns.distplot(data_2['TOEFL Score'])
data_2.columns.values
sns.distplot(data_2['University Rating'])
q = data_2['University Rating'].quantile(0.01)

data_3 = data_2[data_2['University Rating']>q]

data_3.describe(include = 'all')
sns.distplot(data_3['University Rating'])
sns.distplot(data_3['SOP'])
q = data_3['SOP'].quantile(0.01)

data_4 = data_3[data_3['SOP']>q]

data_4.describe(include = 'all')
sns.distplot(data_4['SOP'])
data_4
sns.distplot(data_4['LOR '])
q = data_4['LOR '].quantile(0.01)

data_5 = data_4[data_4['LOR ']>q]

data_5.describe(include = 'all')
sns.distplot(data_5['LOR '])
sns.distplot(data_5['CGPA'])
q = data_5['CGPA'].quantile(0.01)

data_6 = data_5[data_5['CGPA']>q]

data_6.describe(include  = 'all')
sns.distplot(data_6['CGPA'])
sns.distplot(data['Research'])
sns.distplot(data_6['Chance of Admit '])
q = data_6['Chance of Admit '].quantile(0.01)

data_7 = data_6[data_6['Chance of Admit ']>q]

data_7.describe(include = 'all')
sns.distplot(data_7['Chance of Admit '])
data_cleaned = data_7.reset_index(drop = True)
data_cleaned.describe(include = 'all')
f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey=True, figsize=(15,3))

ax1.scatter(data_cleaned['GRE Score'], data_cleaned['Chance of Admit '])

ax1.set_title('Chance of Admit and GRE Score')

ax2.scatter(data_cleaned['TOEFL Score'], data_cleaned['Chance of Admit '])

ax2.set_title('Chance of Admit and TOEFL Score')

ax3.scatter(data_cleaned['CGPA'],data_cleaned['Chance of Admit '])

ax3.set_title('Chance of Admit and CGPA')



plt.show()
f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey=True, figsize=(15,3))

ax1.scatter(data_cleaned['University Rating'], data_cleaned['Chance of Admit '])

ax1.set_title('Chance of Admit and University Rating')

ax2.scatter(data_cleaned['SOP'], data_cleaned['Chance of Admit '])

ax2.set_title('Chance of Admit and SOP')

ax3.scatter(data_cleaned['LOR '],data_cleaned['Chance of Admit '])

ax3.set_title('Chance of Admit and Research')



plt.show()
data_cleaned = data_cleaned.reset_index(drop = True)
data_cleaned.describe(include = 'all')
targets = data_cleaned['Chance of Admit ']

inputs = data_cleaned.drop(['Chance of Admit '], axis = 1)
#using sklearn.preprocessing -> StandardScaler

scaler = StandardScaler()

scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
#using sklearn.modelselection -> train_test_split

x_train,x_test,y_train,y_test = train_test_split(inputs_scaled, targets, test_size = 0.2, random_state = 365)
reg = LinearRegression()

reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)

plt.xlabel('Targets(y_train)', size = 10)

plt.ylabel('Predictions(y_hat)', size = 10)

plt.show()
#Plot For Residuals

sns.distplot(y_train - y_hat)

plt.title('Residual PDFs', size =18)
## R-Squared

reg.score(x_train,y_train)
reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(inputs.columns.values, columns = ['Features'])

reg_summary['Weights'] = reg.coef_

reg_summary
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)

plt.xlabel('Targets(y_test)', size = 18)

plt.ylabel('Predictions(y_hat_test)', size = 18)



plt.show()
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])

df_pf.head()
df_pf['Target'] = np.exp(y_test)

df_pf
y_test= y_test.reset_index(drop=True)
y_test.head()
df_pf['Target'] = np.exp(y_test)

df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Predictions']
df_pf['Difference'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf
pd.options.display.max_rows = 999

pd.set_option('display.float_format', lambda x: '% 2f' % x)

df_pf.sort_values(by = ['Difference'])