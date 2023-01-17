# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importando as bibliotecas

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import datasets, linear_model

from scipy.stats import ttest_ind

import scipy.stats as stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')

df.head().T
df.info()
# isna

print(df.isna().sum())
display(df.shape)
df.describe().T
# Categorias de trabalhos listados

df['JOB'].value_counts()
# Categorias de trabalhos listados

df['BAD'].value_counts(normalize=True)*100
df.sample(20)
df_nan = df.isnull().sum(axis=1)
result = pd.concat([df, df_nan], axis=1)

result.rename(columns={0: 'MISS'}, inplace=True)

ordenado = result.sort_values(by=['MISS'], ascending=False)

ordenado.head(50)
result.groupby(['MISS']).size()
# total de faltam 5 são 275

faltam_5_mais_rec = result[(result['MISS'] > 4) & (result['BAD'] == 1)]

faltam_5_mais_rec.info()
faltam_5_mais_rec.head(58)
faltam_5_mais_acpt = result[(result['MISS'] > 4) & (result['BAD'] == 0)]

faltam_5_mais_acpt.sample(30)
#value missing

df_value_missing = df[df['VALUE'].isna()]

df_value_missing.sample(20)
df_value_missing['BAD'].value_counts()
df_value_missing_acpt = df_value_missing[df_value_missing['BAD'] == 0]

df_value_missing_acpt
sns.boxplot(x='BAD', y='DEBTINC', data=df)

plt.title('Distribuição de endividamento por inadimplência')

plt.figure(figsize=(15,5))

plt.show()
sns.boxplot(x='BAD', y='VALUE', data=df)

plt.title('Distribuição de valor da propriedade por inadimplência')

plt.figure(figsize=(15,5))

plt.show()
relacao_valores = df[['VALUE' , 'LOAN' , 'DELINQ' , 'MORTDUE' , 'DEBTINC']]

sns.pairplot(relacao_valores)
value_not = df[(df['MORTDUE'].notnull()) & (df['VALUE'].notnull())]

x = value_not.MORTDUE.values.reshape(-1,1)

y = value_not.VALUE.values.reshape(-1,1)



regr = linear_model.LinearRegression()

regr.fit(x, y)



regr.coef_
regr.intercept_
df['VALUE'].fillna((df.MORTDUE * 1.07 + 25680.30), inplace=True)

df['MORTDUE'].fillna((df.VALUE / 1.07 - 25680.30), inplace=True)
print(df.isna().sum())
value_not = df[(df['MORTDUE'].isna()) & (df['VALUE'].isna())]

value_not.sample(27)
sns.boxplot(x='BAD', y='CLAGE', data=df)

plt.title('Distribuição de valor da linha mais antiga por inadimplência')

plt.figure(figsize=(15,5))

plt.show()
sns.boxplot(x='BAD', y='DELINQ', data=df)

plt.title('Distribuição de inadimplência por inadimplência')

plt.figure(figsize=(15,5))

plt.show()
# Categorias de trabalhos listados

df['REASON'].value_counts()
numeric_feats = [c for c in df.columns if df[c].dtype != 'object' and c not in ['BAD']]

df_numeric_feats = df[numeric_feats]
plt.figure(figsize=(18,18))

c = 1

for i in df_numeric_feats.columns:

    if c < len(df_numeric_feats.columns):

        plt.subplot(3,3,c)

        sns.boxplot(x='REASON' , y= i, data=df)

        c+=1

    else:

        sns.boxplot(x='REASON' , y= i, data=df)
# teste Shapiro-Wil (Normalidade)

df_reason_homeimp = df[df['REASON']=='HomeImp']['VALUE']

df_reason_debtcon = df[df['REASON']=='DebtCon']['VALUE']

shapiro_stat_reason_homeimp, shapiro_p_valor_reason_homeimp = stats.shapiro(df_reason_homeimp)

shapiro_stat_reason_debtcon, shapiro_p_valor_reason_debtcon = stats.shapiro(df_reason_debtcon)



print('teste de normalidade')

print('reason homeimp: {}'.format(shapiro_p_valor_reason_homeimp))

print('reason_debtcon: {}'.format(shapiro_p_valor_reason_debtcon))
ttest_stats, ttest_p_value = ttest_ind(df_reason_homeimp.dropna(), df_reason_debtcon.dropna(), equal_var=False)



print('T-teste: {}'.format(ttest_p_value))
df_bad1 = df[df['BAD']== 1]['VALUE']

df_bad0 = df[df['BAD']== 0]['VALUE']

shapiro_stat_bad1,  shapiro_p_valor_bad1 = stats.shapiro(df_bad1)

shapiro_stat_bad0, shapiro_p_valor_bad0 = stats.shapiro(df_bad0)



print('teste de normalidade')

print('reason homeimp: {}'.format(shapiro_p_valor_bad1))

print('reason_debtcon: {}'.format(shapiro_p_valor_bad0))
ttest_stats_bad, ttest_p_value_bad = ttest_ind(df_bad1.dropna(), df_bad0.dropna(), equal_var=False)

print('T-teste: {}'.format(ttest_p_value_bad))
jobs = df['JOB'].dropna().unique()

plt.figure(figsize=(14,15))

c=1

for i in jobs:

    plt.subplot(7,1,c)

    plt.title(i)

    df[df['JOB'] == i]['VALUE'].hist(bins=20)

    c+=1

plt.tight_layout() 
anova_value_by_job = {job:df['VALUE'][df['JOB'] == job] for job in jobs}

anova_job_f, anova_job_p = stats.f_oneway(anova_value_by_job['Other'].dropna(),anova_value_by_job['Office'].dropna(),anova_value_by_job['Sales'].dropna(),anova_value_by_job['Mgr'].dropna(),anova_value_by_job['ProfExe'].dropna(), anova_value_by_job['Self'].dropna())



print('One Way Anova: {}'.format(anova_job_p))
value_mean_by_job = df.groupby('JOB')['VALUE'].mean()

value_mean_by_job
df.dropna(thresh=10, inplace=True)

df.dropna(subset=['VALUE'], inplace=True)

df.shape
display(df.groupby('JOB')['VALUE'].mean())

display(df.groupby('JOB')['VALUE'].std())
df['REASON'].fillna('unknown', inplace=True)

df['JOB'].fillna('unknown', inplace = True)
df[['VALUE', 'MORTDUE']].corr()
missing_mortdue = df[df['MORTDUE'].isnull()][['VALUE', 'MORTDUE']]

not_missing_mortdue = df[df['MORTDUE'].notnull()][['VALUE', 'MORTDUE']]
X = not_missing_mortdue['VALUE'].values.reshape(-1, 1)

y = not_missing_mortdue['MORTDUE'].values.reshape(-1, 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()

lr.fit(X_train, y_train)
mortdue_pred = lr.predict(X_test)
real_vs_pred = pd.DataFrame({'Real': y_test.flatten(), 'Predito': mortdue_pred.flatten()})

real_vs_pred.sample(20)
plt.figure(figsize=(10,10))

plt.scatter(X_test, y_test, color='gray')

plt.plot(X_test, mortdue_pred, color='red', linewidth=2)

plt.show()
print('Raiz quadrada do Erro medio ao quadrado: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, mortdue_pred))))
# trantando outliers para tentar diminuir o erro.

# calculando o iqr

q1 = not_missing_mortdue.quantile(0.25)

q3 = not_missing_mortdue.quantile(0.75)



iqr = q3-q1



print(iqr)
#removendo os outliers

not_missing_and_outliers_mortdue = not_missing_mortdue[~((not_missing_mortdue < (q1 - 1.5  * iqr)) | (not_missing_mortdue > (q3 + 1.5 * iqr))).any(axis=1)]
X = not_missing_and_outliers_mortdue['VALUE'].values.reshape(-1, 1)

y = not_missing_and_outliers_mortdue['MORTDUE'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



lr.fit(X_train, y_train)
mortdue_pred = lr.predict(X_test)

real_vs_pred = pd.DataFrame({'Real': y_test.flatten(), 'Predito': mortdue_pred.flatten()})

real_vs_pred.sample(10)
print('Raiz quadrada do Erro medio ao quadrado: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, mortdue_pred))))
plt.figure(figsize=(10,10))

plt.scatter(X_test, y_test, color='gray')

plt.plot(X_test, mortdue_pred, color='red', linewidth=2)

plt.show()
imp_mortdue = pd.Series([])

imp_mortdue = lr.predict(df['VALUE'].values.reshape(-1,1))

imp_mortdue