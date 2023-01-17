# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pandas_profiling

# Matplotlib forms basis for visualization in Python
import matplotlib.pyplot as plt

# We will use the Seaborn library
import seaborn as sns
sns.set()

# Graphics in retina format are more sharp and legible
%config InlineBackend.figure_format = 'retina'

#увеличим дефолтный размер графиков
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Read data
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

# First 5 rows of our data
df.head()
profile = pandas_profiling.ProfileReport(df)
profile
#Count for target values
df.target.value_counts()
df.groupby('target').mean()
# Rows, columns:
df.shape
# summary of dataset
df.info()
# Следующую команду можно использовать частично для информации по №1 (но потом отдельно посчитаем)
df.describe()
features = ['trestbps', 'restecg','thalach']
df[features].hist(figsize=(10, 4));
#Целевой признак по возрасту
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
# Целевой признак по гендерному признаку
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])

# Исходя из графика - у женщин болезни сердца значительно чаще, чем у мужчин. 
# И больных больше, чем здоровых.
sns.distplot(df.chol)
# `pairplot()` may become very slow with the SVG or retina format
%config InlineBackend.figure_format = 'png'
sns.pairplot(df[['thalach','chol', 'trestbps', 'oldpeak']]);
%config InlineBackend.figure_format = 'retina'
# chol - serum cholestoral in mg/dl
# thalach - maximum heart rate achieved
# sex - (1 = male; 0 = female)

sns.lmplot('thalach', 'chol', data=df, hue='sex', fit_reg=False);

# Среднее для определенного столбца
df['age'].mean()
# Медиана аналогично
df['age'].median()
# Дисперсия
df.var()
# Дисперсия, plot
df.var().plot()
# Мода 
df.mode()

# Здесь получилось для холестерина наибольшее число строк сразу для трех значений - 197, 204, 234
# df[df['chol']==197].value_counts('chol')
# Skewness для столбцов
# thalach - maximum heart rate achieved
# chol - serum cholestoral in mg/dl
df[['thalach','chol']].skew()
sns.distplot(df['thalach'])
sns.distplot(df['chol'])
# Kurtosis для столбцов
df.kurtosis()
features = ['thalach', 'chol']
df[features].hist(figsize=(10, 4));
# Box plot
sns.boxplot(x='thalach', data=df);
sns.boxplot(data=df[['thalach','chol', 'trestbps']]);
sns.violinplot(data=df['chol']);
# Scatter plot
plt.scatter(df['trestbps'], df['thalach']);
sns.jointplot(x='thalach', y='chol', 
              data=df, kind='scatter');
# В пандасе не нашел подходящего, возьмем chi2_contingency из SciPy:

import scipy.stats as stats
stat, p, dof, expected = stats.chi2_contingency(df[['chol', 'thalach']])
print('dof=%d' % dof)
prob = 0.95
critical = stats.chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
# Если статистика >= критическое значение: 
# значимый результат, отвергнуть нулевую гипотезу (H0), в зависимости.
# Если статистика < Критическое значение: 
# несущественный результат, неспособность отклонить нулевую гипотезу (H0), независимая.
# Мы гипотезу принимаем (с уровнем значимости 0.05)
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# P значение больше 0.05 - гипотезу о нормальном распределении для df['chol'] отвергаем
stats.kstest(df['chol'], cdf='norm', args=(df['chol'].mean(), df['chol'].std()))


from scipy.stats import boxcox
df['trestbps'].hist()
data=boxcox(df['trestbps'])
df['trestbps_bc']=data[0]
df['trestbps_y']=data[1]
df[['trestbps','trestbps_bc','trestbps_y']]
df[['trestbps','trestbps_bc']].hist()
df.corr(method='pearson')
Var_Corr = df.corr(method='pearson')
# plot the heatmap and annotation on it
plt.subplots(figsize=(15,10))
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
df.corr(method='spearman')
Var_Corr = df.corr(method='spearman')
# plot the heatmap and annotation on it
plt.subplots(figsize=(15,10))
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
df.corr(method='kendall')
Var_Corr = df.corr(method='kendall')
# plot the heatmap and annotation on it
plt.subplots(figsize=(15,10))
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
# Compute the z score of each value in the sample, 
# relative to the sample mean and standard deviation.
stats.zscore(df, axis=1)
df.mad()