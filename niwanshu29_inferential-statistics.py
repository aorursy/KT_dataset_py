# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

import scipy.stats.distributions as dist

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

dataset.head()
dataset.isnull().sum()



# NO null values are there
dataset.describe()
plt.figure(figsize = (15,8))

correlation_matrix = dataset.corr()



sns.heatmap(correlation_matrix , annot = True)

plt.show()
plt.figure(figsize = (15,6))

plt.subplot(1,2,1)

_ = sns.distplot(dataset.Pregnancies).set_ylabel("Distributions" , fontsize = 15)

plt.subplot(1,2,2)

_ = sns.boxplot(dataset.Pregnancies)
dataset['agegrp'] = pd.cut(dataset.Age , [18,30,40,50,60,70,80])

plt.figure(figsize = (15,7))

sns.boxplot(x = dataset.agegrp , y = dataset.BloodPressure)

plt.show()
plt.figure(figsize = (15,7))

dataset['Outcomex'] = dataset.Outcome.replace({1:'Diabetic' , 0:'Non_Diabetic'})

sns.boxplot(x = dataset.agegrp , y = dataset.BloodPressure,hue = dataset.Outcomex)

plt.show()
sample_size_pregnant = dataset[dataset['Outcome'] == 1]['Outcome'].count()

total_size = dataset.shape[0]

unbiased_point_estimate = np.round(sample_size_pregnant / total_size,100)

unbiased_point_estimate
Margin_of_error = 1.96 * np.sqrt(unbiased_point_estimate * (1-unbiased_point_estimate)/total_size)

Margin_of_error
lcb = unbiased_point_estimate - Margin_of_error

ucb = unbiased_point_estimate + Margin_of_error

(lcb,ucb)
sm.stats.proportion_confint(sample_size_pregnant,total_size)
unbiased_point_estimate = dataset[dataset.Outcome == 1]['Pregnancies'].mean()

std = dataset[dataset.Outcome == 1]['Pregnancies'].std()

(unbiased_point_estimate ,std)
Margin_of_error = 1.96 * std/np.sqrt(sample_size_pregnant)

Margin_of_error
lcb = unbiased_point_estimate - Margin_of_error

ucb = unbiased_point_estimate + Margin_of_error

(lcb,ucb)
sm.stats.DescrStatsW(dataset[dataset.Outcome == 1]['Pregnancies']).zconfint_mean()
unbiased_point_estimate = dataset[dataset.Outcome == 0]['Pregnancies'].mean()

std = dataset[dataset.Outcome == 0]['Pregnancies'].std()

(unbiased_point_estimate , std)
Margin_of_error = 1.96 * std/np.sqrt(dataset[dataset.Outcome == 0]['Outcome'].count())

Margin_of_error
lcb = unbiased_point_estimate - Margin_of_error

ucb = unbiased_point_estimate + Margin_of_error

(lcb,ucb)
sm.stats.DescrStatsW(dataset[dataset.Outcome == 0]['Pregnancies']).zconfint_mean()
std1 = dataset[dataset.Outcome == 1]['Pregnancies'].std()

std2 = dataset[dataset.Outcome == 0]['Pregnancies'].std()

(std1**2 , std2**2)
mean1 = dataset[dataset.Outcome == 1]['Pregnancies'].mean()

mean2 = dataset[dataset.Outcome == 0]['Pregnancies'].mean()

print(mean1 - mean2)



n1 = dataset[dataset.Outcome == 1]['Pregnancies'].count()

n2 = dataset[dataset.Outcome == 0]['Pregnancies'].count()



(n1,n2)
t_star = 1.98

Margin_of_error = t_star*np.sqrt(std1**2/n1 + std**2/n2)

Margin_of_error
lcb = (mean1 - mean2) - Margin_of_error

ucb = (mean1 - mean2) + Margin_of_error

(lcb,ucb)
Margin_of_error = t_star * np.sqrt(((n1 - 1)*std1**2 + (n2 - 1)*std2**2)/(n1 + n2 - 2)) * np.sqrt(1/n1 + 1/n2)

Margin_of_error
lcb = (mean1 - mean2) - Margin_of_error

ucb = (mean1 - mean2) + Margin_of_error

(lcb,ucb)
best_estimate = mean1 - mean2

std_error = np.sqrt(std1**2/n1 + std**2/n2)

test_statistic = best_estimate/std_error

p_val = 2*dist.norm.cdf(-np.abs(test_statistic))

(test_statistic , p_val)
best_estimate = mean1 - mean2

std_error = np.sqrt(((n1 - 1)*std1**2 + (n2 - 1)*std2**2)/(n1 + n2 - 2)) * np.sqrt(1/n1 + 1/n2)

test_statistic = best_estimate/std_error

p_val = 2*dist.norm.cdf(-np.abs(test_statistic))

(test_statistic , p_val)
sm.stats.ztest(dataset[dataset.Outcome == 1]['Pregnancies'] , dataset[dataset.Outcome == 0]['Pregnancies'])
dataset['agegrp'] = pd.cut(dataset.Age , [18,30,40,50,60,70,80,90])



dataset['Outcomex'] = dataset.Outcome.replace({1:'Diabetic',0:'Non-Diabetic'})

#Mean

dx_mean = dataset.groupby(['agegrp','Outcomex']).agg({'Pregnancies':[np.mean]}).unstack()

dx_mean.columns = ['Diabetic','Non_Diabetic']



# Standard Deviation

dx_std = dataset.groupby(['agegrp','Outcomex']).agg({'Pregnancies':[np.std]}).unstack()

dx_std.columns = ['Diabetic','Non_Diabetic']



# Size

dx_size = dataset.groupby(['agegrp','Outcomex']).agg({'Pregnancies':[np.size]}).unstack()

dx_size.columns = ['Diabetic','Non_Diabetic']



mean_diff = dx_mean.Diabetic - dx_mean.Non_Diabetic

se = dx_std/np.sqrt(dx_size)

se_diff = np.sqrt(se.Diabetic**2 + se.Non_Diabetic**2)



x = np.arange(dx_size.shape[0])

pp = sns.pointplot(x , mean_diff , color = 'black')

pp.set(xlabel = 'Age group' , ylabel = "Diabetic-Non Diabetic Pregnancies Mean Difference")

sns.pointplot(x , mean_diff - 1.96*se_diff)

sns.pointplot(x , mean_diff + 1.96*se_diff)

pp.set_xticklabels(dx_size.index)

plt.grid(alpha = 0.3)

plt.show()

unbiased_point_estimate = dataset[dataset.Outcome == 1]['BloodPressure'].mean()

std = dataset[dataset.Outcome == 1]['BloodPressure'].std()

(unbiased_point_estimate,std)
Margin_of_error = 1.96 * std/np.sqrt(dataset[dataset.Outcome == 1]['BloodPressure'].count())

Margin_of_error
lcb = unbiased_point_estimate - Margin_of_error

ucb = unbiased_point_estimate + Margin_of_error

(lcb,ucb)
sm.stats.DescrStatsW(dataset[dataset.Outcome == 1]['BloodPressure']).zconfint_mean()
unbiased_point_estimate = dataset[dataset.Outcome == 0]['BloodPressure'].mean()

std = dataset[dataset.Outcome == 0]['BloodPressure'].std()

(unbiased_point_estimate,std)
Margin_of_error = 1.96 * std/np.sqrt(dataset[dataset.Outcome == 0]['BloodPressure'].count())

Margin_of_error
lcb = unbiased_point_estimate - Margin_of_error

ucb = unbiased_point_estimate + Margin_of_error

(lcb,ucb)
sm.stats.DescrStatsW(dataset[dataset.Outcome == 0]['BloodPressure']).zconfint_mean()
mean1 = dataset[dataset.Outcome == 1]['BloodPressure'].mean()

mean2 = dataset[dataset.Outcome == 0]['BloodPressure'].mean()

(mean1 , mean2)

(n1,n2)
std1 = dataset[dataset.Outcome == 0]['BloodPressure'].std()

std2 = dataset[dataset.Outcome == 1]['BloodPressure'].std()

(std1**2 , std2**2)
Margin_of_error = t_star * np.sqrt(((n1 - 1)*std1**2 + (n2 - 1)*std2**2)/(n1 + n2 - 2)) * np.sqrt(1/n1 + 1/n2)

Margin_of_error
lcb = (mean1 - mean2) - Margin_of_error

ucb = (mean1 - mean2) + Margin_of_error

(lcb,ucb)
t_star = 1.98

Margin_of_error = t_star*np.sqrt(std1**2/n1 + std**2/n2)

Margin_of_error
lcb = (mean1 - mean2) - Margin_of_error

ucb = (mean1 - mean2) + Margin_of_error

(lcb,ucb)
best_estimate = mean1 - mean2

std_error = np.sqrt(std1**2/n1 + std**2/n2)

test_statistic = best_estimate/std_error

p_val = 2*dist.norm.cdf(-np.abs(test_statistic))

(test_statistic , p_val)
best_estimate = mean1 - mean2

std_error = np.sqrt(((n1 - 1)*std1**2 + (n2 - 1)*std2**2)/(n1 + n2 - 2)) * np.sqrt(1/n1 + 1/n2)

test_statistic = best_estimate/std_error

p_val = 2*dist.norm.cdf(-np.abs(test_statistic))

(test_statistic , p_val)
dataset['agegrp'] = pd.cut(dataset.Age , [18,30,40,50,60,70,80,90])



dataset['Outcomex'] = dataset.Outcome.replace({1:'Diabetic',0:'Non-Diabetic'})

#Mean

dx_mean = dataset.groupby(['agegrp','Outcomex']).agg({'BloodPressure':[np.mean]}).unstack()

dx_mean.columns = ['Diabetic','Non_Diabetic']



# Standard Deviation

dx_std = dataset.groupby(['agegrp','Outcomex']).agg({'BloodPressure':[np.std]}).unstack()

dx_std.columns = ['Diabetic','Non_Diabetic']



# Size

dx_size = dataset.groupby(['agegrp','Outcomex']).agg({'BloodPressure':[np.size]}).unstack()

dx_size.columns = ['Diabetic','Non_Diabetic']



mean_diff = dx_mean.Diabetic - dx_mean.Non_Diabetic

se = dx_std/np.sqrt(dx_size)

se_diff = np.sqrt(se.Diabetic**2 + se.Non_Diabetic**2)



x = np.arange(dx_size.shape[0])

pp = sns.pointplot(x , mean_diff , color = 'black')

pp.set(xlabel = 'Age group' , ylabel = "Diabetic-Non Diabetic BloodPressure Mean Difference")

sns.pointplot(x , mean_diff - 1.96*se_diff)

sns.pointplot(x , mean_diff + 1.96*se_diff)

pp.set_xticklabels(dx_size.index)

plt.show()
unbiased_point_estimate = dataset[dataset.Outcome == 1]['Glucose'].mean()

unbiased_point_estimate
std = dataset[dataset.Outcome == 1]['Glucose'].std()

std
std_error = std/np.sqrt(dataset[dataset.Outcome == 1]['Glucose'].count())

std_error

                     
lcb = unbiased_point_estimate - 1.96 *std_error

ucb = unbiased_point_estimate + 1.96 *std_error

(lcb , ucb)
sm.stats.DescrStatsW(dataset[dataset.Outcome == 1]['Glucose']).zconfint_mean()
unbiased_point_estimate = dataset[dataset.Outcome == 0]['Glucose'].mean()

std = dataset[dataset.Outcome == 0]['Glucose'].std()

print((unbiased_point_estimate,std))



std_error = std/np.sqrt(dataset[dataset.Outcome == 0]['Glucose'].count())

print(std_error)



lcb = unbiased_point_estimate - 1.96 *std_error

ucb = unbiased_point_estimate + 1.96 *std_error

(lcb , ucb)
sm.stats.DescrStatsW(dataset[dataset.Outcome == 0]['Glucose']).zconfint_mean()
mean1 = dataset[dataset.Outcome == 1]['Glucose'].mean()

mean2 = dataset[dataset.Outcome == 0]['Glucose'].mean()

print(mean1 , mean2)

print(n1,n2)



std1 = dataset[dataset.Outcome == 1]['Glucose'].std()

std2 = dataset[dataset.Outcome == 0]['Glucose'].std()

(std1**2 , std2**2)



t_star = 1.98

Margin_of_error = t_star*np.sqrt(std1**2/n1 + std**2/n2)

Margin_of_error



lcb = (mean1 - mean2) - Margin_of_error

ucb = (mean1 - mean2) + Margin_of_error

(lcb,ucb)
dataset['agegrp'] = pd.cut(dataset.Age , [18,30,40,50,60,70,80,90])



dataset['Outcomex'] = dataset.Outcome.replace({1:'Diabetic',0:'Non-Diabetic'})

#Mean

dx_mean = dataset.groupby(['agegrp','Outcomex']).agg({'Glucose':[np.mean]}).unstack()

dx_mean.columns = ['Diabetic','Non_Diabetic']



# Standard Deviation

dx_std = dataset.groupby(['agegrp','Outcomex']).agg({'Glucose':[np.std]}).unstack()

dx_std.columns = ['Diabetic','Non_Diabetic']



# Size

dx_size = dataset.groupby(['agegrp','Outcomex']).agg({'Glucose':[np.size]}).unstack()

dx_size.columns = ['Diabetic','Non_Diabetic']



mean_diff = dx_mean.Diabetic - dx_mean.Non_Diabetic

se = dx_std/np.sqrt(dx_size)

se_diff = np.sqrt(se.Diabetic**2 + se.Non_Diabetic**2)



x = np.arange(dx_size.shape[0])

pp = sns.pointplot(x , mean_diff , color = 'black')

pp.set(xlabel = 'Age group' , ylabel = "Diabetic-Non Diabetic Glucose Mean Difference")

sns.pointplot(x , mean_diff - 1.96*se_diff)

sns.pointplot(x , mean_diff + 1.96*se_diff)

pp.set_xticklabels(dx_size.index)

plt.show()
dx = dataset.groupby(['agegrp' , 'Outcomex'])['Outcome'].apply(lambda x:x.count()).unstack()

dx['Total'] = dx.sum(axis = 1)

dx = dx.apply(lambda x:x/x.sum(axis = 0))

dx
model = sm.GLM.from_formula('Outcome ~ Pregnancies', family = sm.families.Binomial() , data = dataset)

result = model.fit()

result.summary()
model = sm.GLM.from_formula('Outcome ~ Pregnancies + Glucose', family = sm.families.Binomial() , data = dataset)

result = model.fit()

result.summary()
from statsmodels.sandbox.predict_functional import predict_functional



values = {"Glucose":120,'BloodPressure':80,'SkinThickness':30,'Insulin':0,'BMI':30,'DiabetesPedigreeFunction':0.627,'Age':50,'Outcomex':'Diabeties','agegrp':'[18,30)'}



pr , cb , fv = predict_functional(result , 'Pregnancies' , values = values , ci_method = 'simultaneous')



ax = sns.lineplot(fv , pr , lw = 4)

ax.fill_between(fv , cb[:,0],cb[:,1],color = 'grey',alpha = 0.5)

ax.set_xlabel('Pregnancies')

_ = ax.set_ylabel('Diabeties')
values = {"Pregnancies":3,'BloodPressure':80,'SkinThickness':30,'Insulin':0,'BMI':30,'DiabetesPedigreeFunction':0.627,'Age':50,'Outcomex':'Diabeties','agegrp':'[18,30)'}



pr , cb , fv = predict_functional(result , 'Glucose' , values = values , ci_method = 'simultaneous')



ax = sns.lineplot(fv , pr , lw = 4)

ax.fill_between(fv , cb[:,0],cb[:,1],color = 'grey',alpha = 0.5)

ax.set_xlabel('Glucose')

_ = ax.set_ylabel('Diabeties')
from statsmodels.graphics.regressionplots import add_lowess

fig = result.plot_ceres_residuals("Glucose")

ax = fig.get_axes()[0]

ax.lines[0].set_alpha(0.2)

_ = add_lowess(ax)