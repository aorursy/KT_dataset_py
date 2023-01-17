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
import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



from scipy.special import boxcox, inv_boxcox

from scipy import stats



import warnings

warnings.filterwarnings('ignore')
# Read in data

data = pd.read_csv('/kaggle/input/insurance/insurance.csv')

split = round(len(data) * .8)

train = data[:split]

test = data[split:]
f, ax = plt.subplots(ncols = 2, figsize=(15,5))

sns.distplot(train['charges'], color='g', ax=ax[0])

ax[0].set_title('Train')

sns.distplot(test['charges'], color='g', ax=ax[1])

ax[1].set_title('Test')
train.describe()
train.head()
train.info()
print('Train Shape: {}'.format(train.shape))

print('Test Shape:  {}'.format(test.shape))
numeric_features = train.select_dtypes(include=['int64','float64']).columns

numeric_features
categorical_features = train.select_dtypes(include=['object']).columns

categorical_features
f, ax = plt.subplots(ncols = 2, figsize=(15,5))

sns.distplot(data['charges'], color='g', ax=ax[0])

sns.kdeplot(data['charges'], cumulative=True, shade=True, color='g', ax=ax[1])
f, ax = plt.subplots(ncols = 2, figsize=(15,5))

sns.distplot(data['age'], color='g', ax=ax[0])

sns.kdeplot(data['age'], cumulative=True, shade=True, color='g', ax=ax[1])
f, ax = plt.subplots(ncols = 2, figsize=(15,5))

sns.distplot(data['bmi'], color='g', ax=ax[0])

sns.kdeplot(data['bmi'], cumulative=True, shade=True, color='g', ax=ax[1])

print(data['bmi'].describe())
f, ax = plt.subplots(ncols = 2, figsize=(15,5))

sns.distplot(data['children'], color='g', ax=ax[0])

sns.kdeplot(data['children'], cumulative=True, shade=True, color='g', ax=ax[1])

print(data['children'].describe())
children_tab = pd.crosstab(index=data['children'], columns='Count', colnames=['Frequency'])

children_tab_percentage = children_tab/children_tab.sum()

print('{}\n\n{}'.format(children_tab, children_tab_percentage))
has_children = len(data[data['children'] > 0])/len(data)

sns.barplot(x=data['children'].value_counts().index, y=data['children'].value_counts(), data=data)
print('There are {:.3}% of patients with at least 1 or more children and {:.3}% without children'.format(has_children * 100, (1 - has_children) * 100))
sex_tab = pd.crosstab(index=data['sex'], columns='count', colnames=['Frequency'])

sex_tab_percentage = sex_tab/sex_tab.sum()

print('{}\n\n{}'.format(sex_tab, sex_tab_percentage))

sns.barplot(x=data['sex'].value_counts().index, y=data['sex'].value_counts(), data=data)
smoker_tab = pd.crosstab(index=data['smoker'], columns='count', colnames=['Frequency'])

smoker_tab_percentage = smoker_tab/smoker_tab.sum()

print('{}\n\n{}'.format(smoker_tab, smoker_tab_percentage))

sns.barplot(x=data['smoker'].value_counts().index, y=data['smoker'].value_counts(), data=data)
sns.catplot(x='sex', y='charges', kind='boxen', data=data)

plt.title('Medical Costs of Males and Females')
sns.catplot(x='sex', y='charges', kind='boxen', hue='smoker', data=data)

plt.title('Medical Costs of smokers and non-smokers for males and females')
sns.catplot(x='sex', y='bmi', kind='boxen', data=data)

plt.title('BMI of Males and Females')
sns.catplot(x='region', y='charges', kind='boxen', data=data)

plt.title('Medical Costs of different regions')
sns.catplot(x='sex', y='charges', kind='boxen', hue='region', data=data)

plt.title('Medical Costs of different regions for males and females')
# Distribution of charges by age

with sns.axes_style("white"):

    sns.jointplot(x='age',y='charges',data=data, kind='hex', color='#4CB391')
# Distribution of charges by bmi

with sns.axes_style("white"):

    sns.jointplot(x='bmi',y='charges',data=data, kind='hex', color='#4CB391')
sns.heatmap(train.corr(), annot=True, linewidths=0.5, fmt='.1f', cmap='Blues')

plt.title('Correlation Matrix')
data['bmi_risk'] = np.where(data.bmi < 26, 'Healthy', 

                           np.where((data.bmi > 25) & (data.bmi < 31), 'Overweight', 

                                   np.where((data.bmi > 30) & (data.bmi < 41), 'Obese',

                                           np.where(data.bmi > 40, 'Morbid Obese', data['bmi']))))



y_healthy_smoker_yes = data[(data['bmi_risk'] == 'Healthy') & (data.smoker=='yes')].charges

y_overweight_smoker_yes = data[(data['bmi_risk'] == 'Overweight') & (data.smoker=='yes')].charges

y_obese_smoker_yes = data[(data['bmi_risk'] == 'Obese') & (data.smoker=='yes')].charges

y_morbid_smoker_yes = data[(data['bmi_risk'] == 'Morbid Obese') & (data.smoker=='yes')].charges



sns.kdeplot(y_healthy_smoker_yes, shade=True, label='Healthy')

sns.kdeplot(y_overweight_smoker_yes, shade=True, label='Overweight')

sns.kdeplot(y_obese_smoker_yes, shade=True, label='Obese')

sns.kdeplot(y_morbid_smoker_yes, shade=True, label='Morbid Obese')

plt.title('Smokers')
y_healthy_smoker_no = data[(data['bmi_risk'] == 'Healthy') & (data.smoker=='no')].charges

y_overweight_smoker_no = data[(data['bmi_risk'] == 'Overweight') & (data.smoker=='no')].charges

y_obese_smoker_no = data[(data['bmi_risk'] == 'Obese') & (data.smoker=='no')].charges

y_morbid_smoker_no = data[(data['bmi_risk'] == 'Morbid Obese') & (data.smoker=='no')].charges



sns.kdeplot(y_healthy_smoker_no, shade=True, label='Healthy')

sns.kdeplot(y_overweight_smoker_no, shade=True, label='Overweight')

sns.kdeplot(y_obese_smoker_no, shade=True, label='Obese')

sns.kdeplot(y_morbid_smoker_no, shade=True, label='Morbid Obese')

plt.title('Non-Smokers')
print('Skew:\n{}\n\nKurtosis:\n{}'.format(train.skew(),train.kurt()))
f, ax = plt.subplots(ncols=2, figsize=(25,6))



# distribution of charges

g1 = sns.distplot(train['charges'], fit=stats.norm, ax=ax[0], color='g')



# Get fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['charges'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



# plot the following distribution

g1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')

plt.title('charges distribution')



# plot the QQ-plot

res = stats.probplot(train['charges'], plot=plt)
# Apply boxcox transformation

train["charges"],  maxlog = stats.boxcox(train["charges"])



# distribution of charges

f, ax = plt.subplots(ncols=2, figsize=(25,6))

g1 = sns.distplot(train['charges'], fit=stats.norm, ax=ax[0], color='g')



# Get fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['charges'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



# plot the distribution

g1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')

plt.title('charges distribution')



# plot the QQ-plot

res = stats.probplot(train['charges'], plot=plt)