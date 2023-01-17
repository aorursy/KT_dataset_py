# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Imports



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

from scipy.stats import ttest_ind, ttest_rel, chi2_contingency, chi2

from scipy import stats

import random
# Fetching data



data = pd.read_csv('../input/adult-census-income/adult.csv')

data
data.shape
# Cleaning data



att, counts = np.unique(data['workclass'], return_counts = True)

max_count = att[np.argmax(counts, axis = 0)]

data['workclass'][data['workclass'] == '?'] = max_count 



att, counts = np.unique(data['occupation'], return_counts = True)

max_count = att[np.argmax(counts, axis = 0)]

data['occupation'][data['occupation'] == '?'] = max_count 



att, counts = np.unique(data['native.country'], return_counts = True)

max_count = att[np.argmax(counts, axis = 0)]

data['native.country'][data['native.country'] == '?'] = max_count 
# Converting 'income' values to be '1' for income > 50K and '0' for income <= 50K



data['income']=data['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})

data
# Statistics of numeric columns



data_num = data.copy()

data_num = data.drop(["education.num","income"], axis=1)

data_num.describe()
# Statistics of string columns



data.describe(include=["O"])
data['age'].hist(figsize=(8,8))

plt.show()
data['hours.per.week'].hist(figsize=(8,8))

plt.show()
data['fnlwgt'].hist(figsize=(8,8))

plt.show()
data["capital.gain"].hist(figsize=(8,8))

plt.show()
data["capital.loss"].hist(figsize=(8,8))

plt.show()
sns.relplot('capital.gain','capital.loss', data= data)

plt.xlabel("capital gain")

plt.ylabel("capital loss")

plt.show()
plt.figure(figsize=(12,8))

total = float(len(data["income"]) )

ax = sns.countplot(x="workclass", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(15,8))

total = float(len(data["income"]))

ax = sns.countplot(x="education", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(15,8))

total = float(len(data))

ax = sns.countplot(x="marital.status", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(25,8))

total = float(len(data))

ax = sns.countplot(x="occupation", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(15,8))

total = float(len(data))

ax = sns.countplot(x="relationship", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(15,8))

total = float(len(data))

ax = sns.countplot(x="race", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(8,8))

total = float(len(data))

ax = sns.countplot(x="sex", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(18,8))

total = float(len(data))

ax = sns.countplot(y="native.country", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(7,7))

total = float(len(data))

ax = sns.countplot(x="income", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
fig = plt.figure(figsize=(10,10)) 

sns.boxplot(x="income", y="age", data=data)

plt.show()
data = data[(np.abs(stats.zscore(data["age"])) < 3)] 

income_1 = data[data['income']==1]['age']

income_0 = data[data['income']==0]['age']



income_0 = income_0.values.tolist()

income_0 = random.sample(income_0, 100)

income_1 = income_1.values.tolist()

income_1 = random.sample(income_1, 100)



ttest,pval = ttest_ind(income_1,income_0,equal_var = False)

print("ttest",ttest)

print('p value',pval)





if pval <0.05:

    print("we reject null hypothesis")

else:

    print("we accept null hypothesis")
fig = plt.figure(figsize=(10,10)) 

sns.boxplot(x="income", y="hours.per.week", data=data)

plt.show()
data = data[(np.abs(stats.zscore(data["hours.per.week"])) < 3)] 

income_1 = data[data['income']==1]["hours.per.week"]

income_0 = data[data['income']==0]["hours.per.week"]



income_0 = income_0.values.tolist()

income_0 = random.sample(income_0, 100)

income_1 = income_1.values.tolist()

income_1 = random.sample(income_1, 100)



ttest,pval = ttest_ind(income_1,income_0,equal_var = False)

print("ttest",ttest)

print('p value',format(pval, '.70f'))



if pval <0.05:

    print("we reject null hypothesis")

else:

    print("we accept null hypothesis")
plt.figure(figsize=(10,7))

sns.boxplot(x="income", y="fnlwgt", data=data)

plt.show()
data = data[(np.abs(stats.zscore(data["fnlwgt"])) < 3)] 

income_1 = data[data['income']==1]["fnlwgt"]

income_0 = data[data['income']==0]["fnlwgt"]



income_0 = income_0.values.tolist()

income_0 = random.sample(income_0, 100)

income_1 = income_1.values.tolist()

income_1 = random.sample(income_1, 100)



ttest,pval = ttest_ind(income_1,income_0,equal_var = False)

print("ttest",ttest)

print("p-value",pval)



if pval <0.05:

    print("we reject null hypothesis")

else:

    print("we accept null hypothesis")
plt.figure(figsize=(10,7))

sns.boxplot(x="income", y="capital.gain", data=data)

plt.show()
data = data[(np.abs(stats.zscore(data["capital.gain"])) < 3)] 

income_1 = data[data['income']==1]["capital.gain"]

income_0 = data[data['income']==0]["capital.gain"]



income_0 = income_0.values.tolist()

income_0 = random.sample(income_0, 100)

income_1 = income_1.values.tolist()

income_1 = random.sample(income_1, 100)



ttest,pval = ttest_ind(income_1,income_0,equal_var = False)

print("ttest",ttest)

print("p-value",pval)



if pval <0.05:

    print("we reject null hypothesis")

else:

    print("we accept null hypothesis")
plt.figure(figsize=(10,7))

sns.boxplot(x="income", y="capital.loss", data=data)

plt.show()
income_1 = data[data['income']==1]["capital.loss"]

income_0 = data[data['income']==0]["capital.loss"]



income_0 = income_0.values.tolist()

income_0 = random.sample(income_0, 100)

income_1 = income_1.values.tolist()

income_1 = random.sample(income_1, 100)



ttest,pval = ttest_ind(income_1,income_0,equal_var = False)

print("ttest",ttest)

print("p-value",pval)



if pval <0.05:

    print("we reject null hypothesis")

else:

    print("we accept null hypothesis")
plt.figure(figsize=(12,10))

total = float(len(data["income"]) )

ax = sns.countplot(x="workclass", hue="income", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
# contingency table

c_t = pd.crosstab(data['workclass'].sample(frac=0.002, replace=True, random_state=1),data['income'].sample(frac=0.002, replace=True, random_state=1),margins = False) 

c_t
stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print('p_value', p)

print('expected',expected)



prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('we reject null hypothesis (dependent) ')

else:

    print('we accept null hypothesis (independent)')
plt.figure(figsize=(20,10))

total = float(len(data["income"]))

ax = sns.countplot(x="education", hue="income", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
# contingency table

c_t = pd.crosstab(data['education'].sample(frac=0.002, replace=True, random_state=1),data['income'].sample(frac=0.002, replace=True, random_state=1),margins = False) 

c_t
stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print("p-value", p)

print(expected)



prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))



if abs(stat) >= critical:

    print('we reject null hypothesis (dependent)')

else:

    print('we accept null hypothesis (independent)')
plt.figure(figsize=(17,10))

total = float(len(data))

ax = sns.countplot(x="marital.status", hue="income", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
# contingency table

c_t = pd.crosstab(data['marital.status'].sample(frac=0.002, replace=True, random_state=1),data['income'].sample(frac=0.002, replace=True, random_state=1),margins = False) 

c_t
stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print('p_value', p)

print(expected)



prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('we reject null hypothesis (dependent)')

else:

    print('we accept null hypothesis (independent)')
plt.figure(figsize=(25,10))

total = float(len(data))

ax = sns.countplot(x="occupation", hue="income", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
# contingency table

c_t = pd.crosstab(data['occupation'].sample(frac=0.002, replace=True, random_state=1),data['income'].sample(frac=0.002, replace=True, random_state=1),margins = False) 

c_t
stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print('p_value', p)

print(expected)



prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('we reject null hypothesis (dependent)')

else:

    print('we accept null hypothesis (independent)')
plt.figure(figsize=(17,10))

total = float(len(data))

ax = sns.countplot(x="relationship", hue="income", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
# contingency table

c_t = pd.crosstab(data['relationship'].sample(frac=0.002, replace=True, random_state=1),data['income'].sample(frac=0.002, replace=True, random_state=1),margins = False) 

c_t
stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print('p_value', p)

print(expected)



prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('we reject null hypothesis (dependent)')

else:

    print('we accept null hypothesis (independent)')
plt.figure(figsize=(17,10))

total = float(len(data) )



ax = sns.countplot(x="race", hue="income", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
# contingency table

c_t = pd.crosstab(data['race'].sample(frac=0.002, replace=True, random_state=1),data['income'].sample(frac=0.002, replace=True, random_state=1),margins = False) 

c_t
stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print('p_value', p)

print(expected)



prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('we reject null hypothesis (dependent)')

else:

    print('we accept null hypothesis (independent)')
plt.figure(figsize=(10,10))

total = float(len(data))

ax = sns.countplot(x="sex", hue="income", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
# contingency table

c_t = pd.crosstab(data['sex'].sample(frac=0.002, replace=True, random_state=1),data['income'].sample(frac=0.002, replace=True, random_state=1),margins = False) 

c_t
stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print('p_value', p)

print(expected)



prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('we reject null hypothesis (dependent)')

else:

    print('we accept null hypothesis (independent)')
plt.figure(figsize=(25,20))

total = float(len(data))

ax = sns.countplot(x="sex", hue="native.country", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
# contingency table

c_t = pd.crosstab(data['native.country'].sample(frac=0.002, replace=True, random_state=1),data['income'].sample(frac=0.002, replace=True, random_state=1),margins = False) 

c_t
stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print('p_value', p)

print(expected)



prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('we reject null hypothesis (dependent)')

else:

    print('we accept null hypothesis (independent)')
plt.figure(figsize=(15,10))  

sns.heatmap(data_num.corr(),annot=True,linewidths=.5, cmap="Blues")

plt.title('Heatmap showing correlations between numerical data')

plt.show()
plt.figure(figsize=(12,6))

sns.boxplot(x='income',y ='hours.per.week', hue='sex',data=data)

plt.show()
plt.figure(figsize=(15,10))

sns.boxplot(x="income", y="age",hue="sex",data=data)

plt.show()