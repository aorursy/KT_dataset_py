import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

from scipy.stats import ttest_ind, ttest_rel

from scipy import stats
data = pd.read_csv("../input/adult.csv")

data.head(10)
data.shape
data_num = data.copy()
attrib, counts = np.unique(data['workclass'], return_counts = True)

most_freq_attrib = attrib[np.argmax(counts, axis = 0)]

data['workclass'][data['workclass'] == '?'] = most_freq_attrib 



attrib, counts = np.unique(data['occupation'], return_counts = True)

most_freq_attrib = attrib[np.argmax(counts, axis = 0)]

data['occupation'][data['occupation'] == '?'] = most_freq_attrib 



attrib, counts = np.unique(data['native-country'], return_counts = True)

most_freq_attrib = attrib[np.argmax(counts, axis = 0)]

data['native-country'][data['native-country'] == '?'] = most_freq_attrib 
data.head(10)
data['income']=data['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})

data.head()
data_num = data.drop(["educational-num","income"], axis=1)

data_num.describe()
data.describe(include=["O"])
data['age'].hist(figsize=(8,8))

plt.show()
data[data["age"]>70].shape
data['hours-per-week'].hist(figsize=(8,8))

plt.show()
data['fnlwgt'].hist(figsize=(8,8))

plt.show()
data["capital-gain"].hist(figsize=(8,8))

plt.show()
data["capital-loss"].hist(figsize=(8,8))

plt.show()
data[data["capital-loss"]>0].shape
sns.relplot('capital-gain','capital-loss', data= data)

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
plt.figure(figsize=(20,8))

total = float(len(data["income"]) )



ax = sns.countplot(x="education", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(15,8))

total = float(len(data) )



ax = sns.countplot(x="marital-status", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(25,8))

total = float(len(data) )



ax = sns.countplot(x="occupation", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(15,8))

total = float(len(data) )



ax = sns.countplot(x="relationship", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(15,8))

total = float(len(data) )



ax = sns.countplot(x="race", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(8,8))

total = float(len(data) )



ax = sns.countplot(x="gender", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(18,8))

total = float(len(data) )



ax = sns.countplot(y="native-country", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(7,7))

total = float(len(data) )



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
data[['income', 'age']].groupby(['income'], as_index=False).mean().sort_values(by='age', ascending=False)
import random



data = data[(np.abs(stats.zscore(data["age"])) < 3)] 



income_1 = data[data['income']==1]['age']

income_0 = data[data['income']==0]['age']



income_0 = income_0.values.tolist()

income_0 = random.sample(income_0, 100)

income_1 = income_1.values.tolist()

income_1 = random.sample(income_1, 100)
from scipy.stats import ttest_ind

ttest,pval = ttest_ind(income_1,income_0,equal_var = False)

print("ttest",ttest)

print('p value',pval)





if pval <0.05:

    print("we reject null hypothesis")

else:

    print("we accept null hypothesis")
fig = plt.figure(figsize=(10,10)) 

sns.boxplot(x="income", y="hours-per-week", data=data)

plt.show()
data = data[(np.abs(stats.zscore(data["hours-per-week"])) < 3)] 



income_1 = data[data['income']==1]["hours-per-week"]

income_0 = data[data['income']==0]["hours-per-week"]



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

sns.boxplot(x="income", y="capital-gain", data=data)

plt.show()
data = data[(np.abs(stats.zscore(data["capital-gain"])) < 3)] 



income_1 = data[data['income']==1]["capital-gain"]

income_0 = data[data['income']==0]["capital-gain"]



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

sns.boxplot(x="income", y="capital-loss", data=data)

plt.show()
income_1 = data[data['income']==1]["capital-loss"]

income_0 = data[data['income']==0]["capital-loss"]



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
from scipy.stats import chi2_contingency

from scipy.stats import chi2





stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print('p_value', p)

print(expected)



# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')

plt.figure(figsize=(20,10))

total = float(len(data["income"]) )



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



# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))



if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
plt.figure(figsize=(17,10))

total = float(len(data) )



ax = sns.countplot(x="marital-status", hue="income", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
# contingency table

c_t = pd.crosstab(data['marital-status'].sample(frac=0.002, replace=True, random_state=1),data['income'].sample(frac=0.002, replace=True, random_state=1),margins = False) 

c_t
stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print('p_value', p)

print(expected)



# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))



if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
plt.figure(figsize=(25,10))

total = float(len(data) )



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

print(expected)



# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))



if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
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

print(expected)



# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))



if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
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



# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))



if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
plt.figure(figsize=(10,10))

total = float(len(data) )



ax = sns.countplot(x="gender", hue="income", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
# contingency table

c_t = pd.crosstab(data['gender'].sample(frac=0.002, replace=True, random_state=1),data['income'].sample(frac=0.002, replace=True, random_state=1),margins = False) 

c_t
stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print('p_value', p)

print(expected)



# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))



if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
# contingency table

c_t = pd.crosstab(data['native-country'].sample(frac=0.002, replace=True, random_state=1),data['income'].sample(frac=0.002, replace=True, random_state=1),margins = False) 
stat, p, dof, expected = chi2_contingency(c_t)

print('dof=%d' % dof)

print('p_value', p)

print(expected)



# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))



if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
plt.figure(figsize=(15,10))  

sns.heatmap(data_num.corr(),annot=True,linewidths=.5, cmap="Blues")

plt.title('Heatmap showing correlations between numerical data')

plt.show()
plt.figure(figsize=(12,6))

sns.boxplot(x='income',y ='hours-per-week', hue='gender',data=data)

plt.show()
plt.figure(figsize=(15,10))

sns.boxplot(x="income", y="age",hue="gender",data=data)

plt.show()
fig = plt.figure(figsize = (17,10))

ax = fig.add_subplot(2,1,1)

sns.stripplot('age', 'capital-gain', data = data,

         jitter = 0.2,ax = ax);

plt.xlabel('Age',fontsize = 12);

plt.ylabel('Capital Gain',fontsize = 12);



ax = fig.add_subplot(2,1,2)

sns.stripplot('age', 'capital-gain', data = data,

         jitter = 0.2);

plt.xlabel('Age',fontsize = 12);

plt.ylabel('Capital Gain',fontsize = 12);

plt.ylim(0,40000);
cols = ['workclass','occupation']

cat_col = data.dtypes[data.dtypes == 'object']

for col in cat_col.index:

    if col in cols:

        print(f"======================================={col}=========================")

        print(data[data['age'] == 90][col].value_counts())

    else:

        continue
fig = plt.figure(figsize = (17,10))

ax = fig.add_subplot(2,1,1)

sns.stripplot('hours-per-week', 'capital-gain', data = data,

         jitter = 0.2,ax = ax);

plt.xlabel('Hours per week',fontsize = 12);

plt.ylabel('Capital Gain',fontsize = 12);



ax = fig.add_subplot(2,1,2)

sns.stripplot('hours-per-week', 'capital-gain', data = data,

         jitter = 0.2,ax = ax);

plt.xlabel('Hours per week',fontsize = 12);

plt.ylabel('Capital Gain',fontsize = 12);

plt.ylim(0,40000);
cols = ['workclass','occupation']

cat_col = data.dtypes[data.dtypes == 'object']

for col in cat_col.index:

    if col in cols:

        print(f"======================================={col}=========================")

        print(data[data['hours-per-week'] == 99][col].value_counts())

    else:

        continue
data["capital_change"] = data["capital-gain"] - data["capital-loss"]

data["capital_change"].describe()
data["capital_change"].hist(figsize=(8,8))

plt.show()
income_1 = data[data['income']==1]["capital_change"]

income_0 = data[data['income']==0]["capital_change"]



data = data[(np.abs(stats.zscore(data["age"])) < 3)] 



income_0 = income_0.values.tolist()

income_0 = random.sample(income_0, 50)

income_1 = income_1.values.tolist()

income_1 = random.sample(income_1, 50)



ttest,pval = ttest_ind(income_1,income_0, equal_var=0)

print("ttest",ttest)

print("p-value",pval)



if pval <0.05:

    print("we reject null hypothesis")

else:

    print("we accept null hypothesis")