import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns



from scipy import stats

from scipy.stats import norm



import warnings

warnings.filterwarnings('ignore')
class config:

    data_root = '../input/siim-isic-melanoma-classification'
train = pd.read_csv(config.data_root + '/train.csv')

test = pd.read_csv(config.data_root + '/test.csv')
print('Size of train data: ', train.shape)

print('Size of test data: ', test.shape)
train.head(5)
test.head(5)
print('Cases of melanoma in train data',len(train[train.target == 1]) / len(train) * 100,'%')
print('Male in train data:',len(train[train.sex == 'male'])/len(train))

print('Male in test data:',len(test[test.sex == 'male'])/len(test))
print('Unique patients in train data:', len(train.patient_id.unique())/len(train))

print('Unique patients in test data:', len(test.patient_id.unique())/len(test))
print('Matual patients count',len(set(list(train.patient_id.unique())).intersection(list((test.patient_id.unique())))))
train[train.sex.isnull()].head(8)
len(train[(train.sex.isnull()) & (train.age_approx.isnull())]) == len(train[train.sex.isnull()])
train[(train.sex.isnull()) & (train.age_approx.isnull())].patient_id.unique()
len(train[train.anatom_site_general_challenge.isnull()])/len(train)
train.anatom_site_general_challenge.unique()
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('unknown')

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('unknown')



train['age_approx'] = train['age_approx'].fillna(train['age_approx'].mean())

test['age_approx'] = test['age_approx'].fillna(test['age_approx'].mean())



train['sex'] = train['sex'].fillna('female')

test['sex'] = test['sex'].fillna('male')
train[train.isnull().any(1)]
test[test.isnull().any(1)]
plt.figure(figsize = (10,5))

plt.subplot(1,2,1)

sns.distplot(train.age_approx,fit = norm)

plt.title('Age distribution in train data')

plt.subplot(1,2,2)

sns.distplot(test.age_approx,fit = norm)

plt.title('Age distribution in test data')

plt.show()
patients_age = pd.DataFrame(train.groupby(['patient_id'])['age_approx'].max())

patients_age.head(5)
patients_disease = pd.DataFrame(train.groupby(['patient_id'])['target'].sum())

patients_disease.head(5)
patients_gender = pd.DataFrame(train.groupby(['patient_id'])['sex'].max())

patients_gender.head(5)
patients_history = patients_age.merge(patients_disease, on = 'patient_id')

patients_history = patients_history.merge(patients_gender, on = 'patient_id')

patients_history.head(5)
plt.figure(figsize = (10,5))

plt.subplot(1,2,1)

sns.distplot(patients_history[patients_history.target == 0].age_approx,fit = norm)

plt.title('Benign cases')

plt.subplot(1,2,2)

sns.distplot(patients_history[patients_history.target != 0].age_approx,fit = norm)

plt.title('Melanoma')

plt.show()
plt.figure(figsize = (10,5))

plt.subplot(1,2,1)

stats.probplot(patients_history[patients_history.target == 0].age_approx, plot=plt)

plt.subplot(1,2,2)

stats.probplot(patients_history[patients_history.target != 0].age_approx, plot=plt)

plt.show()
print('M0 = ',patients_history[patients_history.target == 0].age_approx.mean())

print('M1 = ',patients_history[patients_history.target != 0].age_approx.mean())



print('D0 = ',patients_history[patients_history.target == 0].age_approx.var())

print('D1 = ',patients_history[patients_history.target != 0].age_approx.var())
from scipy.stats import ttest_ind
t,p = ttest_ind(patients_history[patients_history.target == 0].age_approx,patients_history[patients_history.target != 0].age_approx)



print('p-value = ', p)

print('p-value < 0.05 - ', p<0.05)
train[train.target == 1].groupby(['anatom_site_general_challenge'])['target'].count()
plt.figure(figsize=(15, 15))



for i,site in enumerate(list(train.anatom_site_general_challenge.unique())):

    plt.subplot(3,3,i+1)

    sns.distplot(train[(train.anatom_site_general_challenge == site) & (train.target == 1)].age_approx,fit = norm)

    plt.title('Age distribution with {} melanoma'.format(site))

plt.show()
head_age = train[(train.target == 1) & (train.anatom_site_general_challenge == 'head/neck')].age_approx

upper_age = train[(train.target == 1) & (train.anatom_site_general_challenge == 'upper extremity')].age_approx

lower_age = train[(train.target == 1) & (train.anatom_site_general_challenge == 'lower extremity')].age_approx

torso_age = train[(train.target == 1) & (train.anatom_site_general_challenge == 'torso')].age_approx
(F,p_value) = stats.f_oneway(head_age,upper_age,lower_age,torso_age)

print('\nF = {:.2f}, p-value = {:.4f}'.format(F,p_value)) 
v = [head_age,upper_age,lower_age,torso_age]

avg = [a.mean() for a in v]

err = [1.96 * (np.std(a)/np.sqrt(len(a))) for a in v]
plt.figsize = (30,45)

plt.errorbar(x = ['head/neck','upper extremity','lower extremity','torso'], y = avg, yerr = err, color="black", capsize=3, marker="s", markersize=5, mfc="red", mec="black",fmt = 'o')

plt.title('The age of the patient with melanoma at various sites')

plt.grid()

plt.xlabel('Site')

plt.ylabel('Age')

plt.show()
p = 0.05

n = 4



c = (n-1)*n/2

p = p / c



print("new p = {:.10f}".format(p))
site_target = train[train.target == 1][(train.anatom_site_general_challenge == 'head/neck') | (train.anatom_site_general_challenge == 'upper extremity') | (train.anatom_site_general_challenge == 'lower extremity') | (train.anatom_site_general_challenge == 'torso')]
import statsmodels.stats.multicomp as multi



test = multi.MultiComparison(site_target['age_approx'], site_target['anatom_site_general_challenge'])

res = test.tukeyhsd(alpha = p)

summary = res.summary()

summary