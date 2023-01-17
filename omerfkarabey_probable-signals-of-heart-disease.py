import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import scipy

import math

from sklearn import preprocessing

from matplotlib.mlab import PCA as mlabPCA

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import scale

from sklearn.decomposition import PCA

import warnings

from scipy.stats.mstats import winsorize

warnings.filterwarnings('ignore')

df = pd.read_csv('../input/heart.csv')

df.head()
df.info()
df.isna().sum()
plt.figure(figsize = (14,20))

plt.subplots_adjust(hspace = 0.5)

plt.subplot(621)

plt.boxplot(df.age)

plt.title('Age')

plt.subplot(622)

plt.boxplot(winsorize(df.age))

plt.title('Winsorized Age')

plt.subplot(623)

plt.boxplot(df.trestbps)

plt.title('Resting blood pressure(trestbps)')

plt.subplot(624)

plt.boxplot(winsorize(df.trestbps, 0.03))

plt.title('Winsorized Resting blood pressure')

plt.subplot(625)

plt.boxplot(df.chol)

plt.title('Cholosterol(chol)')

plt.subplot(626)

plt.boxplot(winsorize(df.chol,0.017))

plt.title('Winsorized Cholosterol')

plt.subplot(627)

plt.boxplot(df.thalach)

plt.title('Max heart rate achieved(thalach)')

plt.subplot(628)

plt.boxplot(winsorize(df.thalach, 0.004, axis = -1))

plt.title('Winsorized max heart rate achieved')

plt.subplot(629)

plt.boxplot(df.oldpeak)

plt.title('Old peak')

plt.subplot(6,2,10)

plt.boxplot(winsorize(df.oldpeak, 0.018))

plt.title('Winsorized old peak')

plt.show()
df.trestbps = winsorize(df.trestbps, 0.03)

df.chol = winsorize(df.chol,0.017)

df.thalach = winsorize(df.thalach, 0.004, axis = -1)

df.oldpeak = winsorize(df.oldpeak, 0.018)
df.describe()
plt.figure(figsize = (6,5))

x = ['no', 'yes']

y = df['target'].value_counts(sort = False).values

plt.bar(x,y)

plt.title('Distribution of People by Heart Disease \n', size = 15)

plt.ylabel('Number of People', size = 15)

plt.show()
fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize = (15,20))

fig.suptitle('\n \n \nCategorical Variable Distributions \n \n', fontsize = 16)

plt.subplots_adjust(wspace=0.3, hspace=0.6)

x = ['Female', 'Male']

y = df[df['target']==1].sex.value_counts(sort = False).values 

axes[0][0].bar(x,y)

axes[0][0].set_title('Sex \n\n')

axes[0][0].set_ylabel('Numer of People')



x = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']

y = df[df['target']==1].cp.value_counts(sort=False).values

axes[0][1].bar(x,y)

axes[0][1].set_title('Chest Pain Type \n \n')

axes[0][1].set_ylabel('Numer of People')



x = ['Healthy', 'Unhealthy']

y = df[df['target']==1].fbs.value_counts(sort = False).values

axes[1][0].bar(x,y)

axes[1][0].set_title('Fasting blood sugar \n\n')

axes[1][0].set_ylabel('Numer of People')



x = ['Regular', 'Abnormality', 'Severe']

y = df[df['target']==1].restecg.value_counts(sort = False).values

axes[1][1].bar(x,y)

axes[1][1].set_title('Resting Electrocardiographic Results \n \n')

axes[1][1].set_ylabel('Numer of People')



x = ['No', 'Yes']

y = df[df['target']==1].exang.value_counts(sort = False).values

axes[2][0].bar(x,y)

axes[2][0].set_title('Exercise induced angina \n\n')

axes[2][0].set_ylabel('Numer of People')



x = ['Downward','Flat','Upward']

y = df[df['target']==1].slope.value_counts(sort = False).values

axes[2][1].bar(x,y)

axes[2][1].set_title('ST excercise peak \n \n')

axes[2][1].set_ylabel('Numer of People')



x = ['None','Normal','Fixed Defect','Reversable Defect']

y = df[df['target']==1].thal.value_counts(sort = False).values

axes[3][0].bar(x,y)

axes[3][0].set_title('Thalium Stress Test \n \n')

axes[3][0].set_ylabel('Numer of People')

plt.show()
sns.set(style="darkgrid")

baslik_font = {'family': 'arial', 'color': 'darkred', 'weight': 'bold', 'size' : 13}

eksen_font = {'family':'arial', 'color':'darkred', 'weight' : 'bold', 'size':13}

plt.figure(figsize = (14,6))

sns.countplot(y = 'cp', hue = 'sex', data = df, palette = 'Greens_d')

plt.title('Chest Pain Distribution \n', fontdict = baslik_font)

plt.ylabel('Chest Pain Type\n 0:Typical Ang., 1:Atypical Ang.\n 2:Non anginal, 3:Asypmtomatic \n', fontdict = eksen_font)

plt.xlabel(('\n Number of People \n0:Female, 1:Male'), fontdict = eksen_font)

plt.show()
corelmat = df[df['target'] == 1].corr()

corelmat
plt.figure(figsize = (8,6))

sns.heatmap(corelmat)

plt.show()
print('Mean age of people', df.age.mean())

print('Mean age of risky people',df[df['target']==1].age.mean())

print('Mean age of not risky people', df[df['target']==0].age.mean())
print('Patients over 54 diagnosis ratio: {} %'.format(round(df[df['age']>54]['target'].mean()*100,2)))

print('Patients under 54 diagnosis ratio: {} %'.format(round(df[df['age']<=54]['target'].mean()*100,2)))
t_age = scipy.stats.ttest_ind(df[df['target']==0].age, df[df['target']==1].age)

t_age
a = df[df['sex'] == 0]['target']

b = df[df['sex'] == 1]['target']

t_sex = scipy.stats.ttest_ind(a,b)

c = df[df['fbs'] == 0]['target']

d = df[df['fbs'] == 1]['target']

t_fbs = scipy.stats.ttest_ind(c,d)

e = df[df['exang'] == 0]['target']

f = df[df['exang'] == 1]['target']

t_exang = scipy.stats.ttest_ind(e,f)



arr_1 = ['Sex', 'Fasting blood sugar', 'Exercise induced angina']

arr_2 = [t_sex, t_fbs, t_exang]

for i in range(0,3):

    

    print('For {}: {}'.format(arr_1[i], arr_2[i]))
pain_types = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']

for i in range (0,4):

    cp_mean = df[df['cp']==i]['target'].mean()

    cp_mean = round(cp_mean, 3)

    print('Heart disease risk for {} is {} %'.format(pain_types[i], cp_mean*100))
arr_ca = df[df['target']==1].ca.value_counts(sort = False)

x = ['0','1','2','3','4']

plt.figure(figsize = (8,5))

plt.bar(x, arr_ca)

plt.title('Major Vessels Distribution \n')

plt.xlabel('\nNumber of Major Vessels')

plt.ylabel('Number of People')

plt.show()
arr_slope = df[df['target']==1].slope.value_counts(sort = False)

x = ['Upslopping', 'Flat', 'Downslopping']

plt.figure(figsize = (8,5))

plt.bar(x,arr_slope)

plt.title('Slopes of the Peak Exercise \n')

plt.ylabel('Number of People')

plt.xlabel('\nSlope Types')

plt.show()
plt.figure(figsize=(16,19))



plt.subplots_adjust(hspace=0.5, wspace=0.3)

plt.suptitle('Distribution of Variables by Age\n Linear Regression Plots', size = 17, color = 'darkred')



plt.subplot(421)

plt.title('Diagnosed People \n\n\n', size = 16,color ='darkred')

sns.regplot(df[df['target']==1]['age'], df[df['target']==1]['thalach'])

plt.xlabel('Age')

plt.ylabel('Maximum Heart Rate Achieved')



plt.subplot(422)

plt.title('Not Diagnosed People \n\n\n', size = 16,color ='darkred')

sns.regplot(df[df['target']==0]['age'], df[df['target']==0]['thalach'])

plt.xlabel('Age')

plt.ylabel('Maximum Heart Rate Achieved')



plt.subplot(423)

sns.regplot(df[df['target']==1]['age'], df[df['target']==1]['chol'])

plt.xlabel('Age')

plt.ylabel('Cholesterol Serum')



plt.subplot(424)

sns.regplot(df[df['target']==0]['age'], df[df['target']==0]['chol'])

plt.xlabel('Age')

plt.ylabel('Cholesterol Serum')



plt.subplot(425)

sns.regplot(df[df['target']==1]['age'], df[df['target']==1]['trestbps'])

plt.xlabel('Age')

plt.ylabel('Resting Blood Pressure')



plt.subplot(426)

sns.regplot(df[df['target']==0]['age'], df[df['target']==0]['trestbps'])

plt.xlabel('Age')

plt.ylabel('Resting Blood Pressure')



plt.subplot(427)

sns.regplot(df[df['target']==1]['age'], df[df['target']==1]['oldpeak'])

plt.xlabel('Age')

plt.ylabel('ST Depression')



plt.subplot(428)

sns.regplot(df[df['target']==0]['age'], df[df['target']==0]['oldpeak'])

plt.xlabel('Age')

plt.ylabel('ST Depression')

plt.show()
t_a = scipy.stats.ttest_ind(df[df['target']==1]['thalach'],df[df['target']==0]['thalach'])

t_b = scipy.stats.ttest_ind(df[df['target']==1]['chol'],df[df['target']==0]['chol'])

t_c = scipy.stats.ttest_ind(df[df['target']==1]['trestbps'],df[df['target']==0]['trestbps'])

t_d = scipy.stats.ttest_ind(df[df['target']==1]['oldpeak'],df[df['target']==0]['oldpeak'])



a_1 = ['Max heart rate achieved', 'Cholosterol', 'Resting blood pleasure', 'ST depression']

a_2 = [t_a,t_b,t_c,t_d]

for i in range(0,4):

    print('For {}: {}'.format(a_1[i], a_2[i]))
df_2 = df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'trestbps','slope']]

plt.figure(figsize = (10,6))

scaled = preprocessing.scale(df_2.T)

pca = PCA()

pca.fit(scaled)

pca_data = pca.transform(scaled)

per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1) 

labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)

plt.ylabel('Percentage of Explained Variance')

plt.xlabel('Principal Component')

plt.title('Scree Plot')

plt.show()
for i in range(0,3):

    print(per_var[i])