# Importing packages

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import scipy.stats # Needed to compute statistics for categorical data
# Importing dataset

heart_data = pd.read_csv('../input/heart-disease/heart.csv')

heart_data.head()
heart_data.isnull().sum()
heart_data['sex'] = heart_data.sex.replace([1,0], ['male', 'female'])

heart_data['cp'] = heart_data.cp.replace([0,1,2,3,4], ['no_cp','typical_ang', 'atypical_ang', 'non_anginal_pain', 'asymptomatic'])

heart_data['fbs'] = heart_data.fbs.replace([1,0], ['true', 'false'])

heart_data['restecg'] = heart_data.restecg.replace([0,1,2], ['normal', 'st_abnormality', 'prob_lvh'])

heart_data['exang'] = heart_data.exang.replace([0,1], ['no', 'yes'])

heart_data['slope'] = heart_data.slope.replace([0,1,2,3], ['no_slope','upsloping', 'flat', 'downsloping'])

heart_data['thal'] = heart_data.thal.replace([3,6,7], ['normal', 'fixed_def', 'rev_def'])

heart_data['target'] = heart_data.target.replace([1,0], ['yes', 'no'])

heart_data.head()
g = sns.pairplot(heart_data, vars =['age', 'trestbps', 'chol', 'thalach', 'oldpeak' ], hue = 'target')

g.map_diag(sns.distplot)

g.add_legend()

g.fig.suptitle('FacetGrid plot', fontsize = 20)

g.fig.subplots_adjust(top= 0.9);
# Plotting correlation matrix

heart_data1 = pd.read_csv('../input/heart-disease/heart.csv')

corr = heart_data1.corr()

corr.style.background_gradient(cmap='RdBu_r')
plt.figure(figsize=(10,4))

plt.legend(loc='upper left')

g = sns.countplot(data = heart_data, x = 'age', hue = 'target')

g.legend(title = 'Heart disease patient?', loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
age_corr = ['age', 'target']

age_corr1 = heart_data[age_corr]

age_corr_y = age_corr1[age_corr1['target'] == 'yes'].groupby(['age']).size().reset_index(name = 'count')

age_corr_y.corr()
sns.regplot(data = age_corr_y, x = 'age', y = 'count').set_title("Correlation graph for Age vs heart disease patient")
age_corr_n = age_corr1[age_corr1['target'] == 'no'].groupby(['age']).size().reset_index(name = 'count')

age_corr_n.corr()
sns.regplot(data = age_corr_n, x = 'age', y = 'count').set_title("Correlation graph for Age vs healthy patient")
# Showing number of heart disease patients based on sex

sex_corr = ['sex', 'target']

sex_corr1 = heart_data[sex_corr]

sex_corr_y = sex_corr1[sex_corr1['target'] == 'yes'].groupby(['sex']).size().reset_index(name = 'count')

sex_corr_y
# Showing number of healthy patients based on sex 

sex_corr_n = sex_corr1[sex_corr1['target'] == 'no'].groupby(['sex']).size().reset_index(name = 'count')

sex_corr_n
g1 = sns.boxplot(data = heart_data, x = 'sex', y = 'age', hue = 'target',palette="Set3")

g1.legend(title = 'Heart disease patient?', loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

g1.set_title('Boxplot showing age vs sex')
# Chi-sq test

cont = pd.crosstab(heart_data["sex"],heart_data["target"])

scipy.stats.chi2_contingency(cont)
# Showing number of heart disease patients based on cp

cp_corr = ['cp', 'target']

cp_corr1 = heart_data[cp_corr]

cp_corr_y = cp_corr1[cp_corr1['target'] == 'yes'].groupby(['cp']).size().reset_index(name = 'count')

cp_corr_y
# Showing number of healthy patients based on cp 

cp_corr_n = cp_corr1[cp_corr1['target'] == 'no'].groupby(['cp']).size().reset_index(name = 'count')

cp_corr_n
# Chi-square test

cont1 = pd.crosstab(heart_data["cp"],heart_data["target"])

scipy.stats.chi2_contingency(cont1)
# Showing number of heart disease patients based on trestbps

restbp_corr = ['trestbps', 'target']

restbp_corr1 = heart_data[restbp_corr]

restbp_corr_y = restbp_corr1[restbp_corr1['target'] == 'yes'].groupby(['trestbps']).size().reset_index(name = 'count')

restbp_corr_y.corr()
sns.regplot(data = restbp_corr_y, x = 'trestbps', y = 'count').set_title('Correlation between resting blood pressure and heart disease patients')
restbp_corr_n = restbp_corr1[restbp_corr1['target'] == 'no'].groupby(['trestbps']).size().reset_index(name = 'count')

restbp_corr_n.corr()
sns.regplot(data = restbp_corr_n, x = 'trestbps', y = 'count').set_title('Correlation between resting blood pressure and healthy patients')
# Showing number of heart disease patients based on serum cholesterol

chol_corr = ['chol', 'target']

chol_corr1 = heart_data[chol_corr]

chol_corr1.chol = chol_corr1.chol.round(decimals=-1)

chol_corr_y = chol_corr1[chol_corr1['target'] == 'yes'].groupby(['chol']).size().reset_index(name = 'count')

chol_corr_y.corr()
sns.regplot(data = chol_corr_y, x = 'chol', y = 'count').set_title('Correlation between serum cholesterol and heart disease patients')
chol_corr_n = chol_corr1[chol_corr1['target'] == 'no'].groupby(['chol']).size().reset_index(name = 'count')

chol_corr_n.corr()
sns.regplot(data = chol_corr_n, x = 'chol', y = 'count').set_title('Correlation between serum cholesterol and healthy patients')
# Showing number of heart disease patients based on fasting blood sugar

fbs_corr = ['fbs', 'target']

fbs_corr1 = heart_data[fbs_corr]

fbs_corr_y = fbs_corr1[fbs_corr1['target'] == 'yes'].groupby(['fbs']).size().reset_index(name = 'count')

fbs_corr_y
# Showing number of healthy patients based on fasting blood sugar

fbs_corr_n = fbs_corr1[fbs_corr1['target'] == 'no'].groupby(['fbs']).size().reset_index(name = 'count')

fbs_corr_n
# Chi-square test

cont3 = pd.crosstab(heart_data["fbs"],heart_data["target"])

scipy.stats.chi2_contingency(cont3)
# Showing number of heart disease patients based on resting ECG results

restecg_corr = ['restecg', 'target']

restecg_corr1 = heart_data[restecg_corr]

restecg_corr_y = restecg_corr1[restecg_corr1['target'] == 'yes'].groupby(['restecg']).size().reset_index(name = 'count')

restecg_corr_y
restecg_corr_n = restecg_corr1[restecg_corr1['target'] == 'no'].groupby(['restecg']).size().reset_index(name = 'count')

restecg_corr_n
# Chi-square test

cont4 = pd.crosstab(heart_data["restecg"],heart_data["target"])

scipy.stats.chi2_contingency(cont4)
# Showing number of heart disease patients based on maximum heart rate

heartrate_corr = ['thalach', 'target']

heartrate_corr1 = heart_data[heartrate_corr]

heartrate_corr_y = heartrate_corr1[heartrate_corr1['target'] == 'yes'].groupby(['thalach']).size().reset_index(name = 'count')

heartrate_corr_y.corr()
sns.regplot(data = heartrate_corr_y, x = 'thalach', y = 'count').set_title('Correlation between maximum heart rate and heart disease patients')
heartrate_corr_n = heartrate_corr1[heartrate_corr1['target'] == 'no'].groupby(['thalach']).size().reset_index(name = 'count')

heartrate_corr_n.corr()
sns.regplot(data = heartrate_corr_n, x = 'thalach', y = 'count').set_title('Correlation between maximum heart rate and healthy patients')