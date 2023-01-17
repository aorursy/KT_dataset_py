import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import gridspec

import math

import numpy as np

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import NearestNeighbors

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

import itertools

import scipy.stats

from scipy import stats

import math as mth
heart_disease_df = pd.read_csv('../input/heart.csv', sep = ',',error_bad_lines = False)
heart_disease_df.info()

print('Data shape: ', heart_disease_df.shape)
heart_disease_df.head()
heart_disease_df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].describe()
plt.figure(figsize = (20, 40))

plt.suptitle('Qualitative features exploration', y = 0.90, fontsize = 16)

gs = gridspec.GridSpec(5, 2)



plt.subplot(gs[0, :])

ax = sns.countplot(heart_disease_df['target'].replace({1 : 'Yes', 0 : 'No'}), palette = 'mako', order = ['Yes', 'No'])

plt.xlabel('Heart disease')

plt.ylabel('Count')

plt.title('Distribution of diagnosis')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 0])

ax = sns.countplot(heart_disease_df['sex'].replace({1: 'Male', 0: 'Female'}), palette = 'mako')

plt.xlabel('Sex')

plt.ylabel('Count')

plt.title('Distribution of sex')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 1])

ax = sns.countplot(heart_disease_df.cp, palette = 'mako', order = [0, 2, 1, 3])

plt.xlabel('Chest pain type')

plt.ylabel('Count')

plt.title('Distribution of chest pain type')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[2, 0])

ax = sns.countplot(heart_disease_df['fbs'].replace({1 : 'Yes', 0 : 'No'}), palette = 'mako', order = ('No', 'Yes'))

plt.xlabel('Fasting blood sugar (>120 mg/dl)')

plt.ylabel('Count')

plt.title('Distribution of fasting blood sugar')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[2, 1])

ax = sns.countplot(heart_disease_df['restecg'], palette = 'mako', order = [1, 0, 2])

plt.xlabel('Resting electrocardiographic results')

plt.ylabel('Count')

plt.title('Distribution of resting electrocardiographic results')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[3, 0])

ax = sns.countplot(heart_disease_df['exang'].replace({1: 'Yes', 0: 'No'}), palette = 'mako')

plt.xlabel('Exercise induced angina')

plt.ylabel('Count')

plt.title('Distribution of exercise induced angina')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[3, 1])

ax = sns.countplot(heart_disease_df['slope'], palette = 'mako', order = [2, 1, 0])

plt.xlabel('The slope of the peak exercise ST segment')

plt.ylabel('Count')

plt.title('Distribution of slope of the peak exercise ST segment')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[4, 0])

ax = sns.countplot(heart_disease_df['ca'], palette = 'mako')

plt.xlabel('CA')

plt.ylabel('Count')

plt.title('Distribution ca')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[4, 1])

ax = sns.countplot(heart_disease_df['thal'], palette = 'mako', order = [2, 3, 1, 0])

plt.xlabel('Thal')

plt.ylabel('Count')

plt.title('Distribution of thal')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
plt.figure(figsize = (20, 40))

plt.suptitle('Quantitative features exploration', y = 0.90, fontsize = 16)

gs = gridspec.GridSpec(5, 2)



plt.subplot(gs[0, 0])

ax = sns.distplot(heart_disease_df['age'], color = '#3daea5')

plt.xlabel('Age')

plt.title('Age distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[0, 1])

ax = sns.distplot(heart_disease_df['trestbps'], color = '#3daea5')

plt.xlabel('Resting blood pressure')

plt.title('Resting blood pressure distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 0])

ax = sns.distplot(heart_disease_df['chol'], color = '#3daea5')

plt.xlabel('Serum cholestoral in mg/dl')

plt.title('Serum cholestoral in mg/dl distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 1])

ax = sns.distplot(heart_disease_df['thalach'], color = '#3daea5')

plt.xlabel('Maximum heart rate achieved')

plt.title('Maximum heart rate achieved distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[2, 0])

ax = sns.distplot(heart_disease_df['oldpeak'], color = '#3daea5')

plt.xlabel('ST depression induced by exercise relative to rest')

plt.title('ST depression induced by exercise relative to rest distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
plt.figure(figsize = (20, 40))

plt.suptitle('Target exploration (qualitative features)', y = 0.90, fontsize = 16)

gs = gridspec.GridSpec(5, 2)



plt.subplot(gs[0, 0])

ax = sns.countplot(heart_disease_df['sex'].replace({0: 'F', 1: 'M'}), palette = 'mako', hue = heart_disease_df['target'].replace({0:'No', 1:'Yes'}))

plt.xlabel('Sex')

plt.ylabel('Count')

plt.title('Distribution of diagnosis')

plt.legend(title = 'Heart disease', loc = 1, frameon = False)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[0, 1])

ax = sns.countplot(heart_disease_df.cp, palette = 'mako', order = [0, 2, 1, 3], hue = heart_disease_df['target'].replace({0:'No', 1:'Yes'}))

plt.xlabel('Chest pain type')

plt.ylabel('Count')

plt.legend(title = 'Heart disease', loc = 1, frameon = False)

plt.title('Distribution of chest pain type')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 0])

ax = sns.countplot(heart_disease_df['fbs'].replace({1 : 'Yes', 0 : 'No'}), palette = 'mako', order = ('No', 'Yes'), hue = heart_disease_df['target'].replace({0:'No', 1:'Yes'}))

plt.xlabel('Fasting blood sugar (>120 mg/dl)')

plt.ylabel('Count')

plt.legend(title = 'Heart disease', loc = 1, frameon = False)

plt.title('Distribution of fasting blood sugar')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 1])

ax = sns.countplot(heart_disease_df['restecg'], palette = 'mako', order = [1, 0, 2], hue = heart_disease_df['target'].replace({0:'No', 1:'Yes'}))

plt.xlabel('Resting electrocardiographic results')

plt.ylabel('Count')

plt.legend(title = 'Heart disease', loc = 1, frameon = False)

plt.title('Distribution of resting electrocardiographic results')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[2, 0])

ax = sns.countplot(heart_disease_df['exang'].replace({1: 'Yes', 0: 'No'}), palette = 'mako', hue = heart_disease_df['target'].replace({0:'No', 1:'Yes'}))

plt.xlabel('Exercise induced angina')

plt.ylabel('Count')

plt.legend(title = 'Heart disease', loc = 1, frameon = False)

plt.title('Distribution of exercise induced angina')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[2, 1])

ax = sns.countplot(heart_disease_df['slope'], palette = 'mako', order = [2, 1, 0], hue = heart_disease_df['target'].replace({0:'No', 1:'Yes'}))

plt.xlabel('The slope of the peak exercise ST segment')

plt.ylabel('Count')

plt.legend(title = 'Heart disease', loc = 1, frameon = False)

plt.title('Distribution of slope of the peak exercise ST segment')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[3, 0])

ax = sns.countplot(heart_disease_df['ca'], palette = 'mako', hue = heart_disease_df['target'].replace({0:'No', 1:'Yes'}))

plt.xlabel('CA')

plt.ylabel('Count')

plt.title('Distribution ca')

plt.legend(title = 'Heart disease', loc = 1, frameon = False)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[3, 1])

ax = sns.countplot(heart_disease_df['thal'], palette = 'mako', order = [2, 3, 1, 0], hue = heart_disease_df['target'].replace({0:'No', 1:'Yes'}))

plt.xlabel('Thal')

plt.ylabel('Count')

plt.legend(title = 'Heart disease', loc = 1, frameon = False)

plt.title('Distribution of thal')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
plt.figure(figsize = (20, 40))

plt.suptitle('Target exploration (quantitative features)', y = 0.90, fontsize = 16)

gs = gridspec.GridSpec(5, 2)



plt.subplot(gs[0, 0])

sns.boxplot(heart_disease_df['target'].replace({0:'No', 1:'Yes'}), heart_disease_df['age'], palette = 'mako')

plt.xlabel('Heart disease')

plt.ylabel('Age')



plt.subplot(gs[0, 1])

sns.boxplot(heart_disease_df['target'].replace({0:'No', 1:'Yes'}), heart_disease_df['chol'], palette = 'mako')

plt.xlabel('Heart disease')

plt.ylabel('Serum cholestoral in mg/dl')



plt.subplot(gs[1, 0])

sns.boxplot(heart_disease_df['target'].replace({0:'No', 1:'Yes'}), heart_disease_df['thalach'], palette = 'mako')

plt.xlabel('Heart disease')

plt.ylabel('Maximum heart rate achieved')





plt.subplot(gs[1, 1])

sns.boxplot(heart_disease_df['target'].replace({0:'No', 1:'Yes'}), heart_disease_df['oldpeak'], palette = 'mako')

plt.xlabel('Heart disease')

plt.ylabel('ST depression induced by exercise relative to rest')
plt.figure(figsize = (20, 40))

plt.suptitle('Sex exploration (qualitative features)', y = 0.90, fontsize = 16)

gs = gridspec.GridSpec(5, 2)



plt.subplot(gs[0, 0])

ax = sns.countplot(heart_disease_df['target'].replace({0:'No', 1:'Yes'}), palette = 'mako', hue = heart_disease_df['sex'].replace({0: 'F', 1: 'M'}))

plt.xlabel('Heart disease')

plt.ylabel('Count')

plt.title('Distribution of diagnosis')

plt.legend(title = 'Sex', loc = 1, frameon = False)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[0, 1])

ax = sns.countplot(heart_disease_df.cp, palette = 'mako', order = [0, 2, 1, 3], hue = heart_disease_df['sex'].replace({0: 'F', 1: 'M'}))

plt.xlabel('Chest pain type')

plt.ylabel('Count')

plt.legend(title = 'Sex', loc = 1, frameon = False)

plt.title('Distribution of chest pain type')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 0])

ax = sns.countplot(heart_disease_df['fbs'].replace({1 : 'Yes', 0 : 'No'}), palette = 'mako', order = ('No', 'Yes'), hue = heart_disease_df['sex'].replace({0: 'F', 1: 'M'}))

plt.xlabel('Fasting blood sugar (>120 mg/dl)')

plt.ylabel('Count')

plt.legend(title = 'Sex', loc = 1, frameon = False)

plt.title('Distribution of fasting blood sugar')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 1])

ax = sns.countplot(heart_disease_df['restecg'], palette = 'mako', order = [1, 0, 2], hue = heart_disease_df['sex'].replace({0: 'F', 1: 'M'}))

plt.xlabel('Resting electrocardiographic results')

plt.ylabel('Count')

plt.legend(title = 'Sex', loc = 1, frameon = False)

plt.title('Distribution of resting electrocardiographic results')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[2, 0])

ax = sns.countplot(heart_disease_df['exang'].replace({1: 'Yes', 0: 'No'}), palette = 'mako', hue = heart_disease_df['sex'].replace({0: 'F', 1: 'M'}))

plt.xlabel('Exercise induced angina')

plt.ylabel('Count')

plt.legend(title = 'Sex', loc = 1, frameon = False)

plt.title('Distribution of exercise induced angina')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[2, 1])

ax = sns.countplot(heart_disease_df['slope'], palette = 'mako', order = [2, 1, 0], hue = heart_disease_df['sex'].replace({0: 'F', 1: 'M'}))

plt.xlabel('The slope of the peak exercise ST segment')

plt.ylabel('Count')

plt.legend(title = 'Sex', loc = 1, frameon = False)

plt.title('Distribution of slope of the peak exercise ST segment')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[3, 0])

ax = sns.countplot(heart_disease_df['ca'], palette = 'mako', hue = heart_disease_df['sex'].replace({0: 'F', 1: 'M'}))

plt.xlabel('CA')

plt.ylabel('Count')

plt.title('Distribution ca')

plt.legend(title = 'Sex', loc = 1, frameon = False)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[3, 1])

ax = sns.countplot(heart_disease_df['thal'], palette = 'mako', order = [2, 3, 1, 0], hue = heart_disease_df['sex'].replace({0: 'F', 1: 'M'}))

plt.xlabel('Thal')

plt.ylabel('Count')

plt.legend(title = 'Sex', loc = 1, frameon = False)

plt.title('Distribution of thal')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
plt.figure(figsize = (20, 40))

plt.suptitle('Sex exploration (quantitative features)', y = 0.90, fontsize = 16)

gs = gridspec.GridSpec(5, 2)



plt.subplot(gs[0, 0])

ax = sns.boxplot(heart_disease_df['sex'].replace({0:'F', 1:'M'}), heart_disease_df['age'], palette = 'mako')

plt.xlabel('Sex')

plt.ylabel('Age')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[0, 1])

ax = sns.boxplot(heart_disease_df['sex'].replace({0:'F', 1:'M'}), heart_disease_df['chol'], palette = 'mako')

plt.xlabel('Sex')

plt.ylabel('Serum cholestoral in mg/dl')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 0])

ax = sns.boxplot(heart_disease_df['sex'].replace({0:'F', 1:'M'}), heart_disease_df['thalach'], palette = 'mako')

plt.xlabel('Sex')

plt.ylabel('Maximum heart rate achieved')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 1])

ax = sns.boxplot(heart_disease_df['sex'].replace({0:'F', 1:'M'}), heart_disease_df['oldpeak'], palette = 'mako')

plt.xlabel('Sex')

plt.ylabel('ST depression induced by exercise relative to rest')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
def chi2_test(df, features_list, feature_name_2):

    result = []

    for feature in features_list:

        if feature == feature_name_2:

            continue

        else:

            chi_2_test_array = []

            array = np.array(pd.crosstab(df[feature], df[feature_name_2]))

            lenght = array.shape[0]

            for i in range(lenght):

                chi_2_test_array.append(array[i, :])

            chi_2, p_val, dof , ex = stats.chi2_contingency(chi_2_test_array, lambda_="log-likelihood")    

            row = [feature + ' - ' + feature_name_2,'{0:.3f}'.format(chi_2), dof, '{0:.3f}'.format(p_val)]

            result.append(row)

    chi_2_test_df = pd.DataFrame(result, columns = ['Features', 'chi_2', 'dof', 'p_val'])

    chi_2_test_df['p_val'] = chi_2_test_df['p_val'].astype('float64')

    return chi_2_test_df
def color_p_val_red(val):

    if val > 0.01:

         color = 'red'

    else:

        color = 'black'

    return 'color: %s' % color
def color_negative_pos_red(val):

    if val < 0:

         color = 'red'

    else:

        color = 'green'

    return 'color: %s' % color
features_list = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

chi2_test_target_df = chi2_test(heart_disease_df, features_list, 'target')

chi2_test_target_df.style.applymap(color_p_val_red,  subset = ['p_val'])
chi2_test_sex_df = chi2_test(heart_disease_df, features_list, 'sex')

chi2_test_sex_df.style.applymap(color_p_val_red,  subset = ['p_val'])
chi2_test_cp_df = chi2_test(heart_disease_df, features_list, 'cp')

chi2_test_cp_df.style.applymap(color_p_val_red,  subset = ['p_val'])
chi2_test_fbs_df = chi2_test(heart_disease_df, features_list, 'fbs')

chi2_test_fbs_df.style.applymap(color_p_val_red,  subset = ['p_val'])
chi2_test_fbs_df = chi2_test(heart_disease_df, features_list, 'restecg')

chi2_test_fbs_df.style.applymap(color_p_val_red,  subset = ['p_val'])
chi2_test_fbs_df = chi2_test(heart_disease_df, features_list, 'thal')

chi2_test_fbs_df.style.applymap(color_p_val_red,  subset = ['p_val'])
heart_disease_df_0 = heart_disease_df[heart_disease_df['target'] == 0]

heart_disease_df_1 = heart_disease_df[heart_disease_df['target'] == 1]

heart_disease_df_0.drop('target', axis = 1, inplace = True)

heart_disease_df_1.drop('target', axis = 1, inplace = True)
drop_list = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

heart_disease_df_0.drop(drop_list, axis = 1, inplace = True)

heart_disease_df_1.drop(drop_list, axis = 1, inplace = True)
result = []

for column in heart_disease_df_0.columns:

    ttest, p_val = stats.ttest_ind(heart_disease_df_0[column], heart_disease_df_1[column])

    row = [column, ttest, '{0:.3f}'.format(p_val)]

    result.append(row)

t_test_df = pd.DataFrame(result, columns = ['Features', 'ttest', 'p_val'])

t_test_df['p_val'] = t_test_df['p_val'].astype('float64') 

t_test_df.style.applymap(color_p_val_red,  subset = ['p_val'])
plt.figure(figsize = (20,20))

plt.suptitle('Confidence intervals', y = 0.90, fontsize = 16)



plt.subplot(321)

ax = sns.pointplot(x = heart_disease_df['target'].replace({0:'No', 1:'Yes'}), y = heart_disease_df['age'], join= False, capsize= 0.1, palette= 'mako')

plt.xlabel('Heart disease')

plt.ylabel('Age')

plt.title('p_val: 0')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(322)

ax = sns.pointplot(x = heart_disease_df['target'].replace({0:'No', 1:'Yes'}), y = heart_disease_df['trestbps'], join= False, capsize= 0.1, palette= 'mako')

plt.xlabel('Heart disease')

plt.ylabel('Resting blood pressure')

plt.title('p_val: 0.012')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(323)

ax = sns.pointplot(x = heart_disease_df['target'].replace({0:'No', 1:'Yes'}), y = heart_disease_df['chol'], join= False, capsize= 0.1, palette= 'mako')

plt.xlabel('Heart disease')

plt.ylabel('Serum cholestoral in mg/dl')

plt.title('p_val: 0.139')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(324)

ax = sns.pointplot(x = heart_disease_df['target'].replace({0:'No', 1:'Yes'}), y = heart_disease_df['thalach'], join= False, capsize= 0.1, palette= 'mako')

plt.xlabel('Heart disease')

plt.ylabel('Maximum heart rate achieved')

plt.title('p_val: 0')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(325)

ax = sns.pointplot(x = heart_disease_df['target'].replace({0:'No', 1:'Yes'}), y = heart_disease_df['oldpeak'], join= False, capsize= 0.1, palette= 'mako')

plt.xlabel('Heart disease')

plt.ylabel('ST depression induced by exercise relative to rest')

plt.title('p_val: 0')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
plt.figure(figsize = (20, 30))

plt.suptitle('Quantitative features exploration', y = 0.90, fontsize = 16)

gs = gridspec.GridSpec(5, 4)



plt.subplot(gs[0, 0])

ax = sns.distplot(heart_disease_df['age'], color =  "#3498db")

plt.xlabel('Age')

plt.title('Age distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[0, 1])

ax = sns.distplot(heart_disease_df['trestbps'], color = "#3498db")

plt.xlabel('Resting blood pressure')

plt.title('Resting blood pressure distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 0])

ax = sns.distplot(heart_disease_df['chol'], color = "#3498db")

plt.xlabel('Serum cholestoral in mg/dl')

plt.title('Serum cholestoral in mg/dl distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 1])

ax = sns.distplot(heart_disease_df['thalach'], color = "#3498db")

plt.xlabel('Maximum heart rate achieved')

plt.title('Maximum heart rate achieved distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)





plt.subplot(gs[0, 2])

ax = sns.distplot(np.log(heart_disease_df['age']), color =  "#e74c3c")

plt.xlabel('Age')

plt.title('Age distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[0, 3])

ax = sns.distplot(np.log(heart_disease_df['trestbps']), color = "#e74c3c")

plt.xlabel('Resting blood pressure')

plt.title('Resting blood pressure distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 2])

ax = sns.distplot(np.log(heart_disease_df['chol']), color = "#e74c3c")

plt.xlabel('Serum cholestoral in mg/dl')

plt.title('Serum cholestoral in mg/dl distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



plt.subplot(gs[1, 3])

ax = sns.distplot(np.log(heart_disease_df['thalach']), color = "#e74c3c")

plt.xlabel('Maximum heart rate achieved')

plt.title('Maximum heart rate achieved distribution')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
heart_disease_df['age'] = np.log(heart_disease_df['age'])

heart_disease_df['trestbps'] = np.log(heart_disease_df['trestbps'])

heart_disease_df['chol'] = np.log(heart_disease_df['chol'])

heart_disease_df['thalach'] = np.log(heart_disease_df['thalach'])
features_list = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

for feature in features_list:

    heart_disease_df[feature] = heart_disease_df[feature].astype('object')    
drop_list = ['sex', 'fbs', 'trestbps', 'chol']

heart_disease_df.drop(drop_list, axis = 1, inplace = True)
heart_disease_df = pd.get_dummies(heart_disease_df, prefix = ['cp_', 'restecg_', 'exang_', 'slope_', 'ca_', 'thal_'])
X = heart_disease_df.drop('target', axis = 1)

Y = heart_disease_df['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 17)
d_tree = DecisionTreeClassifier(max_depth = 10, random_state = 17) 
d_tree.fit(X_train, Y_train)
Y_predict = d_tree.predict(X_test)
print(classification_report(Y_test, Y_predict))
data = {'Test_values' : np.array(Y_test),'Predicted_values' : Y_predict}

cm = pd.DataFrame(data, columns = ['Test_values', 'Predicted_values'])
plt.figure(figsize = (15, 10))

sns.heatmap(pd.crosstab(cm['Test_values'], cm['Predicted_values'], rownames=['Actual'], colnames=['Predicted']), annot=True, cmap = 'mako')
plt.figure(figsize = (15, 10))

ax = sns.barplot(d_tree.feature_importances_, X_test.columns, palette = 'mako_r')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
r_forest = RandomForestClassifier(max_depth = 10, min_samples_split = 10, max_features = 5, n_estimators = 1000, n_jobs = -1)
r_forest.fit(X_train, Y_train)
Y_predict = r_forest.predict(X_test)
print(classification_report(Y_test, Y_predict))
data = {'Test_values' : np.array(Y_test),'Predicted_values' : Y_predict}

cm = pd.DataFrame(data, columns = ['Test_values', 'Predicted_values'])
plt.figure(figsize = (15, 10))

sns.heatmap(pd.crosstab(cm['Test_values'], cm['Predicted_values'], rownames=['Actual'], colnames=['Predicted']), annot=True, cmap = 'mako')
plt.figure(figsize = (15, 10))

ax = sns.barplot(d_tree.feature_importances_, X_test.columns, palette = 'mako_r')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
knn = KNeighborsClassifier(n_neighbors = 2, n_jobs = -1, weights = 'distance', algorithm = 'brute')
knn.fit(X_train, Y_train)
Y_predict = knn.predict(X_test)
print(classification_report(Y_test, Y_predict))
data = {'Test_values' : np.array(Y_test),'Predicted_values' : Y_predict}

cm = pd.DataFrame(data, columns = ['Test_values', 'Predicted_values'])
plt.figure(figsize = (15, 10))

sns.heatmap(pd.crosstab(cm['Test_values'], cm['Predicted_values'], rownames=['Actual'], colnames=['Predicted']), annot=True, cmap = 'mako')
log_reg = LogisticRegressionCV(cv = 5, max_iter = 500, n_jobs = -1, refit = True, random_state = 17)
log_reg.fit(X_test, Y_test)
Y_predict = log_reg.predict(X_test)
print(classification_report(Y_test, Y_predict))
data = {'Test_values' : np.array(Y_test),'Predicted_values' : Y_predict}

cm = pd.DataFrame(data, columns = ['Test_values', 'Predicted_values'])
plt.figure(figsize = (15, 10))

sns.heatmap(pd.crosstab(cm['Test_values'], cm['Predicted_values'], rownames=['Actual'], colnames=['Predicted']), annot=True, cmap = 'mako')
features_list = ['age', 'thalach', 'oldpeak', 'cp__0', 'cp__1', 'cp__2', 'cp__3',

       'restecg__0', 'restecg__1', 'restecg__2', 'exang__0', 'exang__1',

       'slope__0', 'slope__1', 'slope__2', 'ca__0', 'ca__1', 'ca__2', 'ca__3',

       'ca__4', 'thal__0', 'thal__1', 'thal__2', 'thal__3']
result = []

i = 0

for feature in features_list:

    row = [feature, log_reg.coef_[0, i]]

    result.append(row)

    i = i + 1
log_reg_coef_df = pd.DataFrame(result, columns = ['feature', 'coef'])
log_reg_coef_df.style.applymap(color_negative_pos_red, subset = ['coef'])