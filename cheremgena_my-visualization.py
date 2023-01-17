import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from pylab import subplot

import scipy.stats as stats

from statsmodels.stats.weightstats import *

from statsmodels.stats.proportion import proportion_confint

from statsmodels.stats.weightstats import zconfint

import warnings

warnings.filterwarnings('ignore')
path = '../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv'

df = pd.read_csv(path)

df.head()
df.info()
sns.countplot(df.DEATH_EVENT)

plt.title('Classes in target feature')

plt.show()
df.describe()
print('Count uniqie values of features')

for col in df.columns:

    print(col, ':', df[col].nunique())
# function for analysis features

def research_bin(col):

    

    print(col.upper())

    n1 = len(df[df.DEATH_EVENT == 1])

    n0 = len(df[df.DEATH_EVENT == 0])

    

    prop_not_death = proportion_confint(df[df.DEATH_EVENT == 0][col].sum(), n0, method = 'wilson')

    prop_death = proportion_confint(df[df.DEATH_EVENT == 1][col].sum(), n1, method = 'wilson')

    print('95% confidence interval for a anaemia probability in "death" class:', prop_death)

    print('95% confidence interval for a anaemia probability in "alive" class:', prop_not_death)  

    

    z = stats.norm.cdf(1-0.05/2)

    p1 = df[df.DEATH_EVENT == 1][col].sum()/len(df[df.DEATH_EVENT == 1])

    p0 = df[df.DEATH_EVENT == 0][col].sum()/len(df[df.DEATH_EVENT == 0])



    left_bound = p0 - p1 - np.sqrt(p0*(1-p0)/n0 + p1*(1 - p1)/n1)

    right_bound = p0 - p1 + np.sqrt(p0*(1-p0)/n0 + p1*(1 - p1)/n1)

    print("95% confidence interval for a difference between proportions:", [round(left_bound, 5), round(right_bound, 5)])

    P = float(p1*n1 + p0*n0) / (n1 + n0)

    z_stat = (p0 - p1) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n0))

    print("p-value of h0 (proportions are equal): ", round(2 * (1 - stats.norm.cdf(abs(z_stat))), 5))

    

    print('Criterion of chi2 (independence with target feature): p-value = {0}, chi2-statistic = {1}'.format(round(stats.chi2_contingency(pd.crosstab(df[col], 

                                                                                                             df["DEATH_EVENT"]))[1], 5),

                                                                         round(stats.chi2_contingency(pd.crosstab(df[col], 

                                                                                                             df["DEATH_EVENT"]))[0], 5)))

    print('Correlation of Matthew(with target feature):', round(np.sqrt(stats.chi2_contingency(pd.crosstab(df[col], df["DEATH_EVENT"]))[0])/2, 5))
bin_cols = ['anaemia', 'sex', 'smoking', 'diabetes', 'high_blood_pressure']

for col in bin_cols:

    sns.countplot(df.DEATH_EVENT, hue = df[col])

    plt.title(col.upper())

    plt.show()

    research_bin(col)
def analisys_num(col):

    

    plt.figure(figsize = (14, 5))

    subplot(1, 2, 1)

    sns.distplot(df[df.DEATH_EVENT == 1][col])

    sns.distplot(df[df.DEATH_EVENT == 0][col])



    subplot(1, 2, 2)

    sns.boxplot(y = df[col], x = df.DEATH_EVENT)

    plt.show()

    

    print('Statistic conclusion from data (%s):' % col.upper())

    print('95% confidence interval for the mean in "death" class:', zconfint(df[df.DEATH_EVENT == 1][col]))

    print('95% confidence interval for the mean in "alive" class:', zconfint(df[df.DEATH_EVENT == 0][col]))



    print("p-value of h0 (distributions are equal, criterion 'mannwhitneyu'): ", stats.mannwhitneyu(df[df.DEATH_EVENT == 1][col], 

                                                                          df[df.DEATH_EVENT == 0][col])[1])



    df['n'] = (df[col] - df[col].mean())/df[col].std()

    diff = df[df.DEATH_EVENT == 1]['n'].mean() - df[df.DEATH_EVENT == 0]['n'].mean()

    print('Correlation (difference between expectations of two class):', diff)

    
num_cols = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']

for col in num_cols:

    analisys_num(col)

    

del df['n']
# table of pair correlations

df[num_cols].corr()
plt.figure(figsize = (18, 4))

subplot(1, 3, 1)

sns.distplot(df.serum_sodium, color = 'red')



subplot(1, 3, 2)

sns.distplot(df.serum_creatinine, color = 'green')



subplot(1, 3, 3)

sns.regplot(df.serum_sodium, df.serum_creatinine)

plt.show()
from itertools import combinations

comb = combinations(bin_cols, 2)

for i in comb:

    print("Features:", i, ', p_value(hypotesys of independence) = ', round(stats.chi2_contingency(pd.crosstab(df[i[0]], 

                                                                                                             df[i[1]]))[1], 5))

# criterion chi2
plt.figure(figsize = (18, 10))

subplot(2, 3, 1)

sns.countplot(x = df.sex, hue = df.smoking)



subplot(2, 3, 2)

sns.countplot(x = df.sex, hue = df.diabetes)



subplot(2, 3, 3)

sns.countplot(x = df.diabetes, hue = df.smoking)



subplot(2, 3, 4)

sns.countplot(x = df.anaemia, hue = df.smoking)



subplot(2, 3, 5)

sns.countplot(x = df.sex, hue = df.high_blood_pressure)



subplot(2, 3, 6)

sns.countplot(x = df.sex, hue = df.anaemia)



plt.show()
for bin_col in bin_cols:

    print(bin_col.upper())

    i = 0

    for col in num_cols:

        i += 1

        df['n'] = (df[col] - df[col].mean())/df[col].std()

        diff = df[df[bin_col] == 1]['n'].mean() - df[df[bin_col] == 0]['n'].mean()

        print('| {0} - {1} |'.format(col, bin_col), round(diff, 3)) 

    print('\n')

# correlations in pairs(binary - numerical)
plt.figure(figsize = (12, 10))

subplot(2, 2, 1)

sns.violinplot(y = df.time, x = df.high_blood_pressure)



subplot(2, 2, 2)

sns.violinplot(y = df.platelets, x = df.sex)



subplot(2, 2, 3)

sns.violinplot(y = df.time, x = df.anaemia)



subplot(2, 2, 4)

sns.violinplot(y = df.creatinine_phosphokinase, x = df.anaemia)

plt.tight_layout()

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from  xgboost import XGBClassifier 

from sklearn.linear_model import LogisticRegression
X = df.drop('DEATH_EVENT', axis = 1)

y = df['DEATH_EVENT']



train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2) 



def model(cls):

    cls.fit(train_x, train_y)

    y_pred = cls.predict(test_x)

    print('Accuracy-score: {0}, F1-score: {1}, ROC-AUC-score: {2}'.format(round(accuracy_score(y_pred, test_y), 5), 

                                                                       round(f1_score(y_pred, test_y), 5), 

                                                                          round(roc_auc_score(y_pred, test_y), 5)))

    print('Confusion matrix:')

    sns.heatmap(confusion_matrix(y_pred, test_y), annot = True, cmap = plt.cm.Blues)

    plt.show()
logreg = LogisticRegression()

print('Logistic Regression')

model(logreg)
forcls = RandomForestClassifier()

print('Random Forest')

model(forcls)
xgbcls = XGBClassifier(booster = 'gbtree')

print('Gradient booster classifier')

model(xgbcls)