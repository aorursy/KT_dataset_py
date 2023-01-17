!pip install --upgrade pandas

import pandas as pd

import numpy as np

from scipy import stats

import statsmodels.api as sm

import matplotlib.pyplot as plt
heart = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

heart.head()
# How many records are there

len(heart.index)
# Check for the existence of any missing values

heart.count()
cont = ['age','trestbps','chol','thalach','oldpeak']

cat = [x for x in list(heart.columns) if x not in cont]

cat.remove('target')
for val in cat:

    heart[val] = heart[val].astype('category')
heart_gp = heart.groupby('target')

stats_cont = heart_gp[cont].agg([np.mean, np.std]).stack().T

stats_cont
#Split the dataframe into diseased and not diseased

disease = (heart[heart['target'] == 1]).reset_index()

no_disease = (heart[heart['target'] == 0]).reset_index()



results = {}

for val in cont:

    results[val] = stats.levene(disease[val], no_disease[val])

pd.DataFrame.from_dict(results, orient='Index', columns=['statistic','p-value'])
diff = disease['age'] - no_disease['age']

sm.qqplot(diff, line='q')

plt.title('Age Q-Q Plot') 

plt.show()
shapiro = heart_gp.agg(lambda x:stats.shapiro(x)[1]).T

shapiro.loc[cont] < 0.05
fig = plt.figure()



plt.subplot(2, 2, 1)

disease['age'].plot(kind='hist', title='Age - Diseased Group')



plt.subplot(2, 2, 2)

no_disease['age'].plot(kind='hist', title='Age - Non-Diseased Group')
plt.subplot(2, 2, 1)

disease['trestbps'].plot(kind='hist', title='trestbps - Diseased Group')



plt.subplot(2, 2, 2)

no_disease['trestbps'].plot(kind='hist', title='trestbps - Non-Diseased Group')
plt.subplot(2, 2, 1)

disease['chol'].plot(kind='hist', title='chol - Diseased Group')



plt.subplot(2, 2, 2)

no_disease['chol'].plot(kind='hist', title='chol - Non-Diseased Group')
plt.subplot(2, 2, 1)

disease['thalach'].plot(kind='hist', title='thalach - Diseased Group')



plt.subplot(2, 2, 2)

no_disease['thalach'].plot(kind='hist', title='thalach - Non-Diseased Group')
plt.subplot(2, 2, 1)

disease['oldpeak'].plot(kind='hist', title='oldpeak - Diseased Group')



plt.subplot(2, 2, 2)

no_disease['oldpeak'].plot(kind='hist', title='oldpeak - Non-Diseased Group')
results = {}

for val in list(stats_cont.index):

    results[val] = stats.ttest_ind(disease[val],no_disease[val],equal_var=False)

test = pd.DataFrame.from_dict(results, orient='Index', columns=['statistic', 'p-value'])

test = test.drop('oldpeak', axis=0)

test
mannwhit = stats.mannwhitneyu(disease['oldpeak'],no_disease['oldpeak'])

test = test.append(pd.Series({'statistic':mannwhit[0],'p-value':mannwhit[1]},name='oldpeak'))

test  
pd.concat([stats_cont,test['p-value']],axis=1)
heart_gp.sex.value_counts().unstack(0)
heart_gp.cp.value_counts().unstack(0)
heart_gp.fbs.value_counts().unstack(0)
heart_gp.restecg.value_counts().unstack(0)
heart_gp.exang.value_counts().unstack(0)
heart_gp.slope.value_counts().unstack(0)
heart_gp.ca.value_counts().unstack(0)
heart_gp.thal.value_counts().unstack(0)
results = {}

for val in cat:

    results[val] = stats.chi2_contingency(pd.crosstab(heart[val],heart.target))

test = pd.DataFrame.from_dict(results, orient='Index', columns=['statistic', 'p-value','df','expected'])

test.drop(['df','expected'],axis=1,inplace=True)

test.drop(['restecg','ca','thal'], inplace=True)

test
# Run the next line if rpy2 is not already installed

!conda install -c r rpy2 --yes 
import rpy2.robjects.numpy2ri

from rpy2.robjects.packages import importr



rpy2.robjects.numpy2ri.activate()

statsr = importr('stats')
statsr.fisher_test(pd.crosstab(heart.restecg,heart.target).to_numpy())
result = statsr.fisher_test(pd.crosstab(heart.ca,heart.target).to_numpy())

print('p-value: ',result[0])
result = statsr.fisher_test(pd.crosstab(heart.thal,heart.target).to_numpy())

print('p-value: ',result[0])
heart_dum = pd.get_dummies(heart, drop_first=True)

heart_dum.head()
crude_calc = heart_dum.loc[:,['restecg_1','restecg_2','target']]

crude_calc.head()
y = crude_calc['target']

ind = crude_calc[['restecg_1','restecg_2']]

X = sm.add_constant(ind)

logit_model = sm.Logit(y,X)

result = logit_model.fit()

result.summary2()
np.exp(result.params[1])
se = 0.2359

lower = result.params[1] - 1.96*se

upper = result.params[1] + 1.96*se

print("95% CI: (",np.exp(lower),",",np.exp(upper),")")
def backward_elimination(df, dep, ind, alpha, keep):

    y = dep

    x1 = list(ind.columns)

    while len(x1) > 1:

        X = sm.add_constant(df[x1])

        model = sm.Logit(y,X)

        result = model.fit()

        pvalues = pd.Series(result.pvalues.values[1:], index=x1)

        idmax = pvalues.idxmax()

        if idmax in keep:

            pvalues.sort_values(ascending=False, inplace=True)

            pvalues.drop(keep, inplace=True)

            idmax = pvalues.index[0]

        pmax = pvalues.loc[idmax]

        if pmax > alpha:

            x1.remove(idmax)

        else:

            return x1
dep = heart_dum['target']

ind = heart_dum[heart_dum.columns.difference(['target'])]

predictors = backward_elimination(heart_dum, dep, ind, 0.05, ['restecg_1', 'restecg_2'])

predictors
predictors.extend(['ca_4','cp_1','slope_2','thal_3'])

predictors.sort()
x1 = heart_dum[predictors]

X = sm.add_constant(x1)

model = sm.Logit(y,X)

result = model.fit()

result.summary2()
print('Adjusted Odds Ration: ', np.exp(result.params['restecg_1']))



# Confidence Interval

se = 0.3828

lower = result.params['restecg_1'] - 1.96*se

upper = result.params['restecg_1'] + 1.96*se

print('95% CI: (',np.exp(lower),',',np.exp(upper),')')