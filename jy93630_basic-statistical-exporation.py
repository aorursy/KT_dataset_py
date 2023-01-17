import pandas as pd

import numpy as np

from sklearn.cross_validation import train_test_split

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import seaborn

from scipy import stats

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/train.csv")



print('Train shape:', train.shape)

print('Test shape:', test.shape)

print('Columns:', train.columns)



train.head()
train_target1 = train[train['target'] == 1]

train_target0 = train[train['target'] == 0]



print(train_target1.shape)

print(train_target0.shape)

print("[Target = 1] Proportion : ", float(train_target1.shape[0]) / train_target0.shape[0])
train_cat = []

train_bin = []

train_float = []

for eachCol in train.columns:

    astr = eachCol[(len(eachCol)-3):len(eachCol)]

    if (astr == 'cat'):

        train_cat.append(eachCol)

    elif (astr == 'bin'):

        train_bin.append(eachCol)

    else:

        train_float.append(eachCol)



print('There are {} categorical variables'.format(len(train_cat)))

print(train_cat)

print('There are {} binary variables'.format(len(train_bin)))

print(train_bin)

print('There are {} other variables'.format(len(train_float)))

print(train_float)
def frequency_plot(df, var, title) : 

    category = np.unique(df[var]).tolist()

    frequency = df[var].value_counts()

    frequency = frequency[np.sort(frequency.index)]

    

    pos = np.arange(len(category))

    width = 1.0  

    prob_frequency = frequency / np.sum(frequency)



    ax = plt.axes()

    ax.set_xticks(pos)

    ax.set_xticklabels(category)



    plt.bar(pos, prob_frequency, width, color='r', alpha=0.7)

    plt.title(title)

    plt.show()

    

frequency_plot(train, 'ps_ind_01', title="ps_ind_01")
def frequency_overlap_plot(df, var, title) : 

    a = np.unique(df[df['target']== 0][var]).tolist()

    b =  np.unique(df[df['target'] == 1][var]).tolist()

    

    frequency1 = df[df['target']== 0][var].value_counts()

    frequency2 = df[df['target'] == 1][var].value_counts()

    

    if len(a) > len(b) : 

        no_frequency_index = list(set(a) - set(b))

        frequency2.set_value(no_frequency_index[0], 0)

    elif len(b) > len(a) : 

        no_frequency_index = list(set(b) - set(a))

        frequency1.set_value(no_frequency_index[0], 0)



    frequency1 = frequency1[np.sort(frequency1.index)]   

    frequency2 = frequency2[np.sort(frequency2.index)]

    

    ind = np.arange(len(a))

    width = 0.25



    fig = plt.figure()

    ax = fig.add_subplot(111)

    

    ax.bar(ind+width, frequency1 / np.sum(frequency1), width, color='r', alpha=0.7)

    ax.bar(ind+width+0.35, frequency2 / np.sum(frequency2), width, color='g', alpha=0.7)



    ax.set_xticks(ind+width+(width/2))

    ax.set_xticklabels(a)



    ax.yaxis.set_ticks_position("left")



    plt.tight_layout()

    plt.legend("01")

    plt.title(title)

    plt.show()
# RED : target = 0, GREEN : target = 1

for column in train_cat : 

    frequency_overlap_plot(train, column, title=column)
def observed_expected(df, target, var) : 

    tab = pd.crosstab(train[target], train[var], margins=True)



    row = len(np.unique(train[target]))

    col = len(np.unique(train[var]))

    

    observed = tab.ix[0:row,0:col] 

    print("Observed : ")

    print(observed.values)

    

    expected =  np.outer(tab['All'][0:row], tab.ix['All'][0:col]) / len(train['ps_ind_01'])

    print("Expected : ")

    print(expected)

    

observed_expected(train, 'target', 'ps_ind_05_cat')
def chi_test(df, target, var) : 

    row = len(np.unique(df[target]))

    col = len(np.unique(df[var]))

    

    tab = pd.crosstab(df[target], df[var], margins=True)

    

    observed = tab.ix[0:row,0:col] 

    expected =  np.outer(tab['All'][0:row], tab.ix['All'][0:col]) / len(df[var])

    

    chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()

    crit = stats.chi2.ppf(q = 0.95, df = (row-1)*(col-1))   



    print("Variable Name : %s " % var)

    print("Critical value : %s " % crit)

    print("Chi-square : %s " %chi_squared_stat)



    # Find the p-value

    p_value = 1 - stats.chi2.cdf(x=chi_squared_stat, df = (row-1)*(col-1))

    print("P value : %s " %p_value)
for column in train_cat : 

    chi_test(train, 'target', column)
def get_chi_test_pvalue(df, target, var) : 

    row = len(np.unique(df[target]))

    col = len(np.unique(df[var]))

    

    tab = pd.crosstab(df[target], df[var], margins=True)

    

    observed = tab.ix[0:row,0:col] 

    expected =  np.outer(tab['All'][0:row], tab.ix['All'][0:col]) / len(df[var])

    

    chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()

    crit = stats.chi2.ppf(q = 0.95, df = (row-1)*(col-1))   



    # Find the p-value

    p_value = 1 - stats.chi2.cdf(x=chi_squared_stat, 

                                 df = (row-1)*(col-1))

    return var, p_value
for column in train_cat+train_bin : 

    var_name, pvalue = get_chi_test_pvalue(train, 'target', column)

    if pvalue > 0.05 : 

        print("%s p-value : %s" %(var_name, pvalue))
def continuous_overlap_plot(df, var, title) : 

    a = df[df['target'] == 0][var]

    b =  df[df['target'] == 1][var]

    

    plt.hist(a, normed=True, alpha=0.5, color="r", bins=20)

    plt.hist(b, normed=True, alpha=0.5, color="g", bins=20)



    plt.legend("01")

    plt.title(title)

    plt.show()
continuous_var = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14']

for column in continuous_var : 

    continuous_overlap_plot(train, column, title=column)
def t_test(df, var) : 

    a = train[train['target'] == 0][var]

    b = train[train['target'] == 1][var]



    print("Mean of variable %s " % var)

    print("[target = 0] : % s \n[target = 1] : %s "%(np.mean(a), np.mean(b)))

    print("two sample t-test p-value : %s " % stats.ttest_ind(a,b)[1]) 
for column in continuous_var : 

    t_test(train, column)
# get p-value of t-test

def get_ttest_pvalue(df, var) : 

    a = train[train['target'] == 0][var]

    b = train[train['target'] == 1][var]



    return var, stats.ttest_ind(a,b)[1]
for column in continuous_var : 

    var_name, pvalue = get_ttest_pvalue(train, column)

    if pvalue > 0.05 : 

        print("%s p-value : %s" %(var_name, pvalue))