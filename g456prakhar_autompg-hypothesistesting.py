import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import scipy.stats as stats



pd.options.display.max_rows = 1000

pd.options.display.max_columns = 20

path="../input/autompg-dataset/auto-mpg.csv"

data = pd.read_csv(path)

#print(data['car name'])

train_df=data.copy()



from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()

train_df["car name"]=label_encoder.fit_transform(train_df["car name"])

print(train_df["car name"])

print(data["car name"])

del data["car name"]

data["car name"]=train_df["car name"].copy()
#data[data['horsepower']=='?']

median = data[data['horsepower']!='?']['horsepower'].astype(int).median()

print(int(median))

#filling the missing values in the horsepower column



def Fill_Data(d):

    if d == '?':

        return int(median)

    else:

        return int(d)

data['horsepower'] = data['horsepower'].apply(Fill_Data)



#train_df['horsepower']=data['horsepower'].astype('float')

#data = data.replace('?', np.nan)

#data['horsepower'] = data['horsepower'].fillna(data['horsepower'].median())

train_df.horsepower=data.horsepower.astype('float')

print(train_df)

print('?' in train_df.horsepower)
train_df.skew()

#checking skewness
train_df.kurtosis()
train_df.dtypes
def scaled(col):

    res = (col - col.min(axis=0)) / (col.max(axis=0) - col.min(axis=0))

    return res

#implementing a function that performs MinMaxScaler
train_df["mpg"]=scaled(train_df["mpg"])
train_df["displacement"]=scaled(train_df["displacement"])

train_df["weight"]=scaled(train_df["weight"])

train_df["acceleration"]=scaled(train_df["acceleration"])

train_df["horsepower"]=scaled(train_df["horsepower"])

#getting the scaled values
target=train_df["mpg"]

del train_df["mpg"]

#setting the target value
train_df
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(train_df,target,random_state=0)
X_train
X_train.skew()
X_train.kurtosis()
y_train
y_train.skew()
y_train.kurtosis()
# Shapiro-Wilk, D’Agostino’s K^2 test,Anderson Test on dataframe

def normality_tests(df,args):

    """applies Shapiro-Wilk, D’Agostino’s K^2 test and check if it is true for any quantitative data or comes false, i.e. not

    a normal distribution for all data.

    Args:

    df: name of the dataset that takes train and test dataframe splitted according to library named scikit learn. 

    args: takes 2 variables, either "shapiro" or "normaltest".

    """

    if args=='shapiro':

        test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01

        normal = pd.DataFrame(df)

        normal = normal.apply(test_normality)

        return (not normal.any())

    elif args=='normaltest':

        test_normality = lambda x: stats.normaltest(x.fillna(0))[1] < 0.01

        normal = pd.DataFrame(df)

        normal = normal.apply(test_normality)

        return (not normal.any())

    

    
print(normality_tests(X_train,'shapiro'))

print(normality_tests(X_train,'normaltest'))
print(normality_tests(X_test,'shapiro'))

print(normality_tests(X_test,'normaltest'))
print(normality_tests(y_train,'shapiro'))

print(normality_tests(y_train,'normaltest'))
print(normality_tests(y_test,'shapiro'))

print(normality_tests(y_test,'normaltest'))
from scipy.stats import shapiro,normaltest,anderson,ttest_ind,t

import math

from math import sqrt
unique_cylinder=X_train.cylinders.unique()

unique_cylinder
X_train["cylinders"].std()
print(X_test['horsepower'][X_test['cylinders'] == 5])

print(X_test['horsepower'][X_test['cylinders'] == 5].std())
# selecting rows based on condition 

rslt_df = X_train.horsepower[X_train['cylinders'] == unique_cylinder[0]]

rslt_df.std()
def Independent_T_test_unique_variable_between_train_test_datasets(train_dataframe,test_dataframe,unique,args):

    """calculates independent t-values and p-values by doing a test on train and test datasets and then further calculates 

    margin of error between 95% mean confidence interval.

    Args:

    train_dataframe: name of the training dataset splitted according to library named scikit learn. 

    test_dataframe: name of the testing dataset splitted according to library named scikit learn.

    unique: name of a variable either 'origin' or 'cylinders' that is ordinal discrete distribution.

    args: name of other quantitative variables like 'acceleration','weight','displacement','horsepower'.

    """

    #unique_cylinder=X_train.cylinders.unique()

    unique_value=train_dataframe[unique].unique()

    

    for i in range(len(unique_value)):

            if type(unique_value[i])==bool:

                return

            print("i=",str(i))

            #print("For train and test of :-"+"horsepower")

            print("For train and test of :- "+args)

            print("For unique value of :- "+unique+" for unique_value["+str(i)+"] "+str(unique_value[i]))

            # Run independent t-test

            #ind_t_test =ttest_ind(X_train["horsepower"][X_train.cylinders == unique_cylinder[i]],X_test["horsepower"][X_test.cylinders == unique_cylinder[i]])

            ind_t_test =ttest_ind(X_train[args][X_train[unique] == unique_value[i]],X_test[args][X_test[unique] == unique_value[i]])



            # Calculate the mean difference and 95% confidence interval

            #N1 = len(X_train)

            #N2 = len(X_test)

            

            a = X_train[args][X_train[unique] == unique_value[i]]

            b = X_test[args][X_test[unique] == unique_value[i]]

            N1 = len(a)

            N2 = len(b)

            df = (N1 + N2 - 2)

            std1 = a.std()

            std2 = b.std()

            if(math.isnan(std1)):std1=0

            if(math.isnan(std2)):std2=0

            print("std1 => ",std1)

            print("std2 => ",std2)



            std_N1N2 = sqrt( ((N1 - 1)*(std1)**2 + (N2 - 1)*(std2)**2) / df)

            print("std_N1N2 => "+str(std_N1N2))



            diff_mean = a.mean() - b.mean()

            print("mean of train => "+str(a.mean()))

            print("mean of test => "+str(b.mean()))

            MoE = t.ppf(0.975, df) * std_N1N2 * sqrt(1/N1 + 1/N2)

            alpha=0.05

            print('The results of the independent t-test are: \n\tt-value = {}\n\tp-value = {}'.format(ind_t_test[0],ind_t_test[1]))

            if ind_t_test[1]>alpha:

                print("Fail to reject null hypothesis, i.e, normally distributed and population means are equal")

            else:

                print("reject the null hypothesis and accept the alternative hypothesis, which is that the population means are not equal")

            print ('\nThe difference between groups is {:3.6f} [{:3.6f} to {:3.6f}] (mean [95% CI])'.format(diff_mean, diff_mean - MoE, diff_mean + MoE))
#Independent_T_test_unique_variable_between_train_test_datasets(train_dataframe,test_dataframe,unique,args)

values=['horsepower','weight','acceleration','displacement']

for v in values:

    Independent_T_test_unique_variable_between_train_test_datasets(X_train,X_test,'origin',v)
values=['horsepower','weight','acceleration','displacement']

for v in values:

    Independent_T_test_unique_variable_between_train_test_datasets(X_train,X_test,'cylinders',v)
origin_one_train_df=X_train[:][X_train.origin == 1]

#origin_one_train_df
origin_two_train_df=X_train[:][X_train.origin == 2]

#origin_two_train_df
origin_three_train_df=X_train[:][X_train.origin == 3]

#origin_three_train_df
origin_one_test_df=X_test[:][X_test.origin == 1]

#origin_one_test_df
origin_two_test_df=X_test[:][X_test.origin == 2]

#origin_two_test_df
origin_three_test_df=X_test[:][X_test.origin == 3]

#origin_three_test_df
origin_one_data_df=data[:][data.origin == 1]

#origin_one_data_df
origin_two_data_df=data[:][data.origin == 2]

#origin_two_data_df
origin_three_data_df=data[:][data.origin == 3]

#origin_three_data_df
def print_statistics_of_normality_test(df,args):

    """prints statistics of various 3-normality checks and does hypothesis testing.

    Args:

    df: name of any kind of datasets that consists of quantitative variables i.e. multiple dataframes.

    args: takes 3 variables, either "shapiro" or "normaltest" or "anderson".

    """

    if args=='shapiro':

        quantitative_data = [f for f in df.columns]

        

        for q in (quantitative_data):

            data_check=data[q]

            print(q)

            stat, p = shapiro(data_check)

            print('Statistics=%.3f, p=%.3f' % (stat, p))

            # interpret

            alpha = 0.05

            if p > alpha:

                print('Sample looks Gaussian (fail to reject H0)')

            else:

                print('Sample does not look Gaussian (reject H0)')

            print("-"*20)

    elif args=='normaltest':

        quantitative_data = [f for f in df.columns]

        

        for q in (quantitative_data):

            data_check=data[q]

            print(q)

            stat, p = normaltest(data_check)

            print('Statistics=%.3f, p=%.3f' % (stat, p))

            # interpret

            alpha = 0.05

            if p > alpha:

                print('Sample looks Gaussian (fail to reject H0)')

            else:

                print('Sample does not look Gaussian (reject H0)')

            print("-"*20)

    elif args=='anderson':

        quantitative_data = [f for f in df.columns]

        

        for q in (quantitative_data):

            data_check=data[q]

            print(q)

            result = anderson(data_check)

            print('Statistic: %.3f' % result.statistic)

            p = 0

            for i in range(len(result.critical_values)):

                sl, cv = result.significance_level[i], result.critical_values[i]

                if result.statistic < result.critical_values[i]:

                    print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))

                else:

                    print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

            print("-"*20)



        
tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test(origin_one_data_df,test)
tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test(origin_two_data_df,test)
tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test(origin_three_data_df,test)
tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test(origin_one_train_df,test)
tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test(origin_two_train_df,test)
tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test(origin_three_train_df,test)
tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test(origin_one_test_df,test)
tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test(origin_two_test_df,test)
tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test(origin_three_test_df,test)
tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test(X_train,test)

tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test(X_test,test)
def print_statistics_of_normality_test_for_single_dataframe(single_df,args):

    """prints statistics of various 3-normality checks and does hypothesis testing.

    Args:

    single_df: name of any kind of datasets that consists of single column in the dataframe. 

    args: takes 3 variables, either "shapiro" or "normaltest" or "anderson".

    """

    if args=='shapiro':

        data_check=single_df

        #print(single_df.keys())

        stat, p = shapiro(data_check)

        print('Statistics=%.3f, p=%.3f' % (stat, p))

        # interpret

        alpha = 0.05

        if p > alpha:

            print('Sample looks Gaussian (fail to reject H0)')

        else:

            print('Sample does not look Gaussian (reject H0)')

        print("-"*20)

    elif args=='normaltest':

        data_check=single_df

        #print(single_df.keys())

        stat, p = normaltest(data_check)

        print('Statistics=%.3f, p=%.3f' % (stat, p))

        # interpret

        alpha = 0.05

        if p > alpha:

            print('Sample looks Gaussian (fail to reject H0)')

        else:

            print('Sample does not look Gaussian (reject H0)')

        print("-"*20)

    elif args=='anderson':

        data_check=single_df

        result = anderson(data_check)

        print('Statistic: %.3f' % result.statistic)

        p = 0

        for i in range(len(result.critical_values)):

            sl, cv = result.significance_level[i], result.critical_values[i]

            if result.statistic < result.critical_values[i]:

                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))

            else:

                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

        print("-"*20)

        

    
print("For mpg:-")

tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test_for_single_dataframe(y_train,test)
print("For mpg:-")

tests=['shapiro','normaltest','anderson']

for test in tests:

    print("*"*10+test+"*"*10)

    print_statistics_of_normality_test_for_single_dataframe(y_test,test)
def print_levenes_test(train_dataframe,test_dataframe):

    """prints statistics and p-value of various levene's test according to mean, median and trimmed value.

    Args:

    train_dataframe: name of the training dataset splitted according to library named scikit learn. 

    test_dataframe: name of the testing dataset splitted according to library named scikit learn.

    """

    quantitative_data = [f for f in train_dataframe.columns]

    for q in quantitative_data:

        stat_a,p_a=stats.levene(train_dataframe[q],test_dataframe[q],center='median')

        stat_b,p_b=stats.levene(train_dataframe[q],test_dataframe[q],center='mean')

        stat_c,p_c=stats.levene(train_dataframe[q],test_dataframe[q],center='trimmed')

        alpha=0.05

        print(q)

        print("For median : ")

        print(stat_a,p_a)

        if p_a > alpha:

            print('Samples have significantly different variance (fail to reject H0)')

        else:

            print('Samples does not have significantly different variance (reject H0)')

        print("For mean : ")

        print(stat_b,p_b)

        if p_b > alpha:

            print('Samples have significantly different variance (fail to reject H0)')

        else:

            print('Samples does not have significantly different variance (reject H0)')

        print("For trimmed : ")

        print(stat_c,p_c)

        if p_c > alpha:

            print('Samples have significantly different variance (fail to reject H0)')

        else:

            print('Samples does not have significantly different variance (reject H0)')

        

        print("-"*20)
print_levenes_test(X_train,X_test)
from statsmodels.stats import weightstats as stests
def print_z_tests(train_dataframe,test_dataframe):

    """prints statistics and p-value of Z-test between train and test dataset.

    Args:

    train_dataframe: name of the training dataset splitted according to library named scikit learn. 

    test_dataframe: name of the testing dataset splitted according to library named scikit learn.

    """

    quantitative_data = [f for f in train_dataframe.columns]

    for q in quantitative_data:

        print(q)

        ztest ,pval = stests.ztest(train_dataframe[q], x2=test_dataframe[q], value=0,alternative='two-sided')

        print(float(pval))

        if pval<0.05:

            print("H1 : mean of two group is not 0 and reject null hypothesis")

        else:

            print("H0 : mean of two group is 0 and accept null hypothesis")

        print("-"*20)
print_z_tests(X_train,X_test)
from scipy.stats import chisquare,chi2

from scipy import stats

def print_chi_squared_test(dataset,column_name_1,column_name_2,alpha=0.05):

    """prints statistics and p-value of chi-squared test between dataset.

    Args:

    dataset: name of the dataset consists of mutiple quantitative columns.

    column_name_1:name of a variable either 'origin' or 'cylinders' that is ordinal discrete distribution.

    column_name_2:name of a variable either 'origin' or 'cylinders' that is ordinal discrete distribution.

    """

    

    #quantitative_data = [f for f in dataframe.columns]

    #for q in quantitative_data:

    #    stats,p=chisquare(dataframe[q])

    #    print(q)

    #    print("stats,p => ",stats,p)

    #    print("*"*20)

    

    #dataset_table=pd.crosstab(dataset['sex'],dataset['smoker'])

    

    dataset_table=pd.crosstab(dataset[column_name_1],dataset[column_name_2])

    print("dataset_table =>\n",dataset_table)

    Observed_Values = dataset_table.values 

    print("Observed Values =>\n",Observed_Values)

    val=stats.chi2_contingency(dataset_table)

    Expected_Values=val[3]

    print("Expected Values =>\n",Expected_Values)

    no_of_rows=dataset_table.shape[0]

    no_of_columns=dataset_table.shape[1]

    ddof=(no_of_rows-1)*(no_of_columns-1)

    print("Degree of Freedom:-",ddof)

    chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])

    chi_square_statistic=sum(chi_square)

    print("chi-square statistic:-",chi_square_statistic)

    critical_value=chi2.ppf(q=1-alpha,df=ddof)

    print('critical_value:',critical_value)

    #p-value

    p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)

    print('p-value:',p_value)

    print('Significance level: ',alpha)

    print('Degree of Freedom: ',ddof)

    print('p-value:',p_value)

    if chi_square_statistic>=critical_value:

        print("Reject H0,There is a relationship between 2 categorical variables")

    else:

        print("Retain H0,There is no relationship between 2 categorical variables")

    if p_value<=alpha:

        print("Reject H0,There is a relationship between 2 categorical variables")

    else:

        print("Retain H0,There is no relationship between 2 categorical variables")
print_chi_squared_test(X_train,'origin','cylinders')
print_chi_squared_test(X_test,'origin','cylinders')
X_train['cylinders']
quantitative = [f for f in data.columns if data.dtypes[f] != 'object']

print(quantitative)



qualitative = [f for f in data.columns if data.dtypes[f] == 'object']

print(qualitative)
for q in quantitative:

    print(q)

    print(data[quantitative].corr()[q][:])

    print("-"*20)
# Shapiro-Wilk Test

test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01

normal = pd.DataFrame(data[quantitative])

normal = normal.apply(test_normality)

print(not normal.any())
data.info()
missing = data.isnull().sum()

print(missing)

missing = missing[missing > 0]

missing.sort_values(inplace=True)

data['horsepower'][32]

print(type(data['horsepower'][32]))
data['horsepower']
data.head()
# Shapiro-Wilk Test

test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01

normal = pd.DataFrame(data[quantitative])

normal = normal.apply(test_normality)

print(not normal.any())
test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.05

normal = pd.DataFrame(data[quantitative])

normal = normal.apply(test_normality)

print(not normal.any())
# D'Agostino and Pearson's Test

test_normality = lambda x: stats.normaltest(x.fillna(0))[1] < 0.05

normal = pd.DataFrame(data[quantitative])

normal = normal.apply(test_normality)

print(not normal.any())
# Shapiro-Wilk Test

for q in (quantitative):

    data_check=data[q]

    print(q)

    stat, p = shapiro(data_check)

    print('Statistics=%.3f, p=%.3f' % (stat, p))

    # interpret

    alpha = 0.05

    if p > alpha:

        print('Sample looks Gaussian (fail to reject H0)')

    else:

        print('Sample does not look Gaussian (reject H0)')
# D'Agostino and Pearson's Test

for q in (quantitative):

    data_check=data[q]

    print(q)

    stat, p = normaltest(data_check)

    print('Statistics=%.3f, p=%.3f' % (stat, p))

    # interpret

    alpha = 0.05

    if p > alpha:

        print('Sample looks Gaussian (fail to reject H0)')

    else:

        print('Sample does not look Gaussian (reject H0)')
# generate univariate observations

data_q = pd.DataFrame(data[quantitative])

data_q

# Anderson-Darling Test



for q in (quantitative):

    data_check=data[q]

    print(q)

    result = anderson(data_check)

    print('Statistic: %.3f' % result.statistic)

    p = 0

    for i in range(len(result.critical_values)):

        sl, cv = result.significance_level[i], result.critical_values[i]

        if result.statistic < result.critical_values[i]:

            print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))

        else:

            print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

#Doing all three tests on 'horsepower'

q='horsepower'

my_hp_array=[]

data_check=data[q]

for i in data_check:

    my_hp_array.append(int(i))

print(q)

#print(data_check)

stat, p = shapiro(my_hp_array)

stat_1,p_1=normaltest(my_hp_array)

print('Shapiro Test')

print('Statistics=%.3f, p=%.6f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:print('Sample looks Gaussian (fail to reject H0)')

else:print('Sample does not look Gaussian (reject H0)')



print('D\'Agostino and Pearson\'s Test')

print('Statistics=%.3f, p=%.6f' % (stat_1, p_1))

if p_1 > alpha:print('Sample looks Gaussian (fail to reject H0)')

else:print('Sample does not look Gaussian (reject H0)')
my_hp_array
#Heavy Tailed QQ Plots are observed

import statsmodels.api as sm

import pylab

for q in (quantitative):

    data_check=data[q]

    sm.qqplot(data_check, loc = 4, scale = 3, line='s')

pylab.show()
def histogram_plots(quantitative):

    """plots histogram of dataset.

    Args:

    quantitative: name of columns in dataframe.

    """

    for i,q in enumerate(quantitative):

        data_q=data[q]

        plt.subplot(14, 1, i+1)

        plt.hist(data_q)

        #fig=plt.hist(data_q)

        plt.title('Histogram for: '+q)

        #plt.savefig("random_d"+str(i+1)+".png")

    

    #plt.subplots_adjust(left=0.125,

    #                bottom=0.1, 

    #                right=0.9, 

    #                top=0.9, 

    #               wspace=0.2, 

    #                hspace=0.35)

    #plt.subplot_tool()

    plt.suptitle('Randomly Sampled Normal Variables: x_N', fontsize=14)

    plt.show()

histogram_plots(quantitative)
titles=['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'origin','horsepower']





x=np.linspace(0,3,398)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2) 



fig.suptitle('Sharing x per column, y per row')



ax1.plot(x, data[titles[0]])

ax2.plot(x, data[titles[1]], 'tab:orange')

ax3.plot(x, data[titles[2]], 'tab:green')

ax4.plot(x, data[titles[3]], 'tab:red')



for ax in fig.get_axes():

    ax.label_outer()



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.plot(x, data[titles[4]])

ax2.plot(x, data[titles[5]], 'tab:orange')

ax3.plot(x, data[titles[6]], 'tab:green')

ax4.plot(x, my_hp_array, 'tab:red')



for ax in fig.get_axes():

    ax.label_outer()

titles=['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'origin','horsepower']

#displacement, horsepower is giving bimodal distribution

#mpg and weight are showing positive(right) skewed

#acceleration is showing normal dist

#cylinders,origin and model year is showing multimodal dist



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2) 



fig.suptitle('Sharing x per column, y per row')



ax1.hist(data[titles[0]]);ax1.set_title(titles[0])

ax2.hist(data[titles[1]]);ax2.set_title(titles[1])

ax3.hist(data[titles[2]]);ax3.set_title(titles[2])

ax4.hist(data[titles[3]]);ax4.set_title(titles[3])



for ax in fig.get_axes():

    ax.label_outer()



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.hist(data[titles[4]]);ax1.set_title(titles[4])

ax2.hist(data[titles[5]]);ax2.set_title(titles[5])

ax3.hist(data[titles[6]]);ax3.set_title(titles[6])

ax4.hist(my_hp_array);ax4.set_title(titles[7])



for ax in fig.get_axes():

    ax.label_outer()
titles=['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'origin','horsepower']

#cylinders and origin are completely positively skewed

#mpg, displacement,weight and horsepower are somewhat positively skewed



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2) 



fig.suptitle('Sharing x per column, y per row')



ax1.boxplot(data[titles[0]]);ax1.set_title(titles[0])

ax2.boxplot(data[titles[1]]);ax2.set_title(titles[1])

ax3.boxplot(data[titles[2]]);ax3.set_title(titles[2])

ax4.boxplot(data[titles[3]]);ax4.set_title(titles[3])



for ax in fig.get_axes():

    ax.label_outer()



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.boxplot(data[titles[4]]);ax1.set_title(titles[4])

ax2.boxplot(data[titles[5]]);ax2.set_title(titles[5])

ax3.boxplot(data[titles[6]]);ax3.set_title(titles[6])

ax4.boxplot(my_hp_array);ax4.set_title(titles[7])



for ax in fig.get_axes():

    ax.label_outer()
def graph_of_plots(titles,hist_or_box):

    """plots histogram or boxplot from the given titles in dataset.

    Args:

    titles:  name of all 8 columns in dataframe. 

    hist_or_box: consists of variable either 'hist' or 'box'.

    """

    assert len(titles)==8

    if hist_or_box=='hist':

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2) 



        fig.suptitle('Sharing x per column, y per row')



        ax1.hist(data[titles[0]]);ax1.set_title(titles[0])

        ax2.hist(data[titles[1]]);ax2.set_title(titles[1])

        ax3.hist(data[titles[2]]);ax3.set_title(titles[2])

        ax4.hist(data[titles[3]]);ax4.set_title(titles[3])



        for ax in fig.get_axes():

            ax.label_outer()



        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        ax1.hist(data[titles[4]]);ax1.set_title(titles[4])

        ax2.hist(data[titles[5]]);ax2.set_title(titles[5])

        ax3.hist(data[titles[6]]);ax3.set_title(titles[6])

        ax4.hist(my_hp_array);ax4.set_title(titles[7])



        for ax in fig.get_axes():

            ax.label_outer()

    elif hist_or_box=='box':

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2) 



        fig.suptitle('Sharing x per column, y per row')



        ax1.boxplot(data[titles[0]]);ax1.set_title(titles[0])

        ax2.boxplot(data[titles[1]]);ax2.set_title(titles[1])

        ax3.boxplot(data[titles[2]]);ax3.set_title(titles[2])

        ax4.boxplot(data[titles[3]]);ax4.set_title(titles[3])



        for ax in fig.get_axes():

            ax.label_outer()



        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        ax1.boxplot(data[titles[4]]);ax1.set_title(titles[4])

        ax2.boxplot(data[titles[5]]);ax2.set_title(titles[5])

        ax3.boxplot(data[titles[6]]);ax3.set_title(titles[6])

        ax4.boxplot(my_hp_array);ax4.set_title(titles[7])



        for ax in fig.get_axes():

            ax.label_outer()
titles=['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'origin','horsepower']

graph_of_plots(titles,'box')
from scipy.stats import f_oneway

#pvalue is less than 0.05 we accept the null hypothesis(at least one population mean is different from the rest)

titles=['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'origin','horsepower']

group1,group2,group3,group4,group5,group6,group7,group8=data[titles[0]],data[titles[1]],data[titles[2]],data[titles[3]],data[titles[4]],data[titles[5]],data[titles[6]],my_hp_array

#perform one-way ANOVA

stat,p=f_oneway(group1,group2,group3,group4,group5,group6,group7,group8)

print('p value for significance = %.6f' % (p))

if p<0.05:

    print("reject null hypothesis that is, groups does not have equal variance")

else:

    print("accept null hypothesis that is, groups having equal variances")