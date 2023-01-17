import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# let's load the dataset with just a few columns and a few rows

# to speed the demo



# Variable definitions:

#-------------------------

# loan_amnt: loan amount

# int_rate: interest rate

# annual_inc: annual income

# open_acc: open accounts (more on this later)

# loan_status: loan status(paid, defaulted, etc)

# open_il_12m: accounts opened in the last 12 months



use_cols = [

    'loan_amnt', 'int_rate', 'annual_inc', 'open_acc', 'loan_status',

    'open_il_12m','id', 'purpose', 'loan_status', 'home_ownership'

]



# this dataset is very big. To speed things up for the demo

# I will randomly select 10,000 rows when I load the dataset

# so I upload just a sample of the total rows



data = pd.read_csv('../input/lending-club-loan-data/loan.csv', usecols=use_cols).sample(10000, random_state=44)  # set a seed for reproducibility



data.head()
# let's look at the values of the variable loan_amnt

# this is the amount of money requested by the borrower

# in US dollars



# this variable is continuous, it can take in principle,

# any value



data.open_il_12m.unique()
# let's make an histogram to get familiar with the

# distribution of the variable



fig = data.loan_amnt.hist(bins=50)

fig.set_title('Loan Amount Requested')

fig.set_xlabel('Loan Amount')

fig.set_ylabel('Number of Loans')
# let's inspect the values of the variable



# this is a discrete variable



data.open_acc.dropna().unique()
# let's make an histogram to get familiar with the

# distribution of the variable



fig = data.open_acc.hist(bins=100)

fig.set_xlim(0, 30)

fig.set_title('Number of open accounts')

fig.set_xlabel('Number of open accounts')

fig.set_ylabel('Number of Customers')
# let's inspect the variable home ownership,

# which indicates whether the borrowers own their home

# or if they are renting for example, among other things.



data.home_ownership.unique()
# let's make a bar plot, with the number of loans

# for each category of home ownership



# the code below counts the number of observations (borrowers)

# within each category

# and then makes a plot



fig = data['home_ownership'].value_counts().plot.bar()

fig.set_title('Home Ownership')

fig.set_ylabel('Number of customers')
data['home_ownership'].value_counts()
# the "purpose" variable is another categorical variable

# that indicates how the borrowers intend to use the

# money they are borrowing, for example to improve their

# house, or to cancel previous debt.



data.purpose.unique()
# let's make a bar plot with the number of borrowers

# within each category



# the code below counts the number of observations (borrowers)

# within each category

# and then makes a plot



fig = data['purpose'].value_counts().plot.bar()

fig.set_title('Loan Purpose')

fig.set_ylabel('Number of customers')
# let's load the titanic dataset



data = pd.read_csv('../input/titanic/train.csv')

data.head()
# let's inspect the cardinality, this is the number

# of different labels, for the different categorical variables



print('Number of categories in the variable Name: {}'.format(

    len(data.Name.unique())))



print('Number of categories in the variable Gender: {}'.format(

    len(data.Sex.unique())))



print('Number of categories in the variable Ticket: {}'.format(

    len(data.Ticket.unique())))



print('Number of categories in the variable Cabin: {}'.format(

    len(data.Cabin.unique())))



print('Number of categories in the variable Embarked: {}'.format(

    len(data.Embarked.unique())))



print('Total number of passengers in the Titanic: {}'.format(len(data)))
# let's explore the values / categories of Cabin



# we know from the previous cell that there are 148

# different cabins, therefore the variable

# is highly cardinal



data.Cabin.unique()
# let's capture the first letter of Cabin

data['Cabin_reduced'] = data['Cabin'].astype(str).str[0]



data[['Cabin', 'Cabin_reduced']].head()
# let's separate into training and testing set

# in order to build machine learning models

from sklearn.model_selection import train_test_split



use_cols = ['Cabin', 'Cabin_reduced', 'Sex']



# this functions comes from scikit-learn

X_train, X_test, y_train, y_test = train_test_split(

    data[use_cols], 

    data.Survived,  

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
# Let's find out labels present only in the training set



unique_to_train_set = [

    x for x in X_train.Cabin.unique() if x not in X_test.Cabin.unique()

]



len(unique_to_train_set)
# Let's find out labels present only in the test set



unique_to_test_set = [

    x for x in X_test.Cabin.unique() if x not in X_train.Cabin.unique()

]



len(unique_to_test_set)
# Let's find out labels present only in the training set

# for Cabin with reduced cardinality



unique_to_train_set = [

    x for x in X_train['Cabin_reduced'].unique()

    if x not in X_test['Cabin_reduced'].unique()

]



len(unique_to_train_set)
# Let's find out labels present only in the test set

# for Cabin with reduced cardinality



unique_to_test_set = [

    x for x in X_test['Cabin_reduced'].unique()

    if x not in X_train['Cabin_reduced'].unique()

]



len(unique_to_test_set)
# let's load the dataset with the variables

# we need for this demo



# 'X1', 'X2', 'X3', 'X6' are the categorical 

# variables in this dataset

# "y" is the target: time to pass the quality tests



data = pd.read_csv('../input/mercedesbenz-greener-manufacturing/train.csv',

                   usecols=['X1', 'X2', 'X3', 'X6', 'y'])

data.head()
# let's look at the different number of labels

# in each variable (cardinality)



cols_to_use = ['X1', 'X2', 'X3', 'X6']



for col in cols_to_use:

    print('variable: ', col, ' number of labels: ', len(data[col].unique()))



print('total cars: ', len(data))
# let's plot how frequently each label

# appears in the dataset



# in other words, the percentage of cars that

# show each label



total_cars = len(data)



# for each categorical variable

for col in cols_to_use:



    # count the number of cars per category

    # and divide by total cars



    # aka percentage of cars per category



    temp_df = pd.Series(data[col].value_counts() / total_cars)



    # make plot with the above percentages

    fig = temp_df.sort_values(ascending=False).plot.bar()

    fig.set_xlabel(col)



    # add a line at 5 %

    fig.axhline(y=0.05, color='red')

    fig.set_ylabel('percentage of cars')

    plt.show()
# the following function calculates:



# 1) the percentage of cars per category

# 2) the mean time to pass testing per category





def calculate_perc_and_passtime(df, var):



    # total number of cars

    total_cars = len(df)



    # percentage of cars per category

    temp_df = pd.Series(df[var].value_counts() / total_cars).reset_index()

    temp_df.columns = [var, 'perc_cars']



    # add the mean to pass testing time

    # the target in this dataset is called 'y'

    temp_df = temp_df.merge(df.groupby([var])['y'].mean().reset_index(),

                            on=var,

                            how='left')



    return temp_df





# now we use the function for the variable 'X3'

temp_df = calculate_perc_and_passtime(data, 'X3')

temp_df
# Now I create a function to plot of the

# label frequency and mean time to pass testing.



# This will help us visualise the relationship between the

# target and the labels



def plot_categories(df, var):

    

    fig, ax = plt.subplots(figsize=(8, 4))

    plt.xticks(df.index, df[var], rotation=0)



    ax2 = ax.twinx()

    ax.bar(df.index, df["perc_cars"], color='lightgrey')

    ax2.plot(df.index, df["y"], color='green', label='Seconds')

    ax.axhline(y=0.05, color='red')

    ax.set_ylabel('percentage of cars per category')

    ax.set_xlabel(var)

    ax2.set_ylabel('Time to pass testing, in seconds')

    plt.show()
plot_categories(temp_df, 'X3')
# let's plot the remaining categorical variables



for col in cols_to_use:

    

    if col !='X3':

        

        # re using the functions I created

        temp_df = calculate_perc_and_passtime(data, col)

        plot_categories(temp_df, col)
# I will replace all the labels that appear in less than 10%

# of the cars by the label 'rare'



def group_rare_labels(df, var):



    total_cars = len(df)



    # first I calculate the 10% of cars for each category

    temp_df = pd.Series(df[var].value_counts() / total_cars)



    # now I create a dictionary to replace the rare labels with the

    # string 'rare'



    grouping_dict = {

        k: ('rare' if k not in temp_df[temp_df >= 0.1].index else k)

        for k in temp_df.index

    }



    # now I replace the rare categories

    tmp = df[var].map(grouping_dict)



    return tmp
# group rare labels in X1



data['X1_grouped'] = group_rare_labels(data, 'X1')



data[['X1', 'X1_grouped']].head(10)
# let's plot X1 with the grouped categories

# re-using the functions I created above



temp_df = calculate_perc_and_passtime(data, 'X1_grouped')

plot_categories(temp_df, 'X1_grouped')
# let's plot the original X1 for comparison

temp_df = calculate_perc_and_passtime(data, 'X1')

plot_categories(temp_df, 'X1')
# let's group and plot the remaining categorical variables



for col in cols_to_use[1:]:

        

    # re using the functions I created

    data[col+'_grouped'] = group_rare_labels(data, col)

    temp_df = calculate_perc_and_passtime(data, col+'_grouped')

    plot_categories(temp_df, col+'_grouped')
# let's separate into training and testing set

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(data[cols_to_use],

                                                    data.y,

                                                    test_size=0.3,

                                                    random_state=0)



X_train.shape, X_test.shape
# Let's find labels present only in the training set

# I will use X2 as example



unique_to_train_set = [

    x for x in X_train['X2'].unique() if x not in X_test['X2'].unique()

]



print(unique_to_train_set)
# Let's find labels present only in the test set



unique_to_test_set = [

    x for x in X_test['X2'].unique() if x not in X_train['X2'].unique()

]



print(unique_to_test_set)
# print information for boston dataset

from sklearn.datasets import load_boston

print(load_boston().DESCR)
# for Q-Q plots

import pylab

import scipy.stats as stats

import seaborn as sns



# boston house dataset for the demo

from sklearn.datasets import load_boston
# load the the Boston House price data



# load the boston dataset from sklearn

boston_dataset = load_boston()



# create a dataframe with the independent variables

# I will use only 3 of the total variables for this demo



boston = pd.DataFrame(boston_dataset.data,

                      columns=boston_dataset.feature_names)[[

                          'RM', 'LSTAT', 'CRIM'

                      ]]





boston.head()
# load the titanic dataset



titanic = pd.read_csv('../input/titanic/train.csv',

                      usecols=['Age', 'Fare'])



# The variable age has missing values, I will

# remove them for this demo

titanic.dropna(subset=['Age'], inplace=True)



titanic.head()
# function to create histogram, Q-Q plot and

# boxplot





def diagnostic_plots(df, variable):

    # function takes a dataframe (df) and

    # the variable of interest as arguments



    # define figure size

    plt.figure(figsize=(16, 4))



    # histogram

    plt.subplot(1, 3, 1)

    sns.distplot(df[variable], bins=30)

    plt.title('Histogram')



    # Q-Q plot

    plt.subplot(1, 3, 2)

    stats.probplot(df[variable], dist="norm", plot=pylab)

    plt.ylabel('RM quantiles')



    # boxplot

    plt.subplot(1, 3, 3)

    sns.boxplot(y=df[variable])

    plt.title('Boxplot')



    plt.show()
# let's start with the variable RM from the

# boston house dataset.

# RM is the average number of rooms per dwelling



diagnostic_plots(boston, 'RM')
# let's inspect now the variable Age from the titanic

# refers to the age of the passengers on board



diagnostic_plots(titanic, 'Age')
# variable LSTAT from the boston house dataset

# LSTAT is the % lower status of the population



diagnostic_plots(boston, 'LSTAT')
# variable CRIM from the boston house dataset

# CRIM is the per capita crime rate by town



diagnostic_plots(boston, 'CRIM')
# variable Fare from the titanic dataset

# Fare is the price paid for the ticket by

# the passengers



diagnostic_plots(titanic, 'Fare')
# function to find upper and lower boundaries

# for normally distributed variables





def find_normal_boundaries(df, variable):



    # calculate the boundaries outside which sit the outliers

    # for a Gaussian distribution



    upper_boundary = df[variable].mean() + 3 * df[variable].std()

    lower_boundary = df[variable].mean() - 3 * df[variable].std()



    return upper_boundary, lower_boundary
# calculate boundaries for RM

upper_boundary, lower_boundary = find_normal_boundaries(boston, 'RM')

upper_boundary, lower_boundary
# inspect the number and percentage of outliers for RM



print('total number of houses: {}'.format(len(boston)))



print('houses with more than 8.4 rooms (right end outliers): {}'.format(

    len(boston[boston['RM'] > upper_boundary])))



print('houses with less than 4.2 rooms (left end outliers: {}'.format(

    len(boston[boston['RM'] < lower_boundary])))

print()

print('% right end outliers: {}'.format(

    len(boston[boston['RM'] > upper_boundary]) / len(boston)))



print('% left end outliers: {}'.format(

    len(boston[boston['RM'] < lower_boundary]) / len(boston)))
# calculate boundaries for Age in the titanic



upper_boundary, lower_boundary = find_normal_boundaries(titanic, 'Age')

upper_boundary, lower_boundary
# lets look at the number and percentage of outliers



print('total passengers: {}'.format(len(titanic)))



print('passengers older than 73 rooms: {}'.format(

    len(titanic[titanic['Age'] > upper_boundary])))

print()

print('% of passengers older than 73 rooms: {}'.format(

    len(titanic[titanic['Age'] > upper_boundary]) / len(titanic)))
# function to find upper and lower boundaries

# for skewed distributed variables





def find_skewed_boundaries(df, variable, distance):



    # Let's calculate the boundaries outside which sit the outliers

    # for skewed distributions



    # distance passed as an argument, gives us the option to

    # estimate 1.5 times or 3 times the IQR to calculate

    # the boundaries.



    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)



    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)

    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)



    return upper_boundary, lower_boundary
# looking for outliers,

# using the interquantile proximity rule

# IQR * 1.5, the standard metric



# for LSTAT in the boston house dataset



upper_boundary, lower_boundary = find_skewed_boundaries(boston, 'LSTAT', 1.5)

upper_boundary, lower_boundary
# lets look at the number and percentage of outliers

# for LSTAT



print('total houses: {}'.format(len(boston)))



print('houses with LSTAT bigger than 32: {}'.format(

    len(boston[boston['LSTAT'] > upper_boundary])))

print()

print('% houses with LSTAT bigger than 32: {}'.format(

    len(boston[boston['LSTAT'] > upper_boundary])/len(boston)))
# looking for outliers,

# using the interquantile proximity rule

# IQR * 3, now I am looking for extremely high values



upper_boundary, lower_boundary = find_skewed_boundaries(boston, 'CRIM', 3)

upper_boundary, lower_boundary
# lets look at the number and percentage of outliers

# for CRIM



print('total houses: {}'.format(len(boston)))



print('houses with CRIM bigger than 14: {}'.format(

    len(boston[boston['CRIM'] > upper_boundary])))

print()

print('% houses with CRIM bigger than 14s: {}'.format(

    len(boston[boston['CRIM'] > upper_boundary]) / len(boston)))
# finally, identify outliers in Fare in the

# titanic dataset. I will look again for extreme values

# using IQR * 3



upper_boundary, lower_boundary = find_skewed_boundaries(titanic, 'Fare', 3)

upper_boundary, lower_boundary
# lets look at the number and percentage of passengers

# who paid extremely high Fares



print('total passengers: {}'.format(len(titanic)))



print('passengers older than 73 rooms: {}'.format(

    len(titanic[titanic['Fare'] > upper_boundary])))

print()

print('passengers older than 73 rooms: {}'.format(

    len(titanic[titanic['Fare'] > upper_boundary])/len(titanic)))
import datetime



# let's load the Lending Club dataset with a few selected columns

# just a few rows to speed things up



use_cols = ['issue_d', 'last_pymnt_d']

data = pd.read_csv('../input/lending-club-loan-data/loan.csv', usecols=use_cols, nrows=10000)

data.head()
# now let's parse the dates, currently coded as strings, into datetime format



data['issue_dt'] = pd.to_datetime(data.issue_d)

data['last_pymnt_dt'] = pd.to_datetime(data.last_pymnt_d)



data[['issue_d','issue_dt','last_pymnt_d', 'last_pymnt_dt']].head()
# Extracting Month from date



data['issue_dt_month'] = data['issue_dt'].dt.month



data[['issue_dt', 'issue_dt_month']].head()
# Extract quarter from date variable



data['issue_dt_quarter'] = data['issue_dt'].dt.quarter



data[['issue_dt', 'issue_dt_quarter']].head()
# We could also extract semester



data['issue_dt_semester'] = np.where(data.issue_dt_quarter.isin([1,2]),1,2)

data.head()
# day - numeric from 1-31



data['issue_dt_day'] = data['issue_dt'].dt.day



data[['issue_dt', 'issue_dt_day']].head()
# day of the week - from 0 to 6



data['issue_dt_dayofweek'] = data['issue_dt'].dt.dayofweek



data[['issue_dt', 'issue_dt_dayofweek']].head()
# day of the week - name



data['issue_dt_dayofweek'] = data['issue_dt'].dt.day_name()



data[['issue_dt', 'issue_dt_dayofweek']].head()
# was the application done on the weekend?



data['issue_dt_is_weekend'] = np.where(data['issue_dt_dayofweek'].isin(['Sunday', 'Saturday']), 1,0)

data[['issue_dt', 'issue_dt_dayofweek','issue_dt_is_weekend']].head()
# extract year 



data['issue_dt_year'] = data['issue_dt'].dt.year



data[['issue_dt', 'issue_dt_year']].head()
# perhaps more interestingly, extract the date difference between 2 dates



# same as above capturing just the time difference

(data['last_pymnt_dt']-data['issue_dt']).dt.days.head()
# or the time difference to today, or any other day of reference



(datetime.datetime.today() - data['issue_dt']).head()
# let's load the Lending Club dataset with the variable "Number of installment accounts opened in past 24 months"

# installment accounts are those that, at the moment of acquiring them, there is a set period and amount

# of repayments agreed between the lender and borrower. An example of this is a car loan, or a student loan.

# the borrower knows that they are going to pay a certain, fixed amount over, for example 36 months.



data = pd.read_csv('../input/lending-club-loan-data/loan.csv', usecols=['id','open_il_24m'])



# let's replace the NaN with the fictitious codes described below:

# 'A': couldn't identify the person

# 'B': no relevant data

# 'C': person seems not to have any account open

# this is exactly what we did in section 2 of this course for the lecture on mixed types of variables



# select which observations we will replace with each code

indeces_b = data[data.open_il_24m.isnull()].sample(100000, random_state=44).index

indeces_c = data[data.open_il_24m.isnull()].sample(300000, random_state=42).index



# replace NA with the fictitious code

data.open_il_24m.fillna('A', inplace=True)

data.loc[indeces_b, 'open_il_24m']='B'

data.loc[indeces_c, 'open_il_24m']='C'
# let's inspect the mixed variable



data.open_il_24m.unique()
# the variable is also discrete in nature. A person can have 1, 2 accounts but not 2.3 accounts

# let's inspect the number of observations per value of the variable



fig = data.open_il_24m.value_counts().plot.bar()

fig.set_title('Number of installment accounts open')

fig.set_ylabel('Number of borrowers')
# we create 2 variables, a numerical one containing the numerical part, and

# a categorical variable with the codes (strings)



data['open_il_24m_numerical'] = np.where(data.open_il_24m.str.isdigit(), data.open_il_24m, np.nan)

data['open_il_24m_categorical'] = np.where(data.open_il_24m.str.isdigit(), np.nan, data.open_il_24m,)



data.head()
# let's inspect those instances of the dataset where numerical is not null

# we can see that when the numerical variable is not null the categorical is null

# and vice versa



data.dropna(subset = ['open_il_24m_numerical'], axis=0)
# let's load again the titanic dataset for demonstration



data = pd.read_csv('../input/titanic/train.csv', usecols = ['Ticket', 'Cabin', 'Survived'])

data.head()
# for Cabin, it is relatively straightforward, we can extract the letters and the numbers in different variables



data['Cabin_numerical'] = data.Cabin.str.extract('(\d+)') # captures numerical part

data['Cabin_categorical'] = data['Cabin'].str[0] # captures the first letter



data[['Cabin', 'Cabin_numerical', 'Cabin_categorical']].head()
# ticket is not as clear...but we could still capture the first part of the ticket as a code (category)

# and the second part of the ticket as numeric



data.Ticket.unique()
# extract the last bit of ticket as number

data['Ticket_numerical'] = data.Ticket.apply(lambda s: s.split()[-1])

data['Ticket_numerical'] = np.where(data.Ticket_numerical.str.isdigit(), data.Ticket_numerical, np.nan)



# extract the first part of ticket as category

data['Ticket_categorical'] = data.Ticket.apply(lambda s: s.split()[0])

data['Ticket_categorical'] = np.where(data.Ticket_categorical.str.isdigit(), np.nan, data.Ticket_categorical)



data[['Ticket', 'Ticket_numerical','Ticket_categorical']].head(10)
# let's compare the number of categories of the newly designed variables



print('Ticket_original no of labels: ', len(data.Ticket.unique()))

print('Cabin_original no of labels: ', len(data.Cabin.unique()))



print('Ticket_categorical no of labels: ', len(data.Ticket_categorical.unique()))

print('Cabin_categorical no of labels: ', len(data.Cabin_categorical.unique()))

# load the numerical variables of the Titanic Dataset

from sklearn.preprocessing import StandardScaler



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split



data = pd.read_csv('../input/titanic/train.csv', usecols = ['Pclass', 'Age', 'Fare', 'Survived'])

data.head()
# let's have a look at the values of those variables to get an idea of the magnitudes

data.describe()
# let's look at missing data



data.isnull().sum()
# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data[['Pclass', 'Age', 'Fare']],

                                                    data.Survived, test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
# let's fill first the missing data



X_train.Age.fillna(X_train.Age.median(), inplace=True)

X_test.Age.fillna(X_train.Age.median(), inplace=True)
# standarisation: we use the StandardScaler from sklearn



scaler = StandardScaler() # create an object

X_train_scaled = scaler.fit_transform(X_train) # fit the scaler to the train set, and then transform it

X_test_scaled = scaler.transform(X_test) # transform the test set
#let's have a look at the scaled training dataset: mean and standard deviation



print('means (Pclass, Age and Fare): ', X_train_scaled.mean(axis=0))

print('std (Pclass, Age and Fare): ', X_train_scaled.std(axis=0))
# let's look at the transformed min and max values



print('Min values (Pclass, Age and Fare): ', X_train_scaled.min(axis=0))

print('Max values (Pclass, Age and Fare): ', X_train_scaled.max(axis=0))
# let's look at the distribution of the transformed variable Age



plt.hist(X_train_scaled[:,1], bins=20)
# let's look at the distribution of the transformed variable Fare



plt.hist(X_train_scaled[:,2], bins=20)
# let's look at how transformed age looks like compared to the original variable



sns.jointplot(X_train.Age, X_train_scaled[:,1], kind='kde')
# let's look at how transformed Fare looks like compared to the original variable



sns.jointplot(X_train.Fare, X_train_scaled[:,2], kind='kde', xlim=(0,200), ylim=(-1,3))
import pylab 

import scipy.stats as stats
# load the numerical variables of the Titanic Dataset



data = pd.read_csv('../input/titanic/train.csv', usecols = ['Age', 'Fare', 'Survived'])

data.head()
# first I will fill the missing data of the variable age, with a random sample of the variable



def impute_na(data, variable):

    # function to fill na with a random sample

    df = data.copy()

    

    # random sampling

    df[variable+'_random'] = df[variable]

    

    # extract the random sample to fill the na

    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)

    

    # pandas needs to have the same index in order to merge datasets

    random_sample.index = df[df[variable].isnull()].index

    df.loc[df[variable].isnull(), variable+'_random'] = random_sample

    

    return df[variable+'_random']
# fill na

data['Age'] = impute_na(data, 'Age')
# plot the histograms to have a quick look at the distributions

# we can plot Q-Q plots to visualise if the variable is normally distributed



def diagnostic_plots(df, variable):

    # function to plot a histogram and a Q-Q plot

    # side by side, for a certain variable

    

    plt.figure(figsize=(15,6))

    plt.subplot(1, 2, 1)

    df[variable].hist()



    plt.subplot(1, 2, 2)

    stats.probplot(df[variable], dist="norm", plot=pylab)



    plt.show()

    

diagnostic_plots(data, 'Age')
### Logarithmic transformation

data['Age_log'] = np.log(data.Age)



diagnostic_plots(data, 'Age_log')
### Reciprocal transformation

data['Age_reciprocal'] = 1 / data.Age



diagnostic_plots(data, 'Age_reciprocal')
data['Age_sqr'] =data.Age**(1/2)



diagnostic_plots(data, 'Age_sqr')
data['Age_exp'] = data.Age**(1/1.2) # you can vary the exponent as needed



diagnostic_plots(data, 'Age_exp')
data['Age_boxcox'], param = stats.boxcox(data.Age) 



print('Optimal λ: ', param)



diagnostic_plots(data, 'Age_boxcox')