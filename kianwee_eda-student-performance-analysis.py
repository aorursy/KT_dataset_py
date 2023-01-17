import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 

import statsmodels.api as sm

from statsmodels.formula.api import ols
def overview():

    '''

    Read a comma-separated values (csv) file into DataFrame.

    Print 5 rows of data

    Print number of rows and columns

    Print datatype for each column

    Print number of NULL/NaN values for each column

    Print summary data

    

    Return:

    data, rtype: DataFrame

    '''

    data = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")

    print("The first 5 rows if data are:\n", data.head())

    print("\n")

    print("The (Row,Column) is:\n", data.shape)

    print("\n")

    print("Data type of each column:\n", data.dtypes)

    print("\n")

    print("The number of null values in each column are:\n", data.isnull().sum())

    print("\n")

    print("Summary of all the test scores:\n", data.describe())

    return data



df = overview()
def distribution(dataset,variable):

    '''

    Args:

        dataset: Include DataFrame here

        variable: Include which column (categorical) in the data frame should be used for colour encoding.

    

    Returns:

    Seaborn plot with colour encoding

    '''

    g = sns.pairplot(data = dataset, hue = variable)

    g.fig.suptitle('Graph showing distribution between scores and {}'.format(variable), fontsize = 20)

    g.fig.subplots_adjust(top= 0.9)

    return g



distribution(df, 'gender')
distribution(df, 'race/ethnicity')
distribution(df, 'parental level of education')
distribution(df, 'lunch')
distribution(df, 'test preparation course')
df.columns = ['gender', 'race', 'parental_edu', 'lunch', 'test_prep_course', 'math_score', 'reading_score', 'writing_score']



def anova_test(data, variable):

    '''

    Args: data (DataFrame), variable: Categorical columns that you want to do 1-way ANOVA test with

    

    Returns: Nothing

    '''

    x = ['math_score', 'reading_score', 'writing_score']

    for i,k in enumerate(x):

        lm = ols('{} ~ {}'.format(x[i],variable), data = data).fit()

        table = sm.stats.anova_lm(lm)

        print("P-value for 1-way ANOVA test between {} and {} is ".format(x[i],variable),table.loc[variable,'PR(>F)'])



anova_test(df, 'gender')
anova_test(df, 'race')
anova_test(df, 'parental_edu')
anova_test(df, 'lunch')
anova_test(df, 'test_prep_course')
plt.figure(figsize=(12,5))



sns.countplot(data = df, x = 'parental_edu', hue = 'gender')