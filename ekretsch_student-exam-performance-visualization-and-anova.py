import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from statsmodels.formula.api import ols
#create function for summary data

df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

def overview(dataframe):

    #docstring

    '''

    Read a csv file into a DataFrame.

    Print first 5 rows of data.

    Print datatype for each column.

    Print number of NULL/NaN values for each column.

    Print summary data.

    

    Return:

    data, rtype: DataFrame

    '''

    print("The first 5 rows of data are:\n", df.head())

    print("\n")

    print("The (Row,Column) is:\n", df.shape)

    print("\n")

    print("Data type of each column:\n", df.dtypes)

    print("\n")

    print("The number of null values in each column are:\n", df.isnull().sum())

    print("\n")

    print("Summary of data:\n", df.describe())

    return



overview(df)
#Create function to display distribution pairplot

def distribution(dataset, variable):

    '''

    Args:

        dataset: Include the DataFrame here

        variable: Include the column from dataframe used for color encoding

    Returns:

        sns pairplot with color encoding

    '''

    g = sns.pairplot(data = dataset, hue = variable)

    g.fig.suptitle('Graph showing distribution between scores and {}'.format(variable), fontsize=20)

    g.fig.subplots_adjust(top=0.9)

    return g
df.columns
#Score and gender

distribution(df, 'gender')
#score and race

distribution(df, 'race/ethnicity')
#score and parental education level

distribution(df, 'parental level of education')
#Score and lunch

distribution(df, 'lunch')
#Score and test preparation course

distribution(df, 'test preparation course')
#clean up column names for StatsModels

df.columns = ['gender', 'race', 'parental_edu', 'lunch', 'test_prep_course', 'math_score', 'reading_score', 'writing_score']
#Create anova test function

def anova_test(data, variable):

    '''

    Args: 

        data = (DataFrame)

        variable = Categorical column used for 1-way ANOVA test

    Returns: Nothing

    '''

    x = ['math_score', 'reading_score', 'writing_score']

    for i,k in enumerate(x):

        lm = ols('{} ~ {}'.format(x[i],variable), data=data).fit()

        table = sm.stats.anova_lm(lm)

        print("P-value for 1-way ANOVA test between {} and {} is ".format(x[i],variable),table.loc[variable,'PR(>F)'])
#Gender ANOVA

anova_test(df, 'gender')
#Parental education ANOVA

anova_test(df, 'parental_edu')
#Lunch ANOVA

anova_test(df, 'lunch')
#Test Prep ANOVA

anova_test(df, 'test_prep_course')
#Create countplot for parental education and student scores

plt.figure(figsize=(12,5))

sns.countplot(data=df, x='parental_edu', hue='gender')