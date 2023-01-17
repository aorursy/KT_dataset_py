#import pandas for data manipulation and exploratory data analysis
import pandas as pd

#importing matplotlib and seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Read the data into the notebook
young = pd.read_csv('../input/responses.csv')
# It is important to get the sense of the data using the head function
young.head(10)
young.tail(10)
#there are 1010 rows and 150 variables in the dataset
# before starting the analysis get summary of the data
young.describe()
# It is helpful to know which columns have missing data, since we have a lot of columns in this dataset, it's better that we visualize it

nulls = young.isnull().sum().sort_values(ascending=False)
nulls.plot(
    kind='bar', figsize=(23, 5))

# we notice that height and weight have the most missing values but our analysis of the hypothesis testing of phobias does not get affected
#We also notice that gender has 6 missing values, moving on, we can remove them from the dataframe as it would not affect our analysis
young.columns
# we can call the shape function to look at the number of rows and columns of the dataset

young.shape

# our data has 1010 rows and 150 columns, this exercise turned out to be useful as for further analysis, we can focus on the question that needs to be anaswered
# For further analysis, we can use the info method

young.info()
# we already know from the visualization above that the Gender variable has 6 missing columns, we are also interested to know how the data is spread across male and female
young.Gender.value_counts(dropna = False)
#Data visualization is a way to spot obvious errors and outliers
# It is helpful in planning the data cleaning pipeline

#The question of interest for us in the dataset is " Do women fear certain phenomena significantly more than men?
# Hence we will plot Phobias on a histogram and then boxplots for the spread of phobias across men and women 

young.Flying.plot('hist')
young.Storm.plot('hist')
young.Darkness.plot('hist')
young.Heights.plot('hist')
young.Spiders.plot('hist')
young.Snakes.plot('hist')
young.Rats.plot('hist')
young.Ageing.plot('hist')
young['Dangerous dogs'].plot('hist')
young['Fear of public speaking'].plot('hist')
bp = sns.boxplot(x = 'Gender', y = 'Flying', data = young)
bp = sns.boxplot(x = 'Gender', y = 'Storm', data = young)
bp = sns.boxplot(x = 'Gender', y = 'Darkness', data = young)
bp = sns.boxplot(x = 'Gender', y = 'Height', data = young)
bp = sns.boxplot(x = 'Gender', y = 'Spiders', data = young)
bp = sns.boxplot(x = 'Gender', y = 'Snakes', data = young)
bp = sns.boxplot(x = 'Gender', y = 'Rats', data = young)
bp = sns.boxplot(x = 'Gender', y = 'Ageing', data = young)
bp = sns.boxplot(x = 'Gender', y = 'Dangerous dogs', data = young)
bp = sns.boxplot(x = 'Gender', y = 'Fear of public speaking', data = young)
# In this part of the exercise, I am going to drop all the records with missing values in either phobias or gender
young.dropna(subset = ['Flying','Storm', 'Darkness','Heights','Spiders','Snakes','Rats','Ageing','Dangerous dogs',
                       'Fear of public speaking','Gender'])
young.shape
# After dropping the columns, we have 984 records left for Hypothesis testing
# to run the hypothesis test, we will be using scipy.stats package

from scipy.stats import chi2
test = pd.DataFrame()
def table_creation(row, col):
    test = pd.crosstab(index=row,columns=col,margins=True)
    test.columns = ["1.0","2.0","3.0","4.0","5.0","rowtotal"]
    return(test);
    
def chisq_test(t, i):
    # Get table without totals for later use
    observed = t.ix[0:2,0:5]   
    #To get the expected count for a cell.
    expected =  np.outer(t["rowtotal"][0:2],t.ix["All"][0:5])/1010
    expected = pd.DataFrame(expected)
    expected.columns = ["1.0","2.0","3.0","4.0","5.0"]
    expected.index= test.index[0:2]
    #Calculate the chi-sq statistics
    chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()
    print("Chi-sq stat")
    print(chi_squared_stat)
    crit = chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = i)   # *
    print("Critical value")
    print(crit)
    p_value = 1 - chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=i)
    print("P value")
    print(p_value)
    return;
test = table_creation(young["Gender"],young["Flying"])
print(test)
chisq_test(test,4)
test = table_creation(young["Gender"],young["Storm"])
print(test)
chisq_test(test,4)
test = table_creation(young["Gender"],young["Heights"])
print(test)
chisq_test(test,4)
test = table_creation(young["Gender"],young["Spiders"])
print(test)
chisq_test(test,4)
test = table_creation(young["Gender"],young["Snakes"])
print(test)
chisq_test(test,4)
test = table_creation(young["Gender"],young["Rats"])
print(test)
chisq_test(test,4)
test = table_creation(young["Gender"],young["Ageing"])
print(test)
chisq_test(test,4)
test = table_creation(young["Gender"],young["Dangerous dogs"])
print(test)
chisq_test(test,4)
test = table_creation(young["Gender"],young["Fear of public speaking"])
print(test)
chisq_test(test,4)