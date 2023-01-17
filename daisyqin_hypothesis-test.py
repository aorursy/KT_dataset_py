import numpy as np

import pandas as pd

import scipy.stats as stats

import seaborn as sns



# Import data 

dta_all = pd.read_csv('../input/responses.csv')
test = pd.DataFrame()

def table_building(row, col):

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

    crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*

                      df = i)   # *

    print("Critical value")

    print(crit)

    p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value

                             df=i)

    print("P value")

    print(p_value)

    return;
test = table_building(dta_all["Gender"],dta_all["Finances"])

print(test)

chisq_test(test,4)
sns.countplot(x='Finances', hue = 'Gender', data = dta_all)
test = table_building(dta_all["Village - town"],dta_all["Finances"])

print(test)

chisq_test(test,4)
sns.countplot(x='Finances', hue = "Village - town", data = dta_all)