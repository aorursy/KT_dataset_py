# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 

import statsmodels.api as sm

from statsmodels.formula.api import ols
data = pd.read_csv('/kaggle/input/studentperformance/StudentsPerformance.csv')

data.head()
def overview():

        # Creating Docstring

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

        print("The first 5 rows in data are: \n", data.head())

        print("\n")

        print("the (Row,Column) is: \n", data.shape)

        print("\n")

        print("Data type of each column: \n", data.dtypes)

        print("\n")

        print("The number of null values in each column are: \n", data.isnull())

        print("\n")

        print("Summary of all the test scores: \n", data.describe())

        return data

    

df = overview()
def distribution(dataset,variable):

    '''

    Args:

        dataset: Include DataFrame here

        variable: Include which column (categorical) in the data should be used

        

    Returns: 

    Seaborn plot with colo encoding

    '''

    g = sns.pairplot(data = data, hue = variable)

    g.fig.suptitle('Graph showing distribution between scores and {}'.format(variable), fontsize = 20)

    g.fig.subplots_adjust(top= 0.9)

    return g
distribution(data, 'gender')
distribution(data, 'race/ethnicity')
distribution(data,'parental level of education')
distribution(data,'lunch')
distribution(data,'test preparation course')
data.columns = ['gender', 'race', 'parental_edu','lunch', 'test_prep_course', 'math_score', 'reading_score', 'writing_score']

data.head()
def anova_test(data, variable):

    '''

    Args: data (dataFrame), variable: Categorical columns that you want to do 1-way Anova test with

    

    Returns: Nothing

    '''

    x = ['math_score','reading_score','writing_score']

    for i,k in enumerate(x):

        lm = ols('{} ~ {}'.format(x[i],variable),data = data).fit()

        table = sm.stats.anova_lm(lm)

        print("P-value for 1 way ANOVA test between {} and {} is ".format(x[i],variable),table.loc[variable,'PR(>F)'])
anova_test(data,'gender')
anova_test(data,'parental_edu')
anova_test(data,'lunch')
anova_test(data,'test_prep_course')
plt.figure(figsize =(12,5))



sns.countplot(data = data, x = 'parental_edu', hue = 'gender')