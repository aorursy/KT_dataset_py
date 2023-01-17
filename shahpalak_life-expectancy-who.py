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
# import Libraries

import numpy as np 

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

import random
df = pd.read_csv("/kaggle/input/life-expectancy-who/Life Expectancy Data.csv")
df.head()
df.info()
df.describe()
num_col = df.select_dtypes(include=np.number).columns

print("Numerical columns: \n",num_col)



cat_col = df.select_dtypes(exclude=np.number).columns

print("Categorical columns: \n",cat_col)
# Remove the extra space from column names



df = df.rename(columns=lambda x: x.strip())
# Import label encoder 

from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in column 'Status'. 

df['Status']= label_encoder.fit_transform(df['Status'])

  

df.head()
df.isna().sum()
# Replace using mean 

for i in df.columns.drop('Country'):

    df[i].fillna(df[i].mean(), inplace = True)
# Let's check the distribution of y variable (Life Expectancy)

plt.figure(figsize=(8,8), dpi= 80)

sns.boxplot(df['Life expectancy'])

plt.title('Life expectancy Box Plot')

plt.show()
plt.figure(figsize=(8,8))

plt.title('Life expectancy Distribution Plot')

sns.distplot(df['Life expectancy'])
num_col = df.select_dtypes(include=np.number).columns

print("Numerical columns: \n",num_col)



cat_col = df.select_dtypes(exclude=np.number).columns

print("Categorical columns: \n",cat_col)
# Let's check the multicollinearity of features by checking the correlation matric



plt.figure(figsize=(15,15))

p=sns.heatmap(df[num_col].corr(), annot=True,cmap='RdYlGn',center=0) 
# Pair Plots to know the relation between different features

ax = sns.pairplot(df[num_col])
# Train test split

X=df.drop(columns=['Life expectancy','Country'])

y=df[['Life expectancy']]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1234)
## By David Dale https://datascience.stackexchange.com/users/24162/david-dale



df= pd.Series(dtype='float64' 'int64')

def stepwise_selection(X, y, 

                       initial_list=[], 

                       threshold_in=0.01, 

                       threshold_out = 0.05, 

                       verbose=True):

    """ Perform a forward-backward feature selection 

    based on p-value from statsmodels.api.OLS

    Arguments:

        X - pandas.DataFrame with candidate features

        y - list-like with the target

        initial_list - list of features to start with (column names of X)

        threshold_in - include a feature if its p-value < threshold_in

        threshold_out - exclude a feature if its p-value > threshold_out

        verbose - whether to print the sequence of inclusions and exclusions

    Returns: list of selected features 

    Always set threshold_in < threshold_out to avoid infinite looping.

    See https://en.wikipedia.org/wiki/Stepwise_regression for the details

    """

    included = list(initial_list)

    while True:

        changed=False

        # forward step

        excluded = list(set(X.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))



        # backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() # null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included



result = stepwise_selection(X_train, y_train)

print('resulting features:')

print(result)