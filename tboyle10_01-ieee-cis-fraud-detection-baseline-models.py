import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
# setting up default plotting parameters

%matplotlib inline



plt.rcParams['figure.figsize'] = [20.0, 7.0]

plt.rcParams.update({'font.size': 22,})



sns.set_palette('viridis')

sns.set_style('white')

sns.set_context('talk', font_scale=0.8)
#raw_identity = pd.read_csv('../input/train_identity.csv')

raw_transaction = pd.read_csv('../input/train_transaction.csv')



test = pd.read_csv('../input/test_transaction.csv')



#print('train_identity shape: ')

#print(raw_identity.shape)

print('train_transaction shape: ')

print(raw_transaction.shape)
#raw_identity.head()
raw_transaction.head()
# using seaborns countplot to show distribution of questions in dataset

fig, ax = plt.subplots()

g = sns.countplot(raw_transaction.isFraud, palette='viridis')

g.set_xticklabels(['Not Fraud', 'Fraud'])

g.set_yticklabels([])



# function to show values on bars

def show_values_on_bars(axs):

    def _show_on_single_plot(ax):        

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.0f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center") 



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)

show_values_on_bars(ax)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Fraud vs Not Fraud', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
# print percentage of transactions where target == 1

(len(raw_transaction.loc[raw_transaction.isFraud==1])) / (len(raw_transaction.loc[raw_transaction.isFraud == 0])) * 100
#checking for missing values



# from https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%201.ipynb# from  

# Function to calculate missing values by column# Funct 

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(raw_transaction).head(10)
#missing_values_table(raw_identity).head(10)
def drop_missing_values(df, percent_drop):

    """

    Drop columns with missing values.

    

    Args:

        df = dataframe

        percent_drop = percentage of null values above which the column will be dropped

            as decimal between 0 and 1

    Returns:

        df = df where columns above percent_drop are dropped.

    

    """

    to_drop = [column for column in df if (df[column].isnull().sum()/len(df) >= percent_drop)]



    print('Columns to drop: ' , (len(to_drop)))

    # Drop features 

    df = df.drop(columns=to_drop)

    print('Shape: ', df.shape)

    return df
transaction = drop_missing_values(raw_transaction, 0.01)

lst = transaction.columns.values

test = test[test.columns.intersection(lst)]



transaction.shape

test.shape
# info method to get quick description of the data

transaction.info()
# summary of numerical attributes

transaction.describe()
# checking missing values again

missing_values_table(transaction).head(10)
# find correlations to target = isFraud

corr_matrix = transaction.corr().abs()



print(corr_matrix['isFraud'].sort_values(ascending=False).head(20))
# Visualizing the correlation matrix

# Select upper triangle of correlation matrix



upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

sns.heatmap(upper)

plt.show();
#checking data types of columns

transaction.dtypes
# checking for categorical variables

transaction.select_dtypes('object').apply(pd.Series.nunique, axis=0)
y = transaction.isFraud

X = transaction.drop('isFraud', axis=1)
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='median')
categorical = X[['ProductCD', 'card4', 'card6']]

X = X.drop(['ProductCD', 'card4', 'card6'], axis=1)
imputer.fit(X)
X = imputer.transform(X)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.model_selection import cross_val_score

cross_val_score(lr, X_train, y_train, cv=5)
test = test.drop(['ProductCD', 'card4', 'card6'], axis=1)

test = imputer.transform(test)

test_ = lr.predict(test)
sub = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

sub['isFraud'] = test_

sub.to_csv('submission.csv')

sub.head()