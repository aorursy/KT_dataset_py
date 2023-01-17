import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn import tree

%matplotlib inline
def maybe_load_loan_data(threshold=1, path='../input/loan.csv', force='n'):

    def load_data():

        data = pd.read_csv(path, low_memory=False)

        t = len(data) / threshold

        data = data.dropna(thresh=t, axis=1) # Drop any column with more than 50% missing values

        return data



    # conditionally load the data

    try:

        if df.empty or force=='y':

            data = load_data()

        else:

            return df

    except:

        data = load_data()



    return data



df = maybe_load_loan_data(2)
df.columns
def show_stats(df):

    print ("Number of records {}".format(len(df)))

    print ("Dataset Shape {}".format(df.shape))



    sns.distplot(df['loan_amnt'].astype(int))

        

show_stats(df)
# Understand data correlations

numeric_features = df.select_dtypes(include=[np.number])

print(numeric_features.describe())



categoricals = df.select_dtypes(exclude=[np.number])

print(categoricals.describe())



corr = numeric_features.corr()



print (corr['loan_amnt'].sort_values(ascending=False)[:10], '\n')

print (corr['loan_amnt'].sort_values(ascending=False)[-10:])



''' move this to model evaluation section

from sklearn.metrics import confusion_matrix



confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print(confmat)'''

def show_dictionary(path='../input/LCDataDictionary.xlsx'):

    data_dictionary = pd.read_excel(path)



    print(data_dictionary.shape[0])

    print(data_dictionary.columns.tolist())



    data_dictionary.rename(columns={'Name': 'name',

                                    'Description': 'description'})

    return data_dictionary



dict = show_dictionary()

dict.set_index('LoanStatNew', inplace=True)

dict.loc[:]
dict[categoricals]







from pandas.tools.plotting import scatter_matrix



attributes = ['annual_inc','loan_amnt', 'revol_util', 'dti','open_acc','revol_bal','revol_util','total_rec_int' ]

#              'recoveries','acc_now_delinq','delinq_2yrs','emp_length','int_rate','funded_amnt'



scatter_matrix(df[attributes], figsize=(12,8))




def print_data_shape(df):

    print ("No rows: {}".format(df.shape[0]))

    print ("No cols: {}".format(df.shape[1]))

    print (df.head(1).values)

    print ("Columns: " + df.columns)
def proc_emp_length():

    df.replace('n/a', np.nan, inplace=True)

    df.emp_length.fillna(value=0, inplace=True)

    df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)

    df['emp_length'] = df['emp_length'].astype(int)

    #df.emp_length.head()
df.revol_bal.head()

#df.revol_util = pd.Series(df.revol_util).str.replace('%', '').astype(float)
print (df.emp_title.value_counts().head())

print (df.emp_title.value_counts().tail())

df.emp_title.unique().shape
df.verification_status.value_counts()
def proc_desc_len():

    df['desc_lenght'] = df['desc'].fillna(0).str.len()



#df.desc_lenght
def proc_issue_d():

    df['issue_month'], df['issue_year'] = zip(*df.issue_d.str.split('-'))

    df.drop(['issue_d'], 1, inplace=True)
def proc_zip_code():

    df['zip_code'] = df['zip_code'].str.rstrip('x')
print (df.purpose.value_counts())

print ('')

print (df.title.value_counts().head())
#df = maybe_load_loan_data(threshold=2)

df.plot(kind='barh', x='purpose', y='int_rate')
print_data_shape(df)
def proc_loan_status(df):

    #mapping_dict = {'loan_status':{'Fully Paid':0, 'Charged Off': 1, 'Default': 1, 'Current': 0}}

    mapping_dict = {'loan_status':{'Fully Paid':0, 'Charged Off': 1}}

    df = df.replace(mapping_dict)

    df = df[(df['loan_status'] == 1) | (df['loan_status'] == 0)]

    return df

def show_nulls(df):

    nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:25])

    nulls.columns = ['Null Count']   

    nulls.index.name = 'Feature'

    return nulls


















































