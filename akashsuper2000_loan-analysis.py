import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings('ignore')
loan = pd.read_csv('../input/lending-club-loan-data/loan.csv',dtype='object')
loan = loan[['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'disbursement_method', 'debt_settlement_flag']]



print(loan.shape)
loan.head()
loan.describe()
loan.isnull().values.any()
def univariate(df,col,vartype,hue =None): 

    '''

    Univariate function will plot the graphs based on the parameters.

    df      : dataframe name

    col     : Column name

    vartype : variable type : continuos or categorical

                Continuos(0)   : Distribution, Violin & Boxplot will be plotted.

                Categorical(1) : Countplot will be plotted.

    hue     : It's only applicable for categorical analysis.

    

    '''

    sns.set(style="darkgrid")

    

    if vartype == 0:

        fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(20,8))

        ax[0].set_title("Distribution Plot")

        sns.distplot(df[col],ax=ax[0])

    

    if vartype == 1:

        temp = pd.Series(data = hue)

        fig, ax = plt.subplots()

        width = len(df[col].unique()) + 6 + 4*len(temp.unique())

        fig.set_size_inches(width , 7)

        ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue) 

        if len(temp.unique()) > 0:

            for p in ax.patches:

                ax.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(loan))), (p.get_x()+0.05, p.get_height()+20))  

        else:

            for p in ax.patches:

                ax.annotate(p.get_height(), (p.get_x()+0.32, p.get_height()+20)) 

        del temp

    else:

        exit

        

    plt.show()
univariate(df=loan,col='loan_amnt',vartype=0)
univariate(df=loan,col='int_rate',vartype=0)