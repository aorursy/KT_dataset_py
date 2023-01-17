# Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns

import os
print(os.listdir("../input"))

import warnings 
warnings.filterwarnings('ignore')

%matplotlib inline
sns.set_style('darkgrid')
plt.rcParams['figure.figsize']= (12, 6)
kiva_loans= pd.read_csv('../input/kiva_loans.csv')
loan_theme_region= pd.read_csv('../input/loan_themes_by_region.csv')
kiva_mpi_region= pd.read_csv('../input/kiva_mpi_region_locations.csv')
loan_theme_id= pd.read_csv('../input/loan_theme_ids.csv')
print('The shape of Kiva Loans dataset: {}' .format(kiva_loans.shape))
print('The shape of Loan theme by Regions dataset: {}' .format(loan_theme_region.shape))
print('The shape of Kiva Mpi region locations dataset: {}' .format(kiva_mpi_region.shape))
print('The shape of Loan theme by ids dataset: {}' .format(loan_theme_id.shape))
kiva_loans.head()
loan_theme_region.head()
kiva_mpi_region.head()
loan_theme_id.head()
print('The Statistical description of the continuos data of Kiva Loans: ')
kiva_loans.describe(exclude= ['O'])
print('The Statistical description of the categorical data of Kiva Loans: ')
kiva_loans.describe(include= ['O'])
print('The Statistical description of the continuous data of loan theme by regions: ')
loan_theme_region.describe(exclude= ['O'])
print('The statistical description of categorical data of loan theme by regions: ')
loan_theme_region.describe(include= ['O'])
print('The statistical description of continuous data of kiva mpi region locations: ')
kiva_mpi_region.describe(include= ['O'])
print('The Statistical description of categorical data of kiva mpi region locations: ')
kiva_mpi_region.describe(include= ['O'])
print('The Statistical description of continuous data of loan theme by ids: ')
loan_theme_id.describe(exclude= ['O'])
print('The Statistical description of categorical data of loan theme by ids: ')
loan_theme_id.describe(include= ['O'])
print('Kiva loans dataset: ')
null_values_kiva_loan= kiva_loans.isnull().sum().sort_values(ascending= False)
null_pct_kiva_loan= (kiva_loans.isnull().sum()/kiva_loans.shape[0] * 100).sort_values(ascending= False)
null_kiva_loan_df= pd.DataFrame({
    'Missing Values': null_values_kiva_loan,
    'Missing Values Percent': null_pct_kiva_loan
})

null_kiva_loan_df
print('Loan theme by Region dataset: ')
null_values_loan_theme_region= loan_theme_region.isnull().sum().sort_values(ascending= False)
null_pct_loan_theme_region= (loan_theme_region.isnull().sum()/loan_theme_region.shape[0] * 100).sort_values(ascending= False)
null_loan_theme_region_df= pd.DataFrame({
    'Missing Values': null_values_loan_theme_region,
    'Missing Values Percent': null_pct_loan_theme_region
})

null_loan_theme_region_df
print('Kiva mpi region locations dataset: ')
null_kiva_mpi_region= kiva_mpi_region.isnull().sum().sort_values(ascending= False)
null_pct_kiva_mpi_region= (kiva_mpi_region.isnull().sum()/kiva_mpi_region.shape[0] * 100).sort_values(ascending= False)
null_kiva_mpi_region_df= pd.DataFrame({
    'Missing Values': null_kiva_mpi_region,
    'Missing Values Percent': null_pct_kiva_mpi_region
})

null_kiva_mpi_region_df
print('Loan theme ids dataset: ')
null_loan_theme_id= loan_theme_id.isnull().sum().sort_values(ascending= False)
null_pct_loan_theme_id= (loan_theme_id.isnull().sum()/loan_theme_id.shape[0] * 100).sort_values(ascending= False)
null_loan_theme_id_df= pd.DataFrame({
    'Missing Values': null_loan_theme_id,
    'Missing Values Percent': null_pct_loan_theme_id
})

null_loan_theme_id_df
