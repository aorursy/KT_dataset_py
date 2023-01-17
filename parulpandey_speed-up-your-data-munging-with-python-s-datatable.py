# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Install the datatable package : https://github.com/h2oai/datatable/blob/master/docs/index.rst

!pip install datatable # Turn 'ON'the internet option to install the latest version
#Import the datatable package

import datatable as dt

print(dt.__version__)
col_acq = ['LoanID','Channel','SellerName','OrInterestRate','OrUnpaidPrinc','OrLoanTerm','OrDate','FirstPayment','OrLTV','OrCLTV','NumBorrow','DTIRat','CreditScore','FTHomeBuyer','LoanPurpose','PropertyType','NumUnits','OccStatus','PropertyState','Zip','MortInsPerc','ProductType','CoCreditScore','MortInsType','RelocationMort']



col_per = ['LoanID','MonthRep','Servicer','CurrInterestRate','CAUPB','LoanAge','MonthsToMaturity','AdMonthsToMaturity','MaturityDate','MSA','CLDS','ModFlag','ZeroBalCode','ZeroBalDate','LastInstallDate','ForeclosureDate','DispositionDate','ForeclosureCosts','PPRC','AssetRecCost','MHRC','ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP','OFP','NIBUPB','PFUPB','RMWPF',  'FPWA','SERVICING ACTIVITY INDICATOR']
# Reading the data into a Frame object



df_acq = dt.fread('../input/Acquisition_2014Q3.txt',columns=col_acq)

df_per = dt.fread('../input/Performance_2014Q3.txt', columns=col_per)
print(df_acq.shape)

print(df_per.shape)

df_acq.head() 
df_per.head(5)
# Selecting only the LoanID and the ForeclosureDate column and discarding the rest

df_per = df_per[:,['LoanID','ForeclosureDate']]

df_per.head()
# Displaying only the unique Loan IDs in the Performance dataset

dt.unique(df_per[:,"LoanID"]).head(5)
# Filtering

df_per = df_per[-1:,:, dt.by(dt.f.LoanID)]

df_per.head(5)
df_per.names = ['LoanID','Will_Default']

df_per.key = 'LoanID'

df= df_acq[:,:,dt.join(df_per)]
# logical types

df[:,'Will_Default'].ltypes
# Grouping by the 'Will Deafult' column

df[1:,:, dt.by(dt.f.Will_Default)].head(5)
# Replacing the dates in the Will_Default column with '0' and null values with 1

df[:,'Will_Default'] = df[:, {'Will_Default': dt.f['Will_Default']==""}]

df.head(5)
df.shape
#df.to_pandas()

#df.to_csv("out.csv")

#df.to_jay("data.jay")