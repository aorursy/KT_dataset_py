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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
ALdf=pd.read_csv('/kaggle/input/medicaid-2020-drug-util-al/Drug_Utilization_2020_-_Alabama.csv')

# ALdf is the dataframe for Alabamda Medicaid Drug Utilization Data (read from csv file)

ALdf=ALdf.dropna()   #dropping all NaN values
ALdf.shape  #number of rows and columns within dataframe
ALdf.dtypes  #datatypes
ALdf.size  #number of elements within Dataframe
ALdf. describe()  #quick high level statistics
ALdf.head(5)  # quick overview of dataframe showing first 5 rows 
ALdf.tail(5)  # quick overview of dataframe showing bottom 5 rows
# DATA ANALYSIS AT INDIVIDUAL DRUG LEVEL, e.g. TAMOXIFEN

ALdf[ALdf['Product Name']=='TAMOXIFEN']
# DATA ANALYSIS OF DRUGS WITH PRESCRIPTIONS GREATER THAN OR EQUAL TO 1000

ALdf[ALdf['Number of Prescriptions']>=1000]
# DATA ANALYSIS OF DRUGS WITH PRESCRIPTIONS GREATER THAN OR EQUAL TO 1000, SORTED IN DESCENDING ORDER

#NOTE: FOR SORTING BY ASCENDING ORDER, CHANGE ascending parameter below to True

ALdf[ALdf['Number of Prescriptions']>=1000].sort_values(by='Product Name', ascending=False)
# DATA ANALYSIS OF TOP 10 DRUGS WITH PRESCRIPTIONS GREATER THAN OR EQUAL TO 1000

# DATA VISUALIZATION:  DRUG PRODUCT NAME vs UNITS REIMBURSED

E=ALdf[ALdf['Number of Prescriptions']>=1000].sort_values(by='Product Name', ascending=False)[0:10]

E

x=E['Product Name']

y=E['Units Reimbursed']

plt.barh(x,y,color='red')  #plotting using MatplotLib

# DATA ANALYSIS OF TOP 10 DRUGS WITH PRESCRIPTIONS GREATER THAN OR EQUAL TO 1000

# DATA VISUALIZATION:  DRUG PRODUCT NAME vs NUMBER OF PRESCRIPTIONS

x1=E['Product Name']

y1=E['Number of Prescriptions']

plt.figure(figsize=(15,7))

sns.barplot(x1,y1,color='orange')   #plotting 
x1=E['Product Name']

y1=E['Number of Prescriptions']

plt.figure(figsize=(15,7))

sns.barplot(x1,y1)   #plotting with seaborn, without specifying color field (reverts to default colors of seaborn bar plot)
# DATA ANALYSIS- SORTING DATA BY TOTAL AMOUNT REIMBURSED (DESCENDING ORDER)

# FOR ASCENDING ORDER, SIMPLY CHANGE PARAMETER BELOW TO TRUE

ALdf.sort_values(by='Total Amount Reimbursed', ascending=False)
# DATA ANALYSIS- TOP 10 TOTAL AMOUNT REIMBURSED (DESCENDING ORDER)

# FOR ASCENDING ORDER, SIMPLY CHANGE PARAMETER BELOW TO TRUE



ALdf.sort_values(by='Total Amount Reimbursed', ascending=False)[0:10]
#Top 10  By Number of Prescriptions

ALdf.sort_values(by='Number of Prescriptions', ascending=False)[0:10]
#Top 10  By Number of Units Reimbursed

ALdf.sort_values(by='Units Reimbursed', ascending=False)[0:10]
#Top 10  By Medicaid Amount Reimbursed

ALdf.sort_values(by='Medicaid Amount Reimbursed', ascending=False)[0:10]
#Top 10  By Non-Medicaid Amount Reimbursed

ALdf.sort_values(by='Non Medicaid Amount Reimbursed', ascending=False)[0:10]
#Top 10  By Total Amount Reimbursed   #NOTE- Repeated again to flow consistently with the Medicaid/Non-Medicaid Amount analysis

ALdf.sort_values(by='Total Amount Reimbursed', ascending=False)[0:10]
# MULTI COLUMN CONDITIONAL ARGUMENTS/SLICING OF DATAFRAME
#DATA ANALYSIS OF DRUGS WITH PRESCRIPTIONS GREATER THAN 10000 AND 'MEDICAID AMOUNT REIMBURSED' GREATER THAN 200000

ALdf[(ALdf['Number of Prescriptions']>10000)& (ALdf['Medicaid Amount Reimbursed']>200000)]

#NOTE- SIMPLY CHANGE THE PARAMETERS WITHIN ABOVE CODE LINE TO DO OTHER FORMS OF ANALYSIS.
ALdf[(ALdf['Number of Prescriptions']>10000)& (ALdf['Medicaid Amount Reimbursed']>200000)].value_counts()

#COUNT ANALYSIS SHOWING 5 values (same as above table)