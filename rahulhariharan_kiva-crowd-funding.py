# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print("test run")
%matplotlib inline

# Any results you write to the current directory are saved as output.
kiva_loans = pd.read_csv("../input/kiva_loans.csv")
kiva_loans.head()
plt.scatter(kiva_loans['loan_amount'],kiva_loans['funded_amount'],marker='.')
plt.ylabel('funded amount')
plt.xlabel('loan amount')
plt.title('funded amount vs loan amount')
df = kiva_loans[['funded_amount','loan_amount','term_in_months','lender_count']]
df.head()
sns.pairplot(df)
loan_themes_by_region = pd.read_csv("../input/loan_themes_by_region.csv")
loan_themes_by_region.head()
result = loan_themes_by_region.groupby(['sector','Field Partner Name','country'])
result.count()
sns.countplot(x='Field Partner Name', data=loan_themes_by_region)