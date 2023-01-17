# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from scipy.stats import probplot # for a qqplot
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind 
import pylab #

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
kiva_loan = '../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv'
loans = pd.read_csv(kiva_loan)
loans.head()
loans.describe()
probplot(loans["loan_amount"], dist="norm", plot=pylab)
# if any of the borrowers are female, then count the group as female 
loans["single_gender"] = ''
x = 0
for i, row in loans.iterrows():
    g = row["borrower_genders"]
    if g!=g:
        gen = 'none'
    elif g.find('female') == 0:
        gen = g[0:6]
    else:
        gen = g[:4]
    row["single_gender"] = gen
    loans.at[i,'single_gender'] = gen
loans.head()    
female_loans = loans[loans["single_gender"] == 'female']['loan_amount']
male_loans = loans[loans["single_gender"] == 'male']['loan_amount']
ttest_ind(female_loans, male_loans,equal_var=False)   
print('mean female loans')
print(female_loans.mean())
print('mean male loans')
print(male_loans.mean())
plt.hist(female_loans, label='female',bins=10,log=True)
plt.hist(male_loans, label='male',bins=10,log=True)
plt.legend(loc='upper right')
plt.xlabel('gender')
plt.ylabel('loans amount')
plt.title('Loans to female borrowers')
plt.show()
plt.scatter(loans["date"],loans["loan_amount"])
plt.xlabel('date')
plt.ylabel('loans amount')
plt.title('Loans over time')
plt.show()
gender_freqTable = loans["single_gender"].value_counts()
gender_freqTable.head()
labels = list(gender_freqTable.index)
positionsForBars = list(range(len(labels)))
plt.bar(positionsForBars, gender_freqTable.values) # plot our bars
plt.xticks(positionsForBars, labels) # add lables
plt.title("Number of loans by Gender")
sns.countplot(loans["single_gender"]).set_title("Number of loans by Gender")