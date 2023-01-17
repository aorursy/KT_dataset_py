# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

loan = pd.read_csv('../input/loan.csv')

#loan.info()
loan=loan[loan.loan_status!='Current']
c=Counter(list(loan.loan_status))
mmp={x[0]:1 for x in c.most_common(20)}
mmp['Fully Paid']=0
mmp['Does not meet the credit policy. Status:Fully Paid']=0
mmp['Issued']=0
loan['target']=loan['loan_status'].map(mmp)

cl2=['term','grade','sub_grade','purpose']

n=1
for i in cl2:
    plt.subplot(2,2,n)
    pd.pivot_table(loan, values='target', index=i).plot(kind='barh',alpha=0.5, figsize=(15, 10))
    n+=1    



# Any results you write to the current directory are saved as output.
