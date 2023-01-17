# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

frame = pd.read_csv('../input/loan.csv', low_memory=False)

# Any results you write to the current directory are saved as output.
def make_bar(attr, title, ylabel):

    heights = frame[attr].value_counts().tolist()

    names = []

    for k, v in frame[attr].value_counts().items():

        names.append(k)

        

    for ii, height in enumerate(heights):

        color = np.random.random_sample(3)

        plt.bar(ii, height, color=color)

        

    plt.title(title)

    plt.ylabel(ylabel)

    plt.gca().set_xticklabels(names)

    plt.gca().set_xticks(np.arange(len(names)) + .4)

    if len(names) > 5:

        plt.xticks(rotation=90)

    plt.show()

    

    
make_bar('emp_length', 'Length of Employment of Borrowers', 'Borrowers')

make_bar('grade', 'Grades of Loans', 'Loans')

make_bar('term', 'Terms of Loans', 'Loans')

make_bar('purpose', 'Purpose of Loans', 'Loans')

make_bar('loan_status', 'Loan Statuses', 'Loans')

make_bar('application_type', 'Application Types', 'Loans')
plt.hist(frame['int_rate'], bins=30)

plt.title('Distribution of Interest Rates')

plt.xlabel("Interest Rates")

plt.show()



plt.hist(frame['loan_amnt'], bins=15)

plt.title('Distribution of Loan Amounts')

plt.xlabel("Loan Amounts")

plt.show()



plt.hist(frame['installment'], bins=15)

plt.title('Distribution of Installments')

plt.xlabel("Installments")

plt.show()
sns.boxplot(x='grade', y='int_rate', data=frame, order = 'ABCDEFG')
sns.boxplot(x='grade', y='loan_amnt', data=frame, order = 'ABCDEFG')
sns.boxplot(x='term', y='int_rate', data=frame)
sns.boxplot(x='emp_length', y='int_rate', data=frame)

plt.xticks(rotation=50)
sns.boxplot(x='purpose', y='int_rate', data=frame)

plt.xticks(rotation=90)
sns.boxplot(x='application_type', y='int_rate', data=frame)

plt.xticks(rotation=90)
sns.jointplot(x='loan_amnt', y='int_rate', data=frame)
sns.jointplot(x='installment', y='int_rate', data=frame)