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
import pandas as pd

import numpy as np

import seaborn as ss

import matplotlib.pyplot as plt
column=["status","duration(months)","credit_history","purpose","credit_amount","saving_account","employment_since","installment_rate","status/sex","guarantors","residence_since","property","age","installment_plans","housing","credits","job","liable people","telephone","foreig_worker","label"]

df = pd.read_csv("/kaggle/input/german.data",sep=r"\s+",names=column)
df.head()
job = df.job.loc[df.label == 1]

job1= df.job.loc[df.label == 2]

plt.hist([job,job1],label=['good', 'bad'])

plt.legend(loc='upper right')

y_pos = np.arange(5)

plt.xticks(y_pos,['unemployed','unskilled ','skilled employee','management','highly qualified'])

plt.title("Job")



#Property Distribution among labels
status = df.status.loc[df.label == 1]

status1 = df.status.loc[df.label == 2]

plt.hist([status,status1],label=['good', 'bad'])

plt.legend(loc='upper right')

plt.title("Status Graph")

plt.show()

#Status Distribution among labels

#there is more probablity that if status of checking amount

# is none
from scipy.stats import wilcoxon

data1= df.query('status == "A14"')

#data2 = df.label.loc[df.status == 'A14']

#stat, p = wilcoxon(data1.status, data1.label)

#print(stat,p)

data = df;

from sklearn.preprocessing import LabelEncoder



#Auto encodes any dataframe column of type category or object.

# cleanup_nums = {"status":     {"A11": 1, "A12": 2,"A13":3,"A14":4},

#                "saving_account": {"A61": 1, "A62": 2, "A63":3 , "A64": 4,

#                                   "A65": 5, }}



cleanup_nums = { "credit_history": {"A30":1,"A31":1,"A32":2,"A33":3,"A34":4}}

data.replace(cleanup_nums, inplace=True)

# data.head(10)

stat, p = wilcoxon(data.credit_history, data.credit_amount)

print("credit_history - credit_amount ")

print(stat,p)

stat, p = wilcoxon(data.credit_amount, data.credits)

print("credit_amount - credits ")

print(stat,p)

stat, p = wilcoxon(data.credits, data.credit_history)

print(" credit - credit_history")

print(stat,p)


df.query(' status == "A14"').count()
credit = df.credit_history.loc[df.label == 1]

credit1= df.credit_history.loc[df.label == 2]

plt.hist([credit,credit1],label=['good', 'bad'])

plt.legend(loc='upper right')

# y_pos = np.arange(5)

# plt.xticks(y_pos,['no/all credits','paid back dully ','no credits','delay in paying','critical account'])

plt.title("Credit History")

plt.show()



#Credit History among labels

#no DAta for class A33
foreign = df.foreig_worker.loc[df.label == 1]

foreign1= df.foreig_worker.loc[df.label == 2]

plt.hist([foreign,foreign1],2,label=['good', 'bad'])

plt.legend(loc='upper right')

y_pos = np.arange(2)

plt.xticks(y_pos,['Yes','No'])

plt.title("Credit History")

plt.show()

#if you are non foreign worker, there is more chance for allocation of loan
import pandas_profiling as pf
#pf.ProfileReport(df)