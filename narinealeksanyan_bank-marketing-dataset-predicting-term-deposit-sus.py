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

df= pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')

df.head()
df.shape
df.describe()
df.info()
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')



df.hist(bins=20, figsize=(14,10), color='#E14906')

plt.show()
value_counts = df['deposit'].value_counts()



value_counts.plot.bar(title = 'Deposit value counts')
j_df = pd.DataFrame()



j_df['yes'] = df[df['deposit'] == 'yes']['job'].value_counts()

j_df['no'] = df[df['deposit'] == 'no']['job'].value_counts()



j_df.plot.bar(title = 'Job and deposit')
D_corr = df.corr().loc[['age', 'balance', 'day' ,'duration','campaign','pdays','previous'],

                      ['age', 'balance', 'day' ,'duration','campaign','pdays','previous']

                     ]
import seaborn as sns

plt.figure(figsize=(12, 8))

sns.heatmap(D_corr, annot=True)

plt.show()
sns.pairplot(df, size=2)

plt.show()
from sklearn.preprocessing import LabelEncoder



categorical_column = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',

                       'poutcome']



for i in categorical_column:

    le = LabelEncoder()

    df[i] = le.fit_transform(df[i])

print(df.head())
target = pd.get_dummies(df.deposit)

data = df.drop(['deposit'],axis = 'columns')
(X_train,

 X_test, 

 Y_train, 

 Y_test) = train_test_split(data,target,

                              test_size=0.3,

                              random_state=234)
print(X_train.shape,

 X_test.shape, 

 Y_train.shape, 

 Y_test.shape)
%%time

from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression(solver = 'lbfgs',n_jobs=-1)

logisticRegr.fit(X_train, Y_train)