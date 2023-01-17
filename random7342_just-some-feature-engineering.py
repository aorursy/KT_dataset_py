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
import h2o

#connecting to cluster

h2o.init(strict_version_check=False)
data_csv = "/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv"

data = h2o.import_file(data_csv)
data.describe()
data.rename(columns={"PAY_0": "PAY_1"}) #for consistency

data.rename(columns={'default.payment.next.month': "DEFAULT"}) #easier



cols_names = data.columns #because we know the data type for all the columns (they are all ints)

cols_names
not_categorical = ['ID',

 'LIMIT_BAL',

  'AGE',

 'BILL_AMT1',

 'BILL_AMT2',

 'BILL_AMT3',

 'BILL_AMT4',

 'BILL_AMT5',

 'BILL_AMT6',

 'PAY_AMT1',

 'PAY_AMT2',

 'PAY_AMT3',

 'PAY_AMT4',

 'PAY_AMT5',

 'PAY_AMT6']



target = "DEFAULT"



categorical = [item for item in cols_names if item not in not_categorical and item != target]

categorical
data.head()
#Onehot encoding (as labels are already encoded as numbers)



data_onehot = pd.get_dummies(data.as_data_frame(), columns=categorical)

data_onehot.head()
#Drop the ID column



data_onehot = data_onehot.drop(columns=['ID'])
data_onehot.columns
#Creating equally sized bins for age - 5 categories



print(data_onehot['AGE'].describe())



#add age bins to make it all-inclusive - in case new data may come



data_onehot['AGE_BINS'] = pd.qcut(data_onehot['AGE'], 5)



#Add age bins for ages (0, 20.999] and (79.0, ) - even though there may be no data for this in the present dataset, it is important to do this in case we have future data



data_onehot['AGE_BINS_(0, 20.999]'] = 0 #in the same format as after one hot encoding (doing this two cells later)

data_onehot['AGE_BINS_(79.0, )'] = 0

data_onehot.head() #it works!
#Now we use one hot encoding for these categories



data_age = pd.get_dummies(data_onehot, columns=['AGE_BINS'])

data_age = data_age.drop(columns=['AGE'])

data_age.head()
#some statistical featurs



bill_amt_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']



#mean of Bill_amt and Pay_amt, max, min, std, var



data_age['BILL_AMT_MEAN'] = data_age[bill_amt_cols].mean(axis=1)

data_age['PAY_AMT_MEAN'] = data_age[pay_amt_cols].mean(axis=1)



data_age['BILL_AMT_MAX'] = data_age[bill_amt_cols].max(axis=1)

data_age['PAY_AMT_MAX'] = data_age[pay_amt_cols].max(axis=1)



data_age['BILL_AMT_MIN'] = data_age[bill_amt_cols].min(axis=1)

data_age['PAY_AMT_MIN'] = data_age[pay_amt_cols].min(axis=1)



data_age['BILL_AMT_MED'] = data_age[bill_amt_cols].median(axis=1)

data_age['PAY_AMT_MED'] = data_age[pay_amt_cols].median(axis=1)



data_age['BILL_AMT_STD'] = data_age[bill_amt_cols].std(axis=1)

data_age['PAY_AMT_STD'] = data_age[pay_amt_cols].std(axis=1)



data_age['BILL_AMT_VAR'] = data_age[bill_amt_cols].var(axis=1)

data_age['PAY_AMT_VAR'] = data_age[pay_amt_cols].var(axis=1)





data_age.head()
#some new variables



#payment fraction of bill statement

for i in range(1, 7):        

    data_age['PAY_FRAC_' + str(i)] = data_age[pay_amt_cols[i-1]] / data_age[bill_amt_cols[i-1]]

data_age = data_age.fillna(0)





#fraction of credit limit used (bill_amt / limit_bal)

for i in range(1, 7):        

    data_age['USED_CREDIT' + str(i)] = data_age[bill_amt_cols[i-1]] / data_age['LIMIT_BAL']

data_age = data_age.fillna(0)





data_age.head()
data_age['PAY_FRAC_1'].max()







#There are 540. Three simple ways to deal: delete feature, delete rows, set to zero. Have to test.



#Setting to zero



for i in range (1, 7):

    #print(len(data_age[data_age['PAY_FRAC_' + str(i)] == np.inf])) #0 of them are -np.inf

    data_age['PAY_FRAC_' + str(i)] = data_age['PAY_FRAC_' + str(i)].replace({np.inf: 0})

    #print(len(data_age[data_age['PAY_FRAC_' + str(i)] == np.inf]))
#Scaling



#Using standard scalar scaling

#Multiple methods such as min-max scaling, standard scaling, etc. All have different advantages and depend on the distribution of data.

#Can always change this in the next iterations of the ML pipeline. Trial and error process.



from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



scaled_features = data_age.copy()



col_names = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4' ,'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4' ,'PAY_AMT5', 'PAY_AMT6', 'PAY_FRAC_1', 'PAY_FRAC_2', 'PAY_FRAC_3', 'PAY_FRAC_4', 'PAY_FRAC_5', 'PAY_FRAC_6', 'USED_CREDIT1', 'USED_CREDIT2', 'USED_CREDIT3', 'USED_CREDIT4', 'USED_CREDIT5', 'USED_CREDIT6']

features = scaled_features[col_names]

scaler = StandardScaler().fit(features.values)

features = scaler.transform(features.values)



scaled_features[col_names] = features

scaled_features
scaled_df = pd.DataFrame(scaled_features, columns=['LIMIT_BAL', 'BILL_AMT1', 'PAY_AMT1', 'USED_CREDIT1'])



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))



ax1.set_title('Before Scaling')

sns.kdeplot(data_age['LIMIT_BAL'], ax=ax1) #kernel density estimate plot (non-parametric way to estimate the probability density function of a random variable.)

sns.kdeplot(data_age['BILL_AMT1'], ax=ax1)

sns.kdeplot(data_age['PAY_AMT1'], ax=ax1)

sns.kdeplot(data_age['USED_CREDIT1'], ax=ax1)

ax2.set_title('After Standard Scaler')

sns.kdeplot(scaled_df['LIMIT_BAL'], ax=ax2)

sns.kdeplot(scaled_df['BILL_AMT1'], ax=ax2)

sns.kdeplot(scaled_df['PAY_AMT1'], ax=ax2)

sns.kdeplot(scaled_df['USED_CREDIT1'], ax=ax2)

plt.show()
scaled_features.columns