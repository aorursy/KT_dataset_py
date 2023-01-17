# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Modeling

import lightgbm as lgb



MAX_EVALS = 500

N_FOLDS = 10



import matplotlib.pyplot as plt

import pandas as pd 

import numpy as np

import seaborn as sns

sns.set(style="whitegrid")

np.random.seed(203)



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split



from matplotlib import pyplot as plt

from timeit import default_timer as timer



#Suppress warnings from pandas

import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')



# Memory management

import gc 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



%matplotlib inline



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
train_df.head()
test_df = pd.read_csv('../input/test.csv')
test_df.head()
# Function to calculate missing values by column# Funct 

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
# Missing values statistics

missing_values = missing_values_table(train_df)

missing_values.head(20)
# Missing values statistics

missing_values = missing_values_table(test_df)

missing_values.head(20)
# Number of each type of column

train_df.dtypes.value_counts()
X_train, y_train = train_test_split(train_df, test_size=0.2)



# Extract the labels and format properly

train_labels = np.array(X_train['target'].astype(np.int32)).reshape((-1,))

test_labels = np.array(y_train['target'].astype(np.int32)).reshape((-1,))



# Drop the unneeded columns

train = X_train.drop(columns = ['ID_code', 'target'])

test = y_train.drop(columns = ['ID_code','target'])



# Convert to numpy array for splitting in cross validation

features = np.array(train)

test_features = np.array(test)

labels = train_labels[:]



print('Train shape: ', train.shape)

print('Test shape: ', test.shape)

train.head()
plt.hist(labels, edgecolor = 'k'); 

plt.xlabel('Label'); plt.ylabel('Count'); plt.title('Counts of Labels');
# Model with default hyperparameters

model = lgb.LGBMClassifier()

model
start = timer()

model.fit(features, labels)

train_time = timer() - start



predictions = model.predict_proba(test_features)[:, 1]

auc = roc_auc_score(test_labels, predictions)



print('The baseline score on the test set is {:.4f}.'.format(auc))

print('The baseline training time is {:.4f} seconds'.format(train_time))