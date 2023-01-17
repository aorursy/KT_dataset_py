# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read in dataset - create Pandas Training & Test data objects to work with



train_df = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test_df = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

train_df.info(null_counts=True) #Shows each Column name, Non-Null Count and Object type for that Column
# Analyze dataset as needed to identify possible corrections (for clearly-incorrect entries), estimates 

# (for missing data i.e. null entries) as well as notable trends that may guide the type of predictor used.



train_df.describe(include='all')
train_df[['item_price', 'item_cnt_day']].groupby(['item_price'], as_index=False).mean().sort_values(by='item_cnt_day',ascending=False)
# Prepare dataset as needed; combine Training & Test data if similar formatting changes needed

#ENSURE CATEGORICAL COLUMNS CONVERTED TO NUMERICAL (OR DROPPED IF NOT NEEDED), USING ONE-HOT OR OTHER ENCODING METHOD!!!

import datetime



#combine = [train_df, test_df]



# Use copy of Training data to adjust/convert/etc. so that initial dataset remains intact

train_df_final = train_df.copy()



# Convert 'date' Column into numerical values (Try to find conversion to int equivalent, like MS Excel). Consider dropping 'date_block_num' if not relevant to sales prediction. 

int_date = pd.to_datetime(train_df['date'], format='%d.%m.%Y')

train_df_final['date'] = int_date.astype(int)



train_df_final.head()

#train_df_final.tail()

#train_df_final.info(null_counts=True) #Shows each Column name, Non-Null Count and Object type for that Column

#train_df_final.describe() #Shows Count/Mean/STD/Quartile/Min-Max data for Numeric Columns

#train_df_final.shape #Shows Row-Column counts
# Choose model (or group of models) to test



from sklearn.linear_model import LogisticRegression

#from sklearn.svm import SVC, LinearSVC

#from sklearn.ensemble import RandomForestClassifier

#from sklearn.neighbors import KNeighborsClassifier

#from sklearn.naive_bayes import GaussianNB

#from sklearn.linear_model import Perceptron

#from sklearn.linear_model import SGDClassifier

#from sklearn.tree import DecisionTreeClassifier



from sklearn.model_selection import train_test_split



#Set up Training & Test data objects to use for training & testing, including making training/validation splits



X_train, X_val, y_train, y_val = train_test_split(train_df_final.drop("item_cnt_day", axis=1), train_df_final["item_cnt_day"], random_state=1)

X_test = test_df.copy()



logreg = LogisticRegression()

# Fit model (if doing supervised learning)



#logreg.fit(X_train, y_train)

# Make predictions



#y_pred = logreg.predict(X_val)

# Measure model accuracy



from sklearn.metrics import mean_absolute_error



#mae = mean_absolute_error(y_pred, y_val)

#print(mae)



#acc_log = round(logreg.score(X_train, y_train) * 100, 2)



#acc_log
# Look for ways to improve accuracy w/o overfitting
