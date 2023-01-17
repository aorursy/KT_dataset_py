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
# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Loading data into dataframe
df = pd.read_csv('../input/caravan-insurance-challenge/caravan-insurance-challenge.csv',engine= 'python')
df.head()
#Having a look at the shape and datatypes of columns of dataframe
df.info()
#Checking the buying rate
buy = sum(df['CARAVAN'])/len(df.index)*100
buy
#Looking at the statistical summary
df.describe()
#Looking at the train and test data count
df['ORIGIN'].value_counts()

#Increasing the number of columns in display
pd.set_option('display.max_columns',100)
# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
df.describe(percentiles=[.25, .5, .75, .90, .95, .99])
#correlation matrix
plt.figure(figsize=(30,20))
sns.heatmap(df.corr(),annot=True)
#Loading training data in df_train and test data in df_test
df_train = df.loc[df['ORIGIN'] == 'train']
X_train = df_train.drop(['ORIGIN','CARAVAN'], axis=1)
y_train = df_train['CARAVAN']

df_test = df.loc[df['ORIGIN'] == 'test']
X_test = df_test.drop(['ORIGIN','CARAVAN'], axis=1)
y_test = df_test['CARAVAN']
#Applying MinMax Scaling on the training data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
X_train.head()
X_test.head()
from sklearn.decomposition import PCA
pca = PCA(0.9) #90% variance is passed as argument
df_train_pca2 = pca.fit_transform(X_train) #fitting the Training data
df_train_pca2.shape
#Following this up, we will model the Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model_pca = lr.fit(df_train_pca2, y_train) #Fitting on train dataset
df_test_pca2 = pca.transform(X_test) 
df_test_pca2.shape
#Finding the probablities associated with each data point
pred_probs_test2 = model_pca.predict_proba(df_test_pca2)[:,1]
#Calculating the roc_auc_score for the same model
from sklearn.metrics import roc_auc_score
"{:2.2}".format(roc_auc_score(y_test, pred_probs_test2))