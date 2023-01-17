# Introductory Framework
## The Home Credit Datasets are characterized with high dimensionality, where attributes reach up to 120 features in
## training (excl. target and ID) and testing datasets (excl. ID). Therefore, we exclude decision trees method from the machine
## learning model as Lender needs to merge these data from different sources together and explore how different attributes
## correlate or help predicting the target label. An easy way to do data integration in SQL is using joins to connect tables
## by linking primary keys with foreign ones and, thus, link datasets together and perform a process on. Also, we will
## reduce dimensions by removing highly-correlated features and then apply logistic regression predictive model. As the
## prototype suggests, a default score needs to be produced in a quantitative manner 0.0-1.0 to replace the current target
## where summary statistics are described in either 0's (loan repayments) and 1's (loan defaults).

## Note: numbers in the annotations of this code are rounded to its closest integer

# Preparing modules/libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # Input data files are available in the "../input/" directory.

from numpy.random import randn
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from pylab import rcParams

import scipy
from scipy.stats import spearmanr
from scipy.stats.stats import pearsonr

import seaborn as sb
import sklearn
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

## Setting the plotting parameters
%matplotlib inline
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

## Importing the Training Dataset
add = 'C:/Users/UP821483/Documents/MOOCs/Kaggle/Home Credit Default competition/application_train.csv'
train = pd.read_csv(add)

## Explotatoty Analysis
train.head()
train['Target'].value_counts() # current default rate is 8% in training dataset (24825 / (24825 + 282686))
train['TARGET'].plot(kind='hist')
train.describe() # notice that produced table contains 106 out of the 120 features -> (14 non-numeric, categorical)

# Preprocessing
## Missing Values

df = pd.DataFrame(np.zeros(244), index=['No_of_Missing','Pctg_of_Missing'], columns = train.columns)
for i in len(train.columns):
    missing = train[:,i].isnull()
    df[1,i] = missing.sum()
    df[2,i] = 100* (df[1,i] / len(train) )
df ##deciding on which feature to fill with forward filling method and which feature to eliminate as shown in the next step

# Dimensionality Reduction
## for example, let's try the attribute 'OWN_CAR_AGE'
missing_carage = train['OWN_CAR_AGE'].isnull()
no_of_missing = missing_carage.sum()
from __future__ import division ## for python 2.7
pctg_of_missing = 100*(no_of_missing / len(train))
## note that 66% of the data is missing within this feature. Therefore, this feature is better off execluded
del train['OWN_CAR_AGE']

## removing noise/redundancy
correlations = train.corr()
correlations
## by examining correlations table, remove any of 2 attributes that have Correlation Coefficient > 0.7
## for example, feature 'AMT_GOODS_PRICE' and 'AMT_CREDIT' are 99% correlated, so delete one of them
del train['AMT_GOODS_PRICE']
## Let's assume, after preprocessing, we selected 3 features, for simplicity, keeping in mind: independent predictants
new_train = train[['CNT_CHILDREN','AMT_CREDIT','DAYS_EMPLOYED']].values

# Model
## Setting Variables
x = scale(new_train)
y = train[['TARGET']].values

## Fitting Logistic regression on 'TARGET' binary variable
R2 = LogisticRegression()
R2.fit(x,y)

# Results
R2.score(x,y)
## 92%

y_pred = R2.predict(x)
print(classification_report(y, y_pred))
# Conclusion
## The model achieved a precision of 85% with recall of 92% had a support of 307511 although 120 features were reduced to 3.
## In future, we will use the entropy function in R to select more features with maximum information gain
