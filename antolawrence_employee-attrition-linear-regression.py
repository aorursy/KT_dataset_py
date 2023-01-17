your_local_path="../input/"
# read the data and set the datetime as the index

import pandas as pd

url = your_local_path+'HR_comma_sep.csv'

attrition = pd.read_csv(url)
#bikes.describe

attrition.head()

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (8, 6)

plt.rcParams['font.size'] = 14
attrition.plot(kind='bar', x='satisfaction_level', y='left', alpha=0.2)
attrition.plot(kind='scatter', x='average_montly_hours', y='last_evaluation', alpha=0.2)
sns.lmplot(x='average_montly_hours', y='last_evaluation', data=attrition, aspect=1.5, scatter_kws={'alpha':0.2})
attrition[['satisfaction_level','average_montly_hours']].boxplot()

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LinearRegression

import numpy as np

# define a function that accepts a list of features and returns testing RMSE

def train_test_rmse(feature_cols):

    X = attrition[feature_cols]

    y = attrition.left

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    linreg = LinearRegression()

    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print (train_test_rmse(['average_montly_hours', 'last_evaluation']))

print (train_test_rmse(['average_montly_hours', 'last_evaluation', 'satisfaction_level']))





attrition_new = attrition

attrition_new.drop('sales', axis=1, inplace=True)

attrition_new.drop('salary', axis=1, inplace=True)

df_ = pd.DataFrame(columns=('Features','RMSE'))

df_ = df_.fillna(0)

#df = DataFrame(columns=('Features', 'RMSE'))
list(attrition)

print (train_test_rmse(['satisfaction_level']))



import itertools



attrition_new = attrition

#attrition_new.drop('sales', axis=1, inplace=True)

#attrition_new.drop('salary', axis=1, inplace=True)



stuff = list(attrition_new)

i = 0

#for L in range(0, len(stuff)+1):

for L in range(1, 2):

    for subset in itertools.combinations(stuff, L):

         new = list(subset)         

         #print (new)     

         df_.loc[i] = [str(new), train_test_rmse(new)]

         i = i + 1

         print ("RMSE for ", str(new), train_test_rmse(new))

    
