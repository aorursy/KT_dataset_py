# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Read Data into dataframe
diabetes_data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
## Check if data is imported
diabetes_data.head()
##Exploratory Analysis

diabetes_data.info()
## No missing values, all are either int / float
## Check Correlation btw the variables
## 1 by Pearson method 
corr_data = diabetes_data.corr(method='pearson')
## No two values are highly correalted(close to either -1 or 1)
## Glucose Level, BMI, Age, Pregnancies (in this order only) are the highest factors in determing outcome
## Adding heat map for same
ax  = sns.heatmap(corr_data, vmin = -1, vmax = 1, center = 0, square=True)
##Check for outlier values :

fig = plt.figure()
ax = plt.subplot(111)

sns.boxplot(diabetes_data.iloc[:,0])
sns.boxplot(diabetes_data.iloc[:,1])
sns.boxplot(diabetes_data.iloc[:,3])
sns.boxplot(diabetes_data.iloc[:,4])
sns.boxplot(diabetes_data.iloc[:,5])
sns.boxplot(diabetes_data.iloc[:,6])
sns.boxplot(diabetes_data.iloc[:,7])
independent_df = diabetes_data
dependent_df  = independent_df.iloc[:,-1]
independent_df = independent_df.iloc[:,0:7]

Q1 = diabetes_data.quantile(0.25)
Q2 = diabetes_data.quantile(0.75)
IQR = Q2 - Q1
diabetes_data = diabetes_data[~((diabetes_data < Q1- 1.5* IQR ) |  (diabetes_data > Q2+ 1.5* IQR )).any(axis=1)]
diabetes_data
## Adding Logistic Regression
X_train, X_test, Y_train, Y_test = train_test_split(independent_df, dependent_df, test_size=0.3, random_state = 42)

log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
log_reg
y_pred = log_reg.predict(X_test)
## Calculate Accuracy
log_reg.get_params()
log_reg.score(X_test, Y_test)
confus_matrix =  confusion_matrix(Y_test, y_pred)
confus_matrix
print(classification_report(Y_test, y_pred))
my_submission = pd.DataFrame({'Outcome': y_pred})
my_submission.to_csv('submission.csv')
