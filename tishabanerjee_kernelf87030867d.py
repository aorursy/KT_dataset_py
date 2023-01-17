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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Loading the Dataset
url = ("../input/creditcardfraud/creditcard.csv")
credit_card_data = pd.read_csv(url)
#Looking at the first four rows of the data
credit_card_data.head()

#Columns in the data
column_names = credit_card_data.columns
print(column_names)
#Getting the number of classes in the data
credit_card_data["Class"].value_counts()

credit_card_data["Class"].describe()
credit_card_data.describe()
credit_card_data.hist(bins=50,figsize=(20,15))
plt.show()
#Plotting a scatter matrix
from pandas.plotting import scatter_matrix
attributes = ['Class','Time', 'V1', 'V2', 'V3', 'V4']
scatter_matrix(credit_card_data[attributes],figsize=(20,10))
plt.show()
#Visualing the correlation between attributes
corr_mat = credit_card_data.corr()
corr_mat["Class"].sort_values(ascending = True)
#V4 and V11 tends to have some positive correlation with class. To see the same, lets plot a scatter between the two
attributes = ["Class","V11"]
scatter_matrix(credit_card_data[attributes], figsize=(20,10))
plt.show()
X = credit_card_data.drop(["Class"],axis=1)
Y = credit_card_data["Class"]
credit_card_data.shape
print(X.shape)
print(Y.shape)
#Getting the number of positive and negative instances
outliers = Y[Y==1]
num_outliers = len(outliers)
num_normal = len(Y)-num_outliers
outlier_frac = num_outliers/num_normal
print(len(Y))
print(num_outliers, num_normal)
print(outlier_frac)
#Outlier detection using LOF
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=15,contamination=0.005)
Y_predict = lof.fit_predict(X)

#Having a look at the predictions
print(Y_predict[:20])
print(Y[:20])
#Reshaping the prediction values as per the desired output i.e. 0 for normal 1 for outlier
Y_predict[Y_predict==1]=0
Y_predict[Y_predict==-1]=1
print(Y_predict[4920])
print(Y[4920])
#Getting the negative outlier factor
#Inliers tend to have a LOF score close to 1 (negative_outlier_factor_ close to -1), 
#while outliers tend to have a larger LOF score.

lof_score = lof.negative_outlier_factor_
print(lof_score)
#Getting the number of errors
lof_n_error = (Y_predict!=Y).sum()
print(outliers)
print(lof_n_error)
#Getting the accuracy and classification report
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:{}",format(accuracy_score(Y_predict,Y)))
print("Classification Report:",classification_report(Y_predict,Y))
#Using the isolation forest 
from sklearn.ensemble import IsolationForest

isf = IsolationForest(max_samples=len(X),contamination=outlier_frac,random_state=1)
y_pred = isf.fit_predict(X)
y_pred[y_pred==1]=0
y_pred[y_pred==-1]=1
print(Y_predict[4920])
print(Y[4920])
isf_n_error = (y_pred!=Y).sum()
print(isf_n_error)
print("Accuracy:{}",format(accuracy_score(y_pred,Y)))
print("Classification Report:",classification_report(y_pred,Y))
