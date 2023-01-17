# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(ecommerce-customer-device-usage, Ecommerce Customers))
data= pd.read_csv("../input/ecommerce-customer-device-usage/Ecommerce Customers")
data.head()
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data.describe()
data.info()
%matplotlib inline
data.hist(bins= 8, figsize= (15, 15))
corr_matrix= data.corr()
corr_matrix["Yearly Amount Spent"].sort_values(ascending= False)
from pandas.plotting import scatter_matrix

attributes= ["Yearly Amount Spent","Length of Membership", "Time on App", "Avg. Session Length", "Time on Website"]
scatter_matrix(data[attributes], figsize=(15, 15))
data["Ceil_LoM"]= np.ceil(data["Length of Membership"])
data["Ceil_LoM"].hist(bins= 7)
from sklearn.model_selection import StratifiedShuffleSplit

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state= 42)
for train_index, test_index in split.split(data, data["Ceil_LoM"]):
    strat_train= data.iloc[train_index]
    strat_test= data.iloc[test_index]

#To know the proportion of each data point in the entire data set
data["Ceil_LoM"].value_counts()/ len(data)
#To know the proportion of each data point in the training set
strat_train["Ceil_LoM"].value_counts()/ len(strat_train)
#To know the proportion of each data point in the test set
strat_test["Ceil_LoM"].value_counts()/ len(strat_test)
for set in strat_train, strat_test:
    set.drop(["Ceil_LoM"], axis= 1, inplace= True)
X= strat_train[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]

y= strat_train["Yearly Amount Spent"]

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X, y)

lin_reg.coef_
from sklearn.model_selection import cross_val_score
scores= cross_val_score(lin_reg, X, y, scoring="neg_mean_squared_error", cv=10)
rmse_score= np.sqrt(-scores)

def display_scores(score):
    print("Scores:", score)
    print("Mean:", score.mean())
    print("Standard Deviation:",score.std())
display_scores(rmse_score)
X_test= strat_test[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]

y_test= strat_test["Yearly Amount Spent"] 

predictions= lin_reg.predict(X_test)

plt.scatter(y_test,predictions, )
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
from sklearn import metrics

print('MAE :'," ", metrics.mean_absolute_error(y_test,predictions))
print('MSE :'," ", metrics.mean_squared_error(y_test,predictions))
print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,predictions)))