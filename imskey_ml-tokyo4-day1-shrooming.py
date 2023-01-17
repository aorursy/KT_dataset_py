# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
 #import basic library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
from IPython.core.display import display
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#set data
data = pd.read_csv("../input/mushrooms.csv")
#data overview
display(data.head())
display(data.describe())
#checking the target & featrues
print("shape of data:{}".format(data.shape))
print("names of columns:\n{}".format(data.columns))
data.isnull().sum()
print("unique value in class data:\n{}".format(data['class'].value_counts()))
x = 8
y = 3
figsize = (20,50)

def hist_dataframe(df,x,y,figsize):
    plt.figure(figsize =figsize)
    for i,j in zip(range(1,len(df.columns)+1),range(len(df.columns)+1)):
        plt.subplot(x,y,i)
        plt.hist(df.iloc[:,j])
        plt.title(df.columns[j])

hist_dataframe(data,x,y,figsize)     
# make features dummires
data = data.drop("veil-type",axis = 1) # delete veli-type 
print(data.columns)
data_dummies = pd.get_dummies(data)
print("Features after get dummies:\n{}".format(data_dummies.columns))
display(data_dummies.head())
# Correlation of dummmies
data_dummies.corr().style.background_gradient().format('{:.2f}')
features = data_dummies.loc[:,'cap-shape_b':]
X = features.values
y = data_dummies["class_e"].values
print("X.shape:{} y.shape:{}".format(X.shape,y.shape))
(data_dummies.loc[:,'cap-shape_b':].columns)
# setteing Logistic Regression model
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
print("Train score:{:.2f}".format(logreg.score(X_train,y_train)))
print("Test  score:{:.2f}".format(logreg.score(X_test,y_test)))
#evaluating the model
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_pred = logreg.predict(X_test)
mse = mean_squared_error(y_pred,y_test)
mae = mean_absolute_error(y_pred,y_test)

print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("confusion matrics:\n{}".format(confusion))
# selecting the value with P-value
from sklearn.feature_selection import SelectPercentile

select = SelectPercentile(percentile=20) 
select.fit(X_train,y_train)

X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)

print(X_train.shape)
print(X_train_selected.shape)

mask =select.get_support()
print(mask)
logreg_selected = LogisticRegression()
logreg_selected.fit(X_train_selected,y_train)
print("train score :{}".format(logreg_selected.score(X_train_selected,y_train)))
print("test score :{}".format(logreg_selected.score(X_test_selected,y_test)))

y_pred_selected = logreg_selected.predict(X_test_selected)
mse_selected = mean_squared_error(y_pred_selected,y_test)
mae_selected = mean_absolute_error(y_pred_selected,y_test)

print("MSE=%s"%round(mse_selected,3) )
print("RMSE=%s"%round(np.sqrt(mse_selected), 3) )
print("MAE=%s"%round(mae_selected,3) )

from sklearn.metrics import confusion_matrix
confusion_selected = confusion_matrix(y_test,y_pred_selected)
print("confusion matrics:\n{}".format(confusion_selected))