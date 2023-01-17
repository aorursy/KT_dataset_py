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
# importing libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
data = pd.read_csv("../input/medium-articles-dataset/medium_data.csv")
data.head()
# Checking max value in claps
data['claps'].max()
data.shape
data.columns
data.info()
data.describe()
data_numeric=data._get_numeric_data()
data_numeric.head()
# Checking for missing value
missing_data=data.isnull()
missing_data.head()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
dummy_variable_1=pd.get_dummies(data['subtitle'])
dummy_variable_1.head()
data=pd.concat([data,dummy_variable_1],axis=1)

data.drop("subtitle",axis=1,inplace=True)
data.head()
dummy_variable_2=pd.get_dummies(data['title'])
dummy_variable_2.head()
data=pd.concat([data,dummy_variable_2],axis=1)
data.drop('title',axis=1,inplace=True)
data.head()
dummy_variable_3=pd.get_dummies(data['image'])
dummy_variable_3.head()
data=pd.concat([data,dummy_variable_3],axis=1)

data.drop("image",axis=1,inplace=True)
data.head()
data.drop(['id','url','responses','reading_time','publication','date'],axis=1,inplace=True)
data.head()
test_data=data['claps']
train_data=data.drop('claps',axis=1)
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(train_data, test_data, test_size=0.30, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
lm.score(x_train,y_train)
lm.score(x_test,y_test)
y_hat_train=lm.predict(x_train)
y_hat_train[0:5]
y_hat_test=lm.predict(x_test)
y_hat_test[0:5]
%%capture
! pip install ipywidgets
from IPython.display import display
from IPython.html import widgets 
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Claps')
    plt.ylabel('')

    plt.show()
    plt.close()
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, y_hat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_test, y_hat_test, "Actual Values (Train)", "Predicted Values (Train)", Title)
