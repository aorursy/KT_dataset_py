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
['Graduate_Admission.csv']
import pandas as pd
data = pd.read_csv('../input/Graduate_Admission.csv')
data.head()
data.describe()
data.drop('Serial No.',axis = 1, inplace=True)
data.info()
data.columns = ['gre', 'toefl', 'univrating', 'sop', 'lor', 'cgpa', 'research', 'admitchance']
data.head()
data.info()
import seaborn as sns
sns.pairplot(data = data)
targets = data['admitchance']
features = data[['gre','toefl','cgpa']]
targets.head()
features.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,targets,test_size=0.3)
from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    score = r2_score(y_true,y_predict)
    return score
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
reg_fit = regressor.fit(X_train,y_train)
reg_pred = reg_fit.predict(X_test)
print("regression score of :", performance_metric(y_test,reg_pred))
