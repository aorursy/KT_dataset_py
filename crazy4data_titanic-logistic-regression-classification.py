# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sb
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
titanic_train_data = pd.read_csv('../input/train.csv') # has a column for survived
titanic_test_data = pd.read_csv('../input/test.csv')  # need to predict survived columnx
#titanic_train_data[['PassengerId','Survived']]
titanic_train_data.keys()
df = pd.DataFrame(titanic_train_data)

feature_list = ['Pclass','Parch','Age']
X = titanic_train_data.loc[:,feature_list]
X['Age'] = X['Age'].fillna(value=0)
X.shape
y =titanic_train_data['Survived']
from sklearn.svm import SVC
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X,y)
X_new = titanic_test_data.loc[:,feature_list]
X_new['Age'] = X_new['Age'].fillna(value=0)
pred_survival = svclassifier.predict(X_new)
pred_survival
pd.DataFrame({'PassengerId':titanic_test_data.PassengerId, 'Survived':pred_survival}).set_index('PassengerId').to_csv('ThirdSubmission.csv')
