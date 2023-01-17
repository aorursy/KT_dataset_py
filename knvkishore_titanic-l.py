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
train_data= pd.read_csv('/kaggle/input/titanic/train_data.csv',index_col = 0)
test_data = pd.read_csv('/kaggle/input/titanic/test_data.csv', index_col = 0)
train_data.head()
train_data.drop(columns = ['PassengerId'],inplace = True)
test_data.drop(columns = ['PassengerId'], inplace = True)
train_data.head()
import seaborn as sns
print('imported seaborn')
sns.pairplot(train_data)
y_train = pd.DataFrame(train_data.Survived)
y_test = pd.DataFrame(test_data.Survived)

x_train = train_data.drop(columns = ['Survived'])
x_test = test_data.drop(columns = ['Survived'])
x_test.head()
from sklearn import linear_model
from sklearn import metrics
print('linear_model and metrics are imported .')
model = linear_model.LogisticRegression()
print('made a Logistic Regression model imported from scikit-learn')
model.fit(x_train,y_train)
print('model train')
y_predict=model.predict(x_test)
x_test.shape
y_predict.shape
print('accuracy = ', metrics.accuracy_score(y_predict,y_test))
