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
import pandas as pd
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data.head()
data.isnull().sum()
data.describe()
data.groupby(["Outcome"]).count()
import seaborn as sns
sns.pairplot(data, hue='Outcome')
x = pd.DataFrame(data.iloc[:,:7])
y = pd.DataFrame(data.iloc[:,8])
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 7)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
model_score = model.score(x_test,y_test)
print(model_score)
from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))
# Analyzing the confusion matrix
# True +ve, we correctly predicted that they do have diabetes 45
# True -ve, we correctly predicted that they don't have diabetes 131
# False +ve, we incorrectly predicted that they do have diabetes 16 [Falsely predicted]
# False -ve, we incorrectly predicted that they don't have diabetes 39 [Falsely predicted]
