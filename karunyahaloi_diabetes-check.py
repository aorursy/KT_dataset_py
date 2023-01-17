# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.isnull().sum()
X = data.iloc[: , :-1]
y = data.Outcome
X.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=15)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
model = LogisticRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

y_test
y_predict
confusion_matrix(y_test,y_predict)
pd.crosstab(y_test,y_predict)
sns.set(font_scale=1.5)
cm = confusion_matrix(y_predict, y_test)
sns.heatmap(cm, annot=True, fmt='g')
plt.show()
data.corr()
print(data["Outcome"].value_counts())
data["Outcome"].value_counts().plot(kind="bar")
