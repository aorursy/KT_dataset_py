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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
diabetes = pd.read_csv('/kaggle/input/diabetes-dataset/diabetes2.csv')
diabetes.head()
diabetes.tail()
diabetes.info()
diabetes.isnull()
sns.heatmap(diabetes)
sns.heatmap(diabetes.isnull())
#splitting data

from sklearn.model_selection import train_test_split
X= diabetes[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
Y= diabetes['Outcome']
sns.pairplot(diabetes , height=4, kind="reg",markers=".")
X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size=0.3 , random_state=101)
from sklearn.linear_model import LogisticRegression

clf= LogisticRegression()
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import classification_report
print (classification_report(Y_test, predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, predictions)
