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
path = "/kaggle/input/pima-indians-diabetes-database/diabetes.csv"
df = pd.read_csv(path)
df.head()
X_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction','Age']
X = df[X_features]
y = df.Outcome
X.head()
from sklearn.impute import SimpleImputer

imputer =SimpleImputer(missing_values=0, strategy='mean')
X = imputer.fit_transform(X)
X


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state =0)
from sklearn.tree import DecisionTreeRegressor
model =DecisionTreeRegressor() 
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
 
from sklearn import metrics
print("Accuracy={0:.3f}".format(metrics.accuracy_score(y_test,y_predict)))

