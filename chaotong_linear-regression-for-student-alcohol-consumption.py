# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
student = pd.read_csv("../input/student-mat.csv")
student.describe()
student.head()
list(student.columns)
student.isnull()
import seaborn as sns
b = sns.factorplot(x="school", y="age", hue="sex", data=student, kind="bar", palette="muted" )
jp = sns.jointplot(x="studytime", y="absences", data=student)
sns.distplot(student.G1)
from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_predict

import matplotlib.pyplot as plt
X,y = student.iloc[:,:-3],student.iloc[:,-2]
le=LabelEncoder()

for col in X.columns.values:

    if X[col].dtypes=='object':

        le.fit(X[col].values)

        X[col]=le.transform(X[col])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)
predicted = cross_val_predict(regr, X_test, y_test, cv=10)



fig, ax = plt.subplots()

ax.scatter(y_test, predicted)

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()