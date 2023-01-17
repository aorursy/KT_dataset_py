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



data = pd.read_csv('/kaggle/input/wisconsin-breast-cancer-cytology-features/wisconsin_breast_cancer.csv')

data.head()



new_data = data.dropna()

new_data.info()
import seaborn as sns

sns.pairplot(data=new_data, hue="class", palette="Set2" ,diag_kind="hist")
X = new_data.iloc[0:683 , 1:10]

print(X)

y = new_data.iloc[0:683 , 10:11]

print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1522492)
from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)





pred = model.predict(X_test)

print(pred[0:10])

print(y_test[0:10])
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,pred))
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))