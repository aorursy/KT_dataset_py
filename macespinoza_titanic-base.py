# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from sklearn import linear_model

from sklearn import model_selection

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_submision = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
df_train.head()
df_train = df_train.fillna(0)

df_test = df_test.fillna(0)
X = df_train.iloc[:, [2,5]]

X_test = df_test.iloc[:, [1,4]]
X.head()
X_test.head()
y = df_train["Survived"].values
model = linear_model.LogisticRegression(random_state=0,multi_class='auto',solver='liblinear')
model.fit(X,y)
predictions = model.predict(X_test)
predictions.shape

df_submision.shape
df_submision.head()
df_submision["Survived"] =predictions
df_submision.to_csv( 'titanic_predA.csv' , index = False )