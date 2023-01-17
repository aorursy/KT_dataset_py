# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_frame_train = pd.read_csv('../input/train.csv')

data_frame_test = pd.read_csv("../input/test.csv")
print(data_frame_train.shape)
print(data_frame_train.head(5))
print(data_frame_train.dtypes)
print(data_frame_train.describe())
class_count = data_frame_train.groupby('type').size()

print(class_count)
sns.set()



sns.pairplot(data_frame_train,hue="type")
print(data_frame_test.columns)
df = data_frame_train["type"]

indexes_test = data_frame_test["id"]



data_frame_train = data_frame_train.drop(["type","color","id"],axis=1)

data_frame_test = data_frame_test.drop(["color","id"],axis=1)
data_frame_train = pd.get_dummies(data_frame_train)

data_frame_test = pd.get_dummies(data_frame_test)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(data_frame_train, df, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(penalty='l2',C=1000000)

lr.fit(x_train,y_train)

y_pred= lr.predict(x_test) 



print(classification_report(y_pred,y_test))
y_pred = lr.predict(data_frame_test)



Y = pd.DataFrame()

Y["id"] = indexes_test

Y["type"] = y_pred

Y.to_csv("submission.csv",index=False)