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
import seaborn as sns
credit_df = pd.read_csv("/kaggle/input/credit/Default.csv",index_col=0)

credit_df.head()
credit_df.info()
sns.boxplot(y="balance",x="student",data=credit_df,width=0.8)
sns.pairplot(credit_df,hue="student")
credit_df[credit_df["default"] == "Yes"].head()
sns.heatmap(data=credit_df.corr(), annot=True, linewidths=.5, cmap="coolwarm")
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix



modl = LogisticRegression(random_state=10)



# Create feature set

X = credit_df[['balance','income']]

X["student"] = pd.get_dummies(credit_df["student"],drop_first=True)

y = credit_df["default"]



# create a training and a test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)



modl.fit(X_train,y_train)

y_predict = modl.predict(X_test)
print(confusion_matrix(y_test,y_predict))

print(classification_report(y_test,y_predict))
modl.predict_proba(np.array([1500,25000,0]).reshape(1,-1))