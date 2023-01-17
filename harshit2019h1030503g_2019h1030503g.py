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
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")

df.head()
df.info()
import seaborn as sns

sns.countplot(x="target", data=df)
df.isnull()
#Target data

y = df['target']



X = df.drop(['id','target'],axis=1)
X
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn import tree

model = tree.DecisionTreeClassifier()  

model.fit(X_train, y_train)

y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_predict))

pd.crosstab(y_test, y_predict)
model.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve, roc_auc_score
roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
df_test = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")



df_test.head()
columns = list(df_test.columns.values)



columns.remove('id')



test_data = df_test.loc[:, columns]



Y = model.predict(test_data)



df_test['target'] = Y
submit = df_test.loc[:,['id', 'target']]

submit.head()
submit.to_csv("./submission.csv",index=False)