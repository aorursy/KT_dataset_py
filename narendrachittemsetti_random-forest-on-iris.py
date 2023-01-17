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
df=pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df.head()
df.shape
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df["species"]=le.fit_transform(df["species"])
df["species"].value_counts()
from sklearn.model_selection import train_test_split

x=df[["sepal_length","sepal_width","petal_length","petal_width"]]

y=df["species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100)

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
from sklearn import metrics

print("Accuracy on training data:",metrics.accuracy_score(y_pred,y_test))
clf.predict([[8,6,4,2]])
col=df.columns[:4]

col
feature_imp = pd.Series(clf.feature_importances_,index=col).sort_values(ascending=False)

feature_imp
import seaborn as sns

sns.barplot(x=feature_imp,y=feature_imp.index)

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.legend()

plt.show()
x=df[["petal_width","petal_length","sepal_length"]]

y=df["species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
m2=RandomForestClassifier(n_estimators=100)

m2.fit(x_train,y_train)

y_pred1=m2.predict(x_test)
print("accuracy after eleminating less important parameter:",metrics.accuracy_score(y_test,y_pred1))