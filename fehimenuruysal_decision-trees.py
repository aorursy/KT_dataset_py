# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix , accuracy_score ,classification_report

import xgboost as xgb

from xgboost import plot_importance , plot_tree

import graphviz

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/iris/Iris.csv")

df.head()
df.describe().T
df.info()
df.groupby(by="Species").agg(["mean" , "min" , "max"])
sns.scatterplot(x = "SepalLengthCm" , y = "SepalWidthCm" , hue = "Species" , data = df);
sns.pairplot(df , hue="Species");
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])
df["Species"].value_counts()
df.drop("Id" , inplace=True , axis=1)
df.head()
X, y = df.iloc[: , :-1] , df.iloc[: , -1]
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.30)
dmatrix_train = xgb.DMatrix(data = X_train , label = y_train)

dmatrix_test = xgb.DMatrix(data = X_test , label = y_test)
param = {"max_depth" :5,

         "eta" : 1 ,

         "objective" : "multi:softprob",

        "num_class" : 3}

num_round = 13

model = xgb.train(param , dmatrix_train , num_round)
y_pred = model.predict(dmatrix_test)
best_preds = np.asarray([np.argmax(line) for line in y_pred])
best_preds
accuracy_score(y_test , best_preds)
cm = confusion_matrix(y_test , best_preds)
sns.heatmap(cm, annot = True , cbar=False)
print(classification_report(y_test , best_preds))
xgb.plot_importance(model);
xgb.plot_tree(model);