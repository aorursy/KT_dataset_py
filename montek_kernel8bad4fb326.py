# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn



import os

print(os.listdir("../input"))





data = pd.read_csv("../input/heart.csv")



import matplotlib.pyplot as plt

import seaborn as sns

data.head()





sns.lineplot(x="age", y="cp", hue = "sex", data = data)

plt.show()





sns.lineplot(x="age", y="target", hue = "sex", data = data)

plt.show()





sns.lineplot(x="age", y="thalach", hue = "sex", data=data)

plt.show()



ax = sns.scatterplot(x="age", y = "oldpeak", hue = "sex",data = data)

plt.show()



feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol','fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope','ca', 'thal']

X = data[feature_cols]

y = data.target



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.21,random_state=0)





from sklearn.linear_model import LogisticRegression

one_hot_encoded_training_predictors = pd.get_dummies(X_train)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred=logreg.predict(X_test)





print("Accuracy:",sklearn.metrics.accuracy_score(y_test,y_pred))














