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
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.shape,  df_test.shape
df_train.head()
target = df_train["target"]

df_train = df_train.drop(["target","id"], axis=1)
df_test= df_test.drop(["id"], axis=1)
sum(target)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df_train = sc.fit_transform(df_train)
#Split the dataset into train and validation dataset

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(df_train,target, test_size=0.10, random_state=42)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 100)

x_train = lda.fit_transform(x_train,y_train)

x_valid = lda.transform(x_valid)
from sklearn.linear_model import LogisticRegression

m = LogisticRegression(random_state = 0)

m.fit(x_train,y_train)
#predicting the test set

y_pred = classifier.predict(x_valid)
#check confusion metrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_valid)

cm
#Building a single tree

from sklearn.ensemble import RandomForestClassifier

m = RandomForestClassifier(n_estimators=40, n_jobs=-1)



m.fit(x_train, y_train)
y_pred = m.predict(x_valid)
from sklearn.metrics import confusion_matrix

cm =confusion_matrix(y_pred, y_valid)

cm
#worked on test data

df_test = pd.read_csv("../input/test.csv")



#remove Id

x_test = df_test.drop(["id"], axis=1)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_test = sc.fit_transform(x_test)



#lda

x_test = lda.transform(x_test)





#predict the target value

y_test = m.predict(x_test).astype(int)





# Save predictions in format used for competition scoring

output = pd.DataFrame({'id': df_test.id,'target': y_test})

output.to_csv('submission.csv', index=False)

output.head()