# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



dataset = pd.read_csv("../input/train.csv")

dataset.shape

# Any results you write to the current directory are saved as output.

print(dataset.head())
#Let's try the naive bayes algorithm



from sklearn.ensemble import RandomForestClassifier

shape = dataset.shape

X_train = dataset.iloc[:,1:shape[1]]

y_train = dataset.loc[:,"label"]

testdataset = pd.read_csv("../input/test.csv")

model = RandomForestClassifier()



model.fit(X_train, y_train)

predictions = model.predict(testdataset)
IDs = testdataset.index.values + 1

results = pd.DataFrame({

    "ImageID":IDs,

    "Label":predictions

})

results.to_csv("randomforest.csv", index = False)