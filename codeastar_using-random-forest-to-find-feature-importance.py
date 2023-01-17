import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#get the training data set

df_train = pd.read_csv("../input/train.csv")
df_train.head(5)
df_train_all_num = (df_train.apply(lambda x: pd.factorize(x)[0]))
df_train_all_num.head(5)
#load our model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
#prepare training X, Y data set

train_y = df_train_all_num['Survived']

#drop unused fields

train_x = df_train_all_num.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)      
#thit is how we get the feature importance with simple steps:

model.fit(train_x, train_y)

# display the relative importance of each attribute

importances = model.feature_importances_

#Sort it

print ("Sorted Feature Importance:")

sorted_feature_importance = sorted(zip(importances, list(train_x)), reverse=True)

print (sorted_feature_importance)