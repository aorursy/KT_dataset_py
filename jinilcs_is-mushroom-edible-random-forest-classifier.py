import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.ensemble import RandomForestClassifier



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelBinarizer
#Import the data

df = pd.read_csv('../input/mushrooms.csv')



df.head(3)
df.info()
df['class'].value_counts()
#Get features and target

df_X = df[df.columns[1:]]

df_y = df['class']
#Encode the categorical data into numercal values. The below line will help to do one hot encoding 

#for all columns

df_X_encoded = pd.get_dummies(df_X)
#Split data into training and test set

X_train, X_test, y_train, y_test = train_test_split(df_X_encoded, df_y)
#Random forest classifier

clf = RandomForestClassifier(n_estimators=5, random_state=0)

cross_val_score(clf, X_train, y_train, cv=5)
#Check score on test data

clf.fit(X_train, y_train)

clf.score(X_test, y_test)