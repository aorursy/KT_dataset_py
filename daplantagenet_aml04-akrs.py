!pip install autokeras
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import autokeras as ak
df = pd.read_csv("../input/bank-marketing/bank-additional-full.csv", sep=";") # full data set

df.shape
from sklearn.preprocessing import LabelEncoder



lb = LabelEncoder() 

df['y'] = lb.fit_transform(df['y'])
y = df[df.columns[-1:]] # all rows, last col. to create target dataseries

X = df[df.columns[:-1]] # training features only as dataframe

X
y # check the target
# trn & tst splits (stratify to preserve imblanced class distribution)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    stratify=y, 

                                                    test_size=0.2, 

                                                    random_state=42)
train = pd.concat([X_train, y_train], axis = 1)

test = pd.concat([X_test, y_test], axis = 1)
# remove intermediate DFs

del X

del y 

del X_train 

del X_test 

del y_train 

del y_test
train.to_csv('bank_train.csv')

test.to_csv('bank_test.csv')
!ls
# Initialize the structured data classifier.

# It tries 10 different models...

clf = ak.StructuredDataClassifier(max_trials=3) 

# Feed the structured data classifier with training data.

clf.fit(

    # The path to the train.csv file.

    'bank_train.csv',

    # The name of the label column.

    'y')
# Predict with the best model.

predicted_y = clf.predict('bank_test.csv') 

predicted_y
# Evaluate the best model with testing data.

evl = clf.evaluate(x='bank_test.csv', y='y')

evl
clf