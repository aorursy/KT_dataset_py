%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import svm
letters_df = pd.read_csv("../input/letterdata.csv")
letters_df.head(10)   # all fields except the target ("letter") are numeric. We do not know the scale. So normalize
#Prepare X, y

X, y = letters_df.drop(columns = 'letter'), letters_df.loc[:,'letter'] 
#Should always be written in this format else will throw shape warning

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
clf = svm.SVC(gamma=0.025, C=3)    

# gamma is a measure of influence of a data point. It is inverse of distance of influence. C is complexity of the model

# lower C value creates simple hyper surface while higher C creates complex surface
clf.fit(X_train , y_train)
clf.score(X_test, y_test)
#Predict y_pred values given X_test and stack y_test, y_pred

y_pred = clf.predict(X_test)
y_grid = (np.column_stack([y_test, y_pred]))
print(y_grid)
#Display all the letters actual and predicted

pd.set_option('display.max_columns', 26)



pd.crosstab(y_pred, y_test)
#Lets find all the letters which were incorrectly predicted

unmatched = []

for i in range(len(y_grid)):

    if y_grid[i][0] != y_grid[i][1]:

        unmatched.append(i)
y_grid[unmatched]
#np.savetxt("Text", y_grid , fmt='%s')