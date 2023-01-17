#import numpy as np

import pandas as pd

#import sklearn

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import train_test_split

from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression

from subprocess import check_output



print(check_output(["ls", "../input"]).decode("utf8"))



# Import Data

X = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print(test_df.iloc[0][0])

X_testFinal = test_df.drop('id', axis=1).copy()

X = X.drop('id', axis=1)



#separate training features and training result classes

y = X.target.values

y = LabelEncoder().fit_transform(y)

X = X.drop('target', axis=1)



# Split Train / Test

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=0)



model = LogisticRegression()

model = model.fit(Xtrain, ytrain)



# check the accuracy on the training set

model.score(Xtrain, ytrain)

predicted = model.predict(Xtest)

ypreds = model.predict_proba(Xtest)



y_testFinal = model.predict_proba(X_testFinal)

#print(y_testFinal[0])



y_pd = pd.DataFrame(y_testFinal)



finalTable = pd.DataFrame({ "id": test_df["id"]})



#to feed results into otto's output format

index = 0

classList = range(1, 10)

for i in classList:

    finalTable[str("Class_{}".format(i))] = y_pd.iloc[:,index]

    index += 1



#convert to csv format

finalTable.to_csv('otto.csv', index=False)