import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
id=test.iloc[:,0].values

test.drop('id',axis=1)
id
X = train.iloc[:, 1:11].values

y = train.iloc[:, 0].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 99, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
#for calculating accuracy

(131+6090)/(131+6090+92+241)
test.drop(['id'], axis=1, inplace = True)
test.head()
test
test_pred = classifier.predict(test)
test_pred
submission = pd.DataFrame({'Id':id,'Action':test_pred})
submission
final_submission=submission.iloc[0:58921,:].values
final_submission
final_submission =  pd.DataFrame({'Id':final_submission[:,0],'Action':final_submission[:,-1]})
final_submission
filename = 'Amazon Employee Access .csv'



final_submission.to_csv(filename,index=False)



print('Saved file: ' + filename)