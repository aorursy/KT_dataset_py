#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler #scaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#read tranning , test set
traindf = pd.read_csv('../input/train.csv')
testdf = pd.read_csv('../input/test.csv')
#preprocessing
traindf = traindf.drop(['idhogar','tamhog'], axis=1)
traindf['v18q1'] = traindf['v18q1'].fillna(0.0)
traindf['meaneduc'] = traindf['meaneduc'].fillna(0.0)
traindf['SQBmeaned'] = traindf['SQBmeaned'].fillna(0.0)
traindf['rez_esc'] = traindf['rez_esc'].fillna(0.0)
traindf['dependency'] = traindf['dependency'].replace('yes',1)
traindf['dependency'] = traindf['dependency'].replace('no',0)
traindf['edjefe'] = traindf['edjefe'].replace('yes',1)
traindf['edjefe'] = traindf['edjefe'].replace('no',0)
traindf['edjefa'] = traindf['edjefa'].replace('yes',1)
traindf['edjefa'] = traindf['edjefa'].replace('no',0)

testdf = testdf.drop(['idhogar','tamhog'], axis=1)
testdf['v18q1'] = testdf['v18q1'].fillna(0)
testdf['meaneduc'] = testdf['meaneduc'].fillna(0.0)
testdf['SQBmeaned'] = testdf['SQBmeaned'].fillna(0.0)
testdf['rez_esc'] = testdf['rez_esc'].fillna(0.0)
testdf['dependency'] = testdf['dependency'].replace('yes',1)
testdf['dependency'] = testdf['dependency'].replace('no',0)
testdf['edjefe'] = testdf['edjefe'].replace('yes',1)
testdf['edjefe'] = testdf['edjefe'].replace('no',0)
testdf['edjefa'] = testdf['edjefa'].replace('yes',1)
testdf['edjefa'] = testdf['edjefa'].replace('no',0)

traindf['v2a1'] = traindf['v2a1'].fillna(0.0)
testdf['v2a1'] = testdf['v2a1'].fillna(0.0)
#arrange tranning set
X_t = testdf.iloc[:,1:140].values
Y_id = testdf.iloc[:,0].values

X = traindf.iloc[:,1:140].values
y =  traindf.iloc[:,140].values
#more preprocessing
#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#scaler2 = StandardScaler()
#X_t = scaler2.fit_transform(X_t)
#create model
#knn = KNeighborsClassifier(n_neighbors=8, leaf_size=60)
#knn.fit(X, y)
#y_pred = knn.predict(X_t)
classifier = GaussianNB ()
classifier.fit (X, y)
y_pred = classifier.predict(X_t)
#write output file
odf = pd.DataFrame({'Id':Y_id, 'Target': y_pred})
odf.to_csv('submission.csv', index=False)