import pandas as pd

import numpy as np
train_df = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/train.csv.zip')

test_df = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/test.csv.zip')

test_df1 = test_df.copy()
train_df.info()



train_df.color.value_counts()
train_df = train_df.drop(['id'],axis=1)

test_df = test_df.drop(['id'],axis=1)
x = train_df.iloc[:,[0,1,2,3,4]].values

y = train_df.iloc[: , [5]].values



xx = test_df.iloc[:,[0,1,2,3,4]].values
#### for trainng data  #####



from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

x[:,4] = labelencoder.fit_transform(x[:,4])



labelencoder2 = LabelEncoder()

y = labelencoder2.fit_transform(y)





from keras.utils import to_categorical

p = to_categorical(x[:,4])


#### for test data   #####



from sklearn.preprocessing import LabelEncoder

labelencoder1 = LabelEncoder()

xx[:,4] = labelencoder1.fit_transform(xx[:,4])



from keras.utils import to_categorical

pp = to_categorical(xx[:,4])

x = pd.DataFrame(x)

xx = pd.DataFrame(xx)

p = pd.DataFrame(p)

pp = pd.DataFrame(pp)



x = x.drop([4],axis=1)

xx= xx.drop([4],axis =1)



frames = [x,p]

frames1 = [xx,pp]



x= pd.concat(frames,axis=1)

xx = pd.concat(frames1,axis=1)
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.25)
###############     K-Nearest Neighbors     ################



from sklearn.neighbors import KNeighborsClassifier

cf = KNeighborsClassifier(n_neighbors = 5,p=2)

cf.fit(x_train,y_train)



y_pred = cf.predict(x_test)



from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test , y_pred)

print(ac*100)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = cf, X = x_train, y = y_train, cv = 10)

print(accuracies.mean())

print(accuracies.std())

################          SVM       ###########################



from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()

classifier.fit(x_train,y_train)



y_pred=classifier.predict(x_test)



from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test , y_pred)

print(ac*100)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)

print(accuracies.mean())

print(accuracies.std())

################    Random Forest   #######################



from sklearn.ensemble import RandomForestClassifier

classifier1 = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

classifier1.fit(x_train,y_train)



y_pred=classifier1.predict(x_test)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier1, X = x_train, y = y_train, cv = 10)

print(accuracies.mean())

print(accuracies.std())

###############     Logistic Regression     #################



from sklearn.linear_model import LogisticRegression

cf2 = LogisticRegression()

cf2.fit(x_train,y_train)



y_pred = cf2.predict(x_test)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = cf2, X = x_train, y = y_train, cv = 10)

print(accuracies.mean())

print(accuracies.std())
y_pred = cf2.predict(xx)
y_pred_submit = labelencoder2.inverse_transform(y_pred)
submission = pd.DataFrame({'id':test_df1['id'], 'type':y_pred_submit})

submission.to_csv('../working/submission.csv', index=False)