# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.head()
data.drop('customerID', axis =1, inplace = True)
data.info()
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors = 'coerce')
data.info()
X = data.iloc[:,:-2].values

y = data.loc[:,'Churn'].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
cols = [0,2,3,5,6,7,8,9,10,11,12,13,14,15,16]

for col in cols:

    label_encoder = LabelEncoder()

    X[:,col] = label_encoder.fit_transform(X[:,col])



X
X_1 = X[:,[4,-2,-1]]

X = X[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16]]
onehotencoders = OneHotEncoder(categories='auto', drop = 'first')

X = onehotencoders.fit_transform(X).toarray()

X
X = np.concatenate((X,X_1),axis = 1)

X = np.asarray(X)
y
labelencoder = LabelEncoder()

y = labelencoder.fit_transform(y)

y
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
len(X_test)
from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression(penalty = 'l1')

logclf.fit(X_train,y_train)
logclf.predict(X_test)

log_acc= logclf.score(X_test, y_test)

print(log_acc)
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(random_state = 14)

ada_clf.fit(X_train,y_train)

ada_acc= ada_clf.score(X_test,y_test)

ada_acc
from sklearn import tree
dec_clf = tree.DecisionTreeClassifier(criterion ='entropy', splitter='random', random_state= 53, max_features=8, max_leaf_nodes=20)

dec_clf.fit(X_train,y_train)

dec_acc = dec_clf.score(X_test,y_test)

dec_acc
import keras

from keras.models import Sequential

from keras.layers import Dense



classifier = Sequential()

classifier.add(Dense(output_dim = 30, init='uniform', activation='relu', input_dim = 30))

classifier.add(Dense(output_dim = 50, init='uniform', activation='relu'))

classifier.add(Dense(output_dim = 30, init='uniform', activation='relu'))

classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer = 'adam', loss='binary_crossentropy',metrics = ['accuracy'])

classifier.fit(X_train,y_train, batch_size = 10, nb_epoch=100)
y_pred = classifier.predict(X_test)

y_pred = (y_pred>0.5)

y_pred
cm = confusion_matrix(y_test,y_pred)

cm
ann_acc = classifier.evaluate(X_test, y_test)[1]
print("Logistic Accuracy Score: {}\nAdaBOOST Accuracy Score: {}\nDecision Tree Accuracy Score: {}\nANN Classifier Accuracy Score: {}".format(log_acc,ada_acc,dec_acc,ann_acc))