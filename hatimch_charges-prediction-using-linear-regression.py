%matplotlib inline
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import cross_validation
from sklearn import linear_model,neighbors
import matplotlib.pyplot as plt
import os
df = pd.read_csv('../input/insurance.csv')
df.head()
#redefinig categorial data
from sklearn.preprocessing import LabelEncoder
sex=LabelEncoder()
df['sex']=sex.fit_transform(df['sex'].astype(str))

region=LabelEncoder()
df['region']=region.fit_transform(df['region'].astype(str))

smoker=LabelEncoder()
df['smoker']=smoker.fit_transform(df['smoker'].astype(str))
df.head()
#defining the correlation among the features..It is shown there is a strong relation between smoker-charges and little relation between age-charges

df.corr()
X = np.array(df.drop(['smoker'],1))
y = np.array(df['smoker'])

#pridicting whether the person is smoker or not 

X_train , X_test,y_train , y_test = cross_validation.train_test_split(X,y,test_size=0.2)
    # the linear function
reg_clf = linear_model.LinearRegression()
    # Training
reg_clf.fit(X_train,y_train)
    # Testing
acc = reg_clf.score(X_test,y_test)
acc = acc * 100

print('Linear regression accuracy score: ',acc)

#pridicting whether the person is smoker or not using KNearestNeighbours
    # the KNN function
knn_clf = neighbors.KNeighborsClassifier()
    # Training
knn_clf.fit(X_train,y_train)
    # Testing    
acc_knn = knn_clf.score(X_test,y_test)
acc_knn = acc_knn * 100
print('KNN accuracy score: ',acc_knn)

# predicting charges   using linear regression
XC = np.array(df.drop(['charges'],1))
yC = np.array(df['charges'])
X_trainC , X_testC,y_trainC , y_testC = cross_validation.train_test_split(XC,yC,test_size=0.2)


charges_clf=linear_model.LinearRegression()
charges_clf.fit(X_trainC,y_trainC)
predicted_charges=charges_clf.predict(X_testC)
print(predicted_charges)
charges_accuracy=charges_clf.score(X_testC,y_testC)
print('charges accuracy score using Linear regression: ',charges_accuracy*100)

#try using random forest method
clf_tree = tree.DecisionTreeRegressor()
clf_tree.fit(X_trainC,y_trainC)
charges_accuracy_tree=clf_tree.score(X_testC,y_testC)
print(clf_tree.predict(X_testC))

print('charges accuracy score using Tree regression: ',charges_accuracy_tree*100)
