import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix 
pulsar_data = pd.read_csv("../input/pulsar_stars.csv")

pulsar_data.head()
#Preparing the data set

data_all = list(pulsar_data.shape)[0]

data_categories = list(pulsar_data['target_class'].value_counts())



print("The dataset has {} diagnosis, {} not star and {} star.".format(data_all, 

                                                                                 data_categories[0], 

                                                                                 data_categories[1]))
X = pulsar_data.iloc[:, 1:-1].values

y = pulsar_data.iloc[:, 8].values
#Creating training and Test Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 5)
#Fitting Random forest Classifier

clf=RandomForestClassifier(n_estimators=600)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("Percentage of error:",np.mean(y_pred != y_test))

print("Confusion Matrix")

print(confusion_matrix(y_test, y_pred))  

print("Classification Report")

print(classification_report(y_test, y_pred)) 