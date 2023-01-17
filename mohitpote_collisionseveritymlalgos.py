import pandas as pd
from sklearn import (svm, preprocessing)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (recall_score, precision_score, accuracy_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier


from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns

import collections
import numpy as np
from sklearn.decomposition import PCA

train_data = pd.read_csv("../input/Train.csv")
test_data = pd.read_csv("../input/Test.csv")
train_data.info()
test_data.info()
train_data.head(10)
train_data["Collision Severity"].value_counts()
test_data.head(5)
test_data["Policing Area"].value_counts()
train_data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# lable encoder for data_train

le.fit(train_data['Weekday of Collision'])
train_data['Weekday of Collision'] = le.transform(train_data['Weekday of Collision'])
le.fit(train_data['Policing Area'].astype(str))
train_data['Policing Area'] = le.transform(train_data['Policing Area'].astype(str))
train_data.isnull().sum()
train_data = train_data.fillna(train_data.median())
# lable encoder for data_test

le.fit(test_data['Weekday of Collision'])
test_data['Weekday of Collision'] = le.transform(test_data['Weekday of Collision'])
le.fit(test_data['Policing Area'].astype(str))
test_data['Policing Area'] = le.transform(test_data['Policing Area'].astype(str))
test_data = test_data.fillna(test_data.median())

y = train_data["Collision Severity"]
x = train_data.drop(["Collision Severity", "Collision Reference No."], axis=1)

data_test_x = test_data.drop(["Collision Reference No."], axis=1)
data_test_x.shape
x.shape
x.head(10)
train_data.hist(column="Weekday of Collision")
plt.show()
train_data.hist(figsize=(30,20))
plt.show()
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sns.heatmap(train_data.corr())
plt.show()
train_data.hist(column="Collision Severity")
plt.show()
sns.countplot(x="Collision Severity", data=train_data)
plt.show()


# splitting 
data_variables = train_test_split(x, y, test_size = 0.25, random_state = 42)

x_train, x_test, y_train, y_test = data_variables

scaler = preprocessing.StandardScaler().fit(x_train)
train_data_scaled = scaler.transform(x_train)
test_data_scaled = scaler.transform(x_test)

test_data_scaled_x = scaler.transform(data_test_x)


classifier = XGBClassifier()
classifier.fit(train_data_scaled, y_train)



# Predicting test data with train data using classifier
predict_y = classifier.predict(test_data_scaled)

#print(type(predict_y))
a = np.array(predict_y)
print(collections.Counter(a))

accuracy = classifier.score(test_data_scaled, y_test)

precision = precision_score(y_test, predict_y, average='micro')
recall = recall_score(y_test, predict_y, average='micro')

cmatrix = confusion_matrix(y_test, predict_y)
#print("accuracy  : {}".format(accuracy))
print("accuracy[round]  : {}".format(round(accuracy, 3)))
print("precision : {}".format(precision))
print("recall    : {} \n".format(recall))
print("Confusion matrix \n{}\n\n".format(cmatrix))
#with XGBooost algo

xgboost = XGBClassifier()
xgboost.fit(train_data_scaled, y_train)
Y_pred = xgboost.predict(test_data_scaled)
a = np.array(Y_pred)
predict_xgboost = collections.Counter(a)
acc_xgboost = round(xgboost.score(test_data_scaled, y_test) * 100, 2)
#acc_xgboost
print("Accuracy with XGBoost : {}".format(acc_xgboost))
#with decision tree- gini index

decision_tree = DecisionTreeClassifier(min_samples_leaf=1, max_depth=4, criterion= "gini")
decision_tree.fit(train_data_scaled, y_train)
Y_pred = decision_tree.predict(test_data_scaled)
a = np.array(Y_pred)
predict_decision_tree = collections.Counter(a)
acc_decision_tree = round(decision_tree.score(test_data_scaled, y_test) * 100, 2)
#acc_decision_tree
print("Accuracy with Decision Tree_Gini : {}".format(acc_decision_tree))
#with decisiono tree- entropy

decision_tree = DecisionTreeClassifier(min_samples_leaf=1, max_depth=4, criterion= "entropy")
decision_tree.fit(train_data_scaled, y_train)
Y_pred = decision_tree.predict(test_data_scaled)
a = np.array(Y_pred)
predict_decision_tree = collections.Counter(a)
acc_decision_tree = round(decision_tree.score(test_data_scaled, y_test) * 100, 2)
#acc_decision_tree
print("Accuracy with Decision Tree_entropy : {}".format(acc_decision_tree))
#with random forest n=100

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_data_scaled, y_train)
Y_pred = random_forest.predict(test_data_scaled)
a = np.array(Y_pred)
predict_random_forest = collections.Counter(a)
acc_random_forest = round(random_forest.score(test_data_scaled, y_test) * 100, 2)
#acc_random_forest
print("Accuracy with Random Forest : {}".format(acc_random_forest))
#with logistic reg

logreg = LogisticRegression()
logreg.fit(train_data_scaled, y_train)
Y_pred = logreg.predict(test_data_scaled)
a = np.array(Y_pred)
predict_log = collections.Counter(a)
acc_log = round(logreg.score(test_data_scaled, y_test) * 100, 2)
#acc_log
print("Accuracy with Logistic Regression : {}".format(acc_log))

# 

#to aggregate the data

#it took bit long time to build


collision_ref = pd.read_csv('../input/train_collision_ref_no.csv')
ref_no = collision_ref['Collision Reference No.']
base = pd.read_csv('../input/Base_Vehicle_Data.csv')
vehicle_type = dict.fromkeys(['vehicle_type_%s'%i for i in pd.unique(base['Vehicle Type'])],['No'] * collision_ref.shape[0])
collision_ref = pd.concat([collision_ref, pd.DataFrame(vehicle_type)], axis=1)

for index, i in enumerate(ref_no):
    tmp = base.loc[base['Collision Reference No.'] == i]
    unique_type = ['vehicle_type_%s'%j for j in pd.unique(tmp['Vehicle Type'])]
    print(index)
    for k in unique_type:
        collision_ref[k][index] = 'Yes'
        
temp = pd.DataFrame(collision_ref)
temp.to_csv('collision_no123.csv',index=False)

#data with vehicle type (yes-1/no-2) and collision ref no 

data_with_vehicle_type= pd.read_csv("../input/collision_no_to12.csv")
dwvt= data_with_vehicle_type
dwvt.hist(column="Day of Collision")
plt.show()
dwvt.hist(column="vehicle_type_8")
plt.show()
dwvt.hist(column="Collision Reference 2.")
plt.show()
#import seaborn as sns
dwvt['vehicle_type_8'].value_counts()
sns.countplot(x="vehicle_type_8",data=dwvt)
plt.show()



