'''
Author: Ritwik Biswas
Description: Using SVM and Adaboost Classifiers to predict whether a mushroom is edible or poisonous
'''
import numpy as np 
import pandas as pd 
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score #accuracy scoring
import matplotlib.pyplot as plt
%matplotlib inline
import os

df = pd.read_csv('../input/mushrooms.csv')
df.head()
total_size = df['class'].count()
print(total_size)
print("First entry sample:")
print(df.iloc[0])
class_list = []
feature_list = []

#Hash table for numerical encoding of features
num_lookup = {'a': 1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,
              'n':14,'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,'w':23,'x':24,'y':25,'z':26}
class_lookup = {16: 'poisonus', 5: 'edible'}
def encode(vec):
    '''
    Takes an shroom feature vector and encodes it to a numerical feature space
    '''
    encoded_temp = []
    for i in vec:
        try:
            val= num_lookup[i]
        except:
            val = 0
        encoded_temp.append(val)
    return encoded_temp
#Encode discrete features to numerical feature space and split 
for row in df.iterrows():
    index, data = row
    temp = encode(data.tolist())
    class_list.append(temp[0])
    feature_list.append(temp[1:])
    
print("One data point:")
print(class_list[0])
print(feature_list[0])
training_size = int(0.9*total_size)
train_class = np.array(class_list[:training_size])
train_features = np.array(feature_list[:training_size])
test_class = np.array(class_list[training_size:])
test_features = np.array(feature_list[training_size:])
print("Training Length: " + str(len(train_features)))
print("Testing Length: " + str(len(test_features)))
clf_svm = SVC(kernel="rbf",gamma='auto', C=1.0) 
clf_svm.fit(train_features,train_class)
train_score = str(clf_svm.score(train_features,train_class))
test_score = str(clf_svm.score(test_features,test_class))
print ("SVM Model Train Accuracy: " + train_score[:4])
print ("SVM Model Test Accuracy: " + test_score[:4])
base_model_stack = tree.DecisionTreeClassifier(min_samples_split=15)
clf_ada = AdaBoostClassifier(n_estimators=100, base_estimator=base_model_stack)
clf_ada.fit(train_features,train_class)
train_score = str(clf_ada.score(train_features,train_class))
test_score = str(clf_ada.score(test_features,test_class))
print ("Adaboost Model Train Accuracy: " + train_score[:4])
print ("Adaboost Forest Model Test Accuracy: " + test_score[:4])
test_shroom_1 = ['x', 'y', 'w', 't', 'p', 'f', 'c', 'n', 'n', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 's', 'u'] # poisonous
test_shroom_2 = ['x', 's', 'y', 't', 'a', 'f', 'c', 'b', 'k', 'e', 'c', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'n', 'n', 'g'] # edible
test_shroom_3 = ['x', 's', 'g', 'f', 'n', 'f', 'w', 'b', 'k', 't', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'e', 'n', 'a', 'g'] # edible

#functions to do both svm and adaboost prediction
def svm_predict(shroom):
    return class_lookup[clf_svm.predict([shroom])[0]]
def adaboost_predict(shroom):
    return class_lookup[clf_ada.predict([shroom])[0]]

# predictions
print("Shroom 1 Prediction:")
print("SVM: " + svm_predict(encode(test_shroom_1)))
print("Adaboost: " + adaboost_predict(encode(test_shroom_1)))

print("\nShroom 2 Prediction:")
print("SVM: " + svm_predict(encode(test_shroom_2)))
print("Adaboost: " + adaboost_predict(encode(test_shroom_2)))

print("\nShroom 3 Prediction:")
print("SVM: " + svm_predict(encode(test_shroom_3)))
print("Adaboost: " + adaboost_predict(encode(test_shroom_3)))