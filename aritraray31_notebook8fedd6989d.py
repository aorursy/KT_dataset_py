from matplotlib import pyplot as plt
SUBJECTS=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]
MARKS=[86,83,86,90,88] 
tick_label=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]  
plt.bar(SUBJECTS,MARKS,tick_label=tick_label,width=0.8,color=['green','red','green','green','green'])   
plt.xlabel('SUBJECTS') 
plt.ylabel('MARKS')  
plt.title("STUDENT's MARKS DATASET")
plt.show()
arr1 = []
a = int(input("Size of array:"))
for i in range(a):
    arr1.append(float(input("Element:")))
arr1 = np.array(arr1)
arr2 = []
a = int(input("Size of array:"))
for i in range(a):
    arr2.append(float(input("Element:")))
arr2 = np.array(arr2)
# Checking if arr1 has views to arr2 memmory
print(arr1.base is arr2)
# Checking if arr2 has views to arr1 memmory
print(arr2.base is arr1)
for i in arr1.data:
    if i % 3 == 0:
        print (f"{i} is divisible by 3")
    else:
        print (f"{i} is not divisible by 3")
for i in arr2.data:
    if i % 3 == 0:
        print (f"{i} is divisible by 3")
    else:
        print (f"{i} is not divisible by 3")
        print(np.sort(arr2))
    sum = np.sum(arr1)
print(f"The sum of all elements in arr1 is {sum}")

import numpy as np 
import pandas as pd
iris = pd.read_csv('../input/iriscsv/Iris.csv')
iris.drop('Id', axis=1, inplace=True)
iris.head()
print(iris["Species"].value_counts())
print()
print(iris.info())
print()
print(iris.describe())
def _split_iris_dataset(iris):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    y = iris['Species']
    X = iris.drop('Species', axis=1)

    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.33, random_state=42)
    return (X_train, X_test, y_train, y_test)

def _evaluate_iris_classifier(feature, clf, dataset):
    X_train, X_test, y_train, y_test = _split_iris_dataset(dataset)
    
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import f1_score

    y_train_feature = (y_train == feature)
    y_test_feature = (y_test == feature)

    predict = cross_val_predict(clf, X_train, y_train_feature, cv=3)
    print("confusion matrix on", feature)
    print(confusion_matrix(y_train_feature, predict))
    print("percision:", precision_score(y_train_feature, predict))
    print("recall:", recall_score(y_train_feature, predict))
    print("f1 score:", f1_score(y_train_feature, predict))
    from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)

print("---------- Original iris ----------")
_evaluate_iris_classifier('Iris-versicolor', sgd_clf, iris)
print()
print("---------- Enhanced iris ----------")
iris_sgd = iris.copy()
iris_sgd['SepalWidth_PetalLength'] = iris_sgd['SepalWidthCm'] / iris_sgd['PetalLengthCm']
iris_sgd['SepalWidth_PetalWidth'] = iris_sgd['SepalWidthCm'] / iris_sgd['PetalWidthCm']
_evaluate_iris_classifier('Iris-versicolor', sgd_clf, iris_sgd)
import pandas as pd
titanic_data=pd.read_csv("../input/titanic/train_and_test2.csv")
print("TITANIC DATASET : ")
print(titanic_data.head())
print("TITANIC DATASET SHAPE : ",titanic_data.shape)
print(titanic_data.shape)

titanic_data.dropna(axis=1, how='all')
print("__\nTITANIC DATASET : ")
print(titanic_data.head())
print("TITANIC DATASET SHAPE : ",titanic_data.shape)
print(titanic_data.shape)

print("__\nMean value of first 50 samples: \n",titanic_data[:50].mean())

print("__\nMean of the number of male passengers( Sex=1) on the ship :\n",titanic_data[titanic_data['Sex']==1].mean())

print("__\nHighest fare paid by any passenger: ",titanic_data['Fare'].max())