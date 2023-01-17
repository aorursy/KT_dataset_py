import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
iris_dataset = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
iris = iris_dataset.copy()
iris.tail()
def basic_details(data):
    print(f"shape of dataset:{data.shape}\n Number of rows:{iris.shape[0]} \n Number of Coulmns:{iris.shape[1]}")
    print(f"Coolumn names:{iris.columns}")
    print(f"Data types:{iris.dtypes}")
    
basic_details(iris)
iris['species'].value_counts()
# Tabular method
def UniVariate_tabular(data):
    print("Central tendency:\n",iris.describe())
UniVariate_tabular(iris)
#Graphical Method
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
def UniVariate_graphical(dataset):
    data_types = dataset.select_dtypes([np.int, np.float])
    for i, col in enumerate(data_types.columns):
        plt.figure(i)
        sns.boxplot(x = col,orient='v',width=0.4,data = data_types)
UniVariate_graphical(iris)  
sns.set_style('darkgrid')
sns.scatterplot(x="sepal_length", y="sepal_width", data=iris)
sns.set_style('darkgrid')
sns.scatterplot(x="sepal_length", y="sepal_width",hue='species',data=iris)
sns.set_style('darkgrid')
sns.pairplot(iris, hue='species',height=3,corner= True)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def change_label(col):
    if col == 'Iris-virginica':
        return 1
    elif col == 'Iris-setosa':
        return 2
    elif col == 'Iris-versicolor':
        return 3
iris['species'] = iris['species'].apply(lambda x : change_label(x))
x = iris.iloc[:,:-1] # features
y = iris.iloc[:,-1:] # labels
#print(type(y))
#print(y.head())
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state = 42)

#print(train_x.head())
#print(train_y.head())
#print(test_x.head())
#print(test_y.head())
def model_used(train_x,train_y,test_x,test_y):
    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(train_x,train_y.values.ravel())
    prediction_knn = model_knn.predict(test_x)
    print("prediction_knn",prediction_knn)
    print('The accuracy of the KNN is',accuracy_score(prediction_knn,test_y))
    print("*******Logistic Regression**********")
    model_lr = LogisticRegression()
    model_lr.fit(train_x,train_y.values.ravel())
    prediction_lr = model_lr.predict(test_x)
    print("prediction Logistic Regression",prediction_lr)
    print('The accuracy of the Logistic Regression is',accuracy_score(prediction_lr,test_y))
    
model_used(train_x,train_y,test_x,test_y)
    
    
    
sns.set_style('darkgrid')
sns.scatterplot(x="petal_length", y="petal_width",hue='species',data=iris_dataset)
iris_dataset.species.value_counts()
x_p = iris.iloc[:,[2,3]]
print(x_p.head())
y_p = iris.iloc[:,-1:]
print(y_p.head())
xp_train,xp_test,yp_train,yp_test = train_test_split(x_p,y_p,test_size=0.3)

def model_used_only_petal(xp_train,yp_train,xp_test,yp_test):
    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(xp_train,yp_train.values.ravel())
    prediction_knn = model_knn.predict(xp_test)
    print("prediction_knn",prediction_knn)
    print('The accuracy of the KNN is',accuracy_score(prediction_knn,yp_test))
    print("*******Logistic Regression**********")
    model_lr = LogisticRegression()
    model_lr.fit(xp_train,yp_train.values.ravel())
    prediction_lr = model_lr.predict(xp_test)
    print("prediction Logistic Regression",prediction_lr)
    print('The accuracy of the Logistic Regression is',accuracy_score(prediction_lr,yp_test))
    
model_used_only_petal(xp_train,yp_train,xp_test,yp_test)
    
    
