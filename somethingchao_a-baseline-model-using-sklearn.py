import numpy as np # linear algebra
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style = "white",color_codes = True)


from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#Load data
iris = pd.read_csv("../input/Iris.csv")
iris.head()
train,test = train_test_split(iris,test_size= 0.3)
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y = train.Species
test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_y = test.Species
for strategy in ['stratified','most_frequent','prior',
               'uniform']:
    
    model_dummy = DummyClassifier(strategy = strategy)
    model_dummy.fit(train_X,train_y)
    print("The %s's accuracy is %f"%(strategy,model_dummy.score(test_X,test_y)))
model_svm = svm.SVC()
model_svm.fit(train_X,train_y)
print("The accuracy of the SVM is",model_svm.score(test_X,test_y))
model_log = LogisticRegression(random_state = 0,solver = 'lbfgs',
                              multi_class = 'multinomial')
model_log.fit(train_X,train_y)
print("Logistic Regression accuracy:",model_log.score(test_X,test_y))
model_tree = DecisionTreeClassifier()
model_tree.fit(train_X,train_y)
model_tree.score(test_X,test_y)
