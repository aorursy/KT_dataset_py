import numpy as np
import pandas as pd
import os
# print(os.listdir("../input"))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load train dataset from Chau
train = pd.read_csv('../input/dataset-copd-5-training.csv')

# Config
features = ['ho','khac_dam','kho_khe','kho_tho','nang_nguc','mrc','thuoc_la','mui_hong','tim','phoi']
y = train['nhom'].copy()
X = train[features].copy()
X = pd.get_dummies(X) #one hot code encoding
first_imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
X = pd.DataFrame(first_imputer.fit_transform(X))
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=1,test_size=0.2)

# Training
# def AccuracyTracker(Xtrain,Xtest,ytrain,ytest,n):
#     model = DecisionTreeClassifier(max_leaf_nodes=n,random_state=1)
#     model.fit(Xtrain,ytrain)
#     print(n,accuracy_score(ytest,model.predict(Xtest)))
# for i in range(2,20):
#     AccuracyTracker(Xtrain,Xtest,ytrain,ytest,i)

modeltree = DecisionTreeClassifier(max_leaf_nodes=4,random_state=1)
modeltree.fit(Xtrain,ytrain)
print("--")
print("(Test_size: 20%, Train_size: 80%) from dataset-copd-5-training.csv")
print("Decision Tree Classifier's accuracy: ", accuracy_score(ytest,modeltree.predict(Xtest)))
print("--")

# Load test dataset
test = pd.read_csv('../input/dataset-copd-5-test.csv')
Y_TestDataset = test['nhom'].copy()
X_TestDataset = test[features].copy()
print("(Train_size: 100%) from dataset-copd-5-training.csv")
print("(Test_size: 100%) from dataset-copd-5-test.csv")
print("Decision Tree Classifier's accuracy: ", accuracy_score(Y_TestDataset,modeltree.predict(X_TestDataset)))

tree.plot_tree(modeltree,
               feature_names = features,
               filled = True);